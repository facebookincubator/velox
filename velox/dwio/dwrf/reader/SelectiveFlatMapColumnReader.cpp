/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/dwio/dwrf/reader/SelectiveFlatMapColumnReader.h"

#include "velox/dwio/common/FlatMapHelper.h"
#include "velox/dwio/dwrf/reader/SelectiveDwrfReader.h"
#include "velox/dwio/dwrf/reader/SelectiveStructColumnReader.h"

namespace facebook::velox::dwrf {

namespace {

template <typename T>
dwio::common::flatmap::KeyValue<T> extractKey(const proto::KeyInfo& info) {
  return dwio::common::flatmap::KeyValue<T>(info.intkey());
}

template <>
inline dwio::common::flatmap::KeyValue<StringView> extractKey<StringView>(
    const proto::KeyInfo& info) {
  return dwio::common::flatmap::KeyValue<StringView>(
      StringView(info.byteskey()));
}

template <typename T>
std::string toString(const T& x) {
  if constexpr (std::is_same_v<T, StringView>) {
    return x;
  } else {
    return std::to_string(x);
  }
}

template <typename T>
dwio::common::flatmap::KeyPredicate<T> prepareKeyPredicate(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    StripeStreams& stripe) {
  auto& cs = stripe.getColumnSelector();
  const auto expr = cs.getNode(requestedType->id)->getNode().expression;
  return dwio::common::flatmap::prepareKeyPredicate<T>(expr);
}

// Represent a branch of a value node in a flat map.  Represent a keyed value
// node.
template <typename T>
struct KeyNode {
  dwio::common::flatmap::KeyValue<T> key;
  uint32_t sequence;
  std::unique_ptr<dwio::common::SelectiveColumnReader> reader;
  std::unique_ptr<BooleanRleDecoder> inMap;

  KeyNode(
      const dwio::common::flatmap::KeyValue<T>& key,
      uint32_t sequence,
      std::unique_ptr<dwio::common::SelectiveColumnReader> reader,
      std::unique_ptr<BooleanRleDecoder> inMap)
      : key(key),
        sequence(sequence),
        reader(std::move(reader)),
        inMap(std::move(inMap)) {}
};

template <typename T>
std::vector<KeyNode<T>> getKeyNodes(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    DwrfParams& params,
    common::ScanSpec& scanSpec,
    bool asStruct) {
  using namespace dwio::common::flatmap;

  std::vector<KeyNode<T>> keyNodes;
  std::unordered_set<size_t> processed;

  auto& requestedValueType = requestedType->childAt(1);
  auto& dataValueType = dataType->childAt(1);
  auto& stripe = params.stripeStreams();
  auto keyPredicate = prepareKeyPredicate<T>(requestedType, stripe);

  std::shared_ptr<common::ScanSpec> keysSpec;
  std::shared_ptr<common::ScanSpec> valuesSpec;
  if (!asStruct) {
    if (auto keys = scanSpec.childByName(common::ScanSpec::kMapKeysFieldName)) {
      keysSpec = scanSpec.removeChild(keys);
    }
    if (auto values =
            scanSpec.childByName(common::ScanSpec::kMapValuesFieldName)) {
      valuesSpec = scanSpec.removeChild(values);
    }
  }

  std::unordered_map<KeyValue<T>, common::ScanSpec*, KeyValueHash<T>>
      childSpecs;
  if (asStruct) {
    for (auto& c : scanSpec.children()) {
      if constexpr (std::is_same_v<T, StringView>) {
        childSpecs[KeyValue<T>(StringView(c->fieldName()))] = c.get();
      } else {
        childSpecs[KeyValue<T>(c->subscript())] = c.get();
      }
    }
  }

  // Load all sub streams.
  // Fetch reader, in map bitmap and key object.
  auto streams = stripe.visitStreamsOfNode(
      dataValueType->id, [&](const StreamInformation& stream) {
        auto sequence = stream.getSequence();
        // No need to load shared dictionary stream here.
        if (sequence == 0 || processed.count(sequence) > 0) {
          return;
        }
        EncodingKey seqEk(dataValueType->id, sequence);
        const auto& keyInfo = stripe.getEncoding(seqEk).key();
        auto key = extractKey<T>(keyInfo);
        // Check if we have key filter passed through read schema.
        if (!keyPredicate(key)) {
          return;
        }
        common::ScanSpec* childSpec;
        if (auto it = childSpecs.find(key);
            it != childSpecs.end() && !it->second->isConstant()) {
          childSpec = it->second;
        } else if (asStruct) {
          // Column not selected in 'scanSpec', skipping it.
          return;
        } else {
          if (keysSpec && keysSpec->filter() &&
              !common::applyFilter(*keysSpec->filter(), key.get())) {
            return; // Subfield pruning
          }
          childSpec =
              scanSpec.getOrCreateChild(common::Subfield(toString(key.get())));
          childSpec->setProjectOut(true);
          childSpec->setExtractValues(true);
          if (valuesSpec) {
            *childSpec = *valuesSpec;
          }
          childSpecs[key] = childSpec;
        }
        auto inMap =
            stripe.getStream(seqEk.forKind(proto::Stream_Kind_IN_MAP), true);
        VELOX_CHECK(inMap, "In map stream is required");
        auto inMapDecoder = createBooleanRleDecoder(std::move(inMap), seqEk);
        DwrfParams childParams(
            stripe, FlatMapContext(sequence, inMapDecoder.get()));
        auto reader = SelectiveDwrfReader::build(
            requestedValueType, dataValueType, childParams, *childSpec);
        keyNodes.emplace_back(
            key, sequence, std::move(reader), std::move(inMapDecoder));
        processed.insert(sequence);
      });

  VLOG(1) << "[Flat-Map] Initialized a flat-map column reader for node "
          << dataType->id << ", keys=" << keyNodes.size()
          << ", streams=" << streams;

  return keyNodes;
}

template <typename T>
class SelectiveFlatMapAsStructReader : public SelectiveStructColumnReaderBase {
 public:
  SelectiveFlatMapAsStructReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec,
      const std::vector<std::string>& /*keys*/)
      : SelectiveStructColumnReaderBase(
            requestedType,
            dataType,
            params,
            scanSpec),
        keyNodes_(
            getKeyNodes<T>(requestedType, dataType, params, scanSpec, true)) {
    VELOX_CHECK(
        !keyNodes_.empty(),
        "For struct encoding, keys to project must be configured");
    children_.resize(keyNodes_.size());
    for (int i = 0; i < keyNodes_.size(); ++i) {
      keyNodes_[i].reader->scanSpec()->setSubscript(i);
      children_[i] = keyNodes_[i].reader.get();
    }
  }

 private:
  std::vector<KeyNode<T>> keyNodes_;
};

template <typename T>
class SelectiveFlatMapReader : public SelectiveStructColumnReaderBase {
 public:
  SelectiveFlatMapReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveStructColumnReaderBase(
            requestedType,
            dataType,
            params,
            scanSpec),
        // Copy the scan spec because we need to remove the children.
        structScanSpec_(scanSpec) {
    scanSpec_ = &structScanSpec_;
    keyNodes_ =
        getKeyNodes<T>(requestedType, dataType, params, structScanSpec_, false);
    std::sort(keyNodes_.begin(), keyNodes_.end(), [](auto& x, auto& y) {
      return x.sequence < y.sequence;
    });
    childValues_.resize(keyNodes_.size());
    copyRanges_.resize(keyNodes_.size());
    children_.resize(keyNodes_.size());
    for (int i = 0; i < keyNodes_.size(); ++i) {
      children_[i] = keyNodes_[i].reader.get();
    }
    if (auto type = requestedType_->type->childAt(1); type->isRow()) {
      for (auto& vec : childValues_) {
        vec = BaseVector::create(type, 0, &memoryPool_);
      }
    }
  }

  bool useBulkPath() const override {
    return false;
  }

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override {
    numReads_ = scanSpec_->newRead();
    prepareRead<char>(offset, rows, incomingNulls);
    auto* mapNulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    for (auto* reader : children_) {
      advanceFieldReader(reader, offset);
      reader->read(offset, rows, mapNulls);
      reader->addParentNulls(offset, mapNulls, rows);
    }
    lazyVectorReadOffset_ = offset;
    readOffset_ = offset + rows.back() + 1;
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    for (int k = 0; k < children_.size(); ++k) {
      children_[k]->getValues(rows, &childValues_[k]);
      copyRanges_[k].clear();
    }
    auto offsets =
        AlignedBuffer::allocate<vector_size_t>(rows.size(), &memoryPool_);
    auto sizes =
        AlignedBuffer::allocate<vector_size_t>(rows.size(), &memoryPool_);
    auto* rawOffsets = offsets->template asMutable<vector_size_t>();
    auto* rawSizes = sizes->template asMutable<vector_size_t>();
    vector_size_t totalSize = 0;
    for (vector_size_t i = 0; i < rows.size(); ++i) {
      if (anyNulls_ && bits::isBitNull(rawResultNulls_, i)) {
        continue;
      }
      int currentRowSize = 0;
      for (int k = 0; k < children_.size(); ++k) {
        auto& data = static_cast<const DwrfData&>(children_[k]->formatData());
        auto* inMap = data.inMap();
        if (inMap && bits::isBitNull(inMap, rows[i])) {
          continue;
        }
        copyRanges_[k].push_back({
            .sourceIndex = i,
            .targetIndex = totalSize + currentRowSize,
            .count = 1,
        });
        ++currentRowSize;
      }
      if (currentRowSize > 0) {
        rawOffsets[i] = totalSize;
        rawSizes[i] = currentRowSize;
        totalSize += currentRowSize;
      } else {
        if (!rawResultNulls_) {
          setNulls(AlignedBuffer::allocate<bool>(rows.size(), &memoryPool_));
        }
        bits::setNull(rawResultNulls_, i);
        anyNulls_ = true;
      }
    }
    auto& mapType = requestedType_->type->asMap();
    VectorPtr keys =
        BaseVector::create(mapType.keyType(), totalSize, &memoryPool_);
    VectorPtr values =
        BaseVector::create(mapType.valueType(), totalSize, &memoryPool_);
    auto* flatKeys = keys->asFlatVector<T>();
    T* rawKeys = flatKeys->mutableRawValues();
    [[maybe_unused]] size_t strKeySize;
    [[maybe_unused]] char* rawStrKeyBuffer;
    if constexpr (std::is_same_v<T, StringView>) {
      strKeySize = 0;
      for (int k = 0; k < children_.size(); ++k) {
        if (!keyNodes_[k].key.get().isInline()) {
          strKeySize += keyNodes_[k].key.get().size();
        }
      }
      if (strKeySize > 0) {
        auto buf = AlignedBuffer::allocate<char>(strKeySize, &memoryPool_);
        rawStrKeyBuffer = buf->template asMutable<char>();
        flatKeys->addStringBuffer(buf);
        strKeySize = 0;
        for (int k = 0; k < children_.size(); ++k) {
          auto& s = keyNodes_[k].key.get();
          if (!s.isInline()) {
            memcpy(&rawStrKeyBuffer[strKeySize], s.data(), s.size());
            strKeySize += s.size();
          }
        }
        strKeySize = 0;
      }
    }
    for (int k = 0; k < children_.size(); ++k) {
      [[maybe_unused]] StringView strKey;
      if constexpr (std::is_same_v<T, StringView>) {
        strKey = keyNodes_[k].key.get();
        if (!strKey.isInline()) {
          strKey = {
              &rawStrKeyBuffer[strKeySize],
              static_cast<int32_t>(strKey.size())};
          strKeySize += strKey.size();
        }
      }
      for (auto& r : copyRanges_[k]) {
        if constexpr (std::is_same_v<T, StringView>) {
          rawKeys[r.targetIndex] = strKey;
        } else {
          rawKeys[r.targetIndex] = keyNodes_[k].key.get();
        }
      }
      values->copyRanges(childValues_[k].get(), copyRanges_[k]);
    }
    *result = std::make_shared<MapVector>(
        &memoryPool_,
        requestedType_->type,
        anyNulls_ ? resultNulls_ : nullptr,
        rows.size(),
        std::move(offsets),
        std::move(sizes),
        std::move(keys),
        std::move(values));
  }

 private:
  common::ScanSpec structScanSpec_;
  std::vector<KeyNode<T>> keyNodes_;
  std::vector<VectorPtr> childValues_;
  std::vector<std::vector<BaseVector::CopyRange>> copyRanges_;
};

template <typename T>
std::unique_ptr<dwio::common::SelectiveColumnReader> createReader(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    DwrfParams& params,
    common::ScanSpec& scanSpec) {
  auto& mapColumnIdAsStruct =
      params.stripeStreams().getRowReaderOptions().getMapColumnIdAsStruct();
  auto it = mapColumnIdAsStruct.find(requestedType->id);
  if (it != mapColumnIdAsStruct.end()) {
    return std::make_unique<SelectiveFlatMapAsStructReader<T>>(
        requestedType, dataType, params, scanSpec, it->second);
  } else {
    return std::make_unique<SelectiveFlatMapReader<T>>(
        requestedType, dataType, params, scanSpec);
  }
}

} // namespace

std::unique_ptr<dwio::common::SelectiveColumnReader>
createSelectiveFlatMapColumnReader(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    DwrfParams& params,
    common::ScanSpec& scanSpec) {
  auto kind = dataType->childAt(0)->type->kind();
  switch (kind) {
    case TypeKind::TINYINT:
      return createReader<int8_t>(requestedType, dataType, params, scanSpec);
    case TypeKind::SMALLINT:
      return createReader<int16_t>(requestedType, dataType, params, scanSpec);
    case TypeKind::INTEGER:
      return createReader<int32_t>(requestedType, dataType, params, scanSpec);
    case TypeKind::BIGINT:
      return createReader<int64_t>(requestedType, dataType, params, scanSpec);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return createReader<StringView>(
          requestedType, dataType, params, scanSpec);
    default:
      VELOX_UNSUPPORTED("Not supported key type: {}", kind);
  }
}

} // namespace facebook::velox::dwrf

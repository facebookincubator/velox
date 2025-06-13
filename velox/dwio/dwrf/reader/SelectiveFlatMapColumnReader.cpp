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
#include "velox/dwio/common/SelectiveFlatMapColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveDwrfReader.h"
#include "velox/dwio/dwrf/reader/SelectiveStructColumnReader.h"
#include "velox/vector/FlatMapVector.h"

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
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec,
    dwio::common::flatmap::FlatMapOutput outputType) {
  using namespace dwio::common::flatmap;

  std::vector<KeyNode<T>> keyNodes;
  std::unordered_set<size_t> processed;

  auto& requestedValueType = requestedType->childAt(1);
  auto& dataValueType = fileType->childAt(1);
  auto& stripe = params.stripeStreams();

  common::ScanSpec* keysSpec = nullptr;
  common::ScanSpec* valuesSpec = nullptr;
  std::unordered_map<KeyValue<T>, common::ScanSpec*, KeyValueHash<T>>
      childSpecs;

  // Adjust the scan spec according to the output type.
  switch (outputType) {
    // For a kMap output, just need a scan spec for map keys and one for map
    // values.
    case FlatMapOutput::kMap: {
      keysSpec = scanSpec.getOrCreateChild(common::ScanSpec::kMapKeysFieldName);
      valuesSpec =
          scanSpec.getOrCreateChild(common::ScanSpec::kMapValuesFieldName);
      VELOX_CHECK(!valuesSpec->hasFilter());
      keysSpec->setProjectOut(true);
      valuesSpec->setProjectOut(true);
      break;
    }
    // For a kStruct output, the streams to be read are part of the scan spec
    // already.
    case FlatMapOutput::kStruct: {
      for (auto& c : scanSpec.children()) {
        T key;
        if constexpr (std::is_same_v<T, StringView>) {
          key = StringView(c->fieldName());
        } else {
          key = folly::to<T>(c->fieldName());
        }
        childSpecs[KeyValue<T>(key)] = c.get();
      }
      break;
    }
    // For a kFlatMap output, this is a no-op, we'll project all keys.
    case FlatMapOutput::kFlatMap:
      break;
  }

  // Load all sub streams.
  // Fetch reader, in map bitmap and key object.
  auto streams = stripe.visitStreamsOfNode(
      dataValueType->id(), [&](const StreamInformation& stream) {
        auto sequence = stream.getSequence();
        // No need to load shared dictionary stream here.
        if (sequence == 0 || processed.count(sequence)) {
          return;
        }
        processed.insert(sequence);
        EncodingKey seqEk(dataValueType->id(), sequence);
        const auto& keyInfo = stripe.getEncoding(seqEk).key();
        auto key = extractKey<T>(keyInfo);
        common::ScanSpec* childSpec;
        if (outputType == FlatMapOutput::kFlatMap) {
          childSpec = scanSpec.getOrCreateChild(toString(key.get()));
          childSpec->setProjectOut(true);
          childSpec->setChannel(sequence - 1);
        } else if (auto it = childSpecs.find(key);
                   it != childSpecs.end() && !it->second->isConstant()) {
          childSpec = it->second;
        } else if (outputType == FlatMapOutput::kStruct) {
          // Column not selected in 'scanSpec', skipping it.
          return;
        } else {
          if (keysSpec && keysSpec->filter() &&
              !common::applyFilter(*keysSpec->filter(), key.get())) {
            return; // Subfield pruning
          }
          childSpecs[key] = childSpec = valuesSpec;
        }
        auto labels = params.streamLabels().append(toString(key.get()));
        auto inMap = stripe.getStream(
            seqEk.forKind(proto::Stream_Kind_IN_MAP), labels.label(), true);
        VELOX_CHECK(inMap, "In map stream is required");
        auto inMapDecoder = createBooleanRleDecoder(std::move(inMap), seqEk);
        DwrfParams childParams(
            stripe,
            labels,
            params.runtimeStatistics(),
            FlatMapContext{
                .sequence = sequence,
                .inMapDecoder = inMapDecoder.get(),
                .keySelectionCallback = nullptr});
        auto reader = SelectiveDwrfReader::build(
            columnReaderOptions,
            requestedValueType,
            dataValueType,
            childParams,
            *childSpec);
        keyNodes.emplace_back(
            key, sequence, std::move(reader), std::move(inMapDecoder));
      });

  VLOG(1) << "[Flat-Map] Initialized a flat-map column reader for node "
          << fileType->id() << ", keys=" << keyNodes.size()
          << ", streams=" << streams;

  if (outputType != FlatMapOutput::kStruct) {
    std::sort(keyNodes.begin(), keyNodes.end(), [](auto& x, auto& y) {
      return x.sequence < y.sequence;
    });
  }
  return keyNodes;
}

template <typename T>
class SelectiveFlatMapAsStructReader : public SelectiveStructColumnReaderBase {
 public:
  SelectiveFlatMapAsStructReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveStructColumnReaderBase(
            requestedType,
            fileType,
            params,
            scanSpec),
        keyNodes_(getKeyNodes<T>(
            columnReaderOptions,
            requestedType,
            fileType,
            params,
            scanSpec,
            dwio::common::flatmap::FlatMapOutput::kStruct)) {
    VELOX_CHECK(
        !keyNodes_.empty(),
        "For struct encoding, keys to project must be configured");
    children_.resize(keyNodes_.size());
    for (auto& childSpec : scanSpec.children()) {
      childSpec->setSubscript(kConstantChildSpecSubscript);
    }
    for (int i = 0; i < keyNodes_.size(); ++i) {
      keyNodes_[i].reader->scanSpec()->setSubscript(i);
      children_[i] = keyNodes_[i].reader.get();
    }
  }

 private:
  std::vector<KeyNode<T>> keyNodes_;
};

template <typename T>
class SelectiveFlatMapAsMapReader : public SelectiveStructColumnReaderBase {
 public:
  SelectiveFlatMapAsMapReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveStructColumnReaderBase(
            requestedType,
            fileType,
            params,
            scanSpec),
        flatMap_(
            *this,
            getKeyNodes<T>(
                columnReaderOptions,
                requestedType,
                fileType,
                params,
                scanSpec,
                dwio::common::flatmap::FlatMapOutput::kMap)) {}

  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override {
    flatMap_.read(offset, rows, incomingNulls);
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    flatMap_.getValues(rows, result);
  }

 private:
  dwio::common::SelectiveFlatMapColumnReaderHelper<T, KeyNode<T>, DwrfData>
      flatMap_;
};

template <typename T>
class SelectiveFlatMapReader
    : public dwio::common::SelectiveFlatMapColumnReader {
 public:
  SelectiveFlatMapReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : dwio::common::SelectiveFlatMapColumnReader(
            requestedType,
            fileType,
            params,
            scanSpec),
        keyNodes_(getKeyNodes<T>(
            columnReaderOptions,
            requestedType,
            fileType,
            params,
            scanSpec,
            dwio::common::flatmap::FlatMapOutput::kFlatMap)),
        rowsPerRowGroup_(formatData_->rowsPerRowGroup().value()) {
    // Instantiate and populate distinct keys vector.
    keysVector_ = BaseVector::create(
        CppToType<T>::create(),
        (vector_size_t)keyNodes_.size(),
        &params.pool());
    auto rawKeys = keysVector_->values()->asMutable<T>();
    children_.resize(keyNodes_.size());

    for (int i = 0; i < keyNodes_.size(); ++i) {
      keyNodes_[i].reader->scanSpec()->setSubscript(i);
      children_[i] = keyNodes_[i].reader.get();

      rawKeys[i] = keyNodes_[i].key.get();
    }
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    SelectiveFlatMapColumnReader::getValues<DwrfData>(rows, result);
  }

  void seekTo(int64_t offset, bool /*readsNullsOnly*/) override {
    if (offset == readOffset_) {
      return;
    }
    if (readOffset_ < offset) {
      if (numParentNulls_) {
        VELOX_CHECK_LE(
            parentNullsRecordedTo_,
            offset,
            "Must not seek to before parentNullsRecordedTo_");
      }
      auto distance = offset - readOffset_ - numParentNulls_;
      auto numNonNulls = formatData_->skipNulls(distance);
      // We inform children how many nulls there were between original position
      // and destination. The children will seek this many less. The
      // nulls include the nulls found here as well as the enclosing
      // level nulls reported to this by parents.
      for (auto& child : children_) {
        if (child) {
          child->addSkippedParentNulls(
              readOffset_, offset, numParentNulls_ + distance - numNonNulls);
        }
      }
      numParentNulls_ = 0;
      parentNullsRecordedTo_ = 0;
      readOffset_ = offset;
    } else {
      VELOX_FAIL("Seeking backward on a ColumnReader");
    }
  }

  void seekToRowGroup(int64_t index) override {
    dwio::common::SelectiveStructColumnReaderBase::seekToRowGroup(index);
    if (isTopLevel_ && !formatData_->hasNulls()) {
      readOffset_ = index * rowsPerRowGroup_;
      return;
    }

    // There may be a nulls stream but no other streams for the struct.
    formatData_->seekToRowGroup(index);
    // Set the read offset recursively. Do this before seeking the children
    // because list/map children will reset the offsets for their children.
    setReadOffsetRecursive(index * rowsPerRowGroup_);
    for (auto& child : children_) {
      child->seekToRowGroup(index);
    }
  }

  /// Advance field reader to the row group closest to specified offset by
  /// calling seekToRowGroup.
  void advanceFieldReader(
      dwio::common::SelectiveColumnReader* reader,
      int64_t offset) override {
    if (!reader->isTopLevel()) {
      return;
    }
    const auto rowGroup = reader->readOffset() / rowsPerRowGroup_;
    const auto nextRowGroup = offset / rowsPerRowGroup_;
    if (nextRowGroup > rowGroup) {
      reader->seekToRowGroup(nextRowGroup);
      reader->setReadOffset(nextRowGroup * rowsPerRowGroup_);
    }
  }

 private:
  std::vector<KeyNode<T>> keyNodes_;
  const int32_t rowsPerRowGroup_;
};

template <typename T>
std::unique_ptr<dwio::common::SelectiveColumnReader> createReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec) {
  if (scanSpec.isFlatMapAsStruct()) {
    return std::make_unique<SelectiveFlatMapAsStructReader<T>>(
        columnReaderOptions, requestedType, fileType, params, scanSpec);
  } else if (params.stripeStreams()
                 .rowReaderOptions()
                 .preserveFlatMapsInMemory()) {
    return std::make_unique<SelectiveFlatMapReader<T>>(
        columnReaderOptions, requestedType, fileType, params, scanSpec);
  } else {
    return std::make_unique<SelectiveFlatMapAsMapReader<T>>(
        columnReaderOptions, requestedType, fileType, params, scanSpec);
  }
}

} // namespace

std::unique_ptr<dwio::common::SelectiveColumnReader>
createSelectiveFlatMapColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec) {
  auto kind = fileType->childAt(0)->type()->kind();
  switch (kind) {
    case TypeKind::TINYINT:
      return createReader<int8_t>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::SMALLINT:
      return createReader<int16_t>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::INTEGER:
      return createReader<int32_t>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::BIGINT:
      return createReader<int64_t>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return createReader<StringView>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    default:
      VELOX_UNSUPPORTED("Not supported key type: {}", kind);
  }
}

} // namespace facebook::velox::dwrf

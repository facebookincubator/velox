/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/velox/selective/FlatMapColumnReader.h"

#include "velox/dwio/common/FlatMapHelper.h"
#include "velox/dwio/common/SelectiveFlatMapColumnReader.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/selective/ChunkedDecoder.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/dwio/nimble/velox/selective/StructColumnReader.h"

namespace facebook::nimble {

using namespace facebook::velox;

namespace {

template <typename T>
struct KeyNode {
  dwio::common::flatmap::KeyValue<T> key;
  std::unique_ptr<ChunkedDecoder> inMap;
  std::unique_ptr<dwio::common::SelectiveColumnReader> reader;

  explicit KeyNode(dwio::common::flatmap::KeyValue<T> key)
      : key(std::move(key)) {}
};

template <typename T>
std::vector<KeyNode<T>> makeKeyNodes(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    common::ScanSpec& scanSpec,
    dwio::common::flatmap::FlatMapOutput outputType,
    velox::memory::MemoryPool& memoryPool) {
  using namespace dwio::common::flatmap;
  std::vector<KeyNode<T>> keyNodes;
  auto& requestedValueType = requestedType->childAt(1);
  auto& fileValueType = fileType->childAt(1);
  common::ScanSpec* keysSpec = nullptr;
  common::ScanSpec* valuesSpec = nullptr;

  folly::F14FastMap<KeyValue<T>, common::ScanSpec*, KeyValueHash<T>> childSpecs;

  auto& nimbleType = params.nimbleType()->asFlatMap();
  int childrenCount = nimbleType.childrenCount();

  // When flatmap is empty, writer creates dummy child with empty name to
  // carry schema information. We need to capture actual children count.
  if (childrenCount == 1 && nimbleType.nameAt(0).empty()) {
    childrenCount = 0;
  }

  // Adjust the scan spec according to the output type.
  switch (outputType) {
    // For a kMap and kFlatMap output, just need a scan spec for map keys and
    // one for map values.
    case FlatMapOutput::kMap:
    case FlatMapOutput::kFlatMap: {
      keysSpec = scanSpec.getOrCreateChild(
          common::Subfield(common::ScanSpec::kMapKeysFieldName));
      valuesSpec = scanSpec.getOrCreateChild(
          common::Subfield(common::ScanSpec::kMapValuesFieldName));
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
  }

  // Create column readers for each stream and populate the keyNodes vector.
  for (int i = 0; i < childrenCount; ++i) {
    KeyNode<T> node(parseKeyValue<T>(nimbleType.nameAt(i)));
    common::ScanSpec* childSpec;
    if (auto it = childSpecs.find(node.key);
        it != childSpecs.end() && !it->second->isConstant()) {
      childSpec = it->second;
    } else if (outputType == FlatMapOutput::kStruct) {
      // Column not selected in 'scanSpec', skipping it.
      continue;
    } else {
      if (keysSpec && keysSpec->filter() &&
          !common::applyFilter(*keysSpec->filter(), node.key.get())) {
        continue; // Subfield pruning
      }
      childSpecs[node.key] = childSpec = valuesSpec;
    }
    auto inMapInput = params.streams().enqueue(
        nimbleType.inMapDescriptorAt(i).offset(), params.lazyColumnIo());
    if (inMapInput != nullptr) {
      node.inMap = std::make_unique<ChunkedDecoder>(
          std::move(inMapInput),
          /*streamIndex=*/nullptr,
          /*decodeValuesWithNulls=*/false,
          &params.encodingFactory(),
          &memoryPool);
    } else {
      // Missing in-map stream: either the writer skipped it because all rows
      // are in-map (value streams exist), or the key has no data in this
      // stripe (value streams also absent). Check value streams to distinguish.
      if (!visitValueStreamLeaves(
              *nimbleType.childAt(i),
              [&streams = params.streams()](offset_size offset) {
                return streams.hasStream(offset);
              })) {
        continue;
      }
    }
    auto childParams = params.makeChildParams(nimbleType.childAt(i));
    childParams.setInMapDecoder(node.inMap.get());
    node.reader = buildColumnReader(
        requestedValueType, fileValueType, childParams, *childSpec, false);
    keyNodes.push_back(std::move(node));
  }
  return keyNodes;
}

template <typename T>
class FlatMapAsStructColumnReader : public StructColumnReaderBase {
 public:
  FlatMapAsStructColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      common::ScanSpec& scanSpec)
      : StructColumnReaderBase(
            requestedType,
            fileType,
            params,
            scanSpec,
            false),
        keyNodes_(
            makeKeyNodes<T>(
                requestedType,
                fileType,
                params,
                scanSpec,
                dwio::common::flatmap::FlatMapOutput::kStruct,
                *pool_)) {
    children_.resize(keyNodes_.size());
    for (auto& childSpec : scanSpec.children()) {
      childSpec->setSubscript(kConstantChildSpecSubscript);
    }
    for (int i = 0; i < keyNodes_.size(); ++i) {
      keyNodes_[i].reader->scanSpec()->setSubscript(i);
      children_[i] = keyNodes_[i].reader.get();
    }
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const final {
    auto* nulls = formatData().template as<NimbleData>().nullsDecoder();
    auto* inMap = formatData().template as<NimbleData>().inMapDecoder();
    if (nulls) {
      auto nullsRowCount = nulls->estimateRowCount();
      if (!nullsRowCount.has_value()) {
        return false;
      }
      rowCount = *nullsRowCount;
    } else if (inMap) {
      auto inMapRowCount = inMap->estimateRowCount();
      if (!inMapRowCount.has_value()) {
        return false;
      }
      rowCount = *inMapRowCount;
    } else {
      rowCount = 0;
    }
    size_t rowSize = 0;
    for (auto& child : children_) {
      size_t childByteSize, childRowCount;
      if (!child->estimateMaterializedSize(childByteSize, childRowCount)) {
        return false;
      }
      if (!nulls && !inMap) {
        rowCount = childRowCount;
      }
      if (childRowCount > 0) {
        rowSize += childByteSize / childRowCount;
      }
    }
    byteSize = rowSize * rowCount;
    if (nulls || inMap) {
      byteSize += rowCount / 8;
    }
    return true;
  }

 private:
  std::vector<KeyNode<T>> keyNodes_;
};

template <typename T>
class FlatMapColumnReader
    : public velox::dwio::common::SelectiveFlatMapColumnReader {
 public:
  FlatMapColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveFlatMapColumnReader(
            dwio::common::ColumnReaderOptions{},
            requestedType,
            fileType,
            params,
            scanSpec),
        keyNodes_(
            makeKeyNodes<T>(
                requestedType,
                fileType,
                params,
                scanSpec,
                dwio::common::flatmap::FlatMapOutput::kFlatMap,
                *pool_)) {
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

  const BufferPtr& inMapBuffer(column_index_t childIndex) const override {
    return children_[childIndex]
        ->formatData()
        .template as<NimbleData>()
        .inMapBuffer();
  }

  // Same as FlatMapAsStructColumnReader.
  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const final {
    auto* nulls = formatData().template as<NimbleData>().nullsDecoder();
    auto* inMap = formatData().template as<NimbleData>().inMapDecoder();
    if (nulls) {
      auto nullsRowCount = nulls->estimateRowCount();
      if (!nullsRowCount.has_value()) {
        return false;
      }
      rowCount = *nullsRowCount;
    } else if (inMap) {
      auto inMapRowCount = inMap->estimateRowCount();
      if (!inMapRowCount.has_value()) {
        return false;
      }
      rowCount = *inMapRowCount;
    } else {
      rowCount = 0;
    }
    size_t rowSize = 0;
    for (auto& child : children_) {
      size_t childByteSize, childRowCount;
      if (!child->estimateMaterializedSize(childByteSize, childRowCount)) {
        return false;
      }
      if (!nulls && !inMap) {
        rowCount = childRowCount;
      }
      if (childRowCount > 0) {
        rowSize += childByteSize / childRowCount;
      }
    }
    byteSize = rowSize * rowCount;
    if (nulls || inMap) {
      byteSize += rowCount / 8;
    }
    return true;
  }

  void seekToRowGroup(int64_t /*index*/) final {
    VELOX_UNREACHABLE();
  }

  void advanceFieldReader(SelectiveColumnReader* /*reader*/, int64_t /*offset*/)
      final {
    // No-op, there is no index for fast skipping and we need to skip in the
    // decoders.
  }

  // Same as FlatMapAsMapColumnReader.
  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override {
    numReads_ = scanSpec_->newRead();
    prepareRead<char>(offset, rows, incomingNulls);
    VELOX_DCHECK(!hasDeletion());
    auto activeRows = rows;
    auto* mapNulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    if (scanSpec_->filter()) {
      auto kind = scanSpec_->filter()->kind();
      VELOX_CHECK(
          kind == velox::common::FilterKind::kIsNull ||
          kind == velox::common::FilterKind::kIsNotNull);
      filterNulls<int32_t>(
          rows, kind == velox::common::FilterKind::kIsNull, false);
      if (outputRows_.empty()) {
        for (auto* child : children_) {
          child->addParentNulls(offset, mapNulls, rows);
        }
        readOffset_ = offset + rows.back() + 1;
        return;
      }
      activeRows = outputRows_;
    }
    // Separate the loop to be cache friendly.
    for (auto* child : children_) {
      advanceFieldReader(child, offset);
    }
    for (auto* child : children_) {
      child->read(offset, activeRows, mapNulls);
      child->addParentNulls(offset, mapNulls, rows);
    }
    readOffset_ = offset + rows.back() + 1;
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    SelectiveFlatMapColumnReader::getValues(rows, result);

    // After reading the flat map streams recursively, need to read the in map
    // buffers.
    VELOX_CHECK(result && *result);
    auto flatMapVector = (*result)->as<FlatMapVector>();
    VELOX_CHECK(flatMapVector);

    for (int i = 0; i < keyNodes_.size(); ++i) {
      auto& nimbleData = children_[i]->formatData().template as<NimbleData>();
      flatMapVector->inMapsAt(i, true) = nimbleData.inMapBuffer();
    }
  }

 private:
  std::vector<KeyNode<T>> keyNodes_;
};

template <typename T>
class FlatMapAsMapColumnReader : public StructColumnReaderBase {
 public:
  FlatMapAsMapColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      common::ScanSpec& scanSpec)
      : StructColumnReaderBase(
            requestedType,
            fileType,
            params,
            scanSpec,
            false),
        flatMap_(
            *this,
            makeKeyNodes<T>(
                requestedType,
                fileType,
                params,
                scanSpec,
                dwio::common::flatmap::FlatMapOutput::kMap,
                *pool_)) {}

  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override {
    flatMap_.read(offset, rows, incomingNulls);
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    flatMap_.getValues(rows, result);
  }

  void setIsTopLevel() override {
    // Children are not considered top level since this is materialized as MAP.
    SelectiveColumnReader::setIsTopLevel();
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const final {
    auto* nulls = formatData().template as<NimbleData>().nullsDecoder();
    auto* inMap = formatData().template as<NimbleData>().inMapDecoder();
    if (nulls) {
      auto nullsRowCount = nulls->estimateRowCount();
      if (!nullsRowCount.has_value()) {
        return false;
      }
      rowCount = *nullsRowCount;
    } else if (inMap) {
      auto inMapRowCount = inMap->estimateRowCount();
      if (!inMapRowCount.has_value()) {
        return false;
      }
      rowCount = *inMapRowCount;
    } else {
      rowCount = 0;
    }
    size_t rowSize = 8;
    for (auto& child : children_) {
      size_t childByteSize, childRowCount;
      if (!child->estimateMaterializedSize(childByteSize, childRowCount)) {
        return false;
      }
      if (!nulls && !inMap) {
        rowCount = childRowCount;
      }
      if (childRowCount > 0) {
        rowSize += childByteSize / childRowCount;
        rowSize += sizeof(T);
      }
    }
    byteSize = rowSize * rowCount;
    if (nulls || inMap) {
      byteSize += rowCount / 8;
    }
    return true;
  }

 private:
  dwio::common::SelectiveFlatMapColumnReaderHelper<T, KeyNode<T>, NimbleData>
      flatMap_;
};

template <typename T>
std::unique_ptr<dwio::common::SelectiveColumnReader> createReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    common::ScanSpec& scanSpec) {
  if (scanSpec.isFlatMapAsStruct()) {
    return std::make_unique<FlatMapAsStructColumnReader<T>>(
        requestedType, fileType, params, scanSpec);
  } else if (params.preserveFlatMapsInMemory()) {
    return std::make_unique<FlatMapColumnReader<T>>(
        requestedType, fileType, params, scanSpec);
  } else {
    return std::make_unique<FlatMapAsMapColumnReader<T>>(
        requestedType, fileType, params, scanSpec);
  }
}

} // namespace

std::unique_ptr<velox::dwio::common::SelectiveColumnReader>
createFlatMapColumnReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec) {
  VELOX_DCHECK(requestedType->isMap());
  auto kind = requestedType->childAt(0)->kind();
  switch (kind) {
    case TypeKind::TINYINT:
      return createReader<int8_t>(requestedType, fileType, params, scanSpec);
    case TypeKind::SMALLINT:
      return createReader<int16_t>(requestedType, fileType, params, scanSpec);
    case TypeKind::INTEGER:
      return createReader<int32_t>(requestedType, fileType, params, scanSpec);
    case TypeKind::BIGINT:
      return createReader<int64_t>(requestedType, fileType, params, scanSpec);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return createReader<StringView>(
          requestedType, fileType, params, scanSpec);
    default:
      VELOX_UNSUPPORTED("Not supported key type: {}", kind);
  }
}

} // namespace facebook::nimble

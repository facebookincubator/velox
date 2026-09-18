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
#include "velox/dwio/nimble/index/ClusterIndexWriter.h"

#include "velox/dwio/nimble/index/ClusterIndexConfig.h"

#include "folly/ScopeGuard.h"
#include "folly/String.h"
#include "folly/json/json.h"
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"

namespace facebook::nimble::index {

namespace {

// The key stream uses a placeholder offset value (kKeyStreamId) that does
// not correspond to any actual stream position. This offset is not used for
// data retrieval but is required to conform to Nimble's stream typing
// framework, which expects all streams to have an associated offset.
// Returns a reference to a static descriptor to ensure lifetime validity for
// ContentStreamData which stores a reference to the descriptor.
const StreamDescriptorBuilder& keyStreamDescriptor() {
  static const StreamDescriptorBuilder descriptor{
      kKeyStreamId, ScalarKind::Binary};
  return descriptor;
}

// Returns a reference to a static growth policy to ensure lifetime validity for
// ContentStreamData which stores a reference to the policy.
const InputBufferGrowthPolicy& keyStreamGrowthPolicy() {
  static const auto policy =
      DefaultInputBufferGrowthPolicy::withDefaultRanges();
  return *policy;
}

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

uint32_t accumulateRowCount(
    const std::vector<uint32_t>& accumulatedRows,
    uint32_t newRows) {
  const uint32_t prev = accumulatedRows.empty() ? 0 : accumulatedRows.back();
  NIMBLE_CHECK_LE(
      prev + static_cast<uint64_t>(newRows),
      std::numeric_limits<uint32_t>::max(),
      "Partition accumulated row count overflow: {}",
      prev + static_cast<uint64_t>(newRows));
  return prev + newRows;
}

void validateKeyChunkCompressionType(CompressionType type) {
  NIMBLE_USER_CHECK(
      type == CompressionType::Uncompressed || type == CompressionType::Zstd ||
          type == CompressionType::Lz4,
      "Index key chunk compression only supports Uncompressed, Zstd, or Lz4, but got: {}",
      type);
}

} // namespace

ClusterIndexWriter::Options ClusterIndexWriter::makeOptions(
    const IndexConfig& config) {
  const auto& clusterIndexConfig =
      checkedIndexConfig<ClusterIndexConfig>(config);
  auto options = Options{
      .indexName = config.name,
      .columns = clusterIndexConfig.columns,
      .sortOrders = clusterIndexConfig.sortOrders,
      .encodingLayout = clusterIndexConfig.encodingLayout,
      .enforceKeyOrder = clusterIndexConfig.enforceKeyOrder,
      .noDuplicateKey = clusterIndexConfig.noDuplicateKey,
      .maxRowsPerKeyChunk = clusterIndexConfig.maxRowsPerKeyChunk,
      .keyChunkCompressionType = clusterIndexConfig.keyChunkCompressionType,
  };
  NIMBLE_USER_CHECK(
      !options.columns.empty(), "Cluster index must have at least one column");
  NIMBLE_USER_CHECK(
      options.sortOrders.empty() ||
          options.sortOrders.size() == options.columns.size(),
      "Cluster index columns and sort orders must have the same size");
  validateKeyStreamEncodingLayout(options.encodingLayout);
  validateKeyChunkCompressionType(options.keyChunkCompressionType);
  if (options.sortOrders.empty()) {
    options.sortOrders.assign(
        options.columns.size(), SortOrder{.ascending = true});
  }
  return options;
}

std::unique_ptr<ClusterIndexWriter> ClusterIndexWriter::create(
    const IndexConfig& config,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK_EQ(config.family, IndexFamily::Cluster);
  NIMBLE_CHECK_NOT_NULL(pool, "memory pool must not be null");
  return std::unique_ptr<ClusterIndexWriter>(new ClusterIndexWriter(
      velox::asRowType(inputType), makeOptions(config), pool));
}

ClusterIndexWriter::ClusterIndexWriter(
    const velox::RowTypePtr& inputType,
    Options options,
    velox::memory::MemoryPool* pool)
    : indexName_{std::move(options.indexName)},
      pool_{pool},
      columns_{std::move(options.columns)},
      sortOrders_{std::move(options.sortOrders)},
      keyEncoder_{
          clusterIndexFactory(indexName_)
              .createKeyEncoder(columns_, inputType, sortOrders_, pool)},
      encodingLayout_{std::move(options.encodingLayout)},
      enforceKeyOrder_{options.enforceKeyOrder},
      keyColumnIndices_{getKeyColumnIndices({columns_}, inputType)},
      noDuplicateKey_{options.noDuplicateKey},
      maxRowsPerKeyChunk_{options.maxRowsPerKeyChunk},
      keyCompressionParams_{.type = options.keyChunkCompressionType},
      keyStream_{std::make_unique<ContentStreamData<std::string_view>>(
          *pool_,
          keyStreamDescriptor(),
          keyStreamGrowthPolicy())},
      keyBuffer_{std::make_unique<Buffer>(*pool)} {}

std::unique_ptr<EncodingSelectionPolicy<std::string_view>>
ClusterIndexWriter::createEncodingPolicy() const {
  return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
      encodingLayout_,
      CompressionOptions{},
      [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        return nullptr;
      });
}

void ClusterIndexWriter::write(const velox::VectorPtr& input) {
  checkNotClosed();

  if (input->size() == 0) {
    return;
  }

  validateNoNullKeys(input, keyColumnIndices_);

  const auto newKeyStart = keyStream_->mutableData().size();
  keyStream_->ensureMutableDataCapacity(newKeyStart + input->size());

  encodedKeys_.clear();
  encodedKeys_.reserve(input->size());
  keyEncoder_->encode(input, encodedKeys_, [this](size_t size) {
    return keyBuffer_->reserve(size);
  });
  for (const auto key : encodedKeys_) {
    keyStream_->mutableData().emplace_back(key);
  }

  validateKeyOrder(newKeyStart);

  // Encode chunks when enough keys have accumulated.
  encodeKeyChunks();
}

void ClusterIndexWriter::validateKeyOrder(size_t newKeyStart) {
  if (!enforceKeyOrder_) {
    return;
  }

  const auto& keys = keyStream_->mutableData();

  // Helper to check ordering between two keys.
  auto checkOrder = [&](std::string_view current,
                        std::string_view previous,
                        size_t currentIdx,
                        size_t previousIdx) {
    if (FOLLY_LIKELY(
            noDuplicateKey_ ? current > previous : current >= previous)) {
      return;
    }
    // Format keys as hex for readable error message since encoded keys
    // contain binary data that corrupts terminal output.
    NIMBLE_USER_FAIL(
        noDuplicateKey_
            ? "Encoded keys must be in strictly ascending order (duplicates are not allowed). "
              "Key at index {} (hex: {}) is not greater than key at index {} (hex: {})"
            : "Encoded keys must be in ascending order. "
              "Key at index {} (hex: {}) is less than key at index {} (hex: {})",
        currentIdx,
        folly::hexlify(current),
        previousIdx,
        folly::hexlify(previous));
  };

  // Cross-boundary check: when newKeyStart == 0 (buffer was cleared by
  // encodeKeyChunks), compare the first new key against lastEncodedKey_
  // which tracks the last key across clears.
  if (newKeyStart == 0 && lastEncodedKey_.has_value()) {
    checkOrder(keys[0], lastEncodedKey_.value(), 0, 0);
  }

  // Check ordering of all new keys. When newKeyStart > 0, also verify
  // continuity with the last key from the previous batch still in the buffer.
  const size_t startIdx = newKeyStart > 0 ? newKeyStart : 1;
  for (auto i = startIdx; i < keys.size(); ++i) {
    checkOrder(keys[i], keys[i - 1], i, i - 1);
  }
}

void ClusterIndexWriter::encodeKeyChunks(bool force) {
  const auto& keys = keyStream_->mutableData();
  if (keys.empty()) {
    return;
  }

  if (!force) {
    if (maxRowsPerKeyChunk_ == 0 || keys.size() < maxRowsPerKeyChunk_) {
      return;
    }
  }

  const uint64_t maxRows =
      maxRowsPerKeyChunk_ > 0 ? maxRowsPerKeyChunk_ : keys.size();

  ensureIndexPartition();

  for (size_t offset = 0; offset < keys.size();) {
    const auto remaining = static_cast<uint64_t>(keys.size() - offset);
    // Absorb tail if splitting would create a chunk smaller than maxRows.
    const auto chunkSize = remaining < 2 * maxRows ? remaining : maxRows;

    auto span =
        std::span<const std::string_view>(keys.data() + offset, chunkSize);

    KeyChunk keyChunk;
    encodeKeyChunk(span, keyChunk);

    // Push first key to partition keys for root-level early pruning.
    // Layout: [firstKey, lastKey0, lastKey1, ..., lastKeyN]
    ensureIndexRoot();
    if (indexRoot_->partitionKeys.empty()) {
      indexRoot_->partitionKeys.emplace_back(span[0]);
    }

    // Record chunk metadata directly in the partition.
    const auto chunkOffset = indexPartition_->keyStreamOffset;
    uint32_t chunkEncodedSize = 0;
    for (const auto& content : keyChunk.content) {
      indexPartition_->encodedChunks.emplace_back(content);
      chunkEncodedSize += content.size();
    }
    indexPartition_->keyStreamOffset += chunkEncodedSize;

    indexPartition_->chunkRows.emplace_back(
        accumulateRowCount(indexPartition_->chunkRows, chunkSize));
    indexPartition_->chunkOffsets.emplace_back(chunkOffset);
    indexPartition_->chunkKeys.emplace_back(std::move(keyChunk.key));

    offset += chunkSize;
  }

  // Save last key before clearing keyStream_ so cross-batch order validation
  // can compare against it.
  lastEncodedKey_ = std::string(keys.back());
  keyStream_->reset();
  keyBuffer_ = std::make_unique<Buffer>(*pool_);
}

void ClusterIndexWriter::encodeKeyChunk(
    std::span<const std::string_view> keys,
    KeyChunk& chunk) {
  auto& encodingBuffer = *indexPartition_->encodingBuffer;
  const auto encoded = EncodingFactory::encode<std::string_view>(
      createEncodingPolicy(), keys, encodingBuffer);
  NIMBLE_CHECK(!encoded.empty());

  chunk.rowCount = keys.size();
  chunk.key = std::string(keys.back());

  ChunkedStreamWriter chunkWriter{encodingBuffer, keyCompressionParams_};
  for (auto& contentBuffer : chunkWriter.encode(encoded)) {
    chunk.content.push_back(std::move(contentBuffer));
  }
}

std::optional<IndexDescriptor> ClusterIndexWriter::close(
    const WriteDataFn& /*writeDataFn*/,
    const CreateMetadataSectionFn& createMetadataFn) {
  setClosed();

  SCOPE_EXIT {
    keyStream_->reset();
    keyBuffer_.reset();
    indexPartition_.reset();
    indexRoot_.reset();
  };

  if (indexRoot_ == nullptr) {
    return std::nullopt;
  }

  NIMBLE_CHECK(
      !indexRoot_->indexPartitions.empty(), "Must have at least one partition");
  NIMBLE_CHECK_EQ(
      indexRoot_->partitionKeys.size(),
      indexRoot_->indexPartitions.size() + 1,
      "Partition keys must have one more element than index partitions");

  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);

  auto indexColumnsVector =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          columns_.size(), [&builder, this](size_t i) {
            return builder.CreateString(columns_[i]);
          });

  auto sortOrdersVector =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          sortOrders_.size(), [&builder, this](size_t i) {
            return builder.CreateString(
                folly::toJson(sortOrders_[i].serialize()));
          });

  auto partitionsVector =
      builder.CreateVector<flatbuffers::Offset<serialization::MetadataSection>>(
          indexRoot_->indexPartitions.size(), [&builder, this](size_t i) {
            return serialization::CreateMetadataSection(
                builder,
                indexRoot_->indexPartitions[i].offset(),
                indexRoot_->indexPartitions[i].size(),
                static_cast<serialization::CompressionType>(
                    indexRoot_->indexPartitions[i].compressionType()),
                indexRoot_->indexPartitions[i].uncompressedSize().value_or(
                    indexRoot_->indexPartitions[i].size()));
          });

  auto partitionKeysVector =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          indexRoot_->partitionKeys.size(), [&builder, this](size_t i) {
            return builder.CreateString(indexRoot_->partitionKeys[i]);
          });

  auto partitionRowCountsVector =
      builder.CreateVector(indexRoot_->partitionRowCounts);

  builder.Finish(
      serialization::CreateClusterIndex(
          builder,
          indexColumnsVector,
          sortOrdersVector,
          partitionsVector,
          partitionKeysVector,
          partitionRowCountsVector));

  return IndexDescriptor{
      .family = IndexFamily::Cluster,
      .name = indexName_,
      .root = createMetadataFn(asView(builder))};
}

void ClusterIndexWriter::ensureIndexRoot() {
  if (indexRoot_ == nullptr) {
    indexRoot_ = std::make_unique<IndexRoot>();
  }
}

void ClusterIndexWriter::ensureIndexPartition() {
  if (indexPartition_ == nullptr) {
    indexPartition_ = std::make_unique<IndexPartition>();
    indexPartition_->encodingBuffer = std::make_unique<Buffer>(*pool_);
  }
}

void ClusterIndexWriter::flush(
    const WriteDataFn& writeDataFn,
    const CreateMetadataSectionFn& createMetadataFn) {
  checkNotClosed();

  // Force-encode any remaining accumulated keys. This must happen before the
  // indexPartition_ null check because when key chunking is disabled
  // (maxRowsPerKeyChunk_ == 0), encodeKeyChunks() returns early during write()
  // and the partition is only created here.
  encodeKeyChunks(/*force=*/true);

  NIMBLE_CHECK_NOT_NULL(indexPartition_);
  NIMBLE_CHECK(!indexPartition_->empty());

  // Write key stream data to file.
  const auto [fileOffset, fileSize] =
      writeDataFn(indexPartition_->encodedChunks);
  indexPartition_->keyStreamFileOffset = fileOffset;
  indexPartition_->keyStreamFileSize = fileSize;

  // Build ClusterIndexPartition FlatBuffer.
  ensureIndexRoot();

  flatbuffers::FlatBufferBuilder indexBuilder(kInitialFooterSize);

  auto chunkRowsVec = indexBuilder.CreateVector(indexPartition_->chunkRows);
  auto chunkOffsetsVec =
      indexBuilder.CreateVector(indexPartition_->chunkOffsets);
  auto chunkKeysVec =
      indexBuilder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          indexPartition_->chunkKeys.size(), [&indexBuilder, this](size_t i) {
            return indexBuilder.CreateString(indexPartition_->chunkKeys[i]);
          });

  indexBuilder.Finish(
      serialization::CreateClusterIndexPartition(
          indexBuilder,
          indexPartition_->keyStreamFileOffset,
          indexPartition_->keyStreamFileSize,
          chunkRowsVec,
          chunkOffsetsVec,
          chunkKeysVec));

  indexRoot_->indexPartitions.push_back(createMetadataFn(asView(indexBuilder)));

  // Record last key and row count for root-level index.
  NIMBLE_CHECK(!indexPartition_->chunkKeys.empty());
  indexRoot_->partitionKeys.emplace_back(indexPartition_->chunkKeys.back());
  indexRoot_->partitionRowCounts.emplace_back(
      indexPartition_->chunkRows.back());

  // Reset partition state.
  indexPartition_.reset();
}

} // namespace facebook::nimble::index

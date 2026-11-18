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
#include "velox/dwio/nimble/index/ClusterIndexWriterBase.h"

#include "folly/ScopeGuard.h"
#include "folly/json/json.h"
#include "velox/common/Casts.h"
#include "velox/dwio/nimble/index/KeyChunkBuilder.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"

namespace facebook::nimble::index {

namespace {

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

ClusterIndexWriterBase::~ClusterIndexWriterBase() = default;

ClusterIndexWriterBase::ClusterIndexWriterBase(
    const velox::RowTypePtr& inputType,
    Options options,
    std::unique_ptr<KeyChunkBuilder> keyChunkBuilder,
    velox::memory::MemoryPool* pool)
    : indexName_{std::move(options.indexName)},
      pool_{pool},
      columns_{std::move(options.columns)},
      sortOrders_{std::move(options.sortOrders)},
      keyColumnIndices_{getKeyColumnIndices({columns_}, inputType)},
      maxRowsPerKeyChunk_{options.maxRowsPerKeyChunk},
      keyCompressionParams_{.type = options.keyChunkCompressionType},
      keyStreamLayout_{options.keyStreamLayout},
      keyStreamId_{options.keyStreamId},
      keyChunkBuilder_{std::move(keyChunkBuilder)} {
  NIMBLE_CHECK_NOT_NULL(pool_, "memory pool must not be null");
  NIMBLE_CHECK_NOT_NULL(keyChunkBuilder_);
  NIMBLE_CHECK_EQ(
      keyStreamId_.has_value(),
      keyStreamLayout_ == KeyStreamLayout::kStripe,
      "Only the Stripe key stream layout has a key stream id, which it needs");
  NIMBLE_USER_CHECK(
      !columns_.empty(), "Cluster index must have at least one column");
  NIMBLE_USER_CHECK(
      sortOrders_.size() == columns_.size(),
      "Cluster index columns and sort orders must have the same size");
  validateKeyChunkCompressionType(options.keyChunkCompressionType);
}

void ClusterIndexWriterBase::write(const velox::VectorPtr& input) {
  checkNotClosed();

  if (input->size() == 0) {
    return;
  }

  validateNoNullKeys(input, keyColumnIndices_);

  keyChunkBuilder_->append(input);

  // In the Stripe layout, finishStripe() encodes the keys of each stripe at
  // once, into the buffer of the stripe's streams.
  if (keyStreamLayout_ == KeyStreamLayout::kStripe) {
    return;
  }

  // Encode chunks when enough keys have accumulated.
  encodeKeyChunks();
}

std::vector<Chunk> ClusterIndexWriterBase::finishStripe(Buffer& buffer) {
  checkNotClosed();
  if (keyStreamLayout_ == KeyStreamLayout::kPartition) {
    return {};
  }
  NIMBLE_CHECK_GT(keyChunkBuilder_->size(), 0, "A stripe must have keys");

  ensureIndexPartition();

  // The key chunk of a stripe is its whole key stream.
  auto chunks = encodeBufferedKeys(/*maxRowsPerChunk=*/0, buffer);
  NIMBLE_CHECK_EQ(chunks.size(), 1);
  indexPartition_->keyStreamOffset = 0;
  return chunks;
}

void ClusterIndexWriterBase::encodeKeyChunks(bool force) {
  const auto numKeys = keyChunkBuilder_->size();
  if (numKeys == 0) {
    return;
  }

  if (!force) {
    if (maxRowsPerKeyChunk_ == 0 || numKeys < maxRowsPerKeyChunk_) {
      return;
    }
  }

  ensureIndexPartition();

  // Only the Partition layout encodes keys into a buffer of the partition.
  NIMBLE_CHECK_NOT_NULL(indexPartition_->encodingBuffer);
  for (const auto& chunk : encodeBufferedKeys(
           maxRowsPerKeyChunk_, *indexPartition_->encodingBuffer)) {
    indexPartition_->encodedChunks.insert(
        indexPartition_->encodedChunks.end(),
        chunk.content.begin(),
        chunk.content.end());
  }
}

std::vector<Chunk> ClusterIndexWriterBase::encodeBufferedKeys(
    uint64_t maxRowsPerChunk,
    Buffer& buffer) {
  const auto numKeys = keyChunkBuilder_->size();
  const uint64_t maxRows = maxRowsPerChunk > 0 ? maxRowsPerChunk : numKeys;

  std::vector<Chunk> chunks;
  for (size_t offset = 0; offset < numKeys;) {
    const auto remaining = static_cast<uint64_t>(numKeys - offset);
    // Absorb tail if splitting would create a chunk smaller than maxRows.
    const auto chunkSize = remaining < 2 * maxRows ? remaining : maxRows;
    NIMBLE_CHECK_LE(chunkSize, std::numeric_limits<uint32_t>::max());
    const auto chunkRowCount = static_cast<uint32_t>(chunkSize);

    KeyChunk keyChunk;
    encodeKeyChunk(offset, chunkRowCount, buffer, keyChunk);

    // Push first key to partition keys for root-level early pruning.
    // Layout: [firstKey, lastKey0, lastKey1, ..., lastKeyN]
    ensureIndexRoot();
    if (indexRoot_->partitionKeys.empty()) {
      indexRoot_->partitionKeys.emplace_back(keyChunkBuilder_->keyAt(offset));
    }

    // Record chunk metadata directly in the partition.
    const auto chunkOffset = indexPartition_->keyStreamOffset;
    indexPartition_->keyStreamOffset += keyChunk.contentSize();

    indexPartition_->chunkRows.emplace_back(
        accumulateRowCount(indexPartition_->chunkRows, chunkRowCount));
    indexPartition_->chunkOffsets.emplace_back(chunkOffset);
    indexPartition_->chunkKeys.emplace_back(std::move(keyChunk.key));
    chunks.emplace_back(
        Chunk{
            .rowCount = keyChunk.rowCount,
            .content = std::move(keyChunk.content),
        });

    offset += chunkSize;
  }

  keyChunkBuilder_->clear();
  return chunks;
}

void ClusterIndexWriterBase::encodeKeyChunk(
    size_t offset,
    uint32_t count,
    Buffer& buffer,
    KeyChunk& chunk) {
  const auto encoded = keyChunkBuilder_->encode(offset, count, buffer);
  NIMBLE_CHECK(!encoded.empty());

  chunk.rowCount = count;
  chunk.key = keyChunkBuilder_->keyAt(offset + count - 1);

  ChunkedStreamWriter chunkWriter{buffer, keyCompressionParams_};
  for (auto& contentBuffer : chunkWriter.encode(encoded)) {
    chunk.content.push_back(std::move(contentBuffer));
  }
}

std::optional<IndexDescriptor> ClusterIndexWriterBase::close(
    const WriteDataFn& /*writeDataFn*/,
    const CreateMetadataSectionFn& createMetadataFn) {
  setClosed();

  SCOPE_EXIT {
    keyChunkBuilder_.reset();
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

  // A Stripe layout index locates the ClusterIndexGroup of each stripe group
  // in index_groups, and leaves index_partitions unset.
  const bool stripeLayout = keyStreamLayout_ == KeyStreamLayout::kStripe;
  const decltype(partitionsVector) noSections;
  builder.Finish(
      serialization::CreateClusterIndex(
          builder,
          indexColumnsVector,
          sortOrdersVector,
          stripeLayout ? noSections : partitionsVector,
          partitionKeysVector,
          partitionRowCountsVector,
          static_cast<uint8_t>(
              stripeLayout ? serialization::KeyStreamLayout_Stripe
                           : serialization::KeyStreamLayout_Partition),
          keyStreamId_.value_or(0),
          stripeLayout ? partitionsVector : noSections));

  return IndexDescriptor{
      .family = IndexFamily::Cluster,
      .name = indexName_,
      .root = createMetadataFn(asView(builder))};
}

void ClusterIndexWriterBase::ensureIndexRoot() {
  if (indexRoot_ == nullptr) {
    indexRoot_ = std::make_unique<IndexRoot>();
  }
}

void ClusterIndexWriterBase::ensureIndexPartition() {
  if (indexPartition_ != nullptr) {
    return;
  }
  indexPartition_ = std::make_unique<IndexPartition>();
  if (keyStreamLayout_ == KeyStreamLayout::kPartition) {
    indexPartition_->encodingBuffer =
        std::make_unique<Buffer>(*velox::checkedNotNull(pool_));
  }
}

void ClusterIndexWriterBase::finishStripeGroup(
    const WriteDataFn& writeDataFn,
    const CreateMetadataSectionFn& createMetadataFn) {
  checkNotClosed();

  if (keyStreamLayout_ == KeyStreamLayout::kStripe) {
    writeIndexGroup(createMetadataFn);
    return;
  }
  writeIndexPartition(writeDataFn, createMetadataFn);
}

void ClusterIndexWriterBase::writeIndexPartition(
    const WriteDataFn& writeDataFn,
    const CreateMetadataSectionFn& createMetadataFn) {
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

  finishPartition(createMetadataFn(asView(indexBuilder)));
}

void ClusterIndexWriterBase::writeIndexGroup(
    const CreateMetadataSectionFn& createMetadataFn) {
  NIMBLE_CHECK_EQ(
      keyChunkBuilder_->size(),
      0,
      "Keys written after the last finishStripe() belong to no stripe");
  NIMBLE_CHECK_NOT_NULL(indexPartition_);
  NIMBLE_CHECK(!indexPartition_->empty());

  // Each stripe holds one key chunk, so the chunk keys are the stripe keys.
  // Readers find the key stream and the rows of each stripe through the stripe
  // group metadata.
  ensureIndexRoot();
  flatbuffers::FlatBufferBuilder indexBuilder(kInitialFooterSize);
  auto stripeKeysVec =
      indexBuilder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          indexPartition_->chunkKeys.size(), [&indexBuilder, this](size_t i) {
            return indexBuilder.CreateString(indexPartition_->chunkKeys[i]);
          });
  indexBuilder.Finish(
      serialization::CreateClusterIndexGroup(indexBuilder, stripeKeysVec));

  finishPartition(createMetadataFn(asView(indexBuilder)));
}

void ClusterIndexWriterBase::finishPartition(MetadataSection partitionSection) {
  indexRoot_->indexPartitions.push_back(std::move(partitionSection));

  // Record last key and row count for root-level index.
  NIMBLE_CHECK(!indexPartition_->chunkKeys.empty());
  indexRoot_->partitionKeys.emplace_back(indexPartition_->chunkKeys.back());
  indexRoot_->partitionRowCounts.emplace_back(
      indexPartition_->chunkRows.back());

  // Reset partition state.
  indexPartition_.reset();
}

} // namespace facebook::nimble::index

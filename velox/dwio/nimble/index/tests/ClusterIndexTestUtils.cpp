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

#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"

#include "folly/String.h"
#include "folly/json/json.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/IndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/writer/Writer.h"

namespace facebook::nimble::index::test {

std::vector<velox::RowVectorPtr> generateData(
    const velox::RowTypePtr& rowType,
    size_t numVectors,
    velox::vector_size_t numRowsPerVector,
    const KeyColumnGeneratorMap& keyColumnGenerators,
    velox::memory::MemoryPool* pool,
    double nullRatio) {
  velox::VectorFuzzer::Options options;
  options.vectorSize = numRowsPerVector;
  options.nullRatio = nullRatio;
  velox::VectorFuzzer fuzzer(options, pool);

  std::vector<velox::RowVectorPtr> result;
  result.reserve(numVectors);
  for (size_t vecIdx = 0; vecIdx < numVectors; ++vecIdx) {
    std::vector<velox::VectorPtr> children;
    children.reserve(rowType->size());
    for (auto i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      auto it = keyColumnGenerators.find(name);
      if (it != keyColumnGenerators.end()) {
        children.push_back(it->second(vecIdx, numRowsPerVector));
      } else {
        children.push_back(fuzzer.fuzz(rowType->childAt(i)));
      }
    }
    result.push_back(
        std::make_shared<velox::RowVector>(
            pool, rowType, nullptr, numRowsPerVector, std::move(children)));
  }
  return result;
}

void writeFile(
    const std::string& filePath,
    const std::vector<velox::RowVectorPtr>& data,
    std::shared_ptr<const IndexConfig> clusterIndexConfig,
    velox::memory::MemoryPool& pool,
    std::function<std::unique_ptr<FlushPolicy>()> flushPolicyFactory) {
  NIMBLE_CHECK(!data.empty(), "Data must not be empty");

  WriterOptions options;
  options.enableChunking = true;
  options.clusterIndexConfig = std::move(clusterIndexConfig);
  if (flushPolicyFactory) {
    options.flushPolicyFactory = std::move(flushPolicyFactory);
  }

  auto fs = velox::filesystems::getFileSystem(filePath, {});
  auto writeFile = fs->openFileForWrite(
      filePath,
      {.shouldCreateParentDirectories = true,
       .shouldThrowOnFileAlreadyExists = false});

  auto rowType = velox::asRowType(data[0]->type());
  Writer writer(rowType, std::move(writeFile), pool, std::move(options));
  for (const auto& vector : data) {
    writer.write(vector);
  }
  writer.close();
}

std::vector<Chunk> createChunks(
    Buffer& buffer,
    const std::vector<ChunkSpec>& chunkSpecs) {
  std::vector<Chunk> chunks;
  for (const auto& spec : chunkSpecs) {
    auto pos = buffer.reserve(spec.size);
    std::memset(pos, 'X', spec.size);
    chunks.push_back(
        {.rowCount = spec.rowCount,
         .nullCount = spec.nullCount,
         .content = {{pos, spec.size}}});
  }
  return chunks;
}

Stream createStream(Buffer& buffer, const StreamSpec& spec) {
  return Stream{
      .offset = spec.offset, .chunks = createChunks(buffer, spec.chunks)};
}

PartitionStats ClusterIndexTestHelper::partitionStats(
    uint32_t partitionIndex) const {
  PartitionStats stats;

  const auto* partition = clusterIndex_->loadPartition(partitionIndex);
  const auto* index = partition->index;

  stats.keyStreamOffset = index->key_stream_offset();
  stats.keyStreamSize = index->key_stream_size();

  if (const auto* rows = index->chunk_rows()) {
    stats.chunkRows.reserve(rows->size());
    for (const auto row : *rows) {
      stats.chunkRows.push_back(row);
    }
  }

  if (const auto* offsets = index->chunk_offsets()) {
    stats.chunkOffsets.reserve(offsets->size());
    for (const auto offset : *offsets) {
      stats.chunkOffsets.push_back(offset);
    }
  }

  if (const auto* keys = index->chunk_keys()) {
    stats.chunkKeys.reserve(keys->size());
    for (const auto* key : *keys) {
      stats.chunkKeys.push_back(key->str());
    }
  }

  return stats;
}

StreamStats ChunkStatsTestHelper::streamStats(uint32_t streamId) const {
  StreamStats stats;

  const auto* root = flatbuffers::GetRoot<serialization::StripeChunkStats>(
      chunkStats_->metadata_->content().data());

  const auto* streamChunkCounts = root->stream_chunk_counts();
  if (streamChunkCounts == nullptr) {
    return stats;
  }

  const uint32_t streamCount = chunkStats_->streamCount_;
  if (streamId >= streamCount) {
    return stats;
  }

  const auto* chunkRows = root->stream_chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  const auto* chunkOffsets = root->stream_chunk_offsets();
  NIMBLE_CHECK_NOT_NULL(chunkOffsets);
  const auto* chunkNullCounts = root->stream_chunk_null_counts();

  const uint32_t stripeCount = chunkStats_->stripeCount_;

  uint32_t accumulatedChunkCountForStream = 0;
  for (uint32_t stripeOffset = 0; stripeOffset < stripeCount; ++stripeOffset) {
    const uint32_t idx = stripeOffset * streamCount + streamId;
    const uint32_t globalAccumulatedCount = streamChunkCounts->Get(idx);
    const uint32_t prevGlobalAccumulatedCount =
        idx == 0 ? 0 : streamChunkCounts->Get(idx - 1);
    const uint32_t chunksInThisStripe =
        globalAccumulatedCount - prevGlobalAccumulatedCount;
    accumulatedChunkCountForStream += chunksInThisStripe;
    stats.chunkCounts.push_back(accumulatedChunkCountForStream);

    const uint32_t startChunkIndex = prevGlobalAccumulatedCount;
    for (uint32_t i = 0; i < chunksInThisStripe; ++i) {
      const uint32_t chunkIndex = startChunkIndex + i;
      stats.chunkRows.push_back(chunkRows->Get(chunkIndex));
      stats.chunkOffsets.push_back(chunkOffsets->Get(chunkIndex));
      if (chunkNullCounts != nullptr) {
        stats.chunkNullCounts.push_back(chunkNullCounts->Get(chunkIndex));
      }
    }
  }

  return stats;
}

TestClusterIndexMetadataWriter::TestClusterIndexMetadataWriter(
    velox::memory::MemoryPool& pool,
    std::vector<std::string> columns,
    std::vector<SortOrder> sortOrders,
    bool enforceKeyOrder,
    bool noDuplicateKey)
    : pool_(&pool),
      columns_(std::move(columns)),
      sortOrders_(std::move(sortOrders)),
      enforceKeyOrder_(enforceKeyOrder),
      noDuplicateKey_(noDuplicateKey),
      rootBuffer_(std::make_unique<Buffer>(pool)) {}

void TestClusterIndexMetadataWriter::addStripe(
    const std::vector<KeyChunkSpec>& chunkSpecs) {
  NIMBLE_CHECK(!chunkSpecs.empty());

  if (partitionIndex_ == nullptr) {
    partitionIndex_ = std::make_unique<IndexPartition>();
    partitionIndex_->encodingBuffer = std::make_unique<Buffer>(*pool_);
  }

  // Track stripe keys for root index.
  const bool firstStripe = stripeKeys_.empty();
  if (firstStripe) {
    stripeKeys_.emplace_back(rootBuffer_->writeString(chunkSpecs.front().key));
  }
  const auto& lastKey = chunkSpecs.back().key;

  if (enforceKeyOrder_) {
    NIMBLE_CHECK(!stripeKeys_.empty());
    const auto& prevKey = stripeKeys_.back();
    if (firstStripe) {
      if (noDuplicateKey_ && chunkSpecs.size() > 1) {
        NIMBLE_USER_CHECK_GT(
            lastKey,
            prevKey,
            "Stripe keys must be in strictly ascending order (duplicates are not allowed). "
            "lastKey: {}, firstKey: {}",
            folly::hexlify(lastKey),
            folly::hexlify(prevKey));
      } else {
        NIMBLE_USER_CHECK_GE(
            lastKey,
            prevKey,
            "Stripe keys must be in ascending order (firstKey <= lastKey). "
            "lastKey: {}, firstKey: {}",
            folly::hexlify(lastKey),
            folly::hexlify(prevKey));
      }
    } else if (noDuplicateKey_) {
      NIMBLE_USER_CHECK_GT(
          lastKey,
          prevKey,
          "Stripe keys must be in strictly ascending order (duplicates are not allowed). "
          "newKey: {}, previousKey: {}",
          folly::hexlify(lastKey),
          folly::hexlify(prevKey));
    } else {
      NIMBLE_USER_CHECK_GE(
          lastKey,
          prevKey,
          "Stripe keys must be in ascending order. "
          "newKey: {}, previousKey: {}",
          folly::hexlify(lastKey),
          folly::hexlify(prevKey));
    }
  }
  stripeKeys_.emplace_back(rootBuffer_->writeString(lastKey));
  ++currentPartitionStripes_;

  // Encode each chunk's key via EncodingFactory + ChunkedStreamWriter
  // to produce properly formatted chunked stream data.
  auto& chunks = partitionIndex_->chunks;
  auto& encodingBuffer = *partitionIndex_->encodingBuffer;
  const uint32_t prevAccumulatedRows =
      chunks.chunkRows.empty() ? 0 : chunks.chunkRows.back();
  uint32_t accumulatedRows = prevAccumulatedRows;
  for (const auto& spec : chunkSpecs) {
    const auto chunkOffset = static_cast<uint32_t>(encodedKeyData_.size());

    // Encode the key as a single-element Trivial-encoded chunk.
    std::vector<std::string_view> keys = {spec.key};
    auto policy =
        std::make_unique<ManualEncodingSelectionPolicy<std::string_view>>(
            std::vector<std::pair<EncodingType, float>>{
                {EncodingType::Trivial, 1.0}},
            CompressionOptions{},
            std::nullopt);
    auto encoded = EncodingFactory::encode<std::string_view>(
        std::move(policy), keys, encodingBuffer);
    ChunkedStreamWriter chunkWriter{encodingBuffer};
    for (auto& segment : chunkWriter.encode(encoded)) {
      encodedKeyData_.append(segment.data(), segment.size());
    }

    accumulatedRows += spec.rowCount;
    chunks.chunkRows.emplace_back(accumulatedRows);
    chunks.chunkOffsets.emplace_back(chunkOffset);
    chunks.chunkKeys.emplace_back(encodingBuffer.writeString(spec.key));
  }
}

TabletWriter::StripeGroupFlushCallback
TestClusterIndexMetadataWriter::createStripeGroupFlushCallback() {
  return [this](
             const WriteDataFn& writeDataFn,
             const CreateMetadataSectionFn& createMetadataFn) {
    flushKeyStream(writeDataFn);
    flushPartitionMetadata(createMetadataFn);
  };
}

void TestClusterIndexMetadataWriter::flushKeyStream(
    const WriteDataFn& writeData) {
  if (encodedKeyData_.empty()) {
    return;
  }

  std::vector<std::string_view> segments = {encodedKeyData_};
  auto [offset, size] = writeData(segments);
  partitionIndex_->keyStreamFileOffset = offset;
  partitionIndex_->keyStreamFileSize = size;
  encodedKeyData_.clear();
}

void TestClusterIndexMetadataWriter::flushPartitionMetadata(
    const CreateMetadataSectionFn& createMetadataSection) {
  if (partitionIndex_ == nullptr) {
    return;
  }

  flatbuffers::FlatBufferBuilder indexBuilder(kInitialFooterSize);

  const auto& chunks = partitionIndex_->chunks;

  auto rowsVec = indexBuilder.CreateVector(chunks.chunkRows);
  auto offsetsVec = indexBuilder.CreateVector(chunks.chunkOffsets);
  auto keysVec =
      indexBuilder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          chunks.chunkKeys.size(), [&indexBuilder, &chunks](size_t i) {
            return indexBuilder.CreateString(
                chunks.chunkKeys[i].data(), chunks.chunkKeys[i].size());
          });

  indexBuilder.Finish(
      serialization::CreateClusterIndexPartition(
          indexBuilder,
          partitionIndex_->keyStreamFileOffset,
          partitionIndex_->keyStreamFileSize,
          rowsVec,
          offsetsVec,
          keysVec));

  std::string_view view{
      reinterpret_cast<const char*>(indexBuilder.GetBufferPointer()),
      indexBuilder.GetSize()};
  indexPartitions_.push_back(createMetadataSection(view));
  stripesPerPartition_.push_back(currentPartitionStripes_);
  rowsPerPartition_.push_back(
      chunks.chunkRows.empty() ? 0 : chunks.chunkRows.back());
  currentPartitionStripes_ = 0;

  partitionIndex_.reset();
}

TabletWriter::CloseCallback
TestClusterIndexMetadataWriter::createCloseCallback() {
  return [this](
             const WriteDataFn& /*writeDataFn*/,
             const CreateMetadataSectionFn& createMetadataFn,
             const WriteOptionalSectionFn& writeMetadataFn) {
    writeRoot(createMetadataFn, writeMetadataFn);
  };
}

void TestClusterIndexMetadataWriter::writeRoot(
    const CreateMetadataSectionFn& createMetadataSection,
    const WriteOptionalSectionFn& writeOptionalSection) {
  if (indexPartitions_.empty()) {
    return;
  }

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

  auto indexPartitionsVector =
      builder.CreateVector<flatbuffers::Offset<serialization::MetadataSection>>(
          indexPartitions_.size(), [&builder, this](size_t i) {
            return serialization::CreateMetadataSection(
                builder,
                indexPartitions_[i].offset(),
                indexPartitions_[i].size(),
                static_cast<serialization::CompressionType>(
                    indexPartitions_[i].compressionType()),
                indexPartitions_[i].uncompressedSize().value_or(
                    indexPartitions_[i].size()));
          });

  // Build partition keys: [firstKey, lastKeyPartition0, lastKeyPartition1, ...]
  // stripeKeys_ has per-stripe keys [firstKey, lastStripe0, lastStripe1, ...].
  // Reduce to per-partition by picking the last stripe key in each partition.
  const uint32_t numPartitions = static_cast<uint32_t>(indexPartitions_.size());
  std::vector<std::string_view> partitionKeys;
  partitionKeys.reserve(numPartitions + 1);
  partitionKeys.emplace_back(stripeKeys_[0]);
  uint32_t stripeIdx = 0;
  for (uint32_t p = 0; p < numPartitions; ++p) {
    stripeIdx += stripesPerPartition_[p];
    partitionKeys.emplace_back(stripeKeys_[stripeIdx]);
  }
  auto partitionKeysVector =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          partitionKeys.size(), [&builder, &partitionKeys](size_t i) {
            return builder.CreateString(
                partitionKeys[i].data(), partitionKeys[i].size());
          });

  auto partitionRowCountsVector = builder.CreateVector(rowsPerPartition_);

  builder.Finish(
      serialization::CreateClusterIndex(
          builder,
          indexColumnsVector,
          sortOrdersVector,
          indexPartitionsVector,
          partitionKeysVector,
          partitionRowCountsVector));
  std::string_view view{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
  const auto payloadSection = createMetadataSection(view);

  flatbuffers::FlatBufferBuilder rootBuilder(kInitialFooterSize);
  auto name = rootBuilder.CreateString(kClusterIndexName);
  auto payload = serialization::CreateMetadataSection(
      rootBuilder,
      payloadSection.offset(),
      payloadSection.size(),
      static_cast<serialization::CompressionType>(
          payloadSection.compressionType()),
      payloadSection.uncompressedSize().value_or(payloadSection.size()));
  auto descriptor = serialization::CreateIndexDescriptor(
      rootBuilder, serialization::IndexFamily_Cluster, name, payload);
  auto indexes = rootBuilder.CreateVector(&descriptor, 1);
  rootBuilder.Finish(serialization::CreateIndexRoot(rootBuilder, indexes));
  std::string_view rootView{
      reinterpret_cast<const char*>(rootBuilder.GetBufferPointer()),
      rootBuilder.GetSize()};
  writeOptionalSection(std::string(kIndexSection), rootView);
}

} // namespace facebook::nimble::index::test

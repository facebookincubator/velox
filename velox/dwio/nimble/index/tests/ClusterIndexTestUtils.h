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

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::nimble::index::test {

/// Generator function type for key columns.
/// Takes (vectorIndex, numRows) and returns a VectorPtr.
using KeyColumnGenerator =
    std::function<velox::VectorPtr(size_t, velox::vector_size_t)>;

/// Map from column name to generator function.
using KeyColumnGeneratorMap =
    std::unordered_map<std::string, KeyColumnGenerator>;

/// Generates random data using VectorFuzzer with specified key column
/// generators.
/// @param rowType The type of the row vector.
/// @param numVectors Number of vectors to generate.
/// @param numRowsPerVector Number of rows per vector.
/// @param keyColumnGenerators Map from column name to generator function.
///        Generator function takes (vectorIndex, numRows) and returns a
///        VectorPtr. Key columns are generated without nulls.
/// @param pool Memory pool for vector allocation.
/// @param nullRatio Null ratio for non-key columns (default 0.1).
std::vector<velox::RowVectorPtr> generateData(
    const velox::RowTypePtr& rowType,
    size_t numVectors,
    velox::vector_size_t numRowsPerVector,
    const KeyColumnGeneratorMap& keyColumnGenerators,
    velox::memory::MemoryPool* pool,
    double nullRatio = 0.1);

/// Writes row vectors to a Nimble file with cluster index.
/// @param filePath Path to write the Nimble file.
/// @param data Vector of RowVectorPtr to write.
/// @param clusterIndexConfig Index configuration specifying columns, sort
/// orders, etc.
/// @param pool Memory pool for writing.
/// @param flushPolicyFactory Optional factory for custom flush policy to
/// control
///        stripe sizes. If not provided, uses default flush policy.
void writeFile(
    const std::string& filePath,
    const std::vector<velox::RowVectorPtr>& data,
    std::shared_ptr<const IndexConfig> clusterIndexConfig,
    velox::memory::MemoryPool& pool,
    std::function<std::unique_ptr<FlushPolicy>()> flushPolicyFactory = nullptr);

/// Simple StreamLoader implementation for testing that holds data in memory.
class TestStreamLoader : public StreamLoader {
 public:
  explicit TestStreamLoader(std::string_view data) : data_(data) {}

  const std::string_view getStream() const override {
    return data_;
  }

 private:
  std::string_view data_;
};

/// Wrapper around InMemoryChunkedStream that ensures exactly one chunk is
/// decoded. Use this when the data is expected to contain exactly one chunk.
class SingleChunkDecoder {
 public:
  explicit SingleChunkDecoder(
      velox::memory::MemoryPool& pool,
      std::string_view data)
      : stream_(pool, std::make_unique<TestStreamLoader>(data)) {}

  /// Decodes and returns the single chunk.
  /// Asserts that exactly one chunk exists in the data.
  std::string_view decode() {
    NIMBLE_CHECK(stream_.hasNext(), "Expected at least one chunk");
    auto chunk = stream_.nextChunk();
    NIMBLE_CHECK(!stream_.hasNext(), "Expected exactly one chunk");
    return chunk;
  }

 private:
  InMemoryChunkedStream stream_;
};

/// Holds partition-level cluster index statistics for testing.
struct PartitionStats {
  /// File offset of the key data blob.
  uint64_t keyStreamOffset{0};
  /// Total size of the key data blob.
  uint32_t keyStreamSize{0};
  /// Chunk-level stats (search boundaries).
  std::vector<uint32_t> chunkRows;
  std::vector<uint32_t> chunkOffsets;
  std::vector<std::string> chunkKeys;
};

/// Holds stream position index statistics for a specific stream.
struct StreamStats {
  /// Accumulated chunk count for each stripe for this stream.
  std::vector<uint32_t> chunkCounts;
  /// Accumulated row counts per chunk for this stream.
  std::vector<uint32_t> chunkRows;
  /// Byte offset of each chunk within its stream.
  std::vector<uint32_t> chunkOffsets;
  /// Per-chunk null-value count for this stream. Empty when per-chunk null
  /// statistics are absent from the section.
  std::vector<uint32_t> chunkNullCounts;
};

/// Specification for a single chunk in a stream (for test data creation).
struct ChunkSpec {
  uint32_t rowCount{};
  uint32_t size{};
  uint32_t nullCount{0};
};

/// Specification for a single stream (for test data creation).
struct StreamSpec {
  uint32_t offset;
  std::vector<ChunkSpec> chunks;
};

/// Specification for a key chunk (for test data creation).
struct KeyChunkSpec {
  uint32_t rowCount;
  std::string key;
};

/// Creates chunks for a stream from the given chunk specifications.
/// The buffer is used to store the chunk content.
std::vector<Chunk> createChunks(
    Buffer& buffer,
    const std::vector<ChunkSpec>& chunkSpecs);

/// Creates a Stream with the specified test parameters.
/// The buffer is used to store the chunk content.
Stream createStream(Buffer& buffer, const StreamSpec& spec);

/// Test helper class for ClusterIndex that provides access to private members
/// for testing purposes. This is a friend class of ClusterIndex.
class ClusterIndexTestHelper {
 public:
  static std::unique_ptr<ClusterIndex> create(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      bool pinIndex = false,
      bool preloadIndex = false) {
    return std::unique_ptr<ClusterIndex>(new ClusterIndex(
        std::move(rootSection),
        std::move(metadataInput),
        std::move(dataInput),
        pinIndex,
        preloadIndex,
        pool));
  }

  explicit ClusterIndexTestHelper(const ClusterIndex* clusterIndex)
      : clusterIndex_(clusterIndex) {}

  /// Returns the partition-level statistics for the given partition.
  PartitionStats partitionStats(uint32_t partitionIndex) const;

  /// Returns the partition keys from the root index.
  const std::vector<std::string_view>& partitionKeys() const {
    return clusterIndex_->partitionKeys_;
  }

  /// Lookup chunk by encoded key within a partition.
  ChunkLocation lookupChunk(
      uint32_t partitionIndex,
      std::string_view encodedKey) const {
    return clusterIndex_->lookupChunk(
        clusterIndex_->loadPartition(partitionIndex), encodedKey);
  }

  MetadataSection partitionSection(uint32_t partitionIndex) const {
    return clusterIndex_->partitionSection(partitionIndex);
  }

  velox::common::Region keyStreamRegion(uint32_t partitionIndex) const {
    const auto* partition = clusterIndex_->loadPartition(partitionIndex);
    return velox::common::Region{
        partition->index->key_stream_offset(),
        partition->index->key_stream_size()};
  }

  /// Returns the number of decoded chunks currently cached for the partition.
  /// Counts populated slots in the per-chunk vector. With pinIndex=true the
  /// vector has numChunks slots, one per chunk. With pinIndex=false the
  /// vector has a single shared scratch slot, so the count is at most 1.
  size_t decodedChunkCount(uint32_t partitionIndex) const {
    const auto& decodedChunks =
        clusterIndex_->loadPartition(partitionIndex)->decodedChunks;
    return std::count_if(
        decodedChunks.begin(), decodedChunks.end(), [](const auto& c) {
          return c.data != nullptr;
        });
  }

  /// Returns the raw DecodedKeyChunk pointer for the given chunk slot in a
  /// partition. Used to verify which decode result won the install race.
  const DecodedKeyChunk* decodedChunkData(
      uint32_t partitionIndex,
      uint32_t chunkIndex) const {
    const auto& decodedChunks =
        clusterIndex_->loadPartition(partitionIndex)->decodedChunks;
    NIMBLE_CHECK_LT(chunkIndex, decodedChunks.size());
    return decodedChunks[chunkIndex].data.get();
  }

 private:
  const ClusterIndex* const clusterIndex_;
};

/// Test helper class for ChunkStatsGroup.
class ChunkStatsTestHelper {
 public:
  explicit ChunkStatsTestHelper(const ChunkStatsGroup* chunkIndex)
      : chunkStats_(chunkIndex) {}

  uint32_t firstStripe() const {
    return chunkStats_->firstStripe_;
  }

  uint32_t stripeCount() const {
    return chunkStats_->stripeCount_;
  }

  uint32_t streamCount() const {
    return chunkStats_->streamCount_;
  }

  /// Returns stream position index statistics for the specified stream.
  StreamStats streamStats(uint32_t streamId) const;

 private:
  const ChunkStatsGroup* const chunkStats_;
};

/// Test-only cluster index metadata writer for tablet-level tests.
/// Handles stripe key tracking and FlatBuffer serialization of
/// ClusterIndexPartition and ClusterIndex root index.
/// Does NOT perform key encoding — accepts key chunk specs directly.
///
/// Usage:
///   auto helper = TestClusterIndexMetadataWriter(columns, sortOrders, ...);
///   // For each stripe:
///   helper.addStripe({{.rowCount = 100, .key = "aaa"}});
///   // Wire into TabletWriter options:
///   .stripeGroupFlushCallback = helper.createStripeGroupFlushCallback()
///   .closeCallback = helper.createCloseCallback()
class TestClusterIndexMetadataWriter {
 public:
  TestClusterIndexMetadataWriter(
      velox::memory::MemoryPool& pool,
      std::vector<std::string> columns,
      std::vector<SortOrder> sortOrders,
      bool enforceKeyOrder = false,
      bool noDuplicateKey = false);

  /// Tracks stripe key boundaries and chunk metadata from key chunk specs.
  /// Call once per stripe, before TabletWriter::writeStripe().
  void addStripe(const std::vector<KeyChunkSpec>& chunkSpecs);

  /// Returns a stripe group flush callback for TabletWriter::Options.
  TabletWriter::StripeGroupFlushCallback createStripeGroupFlushCallback();

  /// Returns a close callback for TabletWriter::Options.
  /// Writes the root index as an optional section during file close.
  TabletWriter::CloseCallback createCloseCallback();

 private:
  struct ChunkStats {
    std::vector<uint32_t> chunkRows;
    std::vector<uint32_t> chunkOffsets;
    std::vector<std::string_view> chunkKeys;
  };

  struct IndexPartition {
    ChunkStats chunks;
    std::unique_ptr<Buffer> encodingBuffer;
    // File-level offset and size after flushKeyStream() writes the key data
    // blob.
    uint64_t keyStreamFileOffset{0};
    uint32_t keyStreamFileSize{0};
  };

  void writeRoot(
      const CreateMetadataSectionFn& createMetadataSection,
      const WriteOptionalSectionFn& writeOptionalSection);

  void flushKeyStream(const WriteDataFn& writeData);

  void flushPartitionMetadata(
      const CreateMetadataSectionFn& createMetadataSection);

  velox::memory::MemoryPool* const pool_;
  const std::vector<std::string> columns_;
  const std::vector<SortOrder> sortOrders_;
  const bool enforceKeyOrder_;
  const bool noDuplicateKey_;

  // Buffered encoded key data within current partition.
  std::string encodedKeyData_;

  std::unique_ptr<IndexPartition> partitionIndex_;
  std::vector<std::string_view> stripeKeys_;
  std::vector<MetadataSection> indexPartitions_;
  // Number of stripes in each partition.
  std::vector<uint32_t> stripesPerPartition_;
  // Row count for each partition.
  std::vector<uint32_t> rowsPerPartition_;
  // Number of stripes added since the last partition flush.
  uint32_t currentPartitionStripes_{0};
  std::unique_ptr<Buffer> rootBuffer_;
};

} // namespace facebook::nimble::index::test

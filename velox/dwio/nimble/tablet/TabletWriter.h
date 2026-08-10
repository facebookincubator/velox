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

#include "velox/common/file/File.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Checksum.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/tablet/Chunk.h"
#include "velox/dwio/nimble/tablet/ChunkStatsWriter.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tablet/FooterGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/tablet/StripeGroup.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::nimble {

/// Represents a stream of chunks with a file offset.
struct Stream {
  uint32_t offset{0};
  std::vector<Chunk> chunks;
};

class LayoutPlanner {
 public:
  virtual std::vector<Stream> getLayout(std::vector<Stream>&& streams) = 0;

  virtual ~LayoutPlanner() = default;
};

constexpr uint32_t kMetadataFlushThreshold = 8 * 1024 * 1024; // 8MB
constexpr uint32_t kMetadataCompressionThreshold = 64 * 1024; // 64kB

/// Writes a new nimble file.
class TabletWriter {
 public:
  /// Called AFTER stripe group and chunk stats metadata are written.
  /// Writes partition data (key stream + metadata) aligned with stripe groups.
  using StripeGroupFlushCallback = std::function<void(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn)>;

  /// Called during close() after all stripe groups have been flushed,
  /// before the chunk stats root and footer are written. Used by index
  /// writers to finalize and write the root index as an optional section.
  using CloseCallback = std::function<void(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn,
      const WriteOptionalSectionFn& writeMetadataFn)>;

  struct Options {
    std::unique_ptr<LayoutPlanner> layoutPlanner{nullptr};
    uint32_t metadataFlushThreshold{kMetadataFlushThreshold};
    uint32_t metadataCompressionThreshold{kMetadataCompressionThreshold};
    ChecksumType checksumType{ChecksumType::XXH3_64};
    bool streamDeduplicationEnabled{true};
    // When true, chunk-level position index is built for all streams,
    // enabling O(1) chunk-level seeking within stripes.
    bool enableChunkIndex{false};
    // Skip writing chunk stats for a stripe group if the average number
    // of chunks per stream is below this threshold. 0 disables chunk stats
    // skipping.
    float chunkStatsMinAvgChunks{2};
    // Selects how per-stripe-group stream offsets/sizes are serialized (default
    // kRaw); see StripeGroup::EncodingLayout.
    StripeGroup::EncodingLayout stripeGroupEncodingLayout{
        StripeGroup::EncodingLayout::kRaw};
    // Candidate encodings (with read-cost weights) considered when encoding
    // per-stripe-group offsets/sizes in the encoded layouts. Only O(1)
    // point-access encodings are valid. Empty (default) lets the writer pick
    // its own candidate set (Constant, Trivial, FixedBitWidth).
    std::vector<std::pair<EncodingType, float>>
        stripeGroupEncodingLayoutReadFactors{};
    // Callback invoked at stripe group flush boundaries. Used by index
    // writers (e.g., ClusterIndexWriter) to write partition data aligned
    // with stripe groups.
    StripeGroupFlushCallback stripeGroupFlushCallback;
    CloseCallback closeCallback;
  };

  struct Stats {
    uint64_t duplicateStreamCount{0};
    uint64_t duplicateStreamBytes{0};
  };

  static std::unique_ptr<TabletWriter> create(
      velox::WriteFile* file,
      velox::memory::MemoryPool& pool,
      TabletWriter::Options options);

  // Writes out the footer. Remember that the underlying file is not readable
  // until the write file itself is closed.
  void close();

  // TODO: do we want a stripe header? E.g. per stream min/max (at least for
  // key cols), that sort of thing? We can add later. We'll also want to
  // be able to control the stream order on disk, presumably via a options
  // param to the constructor.
  //
  // The first argument in the map gives the stream name. The second's first
  // gives part gives the uncompressed string returned from a Serialize
  // function, and the second part the compression type to apply when writing to
  // disk. Later we may want to generalize that compression type to include a
  // level or other params.
  //
  // A stream's type must be the same across all stripes.
  void writeStripe(uint32_t rowCount, std::vector<Stream> streams);

  // TODO: with low memory budget, we might want to have a version of this for
  // a vector of content to be concatenated. Right now the Buffer class prevents
  // us from draining its content incrementally.
  void invokeCloseCallback();
  void invokeStripeGroupFlushCallback();

  void writeOptionalSection(std::string name, std::string_view content);

  /// Writes metadata to the file and returns a MetadataSection describing
  /// its location. Used by index writers to store sub-index entries as
  /// separate sections.
  MetadataSection createMetadataSection(std::string_view metadata);

  /// The number of bytes written so far.
  uint64_t size() const {
    return file_->size();
  }

  Stats stats() const {
    return stats_;
  }

 private:
  // Writer state.
  enum class State {
    kActive, // Writer is active and can accept data.
    kClosed, // Writer has been closed and no more data can be written.
  };

  TabletWriter(
      velox::WriteFile* file,
      velox::memory::MemoryPool& pool,
      Options options);

  bool hasChunkStats() const {
    return chunkStatsWriter_ != nullptr;
  }

  // Checks that writer is not closed and throws if it is.
  inline void checkNotClosed() const {
    NIMBLE_USER_CHECK(
        state_ == State::kActive, "TabletWriter is already closed.");
  }

  // Write metadata entry to file. Uses options_.metadataCompressionThreshold
  // to decide whether to compress.
  CompressionType writeMetadata(std::string_view metadata);

  // Write stripe group metadata entry and also add that to footer sections if
  // exceeds metadata flush size.
  void tryWriteStripeGroup(bool force = false);
  bool shouldWriteStripeGroup(bool force) const;
  // Serialize the current stripe group's metadata, dispatching to the
  // layout-specific helper below per options_.stripeGroupEncodingLayout.
  void writeStripeGroup(size_t streamCount, size_t stripeCount);
  // Serialize the current stripe group's per-stream offsets/sizes into
  // `builder` using the kRaw / kStreamMajor layout respectively.
  void writeStripeGroupWithRawLayout(
      flatbuffers::FlatBufferBuilder& builder,
      size_t streamCount,
      size_t stripeCount);
  void writeStripeGroupWithStreamMajorLayout(
      flatbuffers::FlatBufferBuilder& builder,
      size_t streamCount,
      size_t stripeCount);
  void finishStripeGroup();

  // Write stripes metadata section
  MetadataSection writeStripes(size_t stripeCount);

  static flatbuffers::Offset<serialization::OptionalMetadataSections>
  createOptionalMetadataSection(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<std::pair<std::string, MetadataSection>>&
          optionalSections);

  void writeWithChecksum(std::string_view data);
  void writeWithChecksum(const folly::IOBuf& buf);

  void writeStreamWithChecksum(const Stream& stream);

  // Starts chunk stats writing for a new stripe.
  void finishStripeChunkStats(size_t streamCount);

  // Adds chunk-level index data for a stream.
  void addStreamChunkStats(
      uint32_t streamIndex,
      const std::vector<Chunk>& chunks);

  // Writes the chunk stats group for a completed stripe group.
  void writeChunkStatsGroup(size_t streamCount, size_t stripeCount);

  // Writes the chunk stats root.
  void writeChunkStatsRoot();

  velox::WriteFile* const file_;
  velox::memory::MemoryPool* const pool_;
  const Options options_;
  const std::unique_ptr<Checksum> checksum_;
  // Chunk-level position index.
  const std::unique_ptr<ChunkStatsWriter> chunkStatsWriter_;

  // Number of rows in each stripe.
  std::vector<uint32_t> stripeRowCounts_;
  // Offsets from start of file to first byte in each stripe.
  std::vector<uint64_t> stripeOffsets_;
  // Sizes of the stripes
  std::vector<uint32_t> stripeSizes_;

  // For each stripe, which stripe group it belongs to.
  std::vector<uint32_t> stripeGroupIndices_;
  // Accumulated offsets within each stripe relative to start of the stripe,
  // with one value for each seen stream in the current OR PREVIOUS stripes.
  std::vector<std::vector<uint32_t>> streamOffsets_;
  // Accumulated stream sizes within each stripe. Same behavior as
  // streamOffsets_.
  std::vector<std::vector<uint32_t>> streamSizes_;

  // Current stripe group index.
  uint32_t stripeGroupIndex_{0};
  // Stripe groups
  std::vector<MetadataSection> stripeGroups_;
  // Optional sections
  std::unordered_map<std::string, MetadataSection> optionalSections_;
  Stats stats_;

  // Writer state.
  State state_{State::kActive};
};

} // namespace facebook::nimble

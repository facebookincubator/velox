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

#include "velox/dwio/nimble/tablet/TabletWriter.h"

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tablet/Compression.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tablet/StripeGroup.h"

namespace facebook::nimble {

std::unique_ptr<TabletWriter> TabletWriter::create(
    velox::WriteFile* file,
    velox::memory::MemoryPool& pool,
    TabletWriter::Options options) {
  return std::unique_ptr<TabletWriter>(
      new TabletWriter(file, pool, std::move(options)));
}

TabletWriter::TabletWriter(
    velox::WriteFile* file,
    velox::memory::MemoryPool& pool,
    TabletWriter::Options options)
    : file_{file},
      pool_(&pool),
      options_(std::move(options)),
      checksum_{ChecksumFactory::create(options_.checksumType)},
      // TODO: keeps the chunkIndex name for now; rename to the chunkStats
      // naming once per-chunk null/min/max stats are fully rolled out.
      chunkStatsWriter_{
          options_.enableChunkIndex ? std::make_unique<ChunkStatsWriter>(
                                          pool,
                                          options_.chunkStatsMinAvgChunks)
                                    : nullptr} {}

namespace {
template <typename Source, typename Target = Source>
flatbuffers::Offset<flatbuffers::Vector<Target>> createFlattenedVector(
    flatbuffers::FlatBufferBuilder& builder,
    size_t streamCount,
    const std::vector<std::vector<Source>>& source,
    Target defaultValue) {
  // This method is performing the following:
  // 1. Converts the source vector into FlatBuffers representation. The flat
  // buffer representation is a single dimension array, so this method flattens
  // the source vectors into this single dimension array.
  // 2. Source vector contains a child vector per stripe. Each one of those
  // child vectors might contain different entry count than the final number of
  // streams in the tablet. This is because each stripe might have
  // encountered just subset of the streams. Therefore, this method fills up all
  // missing offsets in the target vector with the provided default value.

  auto stripeCount = source.size();
  std::vector<Target> target(stripeCount * streamCount);
  size_t targetIndex = 0;

  for (auto& items : source) {
    NIMBLE_CHECK_LE(
        items.size(),
        streamCount,
        "Corrupted footer. Stream count exceeds expected total number of streams.");

    for (auto i = 0; i < streamCount; ++i) {
      target[targetIndex++] = i < items.size()
          ? static_cast<Target>(items[i])
          : static_cast<Target>(defaultValue);
    }
  }

  return builder.CreateVector<Target>(target);
}

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

struct StreamHash {
  size_t operator()(const Stream* stream) const {
    folly::hash::SpookyHashV2 hash{};
    hash.Init(0, 0);
    for (const auto& chunk : stream->chunks) {
      for (const auto content : chunk.content) {
        hash.Update(content.data(), content.size());
      }
    }

    uint64_t hash1;
    uint64_t hash2;
    hash.Final(&hash1, &hash2);
    return static_cast<std::size_t>(hash1);
  }
};

struct StreamEqual {
  bool operator()(const Stream* lhs, const Stream* rhs) const {
    // Calculate total sizes for early exit
    size_t lhsSize{0};
    for (const auto& chunk : lhs->chunks) {
      for (const auto content : chunk.content) {
        lhsSize += content.size();
      }
    }
    size_t rhsSize{0};
    for (const auto& chunk : rhs->chunks) {
      for (const auto content : chunk.content) {
        rhsSize += content.size();
      }
    }

    if (lhsSize != rhsSize) {
      return false;
    }

    size_t lhsChunkIndex{0}, rhsChunkIndex{0};
    size_t lhsContentIndex{0}, rhsContentIndex{0};
    size_t lhsPosition{0}, rhsPosition{0};

    while (lhsChunkIndex < lhs->chunks.size() &&
           rhsChunkIndex < rhs->chunks.size()) {
      if (lhsContentIndex >= lhs->chunks[lhsChunkIndex].content.size()) {
        ++lhsChunkIndex;
        lhsContentIndex = 0;
        if (lhsChunkIndex >= lhs->chunks.size()) {
          break;
        }
      }
      if (rhsContentIndex >= rhs->chunks[rhsChunkIndex].content.size()) {
        ++rhsChunkIndex;
        rhsContentIndex = 0;
        if (rhsChunkIndex >= rhs->chunks.size()) {
          break;
        }
      }

      const auto& lhsView = lhs->chunks[lhsChunkIndex].content[lhsContentIndex];
      const auto& rhsView = rhs->chunks[rhsChunkIndex].content[rhsContentIndex];

      size_t compareSize =
          std::min(lhsView.size() - lhsPosition, rhsView.size() - rhsPosition);

      if (std::memcmp(
              lhsView.data() + lhsPosition,
              rhsView.data() + rhsPosition,
              compareSize) != 0) {
        return false;
      }

      lhsPosition += compareSize;
      rhsPosition += compareSize;

      if (lhsPosition == lhsView.size()) {
        ++lhsContentIndex;
        lhsPosition = 0;
      }
      if (rhsPosition == rhsView.size()) {
        ++rhsContentIndex;
        rhsPosition = 0;
      }
    }

    return true;
  }
};
} // namespace

void TabletWriter::close() {
  checkNotClosed();
  const auto stripeCount = stripeOffsets_.size();
  NIMBLE_CHECK(
      stripeCount == stripeSizes_.size() &&
          stripeCount == stripeRowCounts_.size() &&
          stripeCount == stripeGroupIndices_.size(),
      "Stripe count mismatch.");

  // Write remaining stripe groups.
  tryWriteStripeGroup(true);

  invokeCloseCallback();

  // Write chunk stats root.
  writeChunkStatsRoot();

  // write stripes
  MetadataSection stripes = writeStripes(stripeCount);

  // When stripeCount > 0, stripe data has already been written to the file,
  // so file_->size() must be > 0 when writeStripes records the offset.
  // An offset of 0 means the footer would point to stripe data instead of the
  // stripes FlatBuffer, producing a corrupt file that crashes readers.
  if (stripeCount > 0) {
    NIMBLE_CHECK(
        stripes.offset() > 0,
        "Stripes metadata offset is 0 but stripeCount is {}. "
        "WriteFile::size() returned 0 before writing stripes metadata. "
        "Current file size: {}, stripes section size: {}.",
        stripeCount,
        file_->size(),
        stripes.size());
  }

  const uint64_t totalRows = std::accumulate(
      stripeRowCounts_.begin(), stripeRowCounts_.end(), uint64_t{0});

  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
  // write footer
  builder.Finish(
      serialization::CreateFooter(
          builder,
          totalRows,
          stripeCount > 0 ? serialization::CreateMetadataSection(
                                builder,
                                stripes.offset(),
                                stripes.size(),
                                static_cast<serialization::CompressionType>(
                                    stripes.compressionType()),
                                stripes.uncompressedSize().value())
                          : 0,
          !stripeGroups_.empty()
              ? builder.CreateVector<
                    flatbuffers::Offset<serialization::MetadataSection>>(
                    stripeGroups_.size(),
                    [this, &builder](size_t i) {
                      return serialization::CreateMetadataSection(
                          builder,
                          stripeGroups_[i].offset(),
                          stripeGroups_[i].size(),
                          static_cast<serialization::CompressionType>(
                              stripeGroups_[i].compressionType()),
                          stripeGroups_[i].uncompressedSize().value());
                    })
              : 0,
          !optionalSections_.empty()
              ? createOptionalMetadataSection(
                    builder,
                    {optionalSections_.begin(), optionalSections_.end()})
              : 0));

  auto footerStart = file_->size();
  auto footerCompressionType = writeMetadata(asView(builder));

  // Write postscript. The first kPostscriptChecksumedSize bytes are included
  // in the file checksum; the rest (including the checksum itself) are not.
  const uint64_t footerSize = file_->size() - footerStart;
  NIMBLE_CHECK_LE(
      footerSize, std::numeric_limits<uint32_t>::max(), "Footer size too big.");
  Postscript ps{
      static_cast<uint32_t>(footerSize),
      footerCompressionType,
      options_.checksumType,
      kVersionMajor,
      kVersionMinor};
  auto psBuf = ps.serialize();
  writeWithChecksum({psBuf.data(), kPostscriptChecksumedSize});
  // Patch the checksum value into the serialized buffer. The checksum field
  // starts after checksumType (1 byte after the checksummed region).
  const uint64_t checksum = checksum_->getChecksum();
  constexpr uint32_t kChecksumOffset = kPostscriptChecksumedSize + 1;
  std::memcpy(psBuf.data() + kChecksumOffset, &checksum, sizeof(checksum));
  file_->append(
      {psBuf.data() + kPostscriptChecksumedSize,
       kPostscriptSize - kPostscriptChecksumedSize});

  state_ = State::kClosed;
}

void TabletWriter::writeStripe(uint32_t rowCount, std::vector<Stream> streams) {
  checkNotClosed();
  if (UNLIKELY(rowCount == 0)) {
    return;
  }

  stripeRowCounts_.push_back(rowCount);
  stripeOffsets_.push_back(file_->size());

  const auto streamCount = streams.empty()
      ? 0
      : std::max_element(
            streams.begin(),
            streams.end(),
            [](const auto& a, const auto& b) {
              return a.offset < b.offset;
            })->offset +
          1;

  auto& stripeStreamOffsets = streamOffsets_.emplace_back(streamCount, 0);
  auto& stripeStreamSizes = streamSizes_.emplace_back(streamCount, 0);

  finishStripeChunkStats(streamCount);

  if (options_.layoutPlanner != nullptr) {
    streams = options_.layoutPlanner->getLayout(std::move(streams));
  }

  if (options_.streamDeduplicationEnabled) {
    folly::F14FastMap<
        const Stream*,
        std::pair<uint32_t, uint32_t>,
        StreamHash,
        StreamEqual>
        uniqueStreams;

    for (const auto& stream : streams) {
      const uint32_t index = stream.offset;
      auto it = uniqueStreams.emplace(
          &stream, std::make_pair<uint32_t, uint32_t>(0, 0));

      NIMBLE_DCHECK_GT(
          stripeStreamOffsets.size(),
          index,
          "Invalid stripe stream offsets capacity.");
      NIMBLE_DCHECK_GT(
          stripeStreamSizes.size(),
          index,
          "Invalid stripe stream sizes capacity.");

      if (it.second) {
        // If we are here, this is a new stream, and needs to be added to the
        // file.
        NIMBLE_CHECK_LT(
            file_->size() - stripeOffsets_.back(),
            std::numeric_limits<uint32_t>::max(),
            "Unexpected stream offset");
        stripeStreamOffsets[index] =
            static_cast<uint32_t>(file_->size() - stripeOffsets_.back());

        writeStreamWithChecksum(stream);
        addStreamChunkStats(stream.offset, stream.chunks);

        NIMBLE_DCHECK_LT(
            file_->size() -
                (stripeStreamOffsets[index] + stripeOffsets_.back()),
            std::numeric_limits<uint32_t>::max(),
            "Unexpected stream size");
        stripeStreamSizes[index] = static_cast<uint32_t>(
            file_->size() -
            (stripeStreamOffsets[index] + stripeOffsets_.back()));

        it.first->second.first = stripeStreamOffsets[index];
        it.first->second.second = stripeStreamSizes[index];
      } else {
        // If we are here, this is a duplicate stream, so we need to reference
        // the original stream instead of this one.
        ++stats_.duplicateStreamCount;
        for (const auto& chunk : stream.chunks) {
          stats_.duplicateStreamBytes += chunk.contentSize();
        }

        // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
        stripeStreamOffsets[index] = it.first->second.first;
        // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
        stripeStreamSizes[index] = it.first->second.second;
        addStreamChunkStats(index, it.first->first->chunks);
      }
    }
  } else {
    for (const auto& stream : streams) {
      const uint32_t index = stream.offset;

      NIMBLE_DCHECK_GT(
          stripeStreamOffsets.size(),
          index,
          "Invalid stripe stream offsets capacity.");
      NIMBLE_DCHECK_GT(
          stripeStreamSizes.size(),
          index,
          "Invalid stripe stream sizes capacity.");

      NIMBLE_DCHECK_LT(
          file_->size() - stripeOffsets_.back(),
          std::numeric_limits<uint32_t>::max(),
          "Unexpected stream offset");
      stripeStreamOffsets[index] =
          static_cast<uint32_t>(file_->size() - stripeOffsets_.back());

      writeStreamWithChecksum(stream);
      addStreamChunkStats(stream.offset, stream.chunks);

      NIMBLE_DCHECK_LT(
          file_->size() - (stripeStreamOffsets[index] + stripeOffsets_.back()),
          std::numeric_limits<uint32_t>::max(),
          "Unexpected stream size");
      // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
      stripeStreamSizes[index] = static_cast<uint32_t>(
          file_->size() - (stripeStreamOffsets[index] + stripeOffsets_.back()));
    }
  }

  stripeSizes_.push_back(file_->size() - stripeOffsets_.back());

  stripeGroupIndices_.push_back(stripeGroupIndex_);
  // Write stripe group if size of column offsets/sizes is too large.
  tryWriteStripeGroup();
}

CompressionType TabletWriter::writeMetadata(std::string_view metadata) {
  const auto size = metadata.size();
  const bool shouldCompress = size > options_.metadataCompressionThreshold;
  CompressionType compressionType{CompressionType::Uncompressed};
  std::optional<velox::BufferPtr> compressed;
  if (shouldCompress) {
    compressed = ZstdCompression::compress(metadata, pool_);
    if (compressed.has_value()) {
      compressionType = CompressionType::Zstd;
      metadata = {compressed.value()->as<char>(), compressed.value()->size()};
    }
  }
  writeWithChecksum(metadata);
  return compressionType;
}

MetadataSection TabletWriter::createMetadataSection(std::string_view metadata) {
  const auto uncompressedSize = static_cast<uint32_t>(metadata.size());
  auto offset = file_->size();
  auto compressionType = writeMetadata(metadata);
  auto size = static_cast<uint32_t>(file_->size() - offset);
  return MetadataSection{offset, size, compressionType, uncompressedSize};
}

void TabletWriter::invokeCloseCallback() {
  if (options_.closeCallback == nullptr) {
    return;
  }
  auto writeDataFn = [this](const std::vector<std::string_view>& segments) {
    const auto offset = file_->size();
    for (const auto& segment : segments) {
      writeWithChecksum(segment);
    }
    return std::make_pair(
        offset, static_cast<uint32_t>(file_->size() - offset));
  };
  auto createMetadataFn = [this](std::string_view metadata) {
    return createMetadataSection(metadata);
  };
  auto writeMetadataFn = [this](std::string name, std::string_view content) {
    writeOptionalSection(std::move(name), content);
  };
  options_.closeCallback(writeDataFn, createMetadataFn, writeMetadataFn);
}

void TabletWriter::invokeStripeGroupFlushCallback() {
  if (options_.stripeGroupFlushCallback == nullptr) {
    return;
  }
  auto writeDataFn = [this](const std::vector<std::string_view>& segments) {
    const auto offset = file_->size();
    for (const auto& segment : segments) {
      writeWithChecksum(segment);
    }
    return std::make_pair(
        offset, static_cast<uint32_t>(file_->size() - offset));
  };
  auto createMetadataFn = [this](std::string_view metadata) {
    return createMetadataSection(metadata);
  };
  options_.stripeGroupFlushCallback(writeDataFn, createMetadataFn);
}

void TabletWriter::writeOptionalSection(
    std::string name,
    std::string_view content) {
  checkNotClosed();
  NIMBLE_CHECK(!name.empty(), "Optional section name cannot be empty.");
  NIMBLE_CHECK(
      optionalSections_.find(name) == optionalSections_.end(),
      "Optional section '{}' already exists.",
      name);
  optionalSections_.try_emplace(
      std::move(name), createMetadataSection(content));
}

bool TabletWriter::shouldWriteStripeGroup(bool force) const {
  const auto stripeCount = streamOffsets_.size();
  if (stripeCount == 0) {
    return false;
  }

  // Estimate size
  // 8 bytes for offsets, 4 for size, 1 for compression type, so 13.
  const size_t estimatedSize =
      4 + stripeCount * streamOffsets_.back().size() * 13;
  if (!force && (estimatedSize < options_.metadataFlushThreshold)) {
    return false;
  }
  return true;
}

namespace {
// Build an encoding policy restricted to the given read factors. Read factors
// must list only encodings with O(1) stateless point access (the EncodingView
// requirement); the selector picks the smallest predicted size among them,
// biased by each candidate's read weight. When `readFactors` is empty the
// writer falls back to `kDefaultReadFactors` below.
template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> makeMetadataEncodingPolicy(
    const std::vector<std::pair<EncodingType, float>>& readFactors) {
  static const std::vector<std::pair<EncodingType, float>> kDefaultReadFactors{
      {EncodingType::Constant, 1.0},
      {EncodingType::Trivial, 1.0},
      {EncodingType::FixedBitWidth, 1.0},
  };
  ManualEncodingSelectionPolicyFactory factory{
      readFactors.empty() ? kDefaultReadFactors : readFactors,
      /*compressionOptions=*/std::nullopt};
  auto base = factory.createPolicy(TypeTraits<T>::dataType);
  return std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(base.release()));
}

// Transposes the per-stripe vectors `source` into a single per-stream vector
// of length `stripeCount`, gathering `source[stripe][streamId]` for one fixed
// streamId across every stripe. Stripes whose recorded streamCount is less
// than `streamId + 1` contribute 0 (the writer's default for absent streams).
std::vector<uint32_t> transposeStreamMetadata(
    uint32_t streamId,
    size_t stripeCount,
    const std::vector<std::vector<uint32_t>>& source) {
  std::vector<uint32_t> streamValues(stripeCount, 0);
  for (size_t stripe = 0; stripe < stripeCount; ++stripe) {
    if (streamId < source[stripe].size()) {
      streamValues[stripe] = source[stripe][streamId];
    }
  }
  return streamValues;
}
} // namespace

void TabletWriter::writeStripeGroupWithRawLayout(
    flatbuffers::FlatBufferBuilder& builder,
    size_t streamCount,
    size_t stripeCount) {
  // Flattened stripe-major arrays — the original wire layout every reader
  // understands. Short stripes are padded to streamCount with 0; the reader
  // derives streamCount from the array length.
  auto streamOffsets = createFlattenedVector<uint32_t, uint32_t>(
      builder, streamCount, streamOffsets_, 0);
  auto streamSizes = createFlattenedVector<uint32_t, uint32_t>(
      builder, streamCount, streamSizes_, 0);
  builder.Finish(
      serialization::CreateStripeGroup(
          builder,
          static_cast<uint32_t>(stripeCount),
          streamOffsets,
          streamSizes));
}

void TabletWriter::writeStripeGroupWithStreamMajorLayout(
    flatbuffers::FlatBufferBuilder& builder,
    size_t streamCount,
    size_t stripeCount) {
  // For each leaf streamId, gather that stream's values across every stripe and
  // encode the resulting stripeCount-length array independently. The per-stream
  // bit width is tight to that stream's value range.
  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedOffsets;
  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedSizes;
  encodedOffsets.reserve(streamCount);
  encodedSizes.reserve(streamCount);

  // Encodes one array, copies it into the flatbuffer, then rewinds the scratch
  // buffer so peak memory stays bounded to a single stream.
  Buffer encodingBuffer{*pool_};
  auto encodeStream = [&](const std::vector<uint32_t>& values) {
    const auto encoded = EncodingFactory::encode<uint32_t>(
        makeMetadataEncodingPolicy<uint32_t>(
            options_.stripeGroupEncodingLayoutReadFactors),
        std::span<const uint32_t>(values),
        encodingBuffer);
    auto encodedStream = serialization::CreateEncodedStream(
        builder,
        builder.CreateVector(
            reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size()));
    encodingBuffer.reset();
    return encodedStream;
  };

  for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
    encodedOffsets.push_back(encodeStream(
        transposeStreamMetadata(streamId, stripeCount, streamOffsets_)));
    encodedSizes.push_back(encodeStream(
        transposeStreamMetadata(streamId, stripeCount, streamSizes_)));
  }

  // Tag the blob with the kStreamMajor file identifier so the reader detects
  // the layout from the buffer itself (kRaw blobs are written bare).
  builder.Finish(
      serialization::CreateStreamMajorStripeGroup(
          builder,
          static_cast<uint32_t>(stripeCount),
          static_cast<uint32_t>(streamCount),
          builder.CreateVector(encodedOffsets),
          builder.CreateVector(encodedSizes)),
      StripeGroup::kStreamMajorLayoutIdentifier.data());
}

void TabletWriter::writeStripeGroup(size_t streamCount, size_t stripeCount) {
  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);

  switch (options_.stripeGroupEncodingLayout) {
    case StripeGroup::EncodingLayout::kRaw:
      writeStripeGroupWithRawLayout(builder, streamCount, stripeCount);
      break;
    case StripeGroup::EncodingLayout::kStreamMajor:
      writeStripeGroupWithStreamMajorLayout(builder, streamCount, stripeCount);
      break;
    default:
      NIMBLE_UNREACHABLE(
          fmt::format(
              "Unknown StripeGroup encoding layout: {}",
              static_cast<int>(options_.stripeGroupEncodingLayout)));
  }

  stripeGroups_.emplace_back(createMetadataSection(asView(builder)));
}

MetadataSection TabletWriter::writeStripes(size_t stripeCount) {
  if (stripeCount == 0) {
    return {};
  }
  flatbuffers::FlatBufferBuilder stripesBuilder(kInitialFooterSize);
  stripesBuilder.Finish(
      serialization::CreateStripes(
          stripesBuilder,
          stripesBuilder.CreateVector(stripeRowCounts_),
          stripesBuilder.CreateVector(stripeOffsets_),
          stripesBuilder.CreateVector(stripeSizes_),
          stripesBuilder.CreateVector(stripeGroupIndices_)));
  return createMetadataSection(asView(stripesBuilder));
}

flatbuffers::Offset<serialization::OptionalMetadataSections>
TabletWriter::createOptionalMetadataSection(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<std::pair<std::string, MetadataSection>>&
        optionalSections) {
  return serialization::CreateOptionalMetadataSections(
      builder,
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          optionalSections.size(),
          [&builder, &optionalSections](size_t i) {
            return builder.CreateString(optionalSections[i].first);
          }),
      builder.CreateVector<uint64_t>(
          optionalSections.size(),
          [&optionalSections](size_t i) {
            return optionalSections[i].second.offset();
          }),
      builder.CreateVector<uint32_t>(
          optionalSections.size(),
          [&optionalSections](size_t i) {
            return optionalSections[i].second.size();
          }),
      builder.CreateVector<uint8_t>(
          optionalSections.size(),
          [&optionalSections](size_t i) {
            return static_cast<uint8_t>(
                optionalSections[i].second.compressionType());
          }),
      builder.CreateVector<uint32_t>(
          optionalSections.size(), [&optionalSections](size_t i) {
            return optionalSections[i].second.uncompressedSize().value();
          }));
}

void TabletWriter::tryWriteStripeGroup(bool force) {
  if (!shouldWriteStripeGroup(force)) {
    return;
  }

  const auto stripeCount = streamOffsets_.size();
  NIMBLE_CHECK_GT(stripeCount, 0);

  const auto maxStreamCountIt = std::max_element(
      streamOffsets_.begin(),
      streamOffsets_.end(),
      [](const auto& a, const auto& b) { return a.size() < b.size(); });
  NIMBLE_CHECK(maxStreamCountIt != streamOffsets_.end());

  const auto streamCount = maxStreamCountIt->size();

  // Each stripe may have different stream count recorded.
  // We need to pad shorter stripes to the full length of stream count.
  // Handled by the |transposeStreamMetadata| helper inside writeStripeGroup.
  writeStripeGroup(streamCount, stripeCount);

  writeChunkStatsGroup(streamCount, stripeCount);

  invokeStripeGroupFlushCallback();

  finishStripeGroup();
}

void TabletWriter::finishStripeGroup() {
  streamOffsets_.clear();
  streamSizes_.clear();
  // Move to the next stripe group
  ++stripeGroupIndex_;
}

void TabletWriter::writeWithChecksum(std::string_view data) {
  file_->append(data);
  checksum_->update(data);
}

void TabletWriter::writeWithChecksum(const folly::IOBuf& buf) {
  for (auto buffer : buf) {
    writeWithChecksum(
        {reinterpret_cast<const char*>(buffer.data()), buffer.size()});
  }
}

void TabletWriter::writeStreamWithChecksum(const Stream& stream) {
  for (const auto& chunk : stream.chunks) {
    for (const auto& content : chunk.content) {
      writeWithChecksum(content);
    }
  }
}

void TabletWriter::finishStripeChunkStats(size_t streamCount) {
  if (hasChunkStats()) {
    chunkStatsWriter_->newStripe(streamCount);
  }
}

void TabletWriter::addStreamChunkStats(
    uint32_t streamIndex,
    const std::vector<Chunk>& chunks) {
  if (!hasChunkStats()) {
    return;
  }
  chunkStatsWriter_->addStream(streamIndex, chunks);
}

void TabletWriter::writeChunkStatsGroup(
    size_t streamCount,
    size_t stripeCount) {
  if (hasChunkStats()) {
    auto createMetadataFn = [this](std::string_view metadata) {
      return createMetadataSection(metadata);
    };
    chunkStatsWriter_->writeGroup(streamCount, stripeCount, createMetadataFn);
  }
}

void TabletWriter::writeChunkStatsRoot() {
  if (hasChunkStats()) {
    auto writeOptionalSectionFn =
        [this](std::string name, std::string_view content) {
          writeOptionalSection(std::move(name), content);
        };
    chunkStatsWriter_->writeRoot(writeOptionalSectionFn);
  }
}
} // namespace facebook::nimble

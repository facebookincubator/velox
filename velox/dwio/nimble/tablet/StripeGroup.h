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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

namespace serialization {
struct StreamMajorStripeGroup;
} // namespace serialization

/// Reads a single stripe group's per-stream metadata: the byte offset and size
/// of every stream within each stripe of the group. The metadata's on-disk
/// layout is one of EncodingLayout, self-identified by the blob's FlatBuffers
/// file identifier.
class StripeGroup {
 public:
  /// Selects how a stripe group's per-stream offsets/sizes are serialized (see
  /// Footer.fbs). kRaw is the default, universally-readable layout;
  /// kStreamMajor (per-stream encoded) is experimental. Each non-kRaw blob
  /// self-identifies via a FlatBuffers file identifier, so no separate on-disk
  /// flag is needed; a kRaw blob carries no identifier, hence an unrecognized
  /// one decodes as kRaw and pre-existing files keep reading as kRaw. The
  /// values are a frozen on-disk contract (encoded in the identifier):
  /// append-only, never renumber/reorder.
  enum class EncodingLayout : uint8_t {
    kRaw = 0,
    kStreamMajor = 1,
  };

  /// Composes an EncodingLayout's 4-byte FlatBuffers file identifier from a
  /// fixed "SGL" (Stripe Group Layout) prefix + the EncodingLayout value as an
  /// ASCII digit. Returns a null-terminated 5-char array so the same storage
  /// serves both FlatBufferBuilder::Finish (which strlen's the id and requires
  /// exactly 4 bytes) and flatbuffers::BufferHasIdentifier (see stripeGroup's
  /// layout detection). The digit is derived from the enum, so the enum values
  /// are a frozen on-disk contract: append-only, never renumber/reorder.
  static constexpr std::array<char, 5> makeLayoutIdentifier(
      EncodingLayout layout) {
    return {
        'S', 'G', 'L', static_cast<char>('0' + static_cast<int>(layout)), '\0'};
  }

  /// kStreamMajor blob identifier ("SGL1"). kRaw blobs are written bare, so an
  /// absent or unrecognized identifier (including pre-existing files) decodes
  /// as kRaw. Spelled out here (rather than makeLayoutIdentifier(kStreamMajor))
  /// because a static constexpr member cannot be initialized by a same-class
  /// static method; the static_assert below keeps the two in sync.
  static constexpr std::array<char, 5> kStreamMajorLayoutIdentifier{
      'S',
      'G',
      'L',
      static_cast<char>('0' + static_cast<int>(EncodingLayout::kStreamMajor)),
      '\0'};

  /// When `encodingViewDeduplicationEnabled` is true, kStreamMajor streams
  /// whose arrays reference the same encoded payload share one EncodingView.
  /// Streams of a sparse group always share them, since the writer always
  /// deduplicates their arrays.
  StripeGroup(
      uint32_t stripeGroupIndex,
      const MetadataBuffer& stripes,
      std::unique_ptr<MetadataBuffer> metadata,
      bool encodingViewDeduplicationEnabled,
      velox::memory::MemoryPool* pool);

  ~StripeGroup();

  uint32_t index() const {
    return index_;
  }

  uint32_t streamCount() const {
    return streamCount_;
  }

  uint32_t firstStripe() const {
    return firstStripe_;
  }

  uint32_t stripeCount() const {
    return stripeCount_;
  }

  /// Random-access read of the byte offset of `streamId` within `stripeIndex`
  /// (relative to the stripe's start). Backed by the EncodingView over
  /// `streamId`'s offsets array, which other streams may share. Stateless and
  /// safe to call concurrently. O(1), except that a stream with a stripe list
  /// first searches it, which is O(log k) for a stream with data in k stripes.
  /// `streamId` must be less than streamCount(). The offset of a stream
  /// without data in the stripe (size 0) carries no meaning. In a sparse group,
  /// it is 0.
  uint32_t streamOffset(uint32_t stripeIndex, uint32_t streamId) const;

  /// Random-access read of the byte length of `streamId` within `stripeIndex`,
  /// with the same cost as streamOffset(). Stateless. `streamId` must be less
  /// than streamCount().
  uint32_t streamSize(uint32_t stripeIndex, uint32_t streamId) const;

  /// Returns the number of encoding views this group holds for random access
  /// to its stream-major metadata. With encoding view deduplication, streams
  /// whose encoded arrays share one payload share one view, so this can be
  /// below the number of arrays: two per stream, three in a sparse group.
  /// Zero for kRaw.
  size_t testingEncodingViewCount() const;

  /// Relative byte location of one stream within a stripe. A zero size means
  /// the stream is absent.
  struct StreamMetadata {
    uint32_t offset{0};
    uint32_t size{0};
  };

  /// Reads locations for all streams in one stripe. `locations.size()` must
  /// equal streamCount().
  void streamLocations(
      uint32_t stripeIndex,
      std::span<StreamMetadata> locations) const;

  /// Reads locations for selected streams in one stripe. Stream IDs beyond
  /// this group's stream count and streams with zero size produce absent
  /// locations. `streamIds` and `locations` must have equal sizes.
  void streamLocations(
      uint32_t stripeIndex,
      std::span<const uint32_t> streamIds,
      std::span<StreamMetadata> locations) const;

  /// Returns the location of `streamId` within `stripeIndex`. The location is
  /// absent (size 0) when the stream has no data in the stripe or `streamId` is
  /// beyond this group's stream count.
  StreamMetadata streamLocation(uint32_t stripeIndex, uint32_t streamId) const;

 private:
  // Maps an absolute stripe index to this group's local [0, stripeCount_)
  // index. Debug-checks that the stripe belongs to this group.
  uint32_t stripeOffset(uint32_t stripeIndex) const;

  // kRaw: raw pointers into the flatbuffer blob, laid out stripe-major
  // (stripeCount x streamCount). The blob is owned by metadata_ and outlives
  // the StripeGroup.
  struct RawMetadata {
    const uint32_t* offsets{nullptr};
    const uint32_t* sizes{nullptr};
  };

  // kStreamMajor: the views over each stream's offsets and sizes arrays and,
  // when it has one, over its stripe list. A stream without a stripe list
  // holds stripeCount values per array. In a dense group (no stream_stripes),
  // every stream carries both arrays, and a size of 0 marks a stripe without
  // the stream. In a sparse group (with stream_stripes), a stream stores only
  // stripes with data, and a stream without data in any stripe omits its
  // arrays. With encoding view deduplication, and always in a sparse group,
  // streams whose arrays share one encoded payload (TabletWriter stripe-group
  // metadata deduplication) share one view, which is safe because view reads
  // are stateless.
  class StreamMajorMetadata {
   public:
    // The last stripe-list search of a batched lookup. Streams that share a
    // list share the stripe's row in it, and a column's streams, such as a
    // flat-map key's, have adjacent ids and the same list, so reusing the last
    // search skips most searches.
    struct StripeSearch {
      // The stripe list searched last. Null before the first search.
      const EncodingView* stripeList{nullptr};

      // The row of the looked-up stripe in `stripeList`, or std::nullopt when
      // the list does not contain it.
      std::optional<uint32_t> row;
    };

    StreamMajorMetadata() = default;

    // Builds the views of every stream in `root`, a group of `stripeCount`
    // stripes, and validates each distinct stripe list.
    StreamMajorMetadata(
        const serialization::StreamMajorStripeGroup& root,
        uint32_t stripeCount,
        bool encodingViewDeduplicationEnabled,
        velox::memory::MemoryPool* pool);

    // Returns the row of group-local stripe `stripeIndex` in the offsets and
    // sizes of `streamId`, or std::nullopt when the stream holds none for it:
    // it has no arrays, or its stripe list does not list the stripe. A batched
    // lookup passes one `lastSearch` for all its streams to reuse the previous
    // stream's search when both share a stripe list.
    std::optional<uint32_t> row(
        uint32_t streamId,
        uint32_t stripeIndex,
        StripeSearch& lastSearch) const;

    // Returns the offset of `streamId` at `row`, which row() returned.
    uint32_t offset(uint32_t streamId, uint32_t row) const;

    // Returns the size of `streamId` at `row`, which row() returned. A sparse
    // group stores only stripes with data, so a size of 0 in one fails as a
    // corrupt file.
    uint32_t size(uint32_t streamId, uint32_t row) const;

    // Returns the location of `streamId` in group-local stripe `stripeIndex`,
    // or an absent location when the stream has no data there.
    StreamMetadata location(
        uint32_t streamId,
        uint32_t stripeIndex,
        StripeSearch& lastSearch) const;

    size_t numViews() const {
      return views_.size();
    }

   private:
    // Views of one stream. `stripeList` lists the group-local stripes where
    // the stream has data and is null when the stream has no stripe list.
    // `offsets` and `sizes` hold one value per listed stripe, or per stripe
    // without a list, and are null when the stream omits its arrays.
    struct StreamViews {
      EncodingView* stripeList{nullptr};
      EncodingView* offsets{nullptr};
      EncodingView* sizes{nullptr};
    };

    // Whether the group has stream_stripes, in which a size of 0 is corrupt.
    const bool sparseLayout_{false};

    // Owns the views that streams_ point at.
    std::vector<std::unique_ptr<EncodingView>> views_;

    // Views of each stream, indexed by stream id.
    std::vector<StreamViews> streams_;
  };

  // Builds the stream-major metadata of a kStreamMajor group with streams, or
  // returns empty metadata for any other group.
  static StreamMajorMetadata makeStreamMajorMetadata(
      EncodingLayout encodingLayout,
      std::string_view content,
      bool encodingViewDeduplicationEnabled,
      velox::memory::MemoryPool* pool);

  const std::unique_ptr<MetadataBuffer> metadata_;
  const uint32_t index_;
  // How this group's per-stream offsets/sizes are laid out; detected once at
  // construction from the blob's FlatBuffers file identifier. Determines which
  // metadata member below is populated. All encoded forms are written
  // uncompressed (which EncodingView requires); every EncodingView-supported
  // encoding provides a stateless const readAt, so point access needs no
  // auxiliary state and is safe for concurrent reads.
  const EncodingLayout encodingLayout_;
  const StreamMajorMetadata streamMajor_;
  uint32_t streamCount_{0};
  uint32_t stripeCount_{0};
  uint32_t firstStripe_{0};
  RawMetadata raw_;
};

static_assert(
    StripeGroup::kStreamMajorLayoutIdentifier ==
        StripeGroup::makeLayoutIdentifier(
            StripeGroup::EncodingLayout::kStreamMajor),
    "kStreamMajorLayoutIdentifier must match makeLayoutIdentifier(kStreamMajor).");
static_assert(
    static_cast<int>(StripeGroup::EncodingLayout::kStreamMajor) == 1,
    "On-disk identifier contract: kStreamMajor must stay 1.");

/// Returns the enumerator name (e.g. "kStreamMajor") for logging/formatting.
std::string_view toString(StripeGroup::EncodingLayout layout);

std::ostream& operator<<(std::ostream& os, StripeGroup::EncodingLayout layout);

} // namespace facebook::nimble

template <>
struct fmt::formatter<facebook::nimble::StripeGroup::EncodingLayout>
    : fmt::formatter<std::string_view> {
  auto format(
      facebook::nimble::StripeGroup::EncodingLayout layout,
      format_context& ctx) const {
    return fmt::formatter<std::string_view>::format(
        facebook::nimble::toString(layout), ctx);
  }
};

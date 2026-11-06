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

/// Reads a single stripe group's per-stream metadata: the byte offset and size
/// of every stream within each stripe of the group. The metadata's on-disk
/// layout is one of EncodingLayout, self-identified by the blob's FlatBuffers
/// file identifier.
class StripeGroup {
 public:
  /// Selects how a stripe group's per-stream offsets/sizes are serialized (see
  /// Footer.fbs). kRaw is the default, universally-readable layout;
  /// kStreamMajor and kStreamMajorSparse are experimental. Each non-kRaw blob
  /// self-identifies via a FlatBuffers file identifier; a kRaw blob carries
  /// none, so pre-existing files keep reading as kRaw. An unknown layout
  /// identifier is rejected as unsupported, but readers that predate a layout
  /// misread its blobs as kRaw, so every reader must support a layout before
  /// any writer uses it. The values are a frozen on-disk contract (encoded in
  /// the identifier): append-only, never renumber/reorder.
  enum class EncodingLayout : uint8_t {
    kRaw = 0,
    kStreamMajor = 1,
    kStreamMajorSparse = 2,
  };

  /// Prefix ("Stripe Group Layout") of every layout identifier, reserved for
  /// stripe-group layouts.
  static constexpr std::string_view kLayoutIdentifierPrefix{"SGL"};

  /// Composes an EncodingLayout's 4-byte FlatBuffers file identifier from
  /// kLayoutIdentifierPrefix + the EncodingLayout value as an ASCII digit.
  /// Returns a null-terminated 5-char array so the same storage serves both
  /// FlatBufferBuilder::Finish (which strlen's the id and requires exactly 4
  /// bytes) and the layout detection, which compares it with a blob's
  /// identifier. The digit is derived from the enum, so the enum values are a
  /// frozen on-disk contract: append-only, never renumber/reorder.
  static constexpr std::array<char, 5> makeLayoutIdentifier(
      EncodingLayout layout) {
    return {
        kLayoutIdentifierPrefix[0],
        kLayoutIdentifierPrefix[1],
        kLayoutIdentifierPrefix[2],
        static_cast<char>('0' + static_cast<int>(layout)),
        '\0',
    };
  }

  /// kStreamMajor blob identifier ("SGL1"). kRaw blobs are written bare, so an
  /// absent identifier (including pre-existing files) decodes as kRaw. Spelled
  /// out here (rather than makeLayoutIdentifier(kStreamMajor)) because a static
  /// constexpr member cannot be initialized by a same-class static method; the
  /// static_assert below keeps the two in sync.
  static constexpr std::array<char, 5> kStreamMajorLayoutIdentifier{
      'S',
      'G',
      'L',
      static_cast<char>('0' + static_cast<int>(EncodingLayout::kStreamMajor)),
      '\0'};

  /// kStreamMajorSparse blob identifier ("SGL2"). Spelled out for the same
  /// reason as kStreamMajorLayoutIdentifier.
  static constexpr std::array<char, 5> kStreamMajorSparseLayoutIdentifier{
      'S',
      'G',
      'L',
      static_cast<char>(
          '0' + static_cast<int>(EncodingLayout::kStreamMajorSparse)),
      '\0'};

  /// When `encodingViewDeduplicationEnabled` is true, kStreamMajor streams
  /// whose arrays reference the same encoded payload share one EncodingView.
  /// kStreamMajorSparse streams always share them, since its writer always
  /// deduplicates arrays.
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
  /// safe to call concurrently. O(1), except that kStreamMajorSparse first
  /// searches the stream's stripe list, which is O(log k) for a stream with
  /// data in k stripes. `streamId` must be less than streamCount(). The offset
  /// of a stream without data in the stripe (size 0) carries no meaning;
  /// kStreamMajorSparse returns 0 for it.
  uint32_t streamOffset(uint32_t stripeIndex, uint32_t streamId) const;

  /// Random-access read of the byte length of `streamId` within `stripeIndex`,
  /// with the same cost as streamOffset(). Stateless. `streamId` must be less
  /// than streamCount().
  uint32_t streamSize(uint32_t stripeIndex, uint32_t streamId) const;

  /// Returns the number of encoding views this group holds for random access
  /// to its stream-major metadata. With encoding view deduplication, streams
  /// whose encoded arrays share one payload share one view, so this can be
  /// below the number of arrays: two per stream, three for kStreamMajorSparse.
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

  // The last stripe-list search of a batched kStreamMajorSparse lookup.
  // Streams that share a list share the stripe's row in it, and a column's
  // streams, such as a flat-map key's, have adjacent ids and the same list, so
  // reusing the last search skips most searches.
  struct StripeSearch {
    // The stripe list searched last. Null before the first search.
    const EncodingView* stripeList{nullptr};

    // The row of the looked-up stripe in `stripeList`, or std::nullopt when the
    // list does not contain it.
    std::optional<uint32_t> row;
  };

  // Returns the kStreamMajorSparse location of `streamId` in group-local
  // stripe `localStripe`, or an absent location when the stream has no data
  // there. A batched lookup passes one `lastSearch` for all its streams to
  // reuse the previous stream's search when both share a stripe list.
  StreamMetadata sparseLocation(
      uint32_t streamId,
      uint32_t localStripe,
      StripeSearch& lastSearch) const;

  // kRaw: raw pointers into the flatbuffer blob, laid out stripe-major
  // (stripeCount x streamCount). The blob is owned by metadata_ and outlives
  // the StripeGroup.
  struct RawMetadata {
    const uint32_t* offsets{nullptr};
    const uint32_t* sizes{nullptr};
  };

  // kStreamMajor: streamCount entries per array, each pointing at the view over
  // that stream's stripeCount-length array. With encoding view deduplication,
  // streams whose arrays share one encoded payload (TabletWriter stripe-group
  // metadata deduplication) share one view, which is safe because view reads
  // are stateless.
  struct StreamMajorMetadata {
    // Owns the views that offsets and sizes point at.
    std::vector<std::unique_ptr<EncodingView>> views;
    std::vector<EncodingView*> offsets;
    std::vector<EncodingView*> sizes;
  };

  // kStreamMajorSparse views of one stream. `stripeList` lists the group-local
  // stripes where the stream has data and is null when every stripe does.
  // `offsets` and `sizes` hold one value per listed stripe and are null when no
  // stripe does.
  struct SparseStreamViews {
    EncodingView* stripeList{nullptr};
    EncodingView* offsets{nullptr};
    EncodingView* sizes{nullptr};
  };

  // kStreamMajorSparse: one SparseStreamViews per stream.
  struct StreamMajorSparseMetadata {
    // Owns the views that streams point at.
    std::vector<std::unique_ptr<EncodingView>> views;
    std::vector<SparseStreamViews> streams;
  };

  const std::unique_ptr<MetadataBuffer> metadata_;
  const uint32_t index_;
  // How this group's per-stream offsets/sizes are laid out; detected once at
  // construction from the blob's FlatBuffers file identifier. Determines which
  // metadata member below is populated. All encoded forms are written
  // uncompressed (which EncodingView requires); every EncodingView-supported
  // encoding provides a stateless const readAt, so point access needs no
  // auxiliary state and is safe for concurrent reads.
  const EncodingLayout encodingLayout_;
  uint32_t streamCount_{0};
  uint32_t stripeCount_{0};
  uint32_t firstStripe_{0};
  RawMetadata raw_;
  StreamMajorMetadata streamMajor_;
  StreamMajorSparseMetadata streamMajorSparse_;
};

static_assert(
    StripeGroup::kStreamMajorLayoutIdentifier ==
        StripeGroup::makeLayoutIdentifier(
            StripeGroup::EncodingLayout::kStreamMajor),
    "kStreamMajorLayoutIdentifier must match makeLayoutIdentifier(kStreamMajor).");
static_assert(
    static_cast<int>(StripeGroup::EncodingLayout::kStreamMajor) == 1,
    "On-disk identifier contract: kStreamMajor must stay 1.");
static_assert(
    StripeGroup::kStreamMajorSparseLayoutIdentifier ==
        StripeGroup::makeLayoutIdentifier(
            StripeGroup::EncodingLayout::kStreamMajorSparse),
    "kStreamMajorSparseLayoutIdentifier must match its layout.");
static_assert(
    static_cast<int>(StripeGroup::EncodingLayout::kStreamMajorSparse) == 2,
    "On-disk identifier contract: kStreamMajorSparse must stay 2.");

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

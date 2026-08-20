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

  StripeGroup(
      uint32_t stripeGroupIndex,
      const MetadataBuffer& stripes,
      std::unique_ptr<MetadataBuffer> metadata,
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

  /// O(1) random-access read of the byte offset of `streamId` within
  /// `stripeIndex` (relative to the stripe's start). Backed by the per-stream
  /// EncodingView for `streamId` — stateless and safe to call concurrently.
  /// `streamId` must be less than streamCount().
  uint32_t streamOffset(uint32_t stripeIndex, uint32_t streamId) const;

  /// O(1) random-access read of the byte length of `streamId` within
  /// `stripeIndex`. Stateless. `streamId` must be less than streamCount().
  uint32_t streamSize(uint32_t stripeIndex, uint32_t streamId) const;

  /// Bulk decode of all `streamCount` byte offsets for `stripeIndex` into the
  /// caller-provided output buffer. `out.size()` must equal `streamCount`.
  /// Intended for callers that scan the whole array (file-layout dump tools).
  void streamOffsets(uint32_t stripeIndex, std::span<uint32_t> out) const;

  /// Bulk decode of all `streamCount` byte sizes for `stripeIndex`.
  void streamSizes(uint32_t stripeIndex, std::span<uint32_t> out) const;

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

  // kStreamMajor: streamCount entries, one EncodingView per leaf stream, each
  // over that stream's stripeCount-length array.
  struct StreamMajorMetadata {
    std::vector<std::unique_ptr<EncodingView>> offsets;
    std::vector<std::unique_ptr<EncodingView>> sizes;
  };

  const std::unique_ptr<MetadataBuffer> metadata_;
  const uint32_t index_;
  // How this group's per-stream offsets/sizes are laid out; detected once at
  // construction from the blob's FlatBuffers file identifier. Determines which
  // metadata member below is populated. All encoded forms are written
  // uncompressed (which EncodingView requires); every EncodingView-supported
  // encoding provides a stateless O(1) const readAt, so point access needs no
  // auxiliary state and is safe for concurrent reads.
  const EncodingLayout encodingLayout_;
  uint32_t streamCount_{0};
  uint32_t stripeCount_{0};
  uint32_t firstStripe_{0};
  RawMetadata raw_;
  StreamMajorMetadata streamMajor_;
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

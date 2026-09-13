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

#include <type_traits>

#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Zigzag.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

/// Represents a slice of another encoding without performing the slice.
///
/// Slicing an encoding normally means decoding enough of it to find the slice
/// boundaries, slicing every child, and re-serializing. For encodings whose
/// structure does not align with row offsets that is expensive: MainlyConstant
/// has to count how many common values precede the slice before it can locate
/// the corresponding sub-range of its otherValues child, which in the general
/// case means materializing every bool up to the slice end.
///
/// This encoding skips all of it. It copies the source encoding verbatim and
/// records the row offset the consumer should start at, moving the work to
/// decode time, where the reader is walking the rows anyway. The trade is size:
/// the payload carries the whole source encoding rather than just the slice, so
/// it is only worthwhile where the source is already compact relative to the
/// cost of slicing it.
///
/// The wrapper also carries a signed value delta that is added to every
/// materialized value at decode time. It is the deferred equivalent of
/// subtracting a baseline from the inner encoding: callers that have already
/// established a per-slice baseline (e.g. so a downstream FBW inner can pack in
/// fewer bits) record the shift here rather than mutating the copied inner
/// bytes.
///
/// Composition: a SliceEncoding may itself be wrapped by another SliceEncoding.
/// Each layer applies its own delta after its inner materialize() completes,
/// so deltas accumulate additively across layers.
///
/// When the inner encoding structurally supports it, wrap() folds the delta
/// into the copied inner bytes at write time (baseline rewrite for
/// FixedBitWidth / PFOR / Constant, per-value shift-during-copy for
/// uncompressed Trivial, recurse into the values child for Nullable). In that
/// case the on-wire delta is zero and the decode-time shift loop is skipped.
/// Otherwise the delta rides on the wire and the reader applies it at
/// materialize(). Push-down declines fall into three categories:
///   * No push-down implemented (RLE, Dictionary, Huffman, MainlyConstant,
///     BlockBitPacking, ...): no baseline field to rewrite.
///   * SharedDictionary: could carry a shift over its resolved values but
///     folding into the shared alphabet would mutate state visible to other
///     consumers, and folding into the indices would resolve to wrong
///     alphabet entries; the read-time loop shifts values after alphabet
///     resolution instead.
///   * Runtime-conditional: compressed Trivial, or Nullable whose values
///     child cannot itself absorb the shift.

namespace facebook::nimble {

// Base class holding SliceEncoding::wrap()'s non-templated push-down helpers.
// Kept as a non-template base so the bodies compile once (moving them onto
// SliceEncoding<T> would recompile per T). SliceEncoding<T> inherits so wrap()
// can call the helpers as inherited protected static methods.
class SliceEncodingBase {
 protected:
  // Runtime probe: returns true iff a subsequent applyValueDeltaPushdown() on
  // the same (inner, delta) pair would succeed. Walks Nullable chains and
  // checks Trivial compression state so per-encoding runtime conditions are
  // honoured. `useVarintRowCount` must match the option under which `inner`
  // was produced (Nimble uses one consistent value across a whole encoding
  // tree, so callers forward the outer SliceEncoding wrap options).
  static bool canApplyValueDeltaPushdown(
      std::string_view inner,
      int64_t delta,
      bool useVarintRowCount);

  // Writes shifted inner bytes into `dest` and returns the number of bytes
  // written (invariant: equal to `inner.size()`). Caller guarantees the
  // destination has that much space reserved. Caller MUST have gotten `true`
  // from canApplyValueDeltaPushdown() first -- hitting an unsupported case
  // here fires NIMBLE_UNREACHABLE. `useVarintRowCount` has the same meaning
  // as in canApplyValueDeltaPushdown(). The returned size is used by wrap()
  // as a drift guard: if a folder ever writes fewer or more bytes than the
  // source inner, the outer post-write assertion fires instead of silently
  // corrupting the wire.
  static size_t applyValueDeltaPushdown(
      std::string_view inner,
      int64_t delta,
      bool useVarintRowCount,
      char* dest);
};

/// Data layout is:
/// Standard Encoding prefix (size varies with Options::useVarintRowCount),
///     whose row count is the SLICE length, not the inner encoding's row count.
/// 4 bytes: row offset into the inner encoding at which the slice begins.
/// 1-10 bytes: zigzag varint value delta added to every materialized value at
///     decode time. Zero delta occupies 1 byte.
/// remaining bytes: the inner encoding, verbatim.
template <typename T>
class SliceEncoding final
    : public SliceEncodingBase,
      public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  SliceEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory = nullptr,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options),
        stringBufferFactory_{std::move(stringBufferFactory)},
        sliceOffset_{readSliceOffset(data, this->dataOffset())},
        valueDelta_{
            readValueDelta(data, this->dataOffset() + sizeof(uint32_t))},
        inner_{readInner(data, this->dataOffset())} {
    // A non-zero delta on a physical type where the shift cannot be applied
    // must never appear on the wire. wrap() blocks it at write time; a mismatch
    // here means the payload was corrupted or produced by a mismatched writer.
    if constexpr (!kSupportsValueDelta) {
      NIMBLE_CHECK_EQ(
          valueDelta_,
          0,
          "SliceEncoding value delta requires a non-bool integer physical type.");
    } else {
      // Accept any delta whose static_cast<physicalType> preserves the intent:
      // the domain spans from the signed min of the physical type (so negative
      // deltas that wrap to the top of the unsigned physical range still round-
      // trip) up to the unsigned max. A wire delta outside this range aliases
      // silently at materialize() -- reject it so a mismatched writer surfaces
      // as a hard error instead of a mangled shift. Skipped when the physical
      // type is already 64 bits wide because valueDelta_ (int64_t) cannot
      // overflow its own type.
      if constexpr (sizeof(physicalType) < sizeof(int64_t)) {
        using signedPhysicalType = std::make_signed_t<physicalType>;
        NIMBLE_CHECK(
            valueDelta_ >=
                    static_cast<int64_t>(
                        std::numeric_limits<signedPhysicalType>::min()) &&
                valueDelta_ <= static_cast<int64_t>(
                                   std::numeric_limits<physicalType>::max()),
            "SliceEncoding valueDelta does not fit in the physical type.");
      }
    }
    reset();
  }

  void reset() final {
    // Recreated rather than reset+skip: an encoding's reset() returns it to row
    // zero, so the skip would have to be replayed on every reset anyway, and
    // constructing is what establishes the child's own read state.
    encoding_ = EncodingFactory{this->options_}.create(
        *this->pool_, inner_, stringBufferFactory_);
    if (sliceOffset_ > 0) {
      encoding_->skip(sliceOffset_);
    }
  }

  void skip(uint32_t rowCount) final {
    encoding_->skip(rowCount);
  }

  void materialize(uint32_t rowCount, void* buffer) final {
    encoding_->materialize(rowCount, buffer);
    // Read-time fallback that applies the on-wire delta to every value. When
    // wrap() folded the delta into the inner encoding's own bytes (baseline
    // rewrite for FixedBitWidth / PFOR / Constant, per-value shift-during-
    // copy for uncompressed Trivial, forward-to-values-child for Nullable),
    // valueDelta_ is zero on the wire and this loop is skipped. It stays as
    // the fallback for encodings that cannot absorb a constant shift
    // structurally (RLE, Dictionary, Huffman, MainlyConstant, BlockBitPacking,
    // ...) and for the runtime-conditional cases where push-down was rejected
    // (compressed Trivial, Nullable whose values child cannot itself absorb
    // the shift).
    if constexpr (kSupportsValueDelta) {
      if (valueDelta_ != 0) {
        auto* typed = static_cast<physicalType*>(buffer);
        const auto delta = static_cast<physicalType>(valueDelta_);
        // Vectorised in-place shift: process one SIMD batch per iteration,
        // then handle the tail scalar. Broadcast the delta once so every add
        // is a straight `batch + batch`. Physical types here are always
        // integral (bool is filtered by kSupportsValueDelta), so xsimd has a
        // batch specialization for each width we hit.
        using Batch = xsimd::batch<physicalType>;
        constexpr uint32_t kLanes = static_cast<uint32_t>(Batch::size);
        const Batch deltaVec = xsimd::broadcast<physicalType>(delta);
        uint32_t i = 0;
        for (; i + kLanes <= rowCount; i += kLanes) {
          (Batch::load_unaligned(typed + i) + deltaVec)
              .store_unaligned(typed + i);
        }
        for (; i < rowCount; ++i) {
          typed[i] = static_cast<physicalType>(typed[i] + delta);
        }
      }
    }
  }

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      override {
    // The compile-time guard in wrap() and the constructor's runtime check
    // ensure delta is zero when the physical type is bool, so no shift is
    // needed here -- just forward.
    encoding_->materializeBoolsAsBits(rowCount, buffer, begin);
  }

  template <typename DecoderVisitor>
  void readWithVisitor(
      DecoderVisitor& /*visitor*/,
      ReadWithVisitorParams& /*params*/) {
    // The selective reader path would need the visitor's row positions
    // translated by sliceOffset_. Nothing produces this encoding for that path
    // today -- it is emitted only by EncodingSliceFactory, whose output is
    // consumed through materialize() -- so fail loudly rather than silently
    // returning rows from the wrong offset.
    NIMBLE_UNSUPPORTED(
        "SliceEncoding does not support the selective reader path.");
  }

  std::string debugString(int offset) const override {
    return fmt::format(
        "{}{}<{}> rowCount={} sliceOffset={} valueDelta={}\n{}",
        std::string(offset, ' '),
        toString(this->encodingType()),
        toString(this->dataType()),
        this->rowCount(),
        sliceOffset_,
        valueDelta_,
        encoding_->debugString(offset + 2));
  }

  /// Returns the signed value delta added to every materialized value at
  /// decode time. Zero when wrap() folded the shift into the inner encoding
  /// at write time.
  int64_t valueDelta() const noexcept {
    return valueDelta_;
  }

  /// Returns the raw bytes of the inner encoding, stored verbatim after the
  /// SliceEncoding header.
  std::string_view innerEncoding() const noexcept {
    return inner_;
  }

  /// Wraps `encoded` so that a consumer sees `length` rows starting at
  /// `offset`, without slicing it. `encoded` is copied into `buffer`.
  ///
  /// `valueDelta` is added to every materialized value at decode time. It is
  /// only representable for non-bool integer physical types; passing a
  /// non-zero delta for any other T is rejected at write time. SharedDictionary
  /// inner is accepted -- push-down cannot fold the delta into the shared
  /// alphabet or the indices, so the delta rides on the wire and is applied
  /// to alphabet-resolved values at materialize().
  ///
  /// When `valueDelta` is non-zero and the inner encoding structurally
  /// supports it, the delta is folded into the copied inner bytes at write
  /// time and the on-wire delta is zero -- the decode-time shift loop then
  /// skips itself. Otherwise the delta rides on the wire and is applied at
  /// materialize().
  ///
  /// Wrapping a SliceEncoding stream inside another wrap() is supported; the
  /// outer delta accumulates with the inner delta at read time (each layer
  /// applies its own delta after its inner materialize()).
  static std::string_view wrap(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      int64_t valueDelta = 0,
      const Encoding::Options& options = {}) {
    if constexpr (!kSupportsValueDelta) {
      NIMBLE_CHECK_EQ(
          valueDelta,
          0,
          "SliceEncoding<T>::wrap: valueDelta requires a non-bool integer T.");
    }
    // SharedDictionary inner is intentionally allowed even with a non-zero
    // delta: write-time push-down declines for it (folding the shift into the
    // shared alphabet would mutate state visible to other consumers, and
    // folding into the indices would resolve to wrong alphabet entries), so
    // the shift stays on the wire and the read-time materialize loop adds
    // delta to the values the alphabet resolves. Physical types that cannot
    // carry a shift (bool, non-integer) are still rejected up-front by the
    // compile-time `kSupportsValueDelta` check above.

    // Decide whether push-down applies. When it can, the on-wire delta is
    // zero and the decode-time loop is skipped; when it can't, the delta
    // rides on the wire and the reader applies it.
    bool pushdownValueDelta = false;
    if (valueDelta != 0) {
      pushdownValueDelta = canApplyValueDeltaPushdown(
          encoded, valueDelta, options.useVarintRowCount);
    }
    const int64_t storedDelta = pushdownValueDelta ? 0 : valueDelta;

    // Compute exact wrapper size and reserve.
    const auto prefixSize =
        EncodingPrefix::serializedSize(length, options.useVarintRowCount);
    const auto zigzaggedDelta = zigzag::zigzagEncode64(storedDelta);
    const auto deltaSize = varint::varintSize(zigzaggedDelta);
    const auto encodingSize =
        prefixSize + sizeof(uint32_t) + deltaSize + encoded.size();
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    EncodingPrefix::serialize(
        EncodingType::Slice,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        pos);
    encoding::writeUint32(offset, pos);
    varint::writeVarint(zigzaggedDelta, &pos);

    // Write the inner bytes. Push-down folders write directly into the
    // reserved region -- no scratch buffer allocation. Folders that use a
    // baseline field (FixedBitWidth, PFOR, Constant) memcpy the source first
    // then rewrite the baseline in place; Trivial shifts values while copying.
    // Each folder returns the number of bytes it wrote so the post-write
    // assertion can catch any drift between the reserved size and what the
    // folder produced.
    const size_t bytesWritten = pushdownValueDelta
        ? applyValueDeltaPushdown(
              encoded, valueDelta, options.useVarintRowCount, pos)
        : (std::memcpy(pos, encoded.data(), encoded.size()), encoded.size());
    pos += bytesWritten;
    NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

 private:
  // Only non-bool integer logical types can carry a shift: floats keep their
  // physical bits in an integer, but their arithmetic is not integer addition
  // on those bits; bool has no additive semantics; string is not additive at
  // all. Gated on the logical T rather than physicalType so float/double are
  // excluded even though their physicalType is uint32_t/uint64_t. Kept as a
  // compile-time constant so both wrap() and the constructor can gate on it
  // via `if constexpr`.
  static constexpr bool kSupportsValueDelta =
      std::is_integral_v<T> && !std::is_same_v<T, bool>;

  // Reads the 4-byte row offset that follows the encoding prefix.
  static uint32_t readSliceOffset(std::string_view data, uint32_t dataOffset) {
    const char* pos = data.data() + dataOffset;
    return encoding::readUint32(pos);
  }

  // Reads the zigzag-varint value delta that follows the row offset.
  static int64_t readValueDelta(std::string_view data, uint32_t dataOffset) {
    const char* pos = data.data() + dataOffset;
    return zigzag::zigzagDecode64(varint::readVarint64(&pos));
  }

  // Returns the inner encoding, stored verbatim after the row offset and the
  // value delta varint.
  static std::string_view readInner(
      std::string_view data,
      uint32_t dataOffset) {
    const char* pos = data.data() + dataOffset + sizeof(uint32_t);
    // Skip past the value delta varint without decoding it again.
    varint::skipVarint(&pos);
    return {pos, static_cast<size_t>(data.data() + data.size() - pos)};
  }

  const std::function<void*(uint32_t)> stringBufferFactory_;
  const uint32_t sliceOffset_;
  // Signed shift applied to every materialized value at decode time. Read
  // from the payload's zigzag varint after `sliceOffset_`. Held as int64_t so
  // negative deltas over any physical type width round-trip through wrap()
  // without pre-truncation; the constructor validates that the stored delta
  // fits the physical type before it is used. Always zero for physical types
  // that cannot carry a shift (bool, non-integer).
  const int64_t valueDelta_;
  const std::string_view inner_;
  // The only mutable member: reset() rebuilds the child rather than rewinding
  // it, so this is reassigned on every reset().
  std::unique_ptr<Encoding> encoding_;
};

} // namespace facebook::nimble

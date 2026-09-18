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

#include "velox/dwio/nimble/encodings/SliceEncoding.h"

#include <cstdint>
#include <cstring>
#include <string_view>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble {

namespace {

// Forward declarations for the dispatch pair -- `applyValueDeltaToNullable`
// needs to recurse into `applyValueDeltaPushdown` for its values child.
bool canApplyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount);

size_t applyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    char* dest);

// Rewrites the physical baseline value stored at `dest` by adding `delta`.
// The baseline occupies sizeof(Physical) bytes at `dest`; the caller
// dispatches to the right Physical based on the inner encoding's DataType
// byte in its prefix (see applyValueDeltaInPlace). Uses memcpy so `dest`
// may be arbitrarily aligned.
template <typename Physical>
void applyValueDeltaInPlaceTyped(char* dest, int64_t delta) {
  Physical value;
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(&value, dest, sizeof(value));
  value = static_cast<Physical>(value + static_cast<Physical>(delta));
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(dest, &value, sizeof(value));
}

void applyValueDeltaInPlace(char* dest, DataType dataType, int64_t delta) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
      applyValueDeltaInPlaceTyped<uint8_t>(dest, delta);
      return;
    case DataType::Int16:
    case DataType::Uint16:
      applyValueDeltaInPlaceTyped<uint16_t>(dest, delta);
      return;
    case DataType::Int32:
    case DataType::Uint32:
      applyValueDeltaInPlaceTyped<uint32_t>(dest, delta);
      return;
    case DataType::Int64:
    case DataType::Uint64:
      applyValueDeltaInPlaceTyped<uint64_t>(dest, delta);
      return;
    default:
      NIMBLE_UNREACHABLE(
          "SliceEncoding push-down expects an integer inner data type, got: {}",
          dataType);
  }
}

// Copies `rowCount` physical values from `src` to `dst`, adding `delta` to
// each value as it is copied. Each byte in the destination range is written
// exactly once (no memcpy-then-edit). memcpy is used rather than typed stores
// because `src`/`dst` may not be aligned for the target type. The per-value
// shift is vectorised via xsimd where a batch fits (bulk lanes at SIMD width
// for the physical type), with a scalar tail handling any remainder. This
// matches velox's own load-add-store idiom (see
// velox/dwio/nimble/encodings/selection/Statistics.cpp) -- velox exposes
// simdFill / setAll / gather helpers but none cover the constant-add case,
// so we use the xsimd primitives velox itself uses.
template <typename Physical>
void applyValueDeltaToTrivialPayloadTyped(
    const char* src,
    uint32_t rowCount,
    int64_t delta,
    char* dst) {
  const auto typedDelta = static_cast<Physical>(delta);
  const size_t stride = sizeof(Physical);
  using Batch = xsimd::batch<Physical>;
  constexpr uint32_t kLanes = static_cast<uint32_t>(Batch::size);
  const Batch deltaVec = Batch::broadcast(typedDelta);
  uint32_t i = 0;
  for (; i + kLanes <= rowCount; i += kLanes) {
    // @lint-ignore NULLSAFECLANG nullable-arithmetic
    (Batch::load_unaligned(
         reinterpret_cast<const Physical*>(src + i * stride)) +
     deltaVec)
        // @lint-ignore NULLSAFECLANG nullable-arithmetic
        .store_unaligned(reinterpret_cast<Physical*>(dst + i * stride));
  }
  for (; i < rowCount; ++i) {
    Physical value;
    // @lint-ignore NULLSAFECLANG nullable-arithmetic
    std::memcpy(&value, src + i * stride, stride);
    value = static_cast<Physical>(value + typedDelta);
    // @lint-ignore NULLSAFECLANG nullable-arithmetic
    std::memcpy(dst + i * stride, &value, stride);
  }
}

void applyValueDeltaToTrivialPayload(
    const char* src,
    uint32_t rowCount,
    DataType dataType,
    int64_t delta,
    char* dst) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
      applyValueDeltaToTrivialPayloadTyped<uint8_t>(src, rowCount, delta, dst);
      return;
    case DataType::Int16:
    case DataType::Uint16:
      applyValueDeltaToTrivialPayloadTyped<uint16_t>(src, rowCount, delta, dst);
      return;
    case DataType::Int32:
    case DataType::Uint32:
      applyValueDeltaToTrivialPayloadTyped<uint32_t>(src, rowCount, delta, dst);
      return;
    case DataType::Int64:
    case DataType::Uint64:
      applyValueDeltaToTrivialPayloadTyped<uint64_t>(src, rowCount, delta, dst);
      return;
    default:
      NIMBLE_UNREACHABLE(
          "SliceEncoding push-down expects an integer Trivial data type, got: {}",
          dataType);
  }
}

// Shared baseline-rewrite folder used by FixedBitWidth, PFOR, and Constant:
// all three push down via "memcpy inner verbatim, then rewrite the
// uncompressed baseline field in the header." They differ only in where the
// baseline sits relative to the encoding prefix:
//   FBW:      [prefix][compression byte][baseline][bitWidth][packed bytes]
//             -> rewriteOffset = 1 (skip compression byte)
//   PFOR:     [prefix][baseline][baseBitWidth][varint numExceptions]...
//             -> rewriteOffset = 0
//             (base residuals and exception values are both baseline-relative,
//              see PFOREncoding::patchExceptions, so one edit shifts both)
//   Constant: [prefix][value]  -- the single stored value IS the baseline
//             -> rewriteOffset = 0
// The compression indicator (when present, FBW) governs only the packed
// region that follows the baseline, so the baseline is always written
// uncompressed and can be rewritten in place.
size_t applyValueDeltaByRewrite(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    DataType dataType,
    uint32_t rewriteOffset,
    char* dest) {
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(dest, inner.data(), inner.size());
  const uint32_t prefixSize =
      EncodingPrefix::prefixSize(inner, useVarintRowCount);
  applyValueDeltaInPlace(dest + prefixSize + rewriteOffset, dataType, delta);
  // Push-down is size-preserving: same # of bytes written as the source.
  return inner.size();
}

// Trivial (numeric, uncompressed) push-down. Wire layout after the standard
// prefix is:
//   1 byte:            compression type (must be Uncompressed here)
//   rowCount*phys:     the raw values
// Header (prefix + compression byte) is copied verbatim; each value is shifted
// while it is copied so no byte is written twice.
size_t applyValueDeltaToTrivial(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    DataType dataType,
    char* dest) {
  const uint32_t prefixSize =
      EncodingPrefix::prefixSize(inner, useVarintRowCount);
  const uint32_t headerSize = prefixSize + sizeof(uint8_t);
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(dest, inner.data(), headerSize);
  const uint32_t rowCount =
      EncodingPrefix::readRowCount(inner, useVarintRowCount);
  applyValueDeltaToTrivialPayload(
      inner.data() + headerSize, rowCount, dataType, delta, dest + headerSize);
  return inner.size();
}

// Nullable push-down. Wire layout after the standard prefix is:
//   4 bytes:           non-null child encoding size (valuesSize)
//   valuesSize bytes:  non-null child encoding
//   remaining bytes:   nulls child encoding
// The nulls child is a bool stream and is copied verbatim; the shift only
// applies to decoded non-null values, so we recurse into the values child.
size_t applyValueDeltaToNullable(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    char* dest) {
  const uint32_t prefixSize =
      EncodingPrefix::prefixSize(inner, useVarintRowCount);

  // Copy `[prefix][4-byte valuesSize]` verbatim.
  const uint32_t headerSize = prefixSize + sizeof(uint32_t);
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(dest, inner.data(), headerSize);

  const char* sizePos = inner.data() + prefixSize;
  const uint32_t valuesSize = encoding::readUint32(sizePos);

  // Recurse into the values child at its destination offset.
  const std::string_view valuesInner{inner.data() + headerSize, valuesSize};
  applyValueDeltaPushdown(
      valuesInner, delta, useVarintRowCount, dest + headerSize);

  // Copy the nulls child verbatim. A valid Nullable input always carries a
  // non-empty nulls sub-stream (the encoding wouldn't be Nullable otherwise),
  // so this is asserted rather than runtime-guarded.
  const size_t offsetAfterValues = headerSize + valuesSize;
  const size_t nullsSize = inner.size() - offsetAfterValues;
  NIMBLE_DCHECK_GT(
      nullsSize, 0, "Nullable input must carry a non-empty nulls sub-stream.");
  // @lint-ignore NULLSAFECLANG nullable-argument
  std::memcpy(
      dest + offsetAfterValues, inner.data() + offsetAfterValues, nullsSize);
  return inner.size();
}

// True iff the inner encoding's DataType is one of the integer widths the
// baseline/payload helpers dispatch on. Everything else (Bool, Float, Double,
// String, Undefined) cannot carry an additive shift.
bool isPushDownableIntegerType(DataType dataType) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Int16:
    case DataType::Uint16:
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Int64:
    case DataType::Uint64:
      return true;
    default:
      return false;
  }
}

// Main can-apply dispatch: returns true iff a subsequent applyValueDelta-
// Pushdown() call on the same (inner, delta) pair would succeed. Walks
// Nullable chains, checks Trivial compression state, and rejects any
// non-integer inner that isn't a structural wrapper. Backs
// SliceEncodingPushDown::canApply().
bool canApplyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount) {
  const auto encodingType = EncodingPrefix::encodingType(inner);
  const auto dataType = EncodingPrefix::dataType(inner);

  // Every push-down case that reads the inner as integer bytes requires an
  // integer physical type; reject non-integer dataTypes here so the switch
  // cases below can assume integer bytes without re-checking. Nullable is
  // the sole exemption -- it does not read the inner as integer bytes at
  // this layer and instead recurses into its values child, whose own call
  // re-checks dataType. Any future structural-wrapper encoding that
  // recurses would also need to be exempted here. Non-push-down encodings
  // (RLE, Dictionary, ...) still reach the switch and return false via the
  // fall-through list; the early-return is just an equivalent short-circuit
  // for their non-integer variant.
  if (encodingType != EncodingType::Nullable &&
      !isPushDownableIntegerType(dataType)) {
    return false;
  }

  // Exhaustive over EncodingType with no default arm so that adding a new
  // encoding type fails the build until the author declares whether its
  // wire layout can absorb a constant shift. See applyValueDeltaPushdown()
  // for the mirror-image dispatch that must be updated in lockstep.
  switch (encodingType) {
    case EncodingType::FixedBitWidth:
    case EncodingType::PFOR:
    case EncodingType::Constant:
      return true;

    case EncodingType::Trivial: {
      // Compressed Trivial cannot be shifted without decompressing first, so
      // let the read-time loop handle the shift instead.
      const uint32_t prefixSize =
          EncodingPrefix::prefixSize(inner, useVarintRowCount);
      const auto compression = static_cast<CompressionType>(inner[prefixSize]);
      return compression == CompressionType::Uncompressed;
    }

    case EncodingType::Nullable: {
      // The nulls child is a bool stream and never shifts. Recurse into the
      // values child -- push-down succeeds iff the values child can itself
      // absorb the shift.
      const uint32_t prefixSize =
          EncodingPrefix::prefixSize(inner, useVarintRowCount);
      const char* pos = inner.data() + prefixSize;
      const uint32_t valuesSize = encoding::readUint32(pos);
      const std::string_view valuesInner{pos, valuesSize};
      // A Nullable wrapper preserves the physical type of its values child, so
      // no need to re-check dataType here -- the recursive call will.
      return canApplyValueDeltaPushdown(valuesInner, delta, useVarintRowCount);
    }

    // Everything below cannot fold the delta into its own bytes at write
    // time. RLE / Dictionary / Huffman / MainlyConstant / BlockBitPacking /
    // ... have no single baseline field to rewrite. SharedDictionary could
    // in principle carry a shift (its values are alphabet-resolved integers)
    // but folding into the shared alphabet would mutate state visible to
    // other consumers, and folding into the indices would resolve to wrong
    // alphabet entries -- so push-down declines and the read-time materialize
    // loop adds the delta to the values after the alphabet resolves them.
    // The other listed encodings are simply not push-down-implemented.
    case EncodingType::RLE:
    case EncodingType::Dictionary:
    case EncodingType::Sentinel:
    case EncodingType::SparseBool:
    case EncodingType::Varint:
    case EncodingType::Delta:
    case EncodingType::MainlyConstant:
    case EncodingType::Prefix:
    case EncodingType::ALP:
    case EncodingType::SimdForBitpack:
    case EncodingType::BlockBitPacking:
    case EncodingType::SubIntSplit:
    case EncodingType::FrequencyPartition:
    case EncodingType::FOR:
    case EncodingType::Fsst:
    case EncodingType::Huffman:
    case EncodingType::DeltaBlock:
    case EncodingType::SharedDictionary:
    case EncodingType::Slice:
    case EncodingType::EliasFano:
    case EncodingType::BitRangeSplit:
      return false;
  }
  NIMBLE_UNREACHABLE(
      "SliceEncoding push-down saw an unknown EncodingType: {}", encodingType);
}

// Main apply dispatch: writes the shifted inner bytes into `dest` and returns
// the number of bytes written (== `inner.size()` -- push-down is size-
// preserving). Caller MUST have gotten `true` from
// canApplyValueDeltaPushdown() on the same (inner, delta). Backs
// SliceEncodingPushDown::apply().
size_t applyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    char* dest) {
  const auto encodingType = EncodingPrefix::encodingType(inner);
  const auto dataType = EncodingPrefix::dataType(inner);

  // Exhaustive over EncodingType with no default arm so that adding a new
  // encoding type fails the build until the author declares whether it has
  // a folder. Must be kept in lockstep with canApplyValueDeltaPushdown() --
  // every encoding that returns true there must have a folder branch here.
  switch (encodingType) {
    case EncodingType::FixedBitWidth:
      // FBW header: [prefix][compression byte][baseline][bitWidth][packed].
      // Skip the 1-byte compression indicator to land on the baseline.
      return applyValueDeltaByRewrite(
          inner,
          delta,
          useVarintRowCount,
          dataType,
          /*rewriteOffset=*/1,
          dest);

    case EncodingType::PFOR:
    case EncodingType::Constant:
      // PFOR:     [prefix][baseline][baseBitWidth][varint numExceptions]...
      // Constant: [prefix][value]  -- the value IS the baseline.
      // Both place the baseline immediately after the standard prefix.
      return applyValueDeltaByRewrite(
          inner,
          delta,
          useVarintRowCount,
          dataType,
          /*rewriteOffset=*/0,
          dest);

    case EncodingType::Trivial:
      return applyValueDeltaToTrivial(
          inner, delta, useVarintRowCount, dataType, dest);

    case EncodingType::Nullable:
      return applyValueDeltaToNullable(inner, delta, useVarintRowCount, dest);

    // Drift guard: canApplyValueDeltaPushdown() returns false for every
    // encoding listed below (and for the recursive Nullable-with-unsupported-
    // child case), so the caller (wrap()) should have taken the on-wire delta
    // path instead of invoking us. Hitting any of these means the two
    // functions diverged and produced inconsistent answers for the same
    // inner.
    case EncodingType::RLE:
    case EncodingType::Dictionary:
    case EncodingType::Sentinel:
    case EncodingType::SparseBool:
    case EncodingType::Varint:
    case EncodingType::Delta:
    case EncodingType::MainlyConstant:
    case EncodingType::Prefix:
    case EncodingType::ALP:
    case EncodingType::SimdForBitpack:
    case EncodingType::BlockBitPacking:
    case EncodingType::SubIntSplit:
    case EncodingType::FrequencyPartition:
    case EncodingType::FOR:
    case EncodingType::Fsst:
    case EncodingType::Huffman:
    case EncodingType::DeltaBlock:
    case EncodingType::SharedDictionary:
    case EncodingType::Slice:
    case EncodingType::EliasFano:
    case EncodingType::BitRangeSplit:
      NIMBLE_UNREACHABLE(
          "SliceEncoding push-down invoked on unsupported inner encoding: {}",
          encodingType);
  }
  NIMBLE_UNREACHABLE(
      "SliceEncoding push-down saw an unknown EncodingType: {}", encodingType);
}

} // namespace

bool SliceEncodingBase::canApplyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount) {
  return ::facebook::nimble::canApplyValueDeltaPushdown(
      inner, delta, useVarintRowCount);
}

size_t SliceEncodingBase::applyValueDeltaPushdown(
    std::string_view inner,
    int64_t delta,
    bool useVarintRowCount,
    char* dest) {
  return ::facebook::nimble::applyValueDeltaPushdown(
      inner, delta, useVarintRowCount, dest);
}

} // namespace facebook::nimble

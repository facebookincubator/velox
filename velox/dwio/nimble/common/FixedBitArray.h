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

#include <cstdint>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Types.h"

/// Packs integers into a fixed number (1-64) of bits and retrieves them.
/// The 'bulk' API provided is particularly efficient at setting/getting
/// ranges of values with small bit widths.
///
/// Typical usage:
///   std::vector<uint32_t> data = ...
///   int bitsRequired = BitsRequired(*maxElement(data.begin(), data.end());
///   auto buffer = std::make_unique<char[]>(
///        FixedBitArray::BufferSize(data.size(), bitsRequired));
///   FixedBitArray fixedBitArray(buffer.get(), bitsRequired);
///   fixedBitArray.bulkSet32(0, data.size(), data.data());
///
/// And then you'd store the buffer somewhere, and later read the values back,
/// recovering the info stored in data. Note that the FBA does not own
/// any data.

namespace facebook::nimble {

class FixedBitArray {
 public:
  /// Computes the buffer size needed to hold |elementCount| values with the
  /// specified |bitWidth|. Note that we allocate a few bytes of 'slop' space
  /// beyond what is mathematically required to speed things up.
  static uint64_t bufferSize(uint64_t elementCount, int bitWidth);

  /// Not legal to use; included so we can default construct on the stack.
  FixedBitArray() = default;

  /// Creates a fixed bit array stored at buffer whose elements can lie within
  /// the range [0, 2^|bitWidth|). The |buffer| must already be preallocated to
  /// the appropriate size, as given by BufferSize.
  FixedBitArray(char* buffer, int bitWidth);

  /// Convenience constructor for reading read-only data. Note that you are NOT
  /// prevented by the code from using the non-const calls on a FBA constructed
  /// via this method, but to do so is undefined behavior.
  FixedBitArray(std::string_view buffer, int bitWidth)
      : FixedBitArray(const_cast<char*>(buffer.data()), bitWidth) {}

  /// Sets the |index|'th slot to the given |value|. If value
  /// is >= 2^|bitWidth| behavior is undefined.
  ///
  /// IMPORTANT NOTE: this does NOT function as normal assignment.
  /// We do NOT first 0 out the slot, i.e. we use or semantics.
  /// If you need to first 0 it out, use ZeroAndSet.
  void set(uint64_t index, uint64_t value);

  /// Zeroes a slot and then sets it (like normal assignment).
  void zeroAndSet(uint64_t index, uint64_t value);

  /// Gets the |index|'th value.
  uint64_t get(uint64_t index) const;

  /// Versions of set/get that only work with bitWidth <= 32, but are slightly
  /// faster. Same assignment caveat applies to set32 as to set.
  void set32(uint64_t index, uint32_t value);

  uint32_t get32(uint64_t index) const;

  /// Retrieves a contiguous subarray from slots [start, start + length).
  /// Considerably faster than looping a get call. Only callable when
  /// bit width <= 32.
  void bulkGet32(uint64_t start, uint64_t length, uint32_t* values) const;

  /// Same as above, but outputs into 8-bytes values. Still requires that bit
  /// width <= 32.
  void bulkGet32Into64(uint64_t start, uint64_t length, uint64_t* values) const;

  /// Same as above. Add baseline to every value.
  void bulkGetWithBaseline32Into64(
      uint64_t start,
      uint64_t length,
      uint64_t* values,
      uint64_t baseline) const;

  /// Unified bulk get for any unsigned integral element type, adding baseline
  /// to each value. Mirror of bulkSetWithBaseline: 4-byte outputs reuse
  /// bulkGetWithBaseline32 and 8-byte outputs reuse bulkGet64WithBaseline (both
  /// zero-copy). Narrow 1- and 2-byte outputs are read into fixed-size stack
  /// chunks via the fast bulkGet32 path and narrowed down -- faster than
  /// per-element get, with no heap allocation.
  template <typename T>
  void bulkGetWithBaseline(
      uint64_t start,
      uint64_t length,
      T* values,
      uint64_t baseline) const {
    static_assert(
        isUnsignedIntegralType<T>(),
        "bulkGetWithBaseline requires an unsigned integral type.");
    if constexpr (isFourByteIntegralType<T>()) {
      bulkGetWithBaseline32(
          start,
          length,
          reinterpret_cast<uint32_t*>(values),
          static_cast<uint32_t>(baseline));
    } else if constexpr (isEightByteIntegralType<T>()) {
      bulkGet64WithBaseline(
          start, length, reinterpret_cast<uint64_t*>(values), baseline);
    } else {
      // Narrow (1- or 2-byte) types cannot be reinterpreted as a wider array in
      // place, so read into fixed-size stack chunks via the fast bulkGet32 path
      // and narrow down -- no heap allocation regardless of length.
      constexpr uint64_t kWidenChunk = 1024;
      uint32_t widened[kWidenChunk];
      const auto baseline32 = static_cast<uint32_t>(baseline);
      for (uint64_t offset = 0; offset < length; offset += kWidenChunk) {
        const uint64_t chunk =
            length - offset < kWidenChunk ? length - offset : kWidenChunk;
        bulkGetWithBaseline32(start + offset, chunk, widened, baseline32);
        for (uint64_t i = 0; i < chunk; ++i) {
          values[offset + i] = static_cast<T>(widened[i]);
        }
      }
    }
  }

  /// Sets a contiguous subarray of slots from [start, start + length).
  /// Considerably faster than looping set calls. Only callable when bitWidth
  /// <= 32. Same semantics as set -- see the warning there.
  void bulkSet32(uint64_t start, uint64_t length, const uint32_t* values);

  /// Unified bulk set for any unsigned integral element type, subtracting
  /// baseline from each value before packing. Dispatches at compile time to the
  /// optimal path: 4-byte values reuse bulkSet32WithBaseline and 8-byte values
  /// reuse bulkSet64WithBaseline (both zero-copy). Narrow 1- and 2-byte values
  /// are widened in fixed-size stack chunks onto the bulkSet32 path -- faster
  /// than per-element packing, with no heap allocation. Requires the same
  /// zeroed-buffer precondition as set.
  template <typename T>
  void bulkSetWithBaseline(
      uint64_t start,
      uint64_t length,
      const T* values,
      uint64_t baseline) {
    static_assert(
        isUnsignedIntegralType<T>(),
        "bulkSetWithBaseline requires an unsigned integral type.");
    if constexpr (isFourByteIntegralType<T>()) {
      bulkSet32WithBaseline(
          start,
          length,
          reinterpret_cast<const uint32_t*>(values),
          static_cast<uint32_t>(baseline));
    } else if constexpr (isEightByteIntegralType<T>()) {
      bulkSet64WithBaseline(
          start, length, reinterpret_cast<const uint64_t*>(values), baseline);
    } else {
      // Narrow (1- or 2-byte) types cannot be reinterpreted as a wider array in
      // place, so widen in fixed-size stack chunks and use the fast bulkSet32
      // path -- no heap allocation regardless of length.
      constexpr uint64_t kWidenChunk = 1024;
      uint32_t widened[kWidenChunk];
      const auto baseline32 = static_cast<uint32_t>(baseline);
      for (uint64_t offset = 0; offset < length; offset += kWidenChunk) {
        const uint64_t chunk =
            length - offset < kWidenChunk ? length - offset : kWidenChunk;
        for (uint64_t i = 0; i < chunk; ++i) {
          widened[i] = static_cast<uint32_t>(values[offset + i]);
        }
        bulkSet32WithBaseline(start + offset, chunk, widened, baseline32);
      }
    }
  }

  /// Fills in the bit-packed |bitVector| with whether each value in
  /// [start, start + length) equals |value|. Should be faster than
  /// doing bulkGet32 and then doing your own equality check + bitpack.
  ///
  /// bitVector must point to a buffer of at least size BufferSize(length, 1);
  ///
  /// REQUIRES: start is a multiple of 64. If you need to run equals on
  /// a range where that isn't true, you'll have to manually do the part
  /// up to the first multiple of 64 yourself + adjust the results accordingly.
  void equals32(
      uint64_t start,
      uint64_t length,
      uint32_t value,
      char* bitVector) const;

  int bitWidth() const {
    return bitWidth_;
  }

 private:
  // Width-specific bulk-set/get implementations. Callers use the public
  // bulkSetWithBaseline / bulkGetWithBaseline, which dispatch to these by
  // element type.

  // Only callable when bitWidth <= 32; subtracts baseline from each value.
  // Same zeroed-buffer precondition as set.
  void bulkSet32WithBaseline(
      uint64_t start,
      uint64_t length,
      const uint32_t* values,
      uint32_t baseline);

  // Packs 64-bit values, subtracting baseline; supports bit widths up to 64.
  // For bitWidth < 32 delegates to bulkSet32; 32/40/48/56/64 use direct stores;
  // up to 58 use a single 64-bit OR per value; 59-63 handle cross-word
  // overflow. Same zeroed-buffer precondition as set.
  void bulkSet64WithBaseline(
      uint64_t start,
      uint64_t length,
      const uint64_t* values,
      uint64_t baseline);

  // Only callable when bitWidth <= 32; adds baseline to every value.
  void bulkGetWithBaseline32(
      uint64_t start,
      uint64_t length,
      uint32_t* values,
      uint32_t baseline) const;

  // Reads 64-bit values, adding baseline; supports bit widths up to 64. For
  // bitWidth < 32 delegates to bulkGet32; 32/40/48/56/64 use direct loads; up
  // to 58 use a single 64-bit load per value; 59-63 handle cross-word overflow.
  void bulkGet64WithBaseline(
      uint64_t start,
      uint64_t length,
      uint64_t* values,
      uint64_t baseline) const;

  char* buffer_;
  int bitWidth_;
  uint64_t mask_;
};

} // namespace facebook::nimble

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

namespace torch::wave {

/// Packed inputs: a graph input that holds the values of many features back to
/// back (the values of a KeyedJaggedTensor, key-major) and is consumed only by
/// one split_with_sizes into per-feature views, one length per feature. Each
/// feature is a "slot". The host re-encodes such an input before its
/// host-to-device copy: each slot is stored at the narrowest byte width its
/// values need, as an offset from the slot's minimum, and the graph decodes it
/// back to int64 per slot with tw.unpack_slot.
///
/// The encoded input is an int64 tensor of words, laid out as
///   header   kPackHeaderWords words: kPackMagic, numSlots, numValues,
///            data offset in bytes, total bytes, then zero
///   desc     numSlots rows of kPackDescWords words:
///              0 byteOffset   from the start of the buffer, 64-byte aligned
///              1 byteLength
///              2 valueCount
///              3 bits         0 (every value == base), 8, 16, 32 or 64
///              4 base         added to each stored value, mod 2^64
///              5 flags        kPackReduced if the values were reduced by the
///                             slot's modulus
///              6 valueOffset  values in the slots before this one
///              7 hash         if > 0, each decoded value v becomes v % hash
///                             (truncated toward zero, as torch.fmod): the
///                             id hashing a trainer applies after H2D
///   data     each slot's values, (value - base) as an unsigned integer of
///            'bits' bits, at byteOffset.
/// Shared by the host encoder (PackedInput.h) and the device decoder
/// (unpack_slot in Unpack.cuh), so it holds only constants.
inline constexpr int64_t kPackMagic = 0x3176'6b63'6150'5754; // "TWPack1v"
inline constexpr int64_t kPackHeaderWords = 8;
inline constexpr int64_t kPackDescWords = 8;
inline constexpr int64_t kPackAlign = 64;
inline constexpr int64_t kPackReduced = 1;

/// Word indices of the header.
inline constexpr int64_t kPackHeaderMagic = 0;
inline constexpr int64_t kPackHeaderNumSlots = 1;
inline constexpr int64_t kPackHeaderNumValues = 2;
inline constexpr int64_t kPackHeaderDataOffset = 3;
inline constexpr int64_t kPackHeaderTotalBytes = 4;

/// Word indices of a slot descriptor.
inline constexpr int64_t kPackDescByteOffset = 0;
inline constexpr int64_t kPackDescByteLength = 1;
inline constexpr int64_t kPackDescValueCount = 2;
inline constexpr int64_t kPackDescBits = 3;
inline constexpr int64_t kPackDescBase = 4;
inline constexpr int64_t kPackDescFlags = 5;
inline constexpr int64_t kPackDescValueOffset = 6;
inline constexpr int64_t kPackDescHash = 7;

} // namespace torch::wave

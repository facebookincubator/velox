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

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/process/ProcessBase.h"

#include <folly/BenchmarkUtil.h>
#include <folly/hash/rapidhash.h>

namespace facebook::velox::bits {

namespace {
// Naive implementation that does not rely on BMI2.
void scatterBitsSimple(
    int32_t numSource,
    int32_t numTarget,
    const char* source,
    const uint64_t* targetMask,
    char* target) {
  int64_t from = numSource - 1;
  for (int64_t to = numTarget - 1; to >= 0; to--) {
    const bool maskIsSet = bits::isBitSet(targetMask, to);
    bits::setBit(target, to, maskIsSet && bits::isBitSet(source, from));
    from -= maskIsSet ? 1 : 0;
  }
}

#ifdef __BMI2__
// Fetches 'numBits' bits of data, from data starting at lastBit -
// numbits (inclusive) and ending at lastBit (exclusive). 'lastBit' is
// updated to be the bit offset of the lowest returned bit. Successive
// calls will go through 'data' from high to low in consecutive chunks
// of up to 64 bits each.
uint64_t getBitField(const char* data, int32_t numBits, int32_t& lastBit) {
  int32_t highByte = lastBit / 8;
  int32_t lowByte = (lastBit - numBits) / 8;
  int32_t lowBit = (lastBit - numBits) & 7;
  uint64_t bits = *reinterpret_cast<const uint64_t*>(data + lowByte) >> lowBit;
  if (numBits + lowBit > 64) {
    auto fromNextByte = numBits + lowBit - 64;
    uint8_t lastBits = *reinterpret_cast<const uint8_t*>(data + highByte) &
        bits::lowMask(fromNextByte);
    bits |= static_cast<uint64_t>(lastBits) << (64 - lowBit);
  }
  lastBit -= numBits;
  return bits;
}
#endif

// Copy bits backward while the remaining data is still larger than size of T.
template <typename T>
inline void copyBitsBackwardImpl(
    uint64_t* bits,
    uint64_t sourceOffset,
    uint64_t targetOffset,
    int64_t& remaining) {
  constexpr int kBits = 8 * sizeof(T);
  for (; remaining >= kBits; remaining -= kBits) {
    T word = detail::loadBits<T>(bits, sourceOffset + remaining - kBits, kBits);
    detail::storeBits<T>(bits, targetOffset + remaining - kBits, word, kBits);
  }
}

} // namespace

void copyBitsBackward(
    uint64_t* bits,
    uint64_t sourceOffset,
    uint64_t targetOffset,
    uint64_t numBits) {
  VELOX_DCHECK_LE(sourceOffset, targetOffset);
  int64_t remaining = numBits;
  // Copy using the largest unit first and narrow down to smaller ones.
  copyBitsBackwardImpl<uint64_t>(bits, sourceOffset, targetOffset, remaining);
  copyBitsBackwardImpl<uint32_t>(bits, sourceOffset, targetOffset, remaining);
  copyBitsBackwardImpl<uint16_t>(bits, sourceOffset, targetOffset, remaining);
  copyBitsBackwardImpl<uint8_t>(bits, sourceOffset, targetOffset, remaining);
  if (remaining > 0) {
    uint8_t byte = detail::loadBits<uint8_t>(bits, sourceOffset, remaining);
    detail::storeBits<uint8_t>(bits, targetOffset, byte, remaining);
  }
}

void toString(const void* bits, int offset, int size, char* out) {
  for (int i = 0; i < size; ++i) {
    out[i] = '0' + isBitSet(reinterpret_cast<const uint8_t*>(bits), offset + i);
  }
}

std::string toString(const void* bits, int offset, int size) {
  std::string ans(size, '\0');
  toString(bits, offset, size, ans.data());
  return ans;
}

void scatterBits(
    int32_t numSource,
    int32_t numTarget,
    const char* source,
    const uint64_t* targetMask,
    char* target) {
  if (!process::hasBmi2()) {
    scatterBitsSimple(numSource, numTarget, source, targetMask, target);
    return;
  }
#ifdef __BMI2__
  int32_t highByte = numTarget / 8;
  int32_t highBit = numTarget & 7;
  int lowByte = std::max(0, highByte - 7);
  auto maskAsBytes = reinterpret_cast<const char*>(targetMask);
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
  int32_t sourceOffset = std::min(0, (numSource / 8) - 7) + 1;
  folly::doNotOptimizeAway(
      *reinterpret_cast<const uint64_t*>(source + sourceOffset));
  folly::doNotOptimizeAway(
      *reinterpret_cast<const uint64_t*>(maskAsBytes + lowByte + 1));
  folly::doNotOptimizeAway(*reinterpret_cast<uint64_t*>(target + lowByte + 1));
#endif
#endif

  // Loop from top to bottom of 'targetMask' up to 64 bits at a time,
  // with a partial word at either end. Count the set bits and fetch
  // as many consecutive bits of source data. Scatter the source bits
  // over the set bits of the target mask with pdep and write the
  // result into 'target'.
  for (;;) {
    auto numBitsToWrite = (highByte - lowByte) * 8 + highBit;
    if (numBitsToWrite == 64) {
      uint64_t mask =
          *(reinterpret_cast<const uint64_t*>(maskAsBytes + lowByte));
      int32_t consume = __builtin_popcountll(mask);
      uint64_t bits = getBitField(source, consume, numSource);
      *reinterpret_cast<uint64_t*>(target + lowByte) = _pdep_u64(bits, mask);
    } else {
      auto writeMask = bits::lowMask(numBitsToWrite);
      uint64_t mask =
          *(reinterpret_cast<const uint64_t*>(maskAsBytes + lowByte)) &
          writeMask;
      int32_t consume = __builtin_popcountll(mask);
      uint64_t bits = getBitField(source, consume, numSource);
      auto targetPtr = reinterpret_cast<uint64_t*>(target + lowByte);
      uint64_t newBits = _pdep_u64(bits, mask);
      *targetPtr = (*targetPtr & ~writeMask) | (newBits & writeMask);
    }
    VELOX_DCHECK_GE(numSource, 0);
    if (!lowByte) {
      break;
    }
    highByte = lowByte;
    highBit = 0;
    lowByte = std::max(lowByte - 8, 0);
  }
  VELOX_DCHECK_EQ(
      numSource,
      0,
      "scatterBits expects to have numSource bits set in targetMask");
#else
  VELOX_UNREACHABLE();
#endif
}

uint64_t hashBytes(uint64_t seed, const char* data, size_t size) {
  return folly::hash::rapidhashNano_with_seed(data, size, seed);
}

void packBitmap(std::span<const bool> bools, char* bitmap) {
  uint64_t* word = reinterpret_cast<uint64_t*>(bitmap);
  const uint64_t loopCount = bools.size() >> 6;
  const uint64_t remainder = bools.size() - (loopCount << 6);
  const bool* rawBools = bools.data();
  for (uint64_t i = 0; i < loopCount; ++i) {
    for (int j = 0; j < 64; ++j) {
      *word |= static_cast<uint64_t>(*rawBools++) << j;
    }
    ++word;
  }
  for (int j = 0; j < remainder; ++j) {
    *word |= static_cast<uint64_t>(*rawBools++) << j;
  }
}

uint32_t
findSetBit(const char* bitmap, uint32_t begin, uint32_t end, uint32_t n) {
  if (begin >= end || n == 0) {
    return begin;
  }

  const uint64_t* wordPtr = reinterpret_cast<const uint64_t*>(bitmap);

  // Handle bits in the first partial word
  uint32_t wordIdx = begin >> 6;
  uint32_t bitOffset = begin & 63;
  uint64_t word = wordPtr[wordIdx];

  // Mask out bits before 'begin'
  word &= ~((1ULL << bitOffset) - 1);

  while (true) {
    // Count set bits in current word
    uint32_t setBitsInWord = __builtin_popcountll(word);

    if (setBitsInWord >= n) {
      // The n'th set bit is in this word
      while (n > 0) {
        uint32_t firstSetBit = __builtin_ffsll(static_cast<long long>(word));
        if (firstSetBit == 0) {
          break; // No more set bits
        }

        // __builtin_ffsll returns the index plus one, so subtract 1
        --firstSetBit;

        word &= ~(1ULL << firstSetBit);
        --n;

        if (n == 0) {
          uint32_t result = (wordIdx << 6) + firstSetBit;
          return result < end ? result : end;
        }
      }
    }

    // Move to next word
    n -= setBitsInWord;
    ++wordIdx;
    bitOffset = 0;

    // Check if we've reached the end
    uint32_t nextWordStart = wordIdx << 6;
    if (nextWordStart >= end) {
      return end;
    }

    word = wordPtr[wordIdx];

    // Mask out bits beyond 'end' if this is the last word
    if (nextWordStart + 64 > end) {
      word &= (1ULL << (end - nextWordStart)) - 1;
    }

    // If no bits set in this word, continue to next word
    if (word == 0) {
      continue;
    }
  }
}

void BitmapBuilder::copy(const Bitmap& other, uint32_t begin, uint32_t end) {
  auto source = static_cast<const char*>(other.bits());
  auto dest = static_cast<char*>(bitmap_);
  auto firstByte = begin / 8;
  if (begin % 8) {
    uint8_t mask = (1 << (begin % 8)) - 1;
    dest[firstByte] = static_cast<char>(
        (dest[firstByte] & mask) | (source[firstByte] & ~mask));
    ++firstByte;
  }
  // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
  std::memcpy(
      dest + firstByte, source + firstByte, bits::nbytes(end) - firstByte);
}

} // namespace facebook::velox::bits

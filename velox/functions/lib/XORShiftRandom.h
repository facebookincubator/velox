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

namespace facebook::velox::functions {

/// XORShift random number generator matching Spark's XORShiftRandom.
/// Based on Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical
/// Software, Vol. 8, Issue 14.
///
/// Spark hashes the seed with MurmurHash3 before use. This implementation
/// matches that behavior for reproducibility.
class XORShiftRandom {
 public:
  XORShiftRandom() = default;

  void setSeed(int64_t seed) {
    seed_ = hashSeed(seed);
  }

  /// Returns a random 32-bit integer (like Java's nextInt() with no argument).
  int32_t nextInt() {
    return next(32);
  }

  /// Returns a random integer in [0, bound) using Java's algorithm.
  int32_t nextInt(int32_t bound) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(next(31)) * static_cast<uint64_t>(bound)) >> 31);
  }

  /// Returns a random double in [0.0, 1.0) matching Java's Random.nextDouble().
  /// Uses 53 bits of randomness (26 + 27) to fill the mantissa of a double.
  double nextDouble() {
    int64_t bits =
        (static_cast<int64_t>(next(26)) << 27) + static_cast<int64_t>(next(27));
    return static_cast<double>(bits) / static_cast<double>(1LL << 53);
  }

 private:
  /// Generates the next random bits.
  int32_t next(int32_t bits) {
    int64_t nextSeed = seed_ ^ (seed_ << 21);
    nextSeed ^= (static_cast<uint64_t>(nextSeed) >> 35);
    nextSeed ^= (nextSeed << 4);
    seed_ = nextSeed;
    return static_cast<int32_t>(nextSeed & ((1LL << bits) - 1));
  }

  /// Hashes the seed using MurmurHash3 to distribute bits, matching Spark's
  /// XORShiftRandom.hashSeed().
  static int64_t hashSeed(int64_t seed) {
    // Convert seed to bytes in big-endian order (Java's ByteBuffer default).
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
      bytes[i] = static_cast<uint8_t>((seed >> (56 - i * 8)) & 0xFF);
    }
    // Spark uses MurmurHash3.arraySeed (0x3c074a61) as initial seed.
    int32_t lowBits = murmurHash3(bytes, 8, 0x3c074a61);
    int32_t highBits = murmurHash3(bytes, 8, lowBits);
    return (static_cast<int64_t>(highBits) << 32) |
        (static_cast<int64_t>(lowBits) & 0xFFFFFFFFL);
  }

  /// MurmurHash3 32-bit implementation matching Scala's MurmurHash3.bytesHash.
  static int32_t murmurHash3(const uint8_t* data, int32_t len, int32_t seed) {
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    uint32_t h1 = seed;
    int32_t i = 0;

    // Body: process 4-byte chunks.
    while (i + 4 <= len) {
      uint32_t k1 = static_cast<uint32_t>(data[i]) |
          (static_cast<uint32_t>(data[i + 1]) << 8) |
          (static_cast<uint32_t>(data[i + 2]) << 16) |
          (static_cast<uint32_t>(data[i + 3]) << 24);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
      i += 4;
    }

    // Tail: process remaining bytes.
    uint32_t k1 = 0;
    switch (len - i) {
      case 3:
        k1 ^= static_cast<uint32_t>(data[i + 2]) << 16;
        [[fallthrough]];
      case 2:
        k1 ^= static_cast<uint32_t>(data[i + 1]) << 8;
        [[fallthrough]];
      case 1:
        k1 ^= static_cast<uint32_t>(data[i]);
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        break;
      default:
        break;
    }

    // Finalization.
    h1 ^= len;
    h1 = fmix32(h1);
    return static_cast<int32_t>(h1);
  }

  static uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
  }

  static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  int64_t seed_{0};
};

} // namespace facebook::velox::functions

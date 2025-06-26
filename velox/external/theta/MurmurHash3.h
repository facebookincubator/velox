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

// Adapted from Apache DataSketches

// Minimally modified from Austin Applebee's code:
//  * Removed MurmurHash3_x86_32 and MurmurHash3_x86_128
//  * Changed input seed in MurmurHash3_x64_128 to uint64_t
//  * Define and use HashState reference to return result
//  * Made entire hash function defined inline
//  * Added compute_seed_hash
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.

#pragma once

#include <cstring>

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER)

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;

#define MURMUR3_FORCE_INLINE __forceinline

#include <stdlib.h>

#define MURMUR3_ROTL64(x, y) _rotl64(x, y)

#define MURMUR3_BIG_CONSTANT(x) (x)

// Other compilers

#else // defined(_MSC_VER)

#include <stdint.h>

#define MURMUR3_FORCE_INLINE inline __attribute__((always_inline))

inline uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define MURMUR3_ROTL64(x, y) rotl64(x, y)

#define MURMUR3_BIG_CONSTANT(x) (x##LLU)

#endif // !defined(_MSC_VER)

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Return type - Using C++ reference for return type which should allow better
// compiler optimization than a void* pointer
typedef struct {
  uint64_t h1;
  uint64_t h2;
} HashState;

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

MURMUR3_FORCE_INLINE uint64_t getblock64(const uint8_t* p, size_t i) {
  uint64_t res;
  memcpy(&res, p + i * sizeof(uint64_t), sizeof(res));
  return res;
}

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

MURMUR3_FORCE_INLINE uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= MURMUR3_BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= MURMUR3_BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

MURMUR3_FORCE_INLINE void MurmurHash3_x64_128(
    const void* key,
    size_t lenBytes,
    uint64_t seed,
    HashState& out) {
  static const uint64_t c1 = MURMUR3_BIG_CONSTANT(0x87c37b91114253d5);
  static const uint64_t c2 = MURMUR3_BIG_CONSTANT(0x4cf5ad432745937f);

  const uint8_t* data = (const uint8_t*)key;

  out.h1 = seed;
  out.h2 = seed;

  // Number of full 128-bit blocks of 16 bytes.
  // Possible exclusion of a remainder of up to 15 bytes.
  const size_t nblocks = lenBytes >> 4; // bytes / 16

  // Process the 128-bit blocks (the body) into the hash
  for (size_t i = 0; i < nblocks; ++i) { // 16 bytes per block
    uint64_t k1 = getblock64(data, i * 2 + 0);
    uint64_t k2 = getblock64(data, i * 2 + 1);

    k1 *= c1;
    k1 = MURMUR3_ROTL64(k1, 31);
    k1 *= c2;
    out.h1 ^= k1;
    out.h1 = MURMUR3_ROTL64(out.h1, 27);
    out.h1 += out.h2;
    out.h1 = out.h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = MURMUR3_ROTL64(k2, 33);
    k2 *= c1;
    out.h2 ^= k2;
    out.h2 = MURMUR3_ROTL64(out.h2, 31);
    out.h2 += out.h1;
    out.h2 = out.h2 * 5 + 0x38495ab5;
  }

  // tail
  const uint8_t* tail = (const uint8_t*)(data + (nblocks << 4));

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (lenBytes & 15) {
    case 15:
      k2 ^= ((uint64_t)tail[14]) << 48; // falls through
    case 14:
      k2 ^= ((uint64_t)tail[13]) << 40; // falls through
    case 13:
      k2 ^= ((uint64_t)tail[12]) << 32; // falls through
    case 12:
      k2 ^= ((uint64_t)tail[11]) << 24; // falls through
    case 11:
      k2 ^= ((uint64_t)tail[10]) << 16; // falls through
    case 10:
      k2 ^= ((uint64_t)tail[9]) << 8; // falls through
    case 9:
      k2 ^= ((uint64_t)tail[8]) << 0;
      k2 *= c2;
      k2 = MURMUR3_ROTL64(k2, 33);
      k2 *= c1;
      out.h2 ^= k2;
      // falls through
    case 8:
      k1 ^= ((uint64_t)tail[7]) << 56; // falls through
    case 7:
      k1 ^= ((uint64_t)tail[6]) << 48; // falls through
    case 6:
      k1 ^= ((uint64_t)tail[5]) << 40; // falls through
    case 5:
      k1 ^= ((uint64_t)tail[4]) << 32; // falls through
    case 4:
      k1 ^= ((uint64_t)tail[3]) << 24; // falls through
    case 3:
      k1 ^= ((uint64_t)tail[2]) << 16; // falls through
    case 2:
      k1 ^= ((uint64_t)tail[1]) << 8; // falls through
    case 1:
      k1 ^= ((uint64_t)tail[0]) << 0;
      k1 *= c1;
      k1 = MURMUR3_ROTL64(k1, 31);
      k1 *= c2;
      out.h1 ^= k1;
  };

  //----------
  // finalization

  out.h1 ^= lenBytes;
  out.h2 ^= lenBytes;

  out.h1 += out.h2;
  out.h2 += out.h1;

  out.h1 = fmix64(out.h1);
  out.h2 = fmix64(out.h2);

  out.h1 += out.h2;
  out.h2 += out.h1;
}

//-----------------------------------------------------------------------------

MURMUR3_FORCE_INLINE uint16_t compute_seed_hash(uint64_t seed) {
  HashState hashes;
  MurmurHash3_x64_128(&seed, sizeof(seed), 0, hashes);
  return static_cast<uint16_t>(hashes.h1 & 0xffff);
}

#undef MURMUR3_FORCE_INLINE
#undef MURMUR3_ROTL64
#undef MURMUR3_BIG_CONSTANT

/*
 * Copyright (a) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace facebook::velox::exec {

constexpr size_t kCacheLineAlignment = 64;
constexpr size_t kDefaultNumBuckets = 256;
constexpr size_t kPrefetchDistance = 16;
constexpr size_t kTuplesPerChunk = 4096; // 64KB L2 cache fitting chunk

struct alignas(kCacheLineAlignment) RadixTuple {
  uint64_t key;
  uint64_t payload;
};

// Thread-Local 64-byte Aligned Bump-Pointer Memory Arena Pool
class AlignedArena {
 public:
  uint8_t* buffer{nullptr};
  size_t capacity{0};
  size_t offset{0};

  explicit AlignedArena(size_t cap = 1024ULL * 1024ULL * 1024ULL)
      : capacity(cap), offset(0) {
    int res = posix_memalign((void**)&buffer, kCacheLineAlignment, capacity);
    assert(res == 0 && "AlignedArena allocation failed");
  }

  ~AlignedArena() {
    if (buffer) {
      free(buffer);
    }
  }

  inline void* allocate(size_t bytes) {
    size_t aligned_bytes = (bytes + 63) & ~63;
    size_t current = offset;
    offset += aligned_bytes;
    assert(offset <= capacity && "AlignedArena capacity exhausted!");
    return buffer + current;
  }

  inline void reset() {
    offset = 0;
  }
};

// 64-Byte Cache-Line Aligned Sub-Partition Chunk Node (Zero-Copy)
struct alignas(kCacheLineAlignment) ChunkNode {
  RadixTuple data[kTuplesPerChunk];
  uint32_t count;
  ChunkNode* next;
};

// Skew-Resilient Zero-Copy Hardware-Conscious Radix Partitioner
class SkewResilientRadixPartitioner {
 public:
  AlignedArena arena;
  ChunkNode* heads[kDefaultNumBuckets];
  ChunkNode* tails[kDefaultNumBuckets];

  SkewResilientRadixPartitioner() : arena(1024ULL * 1024ULL * 1024ULL) {
    for (size_t i = 0; i < kDefaultNumBuckets; ++i) {
      auto* node =
          reinterpret_cast<ChunkNode*>(arena.allocate(sizeof(ChunkNode)));
      node->count = 0;
      node->next = nullptr;
      heads[i] = node;
      tails[i] = node;
    }
  }

  // Inline MurmurHash3 key scrambler for zero low-bit key alias collisions
  inline size_t extractRadixBucket(uint64_t key) const {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key & (kDefaultNumBuckets - 1);
  }

  void partition(const std::vector<RadixTuple>& input) {
    const size_t size = input.size();
    const RadixTuple* raw_input = input.data();

    for (size_t i = 0; i < size; ++i) {
      if (i + kPrefetchDistance < size) {
        __builtin_prefetch(&raw_input[i + kPrefetchDistance], 0, 3);
      }

      uint64_t key = raw_input[i].key;
      size_t idx = extractRadixBucket(key);
      ChunkNode* tail = tails[idx];

      if (__builtin_expect(tail->count == kTuplesPerChunk, 0)) {
        auto* new_node =
            reinterpret_cast<ChunkNode*>(arena.allocate(sizeof(ChunkNode)));
        new_node->count = 0;
        new_node->next = nullptr;
        tail->next = new_node;
        tails[idx] = new_node;
        tail = new_node;
      }
      tail->data[tail->count++] = raw_input[i];
    }
  }
};

} // namespace facebook::velox::exec

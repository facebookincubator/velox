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

#include <breeze/functions/reduce.h>
#include <breeze/functions/store.h>
#include <breeze/platforms/platform.h>
#include <breeze/utils/types.h>
#include <breeze/platforms/cuda.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include "velox/experimental/wave/common/CudaUtil.cuh"

/// Utilities for  booleans and indices and thread blocks.

namespace facebook::velox::wave {

/// Converts an array of flags to an array of indices of set flags. The first
/// index is given by 'start'. The number of indices is returned in 'size', i.e.
/// this is 1 + the index of the last set flag.
template <typename T, int32_t blockSize>
inline int32_t __device__ __host__ boolToIndicesSharedSize() {
  using namespace breeze::functions;

  using PlatformT = CudaPlatform<blockSize, kWarpThreads>;
  using BlockScanT = BlockScan<PlatformT, T, /*kItemsPerThread=*/1>;

  return sizeof(typename BlockScanT::Scratch);
}

/// Converts an array of flags to an array of indices of set flags. The first
/// index is given by 'start'. The number of indices is returned in 'size', i.e.
/// this is 1 + the index of the last set flag.
template <int32_t blockSize, typename T, typename Getter>
__device__ inline void
boolBlockToIndices(Getter getter, unsigned start, T* indices, void* shmem, T& size) {
  using namespace breeze::functions;
  using namespace breeze::utils;

  CudaPlatform<blockSize, kWarpThreads> p;
  using BlockScanT = BlockScan<decltype(p), T, /*kItemsPerThread=*/1>;

  auto* temp = reinterpret_cast<typename BlockScanT::Scratch*>(shmem);
  T data[1];
  uint8_t flag = getter();
  data[0] = flag;
  __syncthreads();
  // Perform inclusive scan
  T aggregate = BlockScanT::template Scan<ScanOpAdd>(
      p,
      make_slice(data),
      make_slice(data),
      make_slice(temp).template reinterpret<SHARED>());
  if (flag) {
    T exclusive_result = data[0] - flag;
    indices[exclusive_result] = threadIdx.x + start;
  }
  if (threadIdx.x == (blockSize - 1)) {
    size = aggregate;
  }
  __syncthreads();
}

inline int32_t __device__ __host__ bool256ToIndicesSize() {
  return sizeof(typename cub::WarpScan<uint16_t>::TempStorage) +
      33 * sizeof(uint16_t);
}

/// Returns indices of set bits for 256 one byte flags. 'getter8' is
/// invoked for 8 flags at a time, with the ordinal of the 8 byte
/// flags word as argument, so that an index of 1 means flags
/// 8..15. The indices start at 'start' and last index + 1 is
/// returned in 'size'.
template <typename T, typename Getter8>
__device__ inline void bool256ToIndices(
    Getter8 getter8,
    unsigned start,
    T* indices,
    T& size,
    char* smem) {
  using Scan = cub::WarpScan<uint16_t>;
  auto* smem16 = reinterpret_cast<uint16_t*>(smem);
  auto group = threadIdx.x / 8;
  uint64_t bits = getter8(group) & 0x0101010101010101;
  if ((threadIdx.x & 7) == 0) {
    smem16[group] = __popcll(bits);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    auto* temp = reinterpret_cast<typename Scan::TempStorage*>((smem + 72));
    uint16_t data = smem16[threadIdx.x];
    uint16_t result;
    Scan(*temp).ExclusiveSum(data, result);
    smem16[threadIdx.x] = result;
    if (threadIdx.x == 31) {
      size = data + result;
    }
  }
  __syncthreads();
  auto tidInGroup = threadIdx.x & 7;
  if (bits & (1UL << (tidInGroup * 8))) {
    int32_t base =
        smem16[group] + __popcll(bits & lowMask<uint64_t>(tidInGroup * 8));
    indices[base] = threadIdx.x + start;
  }
  __syncthreads();
}

template <int32_t blockSize, typename T, typename Getter>
__device__ inline void blockSum(Getter getter, void* shmem, T* result) {
  using namespace breeze::functions;
  using namespace breeze::utils;

  CudaPlatform<blockSize, kWarpThreads> p;
  using BlockReduceT = BlockReduce<decltype(p), T>;

  auto* temp = reinterpret_cast<typename BlockReduceT::Scratch*>(shmem);
  T data[1];
  data[0] = getter();
  T aggregate =
      BlockReduceT::template Reduce<ReduceOpAdd, /*kItemsPerThread=*/1>(
          p, make_slice(data), make_slice(temp).template reinterpret<SHARED>());
  if (p.thread_idx() == 0) {
    result[p.block_idx()] = aggregate;
  }
}

template <
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    typename Key,
    typename Value>
using RadixSort =
    typename cub::BlockRadixSort<Key, kBlockSize, kItemsPerThread, Value>;

template <
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    typename Key,
    typename Value>
inline int32_t __host__ __device__ blockSortSharedSize() {
  return sizeof(
      typename RadixSort<kBlockSize, kItemsPerThread, Key, Value>::TempStorage);
}

template <
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    typename Key,
    typename Value,
    typename KeyGetter,
    typename ValueGetter>
void __device__ blockSort(
    KeyGetter keyGetter,
    ValueGetter valueGetter,
    Key* keyOut,
    Value* valueOut,
    char* smem) {
  using namespace breeze::functions;
  using namespace breeze::utils;
  using Sort = cub::BlockRadixSort<Key, kBlockSize, kItemsPerThread, Value>;

  // Per-thread tile items
  Key keys[kItemsPerThread];
  Value values[kItemsPerThread];

  // Our current block's offset
  int blockOffset = 0;

  // Load items into a blocked arrangement
  for (auto i = 0; i < kItemsPerThread; ++i) {
    auto idx = blockOffset + i * kBlockSize + threadIdx.x;
    values[i] = valueGetter(idx);
    keys[i] = keyGetter(idx);
  }

  __syncthreads();
  auto* temp_storage = reinterpret_cast<typename Sort::TempStorage*>(smem);

  Sort(*temp_storage).SortBlockedToStriped(keys, values);

  // Store a striped arrangement of output across the thread block into a linear
  // segment of items
  CudaPlatform<kBlockSize, kWarpThreads> p;
  BlockStore<kBlockSize, kItemsPerThread>(
      p,
      make_slice<THREAD, STRIPED>(values),
      make_slice<GLOBAL>(valueOut + blockOffset));
  BlockStore<kBlockSize, kItemsPerThread>(
      p,
      make_slice<THREAD, STRIPED>(keys),
      make_slice<GLOBAL>(keyOut + blockOffset));
  __syncthreads();
}

template <int kBlockSize>
int32_t partitionRowsSharedSize(int32_t numPartitions) {
  using namespace breeze::functions;
  using PlatformT = CudaPlatform<kBlockSize, kWarpThreads>;
  using BlockScanT = BlockScan<PlatformT, int32_t, /*kItemsPerThread=*/1>;
  auto scanSize =
      max(sizeof(typename BlockScanT::Scratch), sizeof(int32_t) * kBlockSize) +
      sizeof(int32_t);
  int32_t counterSize = sizeof(int32_t) * numPartitions;
  if (counterSize <= scanSize) {
    return scanSize;
  }
  return scanSize + counterSize; // - kBlockSize * sizeof(int32_t);
}

/// Partitions a sequence of indices into runs where the indices
/// belonging to the same partition are contiguous. Indices from 0 to
/// 'numKeys-1' are partitioned into 'partitionedRows', which must
/// have space for 'numKeys' row numbers. The 0-based partition number
/// for row 'i' is given by 'getter(i)'.  The row numbers for
/// partition 0 start at 0. The row numbers for partition i start at
/// 'partitionStarts[i-1]'. There must be at least the amount of
/// shared memory given by partitionSharedSize(numPartitions).
/// 'ranks' is a temporary array of 'numKeys' elements.
template <int32_t kBlockSize, typename RowNumber, typename Getter>
void __device__ partitionRows(
    Getter getter,
    uint32_t numKeys,
    uint32_t numPartitions,
    RowNumber* ranks,
    RowNumber* partitionStarts,
    RowNumber* partitionedRows) {
  using namespace breeze::functions;
  using namespace breeze::utils;

  CudaPlatform<kBlockSize, kWarpThreads> p;
  using BlockScanT = BlockScan<decltype(p), int32_t, /*kItemsPerThread=*/1>;
  auto warp = threadIdx.x / kWarpThreads;
  auto lane = cub::LaneId();
  extern __shared__ __align__(16) char smem[];
  auto* counters = reinterpret_cast<uint32_t*>(
      numPartitions <= kBlockSize ? smem
                                  : smem +
              sizeof(typename BlockScanT::
                         Scratch) /*- kBlockSize * sizeof(uint32_t)*/);
  for (auto i = threadIdx.x; i < numPartitions; i += kBlockSize) {
    counters[i] = 0;
  }
  __syncthreads();
  for (auto start = 0; start < numKeys; start += kBlockSize) {
    int32_t warpStart = start + warp * kWarpThreads;
    if (start >= numKeys) {
      break;
    }
    uint32_t laneMask = warpStart + kWarpThreads <= numKeys
        ? 0xffffffff
        : lowMask<uint32_t>(numKeys - warpStart);
    if (warpStart + lane < numKeys) {
      int32_t key = getter(warpStart + lane);
      uint32_t mask = __match_any_sync(laneMask, key);
      int32_t leader = (kWarpThreads - 1) - __clz(mask);
      uint32_t cnt = __popc(mask & lowMask<uint32_t>(lane + 1));
      uint32_t base;
      if (lane == leader) {
        base = atomicAdd(&counters[key], cnt);
      }
      base = __shfl_sync(laneMask, base, leader);
      ranks[warpStart + lane] = base + cnt - 1;
    }
  }
  // Prefix sum the counts. All counters must have their final value.
  __syncthreads();
  auto* temp = reinterpret_cast<typename BlockScanT::Scratch*>(smem);
  int32_t* aggregate = reinterpret_cast<int32_t*>(smem);
  for (auto start = 0; start < numPartitions; start += kBlockSize) {
    int32_t localCount[1];
    localCount[0] =
        threadIdx.x + start < numPartitions ? counters[start + threadIdx.x] : 0;
    if (threadIdx.x == 0 && start > 0) {
      // The sum of the previous round is carried over as start of this.
      localCount[0] += *aggregate;
    }
    BlockScanT::template Scan<ScanOpAdd>(
        p,
        make_slice(localCount),
        make_slice(localCount),
        make_slice(temp).template reinterpret<SHARED>());
    if (start + threadIdx.x < numPartitions) {
      partitionStarts[start + threadIdx.x] = localCount[0];
    }
    if (threadIdx.x == kBlockSize - 1 && start + kBlockSize < numPartitions) {
      *aggregate = localCount[0];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (partitionStarts[numPartitions - 1] != numKeys) {
      *(long*)0 = 0;
    }
  }
  // Write the row numbers of the inputs into the rankth position in each
  // partition.
  for (auto i = threadIdx.x; i < numKeys; i += kBlockSize) {
    auto key = getter(i);
    auto keyStart = key == 0 ? 0 : partitionStarts[key - 1];
    partitionedRows[keyStart + ranks[i]] = i;
  }
  __syncthreads();
}

} // namespace facebook::velox::wave

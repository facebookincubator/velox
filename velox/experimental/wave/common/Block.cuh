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

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

/// Utilities for  booleans and indices and thread blocks.

namespace facebook::velox::wave {

template <
    typename T,
    int32_t blockSize,
    cub::BlockScanAlgorithm Algorithm = cub::BLOCK_SCAN_RAKING>
inline int32_t __device__ __host__ boolToIndicesSharedSize() {
  typedef cub::BlockScan<T, blockSize, Algorithm> BlockScanT;

  return sizeof(typename BlockScanT::TempStorage);
}

/// Converts an array of flags to an array of indices of set flags. The first
/// index is given by 'start'. The number of indices is returned in 'size', i.e.
/// this is 1 + the index of the last set flag.
template <
    int32_t blockSize,
    typename T,
    cub::BlockScanAlgorithm Algorithm = cub::BLOCK_SCAN_RAKING,
    typename Getter>
__device__ inline void
boolBlockToIndices(Getter getter, T start, T* indices, void* shmem, T& size) {
  typedef cub::BlockScan<T, blockSize, Algorithm> BlockScanT;

  auto* temp = reinterpret_cast<typename BlockScanT::TempStorage*>(shmem);
  T data[1];
  uint8_t flag = getter();
  data[0] = flag;
  __syncthreads();
  T aggregate;
  BlockScanT(*temp).ExclusiveSum(data, data, aggregate);
  if (flag) {
    indices[data[0]] = threadIdx.x + start;
  }
  if (threadIdx.x == 0) {
    size = aggregate;
  }
  __syncthreads();
}

template <int32_t blockSize, typename T, typename Getter>
__device__ inline void blockSum(Getter getter, void* shmem, T* result) {
  typedef cub::BlockReduce<T, blockSize> BlockReduceT;

  auto* temp = reinterpret_cast<typename BlockReduceT::TempStorage*>(shmem);
  T data[1];
  data[0] = getter();
  T aggregate = BlockReduceT(*temp).Reduce(data, cub::Sum());

  if (threadIdx.x == 0) {
    result[blockIdx.x] = aggregate;
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
  using Sort = cub::BlockRadixSort<Key, kBlockSize, kItemsPerThread, Value>;

  // Per-thread tile items
  Key keys[kItemsPerThread];
  Value values[kItemsPerThread];

  // Our current block's offset
  int blockOffset = 0;

  // Load items into a blocked arrangement
  for (auto i = 0; i < kItemsPerThread; ++i) {
    int32_t idx = blockOffset + i * kBlockSize + threadIdx.x;
    values[i] = valueGetter(idx);
    keys[i] = keyGetter(idx);
  }

  __syncthreads();
  auto* temp_storage = reinterpret_cast<typename Sort::TempStorage*>(smem);

  Sort(*temp_storage).SortBlockedToStriped(keys, values);

  // Store output in striped fashion
  cub::StoreDirectStriped<kBlockSize>(
      threadIdx.x, valueOut + blockOffset, values);
  cub::StoreDirectStriped<kBlockSize>(threadIdx.x, keyOut + blockOffset, keys);
  __syncthreads();
}

} // namespace facebook::velox::wave

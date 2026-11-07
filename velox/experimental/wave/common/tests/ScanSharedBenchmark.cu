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

/// Measures the shared memory transpose technique for multi element per thread
/// scans and stream compaction. ScanBenchmark.cu is the baseline; this file
/// adds the striped layouts and keeps the baseline layouts as controls so the
/// two effects are visible side by side in one table.
///
/// A block owns TILE = blockSize * itemsPerThread consecutive elements. There
/// are two ways to hand them to threads:
///  - Blocked: thread t owns the contiguous run [t * items, (t + 1) * items).
///    The scan math is trivial but global accesses at a fixed item are
///    itemsPerThread apart.
///  - Striped: thread t owns {t, t + blockSize, ...}. Global accesses are dense
///    but a thread's elements are not contiguous in logical order, so the
///    "local scan plus scan of thread totals" form does not apply.
/// The transpose runs global I/O striped and the scan math blocked, paying the
/// strided access in shared memory instead of in HBM.
///
/// The variants, all doing the same reduce then scan pipeline as the baseline
/// benchmark:
///   base  1 element per thread. Coalesced by construction.
///   blk   Blocked, scalar global accesses. The trap the technique avoids.
///   vec   Blocked, but each thread's run loaded as 16 byte vectors. What
///         ScanBenchmark.cu does, and much better than 'blk'.
///   str   Striped global I/O with a shared memory transpose. The scanned
///         elements live in registers between the two shared passes.
///   shm   Same transpose, but the per thread scan runs directly on shared
///         memory, so no itemsPerThread sized register array exists. Trades
///         shared memory traffic for registers and hence for occupancy.
///   shm+  'shm' with the transpose buffer padded to an odd stride per thread
///         run. 'shm' touches shared three times per element instead of twice,
///         which is where the blocked bank conflicts start to show.
/// cub::DeviceScan / cub::DeviceSelect and a device to device copy of the
/// payload are printed as the vendor reference and the bandwidth floor.

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/Scan.cuh"

DEFINE_int64(
    scan_bytes,
    200 << 20,
    "Size of the scanned/compacted int64 data in bytes.");

DEFINE_double(
    scan_selectivity,
    0.8,
    "Fraction of set flags in the stream compaction benchmark.");

DEFINE_int32(scan_repeats, 10, "Number of timed repetitions after one warmup.");

DEFINE_bool(
    scan_check,
    true,
    "Verify every variant against a CPU reference before timing it.");

namespace facebook::velox::wave {
namespace {

using ScanType = int64_t;
using CountType = int32_t;

// How a tile of blockSize * itemsPerThread elements is handed to the threads.
enum class Layout {
  // Thread t owns [t * items, (t + 1) * items), one element per access.
  kBlockedScalar,
  // Same ownership, but the run is moved as 16 byte vectors.
  kBlockedVector,
  // Striped global I/O transposed through shared memory, scanned in registers.
  kStripedRegisters,
  // Striped global I/O transposed through shared memory, scanned in place in
  // shared memory so that no per item registers are needed.
  kStripedShared,
  // As kStripedShared, but the transpose buffer gives every thread run an odd
  // stride so that the blocked shared accesses spread over all banks.
  kStripedSharedPadded,
};

template <Layout kLayout>
constexpr bool kIsStriped =
    kLayout == Layout::kStripedRegisters || kLayout == Layout::kStripedShared ||
    kLayout == Layout::kStripedSharedPadded;

template <Layout kLayout>
constexpr bool kScansInShared = kLayout == Layout::kStripedShared ||
    kLayout == Layout::kStripedSharedPadded;

const char* layoutName(Layout layout, int32_t itemsPerThread) {
  if (itemsPerThread == 1) {
    return "base";
  }
  switch (layout) {
    case Layout::kBlockedScalar:
      return "blk";
    case Layout::kBlockedVector:
      return "vec";
    case Layout::kStripedRegisters:
      return "str";
    case Layout::kStripedShared:
      return "shm";
    case Layout::kStripedSharedPadded:
      return "shm+";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Layout primitives.
// ---------------------------------------------------------------------------

// Widest vector type that moves 'kBytesPerThread' bytes in one access.
template <int32_t kBytesPerThread>
struct BlockedVector {
  using Type = int4;
};

template <>
struct BlockedVector<1> {
  using Type = uint8_t;
};

template <>
struct BlockedVector<2> {
  using Type = uint16_t;
};

template <>
struct BlockedVector<4> {
  using Type = uint32_t;
};

template <>
struct BlockedVector<8> {
  using Type = uint2;
};

// Distance between the starts of two thread runs in the transpose buffer. With
// the natural stride a blocked access at a fixed item lands on a stride of
// itemsPerThread elements, which for a power of two item count maps a whole
// warp onto a few banks. Adding one element per run makes the stride odd and
// spreads the access over all banks, at the cost of a multiply in the index.
// kStripedSharedPadded measures whether that trade pays off.
template <Layout kLayout, int32_t kItemsPerThread>
constexpr int32_t kSharedStride =
    kLayout == Layout::kStripedSharedPadded ? kItemsPerThread + 1
                                            : kItemsPerThread;

template <int32_t kItemsPerThread, int32_t kStride>
__device__ inline int32_t sharedIndex(int32_t logicalIndex) {
  return (logicalIndex / kItemsPerThread) * kStride +
      logicalIndex % kItemsPerThread;
}

template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void loadBlockedScalar(
    const T* tile,
    T (&items)[kItemsPerThread]) {
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    items[i] = tile[static_cast<int32_t>(threadIdx.x) * kItemsPerThread + i];
  }
}

template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void storeBlockedScalar(
    T* tile,
    const T (&items)[kItemsPerThread]) {
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    tile[static_cast<int32_t>(threadIdx.x) * kItemsPerThread + i] = items[i];
  }
}

template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void loadBlockedVector(
    const T* tile,
    T (&items)[kItemsPerThread]) {
  using Vector = typename BlockedVector<static_cast<int32_t>(
      kItemsPerThread * sizeof(T))>::Type;
  constexpr int32_t kItemsPerVector = sizeof(Vector) / sizeof(T);
  constexpr int32_t kVectorsPerThread = kItemsPerThread / kItemsPerVector;
  Vector vectors[kVectorsPerThread];
  const auto* from =
      reinterpret_cast<const Vector*>(tile) + threadIdx.x * kVectorsPerThread;
#pragma unroll
  for (int32_t i = 0; i < kVectorsPerThread; ++i) {
    vectors[i] = from[i];
  }
  const auto* raw = reinterpret_cast<const T*>(vectors);
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    items[i] = raw[i];
  }
}

template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void storeBlockedVector(
    T* tile,
    const T (&items)[kItemsPerThread]) {
  using Vector = typename BlockedVector<static_cast<int32_t>(
      kItemsPerThread * sizeof(T))>::Type;
  constexpr int32_t kItemsPerVector = sizeof(Vector) / sizeof(T);
  constexpr int32_t kVectorsPerThread = kItemsPerThread / kItemsPerVector;
  Vector vectors[kVectorsPerThread];
  auto* raw = reinterpret_cast<T*>(vectors);
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    raw[i] = items[i];
  }
  auto* to = reinterpret_cast<Vector*>(tile) + threadIdx.x * kVectorsPerThread;
#pragma unroll
  for (int32_t i = 0; i < kVectorsPerThread; ++i) {
    to[i] = vectors[i];
  }
}

// Blocked accesses for the ragged last tile. Elements at or after 'numValid'
// take 'fill', which is the identity of the sum.
template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void loadBlockedPartial(
    const T* tile,
    int32_t numValid,
    T fill,
    T (&items)[kItemsPerThread]) {
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    const int32_t index =
        static_cast<int32_t>(threadIdx.x) * kItemsPerThread + i;
    items[i] = index < numValid ? tile[index] : fill;
  }
}

template <typename T, int32_t kBlockSize, int32_t kItemsPerThread>
__device__ inline void storeBlockedPartial(
    T* tile,
    int32_t numValid,
    const T (&items)[kItemsPerThread]) {
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    const int32_t index =
        static_cast<int32_t>(threadIdx.x) * kItemsPerThread + i;
    if (index < numValid) {
      tile[index] = items[i];
    }
  }
}

// Striped global read into the shared transpose buffer. Side by side threads
// touch side by side addresses, so every access is dense.
template <
    typename T,
    typename U,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride>
__device__ inline void
loadStriped(const U* tile, int32_t numValid, T fill, T* shared) {
  const auto tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
  for (int32_t j = 0; j < kItemsPerThread; ++j) {
    const int32_t index = j * kBlockSize + tid;
    shared[sharedIndex<kItemsPerThread, kStride>(index)] =
        index < numValid ? static_cast<T>(tile[index]) : fill;
  }
}

template <
    typename T,
    typename U,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride>
__device__ inline void loadStripedFull(const U* tile, T* shared) {
  const auto tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
  for (int32_t j = 0; j < kItemsPerThread; ++j) {
    const int32_t index = j * kBlockSize + tid;
    shared[sharedIndex<kItemsPerThread, kStride>(index)] =
        static_cast<T>(tile[index]);
  }
}

template <
    typename T,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride>
__device__ inline void
storeStriped(const T* shared, int32_t numValid, T* tile) {
  const auto tid = static_cast<int32_t>(threadIdx.x);
  // The trailing barrier is part of the contract: the next tile overwrites the
  // buffer these reads come from.
#pragma unroll
  for (int32_t j = 0; j < kItemsPerThread; ++j) {
    const int32_t index = j * kBlockSize + tid;
    // The load filled every slot, so the read is safe past 'numValid'; only
    // the store is guarded.
    const T value = shared[sharedIndex<kItemsPerThread, kStride>(index)];
    if (index < numValid) {
      tile[index] = value;
    }
  }
  __syncthreads();
}

template <
    typename T,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride>
__device__ inline void storeStripedFull(const T* shared, T* tile) {
  const auto tid = static_cast<int32_t>(threadIdx.x);
  // The trailing barrier is part of the contract: the next tile overwrites the
  // buffer these reads come from.
#pragma unroll
  for (int32_t j = 0; j < kItemsPerThread; ++j) {
    const int32_t index = j * kBlockSize + tid;
    tile[index] = shared[sharedIndex<kItemsPerThread, kStride>(index)];
  }
  __syncthreads();
}

// Scratch for one tile. The transpose buffer and the scan scratch are live at
// the same time, so they cannot overlap.
template <
    typename T,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride,
    Layout kLayout>
struct TileStorage {
  static constexpr int32_t kTransposeSize =
      kIsStriped<kLayout> ? kBlockSize * kStride : 1;

  T transpose[kTransposeSize];
  T warpSums[kBlockSize / kWarpThreads];
  T total;
};

// Inclusive scan of this thread's contiguous run of the transpose buffer,
// combined with the block wide offset and with 'running', the prefix of
// everything before the tile. Updates 'running' to include the tile. Holds the
// run in registers between the two shared memory passes.
template <
    typename T,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride,
    typename Storage>
__device__ inline void scanRunInRegisters(Storage& storage, T& running) {
  const int32_t first = static_cast<int32_t>(threadIdx.x) * kStride;
  T items[kItemsPerThread];
  T threadSum = 0;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    items[i] = storage.transpose[first + i];
    threadSum += items[i];
  }
  __syncthreads();
  T offset =
      exclusiveSum<T, kBlockSize>(threadSum, &storage.total, storage.warpSums) +
      running;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    offset += items[i];
    storage.transpose[first + i] = offset;
  }
  running += storage.total;
}

// Same, but the run never leaves shared memory, so the kernel needs no
// itemsPerThread sized register array.
template <
    typename T,
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride,
    typename Storage>
__device__ inline void scanRunInShared(Storage& storage, T& running) {
  const int32_t first = static_cast<int32_t>(threadIdx.x) * kStride;
  T threadSum = 0;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    threadSum += storage.transpose[first + i];
  }
  __syncthreads();
  T offset =
      exclusiveSum<T, kBlockSize>(threadSum, &storage.total, storage.warpSums) +
      running;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    offset += storage.transpose[first + i];
    storage.transpose[first + i] = offset;
  }
  running += storage.total;
}

// Replaces the flags in this thread's run of the transpose buffer with the
// destination offset of the selected elements and -1 for the dropped ones.
template <
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride,
    typename Storage>
__device__ inline void rankRunInRegisters(
    Storage& storage,
    CountType& running) {
  const int32_t first = static_cast<int32_t>(threadIdx.x) * kStride;
  CountType items[kItemsPerThread];
  CountType threadCount = 0;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    items[i] = storage.transpose[first + i] != 0;
    threadCount += items[i];
  }
  __syncthreads();
  CountType offset = exclusiveSum<CountType, kBlockSize>(
                         threadCount, &storage.total, storage.warpSums) +
      running;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    storage.transpose[first + i] = items[i] != 0 ? offset : -1;
    offset += items[i];
  }
  running += storage.total;
}

template <
    int32_t kBlockSize,
    int32_t kItemsPerThread,
    int32_t kStride,
    typename Storage>
__device__ inline void rankRunInShared(Storage& storage, CountType& running) {
  const int32_t first = static_cast<int32_t>(threadIdx.x) * kStride;
  CountType threadCount = 0;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    threadCount += storage.transpose[first + i] != 0;
  }
  __syncthreads();
  CountType offset = exclusiveSum<CountType, kBlockSize>(
                         threadCount, &storage.total, storage.warpSums) +
      running;
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    const CountType flag = storage.transpose[first + i] != 0;
    storage.transpose[first + i] = flag != 0 ? offset : -1;
    offset += flag;
  }
  running += storage.total;
}

// Start and end of the partition of 'numElements' owned by this block. A
// partition is a whole number of tiles, so only the last tile of the last non
// empty partition can be ragged.
__device__ inline void partitionRange(
    int64_t numElements,
    int64_t tilesPerBlock,
    int32_t tileSize,
    int64_t& start,
    int64_t& end) {
  const int64_t partition = tilesPerBlock * tileSize;
  start = static_cast<int64_t>(blockIdx.x) * partition;
  end = start + partition;
  if (end > numElements) {
    end = numElements;
  }
}

// Exclusive scan of the per block totals into the per block seeds. Out of place
// so that repeating it for timing does not change its input.
template <typename T, int32_t kBlockSize>
__global__ void
exclusiveScanSeedsKernel(const T* totals, T* seeds, int32_t numSeeds) {
  __shared__ T warpSums[kBlockSize / kWarpThreads];
  __shared__ T total;
  T carry = 0;
  for (int32_t base = 0; base < numSeeds; base += kBlockSize) {
    const int32_t index = base + static_cast<int32_t>(threadIdx.x);
    const T value = index < numSeeds ? totals[index] : T(0);
    __syncthreads();
    const T offset = exclusiveSum<T, kBlockSize>(value, &total, warpSums);
    if (index < numSeeds) {
      seeds[index] = carry + offset;
    }
    carry += total;
  }
}

// ---------------------------------------------------------------------------
// Prefix sum over int64.
// ---------------------------------------------------------------------------

// Pass one. Read bound with an output of one element per block, so no
// transpose is needed. Wide loads are the whole win here; only the scalar
// layout leaves them on the table.
template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
__global__ void reduceTilesKernel(
    const ScanType* input,
    int64_t numElements,
    int64_t tilesPerBlock,
    ScanType* blockSums) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  __shared__ ScanType warpSums[kBlockSize / kWarpThreads];
  __shared__ ScanType total;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  ScanType threadSum = 0;
  for (int64_t base = start; base < end; base += kTileSize) {
    ScanType items[kItemsPerThread];
    const int64_t remaining = end - base;
    if (remaining < kTileSize) {
      loadBlockedPartial<ScanType, kBlockSize, kItemsPerThread>(
          input + base, static_cast<int32_t>(remaining), ScanType(0), items);
    } else if constexpr (kLayout == Layout::kBlockedScalar) {
      loadBlockedScalar<ScanType, kBlockSize, kItemsPerThread>(
          input + base, items);
    } else {
      loadBlockedVector<ScanType, kBlockSize, kItemsPerThread>(
          input + base, items);
    }
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      threadSum += items[i];
    }
  }
  __syncthreads();
  exclusiveSum<ScanType, kBlockSize>(threadSum, &total, warpSums);
  if (threadIdx.x == 0) {
    blockSums[blockIdx.x] = total;
  }
}

// Pass two. Write bound with an output of one element per input element, so
// this is where the layout matters.
template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
__global__ void scanTilesKernel(
    const ScanType* input,
    ScanType* output,
    int64_t numElements,
    int64_t tilesPerBlock,
    const ScanType* blockSeeds) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  constexpr int32_t kStride = kSharedStride<kLayout, kItemsPerThread>;
  __shared__
      TileStorage<ScanType, kBlockSize, kItemsPerThread, kStride, kLayout>
          storage;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  ScanType running = blockSeeds[blockIdx.x];
  for (int64_t base = start; base < end; base += kTileSize) {
    const int64_t remaining = end - base;
    const int32_t numValid =
        remaining >= kTileSize ? kTileSize : static_cast<int32_t>(remaining);
    if constexpr (kIsStriped<kLayout>) {
      if (numValid == kTileSize) {
        loadStripedFull<
            ScanType,
            ScanType,
            kBlockSize,
            kItemsPerThread,
            kStride>(input + base, storage.transpose);
      } else {
        loadStriped<ScanType, ScanType, kBlockSize, kItemsPerThread, kStride>(
            input + base, numValid, ScanType(0), storage.transpose);
      }
      __syncthreads();
      if constexpr (kScansInShared<kLayout>) {
        scanRunInShared<ScanType, kBlockSize, kItemsPerThread, kStride>(
            storage, running);
      } else {
        scanRunInRegisters<ScanType, kBlockSize, kItemsPerThread, kStride>(
            storage, running);
      }
      __syncthreads();
      if (numValid == kTileSize) {
        storeStripedFull<ScanType, kBlockSize, kItemsPerThread, kStride>(
            storage.transpose, output + base);
      } else {
        storeStriped<ScanType, kBlockSize, kItemsPerThread, kStride>(
            storage.transpose, numValid, output + base);
      }
    } else {
      ScanType items[kItemsPerThread];
      if (numValid < kTileSize) {
        loadBlockedPartial<ScanType, kBlockSize, kItemsPerThread>(
            input + base, numValid, ScanType(0), items);
      } else if constexpr (kLayout == Layout::kBlockedScalar) {
        loadBlockedScalar<ScanType, kBlockSize, kItemsPerThread>(
            input + base, items);
      } else {
        loadBlockedVector<ScanType, kBlockSize, kItemsPerThread>(
            input + base, items);
      }
      ScanType threadSum = 0;
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        threadSum += items[i];
      }
      __syncthreads();
      ScanType offset = exclusiveSum<ScanType, kBlockSize>(
                            threadSum, &storage.total, storage.warpSums) +
          running;
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        offset += items[i];
        items[i] = offset;
      }
      running += storage.total;
      if (numValid < kTileSize) {
        storeBlockedPartial<ScanType, kBlockSize, kItemsPerThread>(
            output + base, numValid, items);
      } else if constexpr (kLayout == Layout::kBlockedScalar) {
        storeBlockedScalar<ScanType, kBlockSize, kItemsPerThread>(
            output + base, items);
      } else {
        storeBlockedVector<ScanType, kBlockSize, kItemsPerThread>(
            output + base, items);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stream compaction. int64 values, one byte flags, int32 sum type.
// ---------------------------------------------------------------------------

template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
__global__ void countFlagsKernel(
    const uint8_t* flags,
    int64_t numElements,
    int64_t tilesPerBlock,
    CountType* blockCounts) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  __shared__ CountType warpSums[kBlockSize / kWarpThreads];
  __shared__ CountType total;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  CountType threadCount = 0;
  for (int64_t base = start; base < end; base += kTileSize) {
    uint8_t items[kItemsPerThread];
    const int64_t remaining = end - base;
    if (remaining < kTileSize) {
      loadBlockedPartial<uint8_t, kBlockSize, kItemsPerThread>(
          flags + base, static_cast<int32_t>(remaining), uint8_t(0), items);
    } else if constexpr (kLayout == Layout::kBlockedScalar) {
      loadBlockedScalar<uint8_t, kBlockSize, kItemsPerThread>(
          flags + base, items);
    } else {
      loadBlockedVector<uint8_t, kBlockSize, kItemsPerThread>(
          flags + base, items);
    }
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      threadCount += items[i] != 0;
    }
  }
  __syncthreads();
  exclusiveSum<CountType, kBlockSize>(threadCount, &total, warpSums);
  if (threadIdx.x == 0) {
    blockCounts[blockIdx.x] = total;
  }
}

template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
__global__ void compactKernel(
    const ScanType* values,
    const uint8_t* flags,
    int64_t numElements,
    int64_t tilesPerBlock,
    const CountType* blockSeeds,
    ScanType* output) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  constexpr int32_t kStride = kSharedStride<kLayout, kItemsPerThread>;
  __shared__
      TileStorage<CountType, kBlockSize, kItemsPerThread, kStride, kLayout>
          storage;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  CountType running = blockSeeds[blockIdx.x];
  const auto tid = static_cast<int32_t>(threadIdx.x);
  for (int64_t base = start; base < end; base += kTileSize) {
    const int64_t remaining = end - base;
    const int32_t numValid =
        remaining >= kTileSize ? kTileSize : static_cast<int32_t>(remaining);
    if constexpr (kIsStriped<kLayout>) {
      // The transpose buffer holds the flag, then the destination offset of a
      // selected element or -1 for a dropped one.
      if (numValid == kTileSize) {
        loadStripedFull<
            CountType,
            uint8_t,
            kBlockSize,
            kItemsPerThread,
            kStride>(flags + base, storage.transpose);
      } else {
        loadStriped<CountType, uint8_t, kBlockSize, kItemsPerThread, kStride>(
            flags + base, numValid, CountType(0), storage.transpose);
      }
      __syncthreads();
      if constexpr (kScansInShared<kLayout>) {
        rankRunInShared<kBlockSize, kItemsPerThread, kStride>(storage, running);
      } else {
        rankRunInRegisters<kBlockSize, kItemsPerThread, kStride>(
            storage, running);
      }
      __syncthreads();
      // Striped scatter. The value read is dense and, because the selected
      // elements of consecutive threads land at consecutive destinations, so
      // is the write.
#pragma unroll
      for (int32_t j = 0; j < kItemsPerThread; ++j) {
        const int32_t index = j * kBlockSize + tid;
        if (index < numValid) {
          const CountType destination =
              storage.transpose[sharedIndex<kItemsPerThread, kStride>(index)];
          if (destination >= 0) {
            output[destination] = values[base + index];
          }
        }
      }
      __syncthreads();
    } else {
      uint8_t flagItems[kItemsPerThread];
      ScanType valueItems[kItemsPerThread];
      if (numValid < kTileSize) {
        loadBlockedPartial<uint8_t, kBlockSize, kItemsPerThread>(
            flags + base, numValid, uint8_t(0), flagItems);
        loadBlockedPartial<ScanType, kBlockSize, kItemsPerThread>(
            values + base, numValid, ScanType(0), valueItems);
      } else if constexpr (kLayout == Layout::kBlockedScalar) {
        loadBlockedScalar<uint8_t, kBlockSize, kItemsPerThread>(
            flags + base, flagItems);
        loadBlockedScalar<ScanType, kBlockSize, kItemsPerThread>(
            values + base, valueItems);
      } else {
        loadBlockedVector<uint8_t, kBlockSize, kItemsPerThread>(
            flags + base, flagItems);
        loadBlockedVector<ScanType, kBlockSize, kItemsPerThread>(
            values + base, valueItems);
      }
      CountType threadCount = 0;
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        threadCount += flagItems[i] != 0;
      }
      __syncthreads();
      CountType offset = exclusiveSum<CountType, kBlockSize>(
                             threadCount, &storage.total, storage.warpSums) +
          running;
      // Padding in a ragged tile has a cleared flag, so it is never written.
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        if (flagItems[i] != 0) {
          output[offset] = valueItems[i];
          ++offset;
        }
      }
      running += storage.total;
    }
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(int64_t numElements) : numElements_(numElements) {
    CUDA_CHECK(cudaMalloc(&data_, numElements * sizeof(T)));
  }

  ~DeviceArray() {
    cudaFree(data_);
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  T* data() const {
    return data_;
  }

  void fromHost(const T* host) {
    CUDA_CHECK(cudaMemcpy(
        data_, host, numElements_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void toHost(T* host, int64_t numElements) const {
    CUDA_CHECK(cudaMemcpy(
        host, data_, numElements * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void fill(int32_t byte) {
    CUDA_CHECK(cudaMemset(data_, byte, numElements_ * sizeof(T)));
  }

 private:
  T* data_{nullptr};
  int64_t numElements_;
};

struct Timing {
  float min{0};
  float avg{0};
  float max{0};
};

// Runs 'body' once as a warmup and then FLAGS_scan_repeats times.
template <typename Body>
Timing timeKernel(Body body) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  body();
  CUDA_CHECK(cudaDeviceSynchronize());
  Timing timing;
  timing.min = std::numeric_limits<float>::max();
  double sum = 0;
  for (int32_t i = 0; i < FLAGS_scan_repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    body();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    timing.min = std::min(timing.min, milliseconds);
    timing.max = std::max(timing.max, milliseconds);
    sum += milliseconds;
  }
  timing.avg = static_cast<float>(sum / FLAGS_scan_repeats);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return timing;
}

int32_t numMultiProcessors() {
  int32_t device;
  CUDA_CHECK(cudaGetDevice(&device));
  int32_t value;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
  return value;
}

// Peak transfer rate from the memory clock and bus width. The practical
// ceiling is the device to device copy, which lands well below this.
double nominalGbPerSecond() {
  int32_t device;
  CUDA_CHECK(cudaGetDevice(&device));
  int32_t clockKhz;
  int32_t busBits;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&clockKhz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &busBits, cudaDevAttrGlobalMemoryBusWidth, device));
  return 2.0 * clockKhz * 1e3 * (busBits / 8.0) / 1e9;
}

struct KernelResources {
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
};

template <typename Kernel>
KernelResources kernelResources(Kernel kernel, int32_t blockSize) {
  KernelResources resources;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resources.blocksPerSm, kernel, blockSize, 0));
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  resources.numRegisters = attributes.numRegs;
  resources.sharedBytes = static_cast<int32_t>(attributes.sharedSizeBytes);
  return resources;
}

struct Measurement {
  std::string variant;
  int32_t blockSize{0};
  int32_t itemsPerThread{0};
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
  float firstPassMs{0};
  float seedMs{0};
  Timing secondPass;
  int64_t movedBytes{0};
  bool checked{true};
  bool correct{true};

  float totalMs() const {
    return firstPassMs + seedMs + secondPass.min;
  }

  double gbPerSecond() const {
    return static_cast<double>(movedBytes) / (totalMs() * 1e-3) / 1e9;
  }
};

std::string optional(int32_t value) {
  return value == 0 ? std::string("-") : std::to_string(value);
}

std::string optional(float milliseconds) {
  return milliseconds == 0 ? std::string("-")
                           : fmt::format("{:.3f}", milliseconds);
}

void printMeasurements(
    const std::string& firstPassName,
    const std::string& secondPassName,
    double floorGbPerSecond,
    std::vector<Measurement> measurements) {
  // Fastest first. The order the variants were run in carries no information.
  std::sort(
      measurements.begin(),
      measurements.end(),
      [](const Measurement& left, const Measurement& right) {
        return left.gbPerSecond() > right.gbPerSecond();
      });
  const double peak = nominalGbPerSecond();
  std::cout << fmt::format(
      "{:>5} {:>4} {:>4} {:>5} {:>5} {:>6} {:>8} {:>7} {:>9} {:>8} {:>8}"
      " {:>8} {:>7} {:>6} {:>6} {:>4}\n",
      "var",
      "tpb",
      "ipt",
      "bl/SM",
      "regs",
      "smem",
      firstPassName,
      "seed",
      secondPassName,
      "p2 avg",
      "p2 max",
      "total",
      "GB/s",
      "%peak",
      "%copy",
      "chk");
  for (const auto& measurement : measurements) {
    const double rate = measurement.gbPerSecond();
    std::cout << fmt::format(
        "{:>5} {:>4} {:>4} {:>5} {:>5} {:>6} {:>8} {:>7} {:>9.3f} {:>8.3f}"
        " {:>8.3f} {:>8.3f} {:>7.1f} {:>6.1f} {:>6.1f} {:>4}\n",
        measurement.variant,
        optional(measurement.blockSize),
        optional(measurement.itemsPerThread),
        optional(measurement.blocksPerSm),
        optional(measurement.numRegisters),
        optional(measurement.sharedBytes),
        optional(measurement.firstPassMs),
        optional(measurement.seedMs),
        measurement.secondPass.min,
        measurement.secondPass.avg,
        measurement.secondPass.max,
        measurement.totalMs(),
        rate,
        100.0 * rate / peak,
        100.0 * rate / floorGbPerSecond,
        !measurement.checked ? "-" : (measurement.correct ? "ok" : "FAIL"));
  }
}

struct ScanFixture {
  const ScanType* input;
  ScanType* output;
  int64_t numElements;
  int64_t movedBytes;
  const std::vector<ScanType>* expected;
  std::vector<ScanType>* result;
};

template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
Measurement runScanVariant(const ScanFixture& fixture) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  const int64_t numElements = fixture.numElements;
  const int64_t numTiles = (numElements + kTileSize - 1) / kTileSize;

  auto* scanKernel = scanTilesKernel<kBlockSize, kItemsPerThread, kLayout>;
  auto* reduceKernel = reduceTilesKernel<kBlockSize, kItemsPerThread, kLayout>;

  Measurement measurement;
  measurement.variant = layoutName(kLayout, kItemsPerThread);
  measurement.blockSize = kBlockSize;
  measurement.itemsPerThread = kItemsPerThread;
  measurement.movedBytes = fixture.movedBytes;
  const KernelResources resources = kernelResources(scanKernel, kBlockSize);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  const auto numBlocks = static_cast<int32_t>(std::min<int64_t>(
      static_cast<int64_t>(resources.blocksPerSm) * numMultiProcessors(),
      numTiles));
  const int64_t tilesPerBlock = (numTiles + numBlocks - 1) / numBlocks;

  DeviceArray<ScanType> blockSums(numBlocks);
  DeviceArray<ScanType> blockSeeds(numBlocks);

  auto reducePass = [&]() {
    reduceKernel<<<numBlocks, kBlockSize>>>(
        fixture.input, numElements, tilesPerBlock, blockSums.data());
  };
  auto seedPass = [&]() {
    exclusiveScanSeedsKernel<ScanType, kBlockSize>
        <<<1, kBlockSize>>>(blockSums.data(), blockSeeds.data(), numBlocks);
  };
  auto scanPass = [&]() {
    scanKernel<<<numBlocks, kBlockSize>>>(
        fixture.input,
        fixture.output,
        numElements,
        tilesPerBlock,
        blockSeeds.data());
  };

  measurement.checked = FLAGS_scan_check;
  if (FLAGS_scan_check) {
    CUDA_CHECK(
        cudaMemset(fixture.output, 0xff, numElements * sizeof(ScanType)));
    reducePass();
    seedPass();
    scanPass();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        fixture.result->data(),
        fixture.output,
        numElements * sizeof(ScanType),
        cudaMemcpyDeviceToHost));
    measurement.correct = std::equal(
        fixture.expected->begin(),
        fixture.expected->end(),
        fixture.result->begin());
  }

  measurement.firstPassMs = timeKernel(reducePass).min;
  measurement.seedMs = timeKernel(seedPass).min;
  measurement.secondPass = timeKernel(scanPass);
  CUDA_CHECK(cudaGetLastError());
  return measurement;
}

template <int32_t kBlockSize, int32_t kItemsPerThread>
void addScanLayouts(
    const ScanFixture& fixture,
    std::vector<Measurement>& measurements) {
  measurements.push_back(
      runScanVariant<kBlockSize, kItemsPerThread, Layout::kBlockedScalar>(
          fixture));
  measurements.push_back(
      runScanVariant<kBlockSize, kItemsPerThread, Layout::kBlockedVector>(
          fixture));
  measurements.push_back(
      runScanVariant<kBlockSize, kItemsPerThread, Layout::kStripedRegisters>(
          fixture));
  measurements.push_back(
      runScanVariant<kBlockSize, kItemsPerThread, Layout::kStripedShared>(
          fixture));
  measurements.push_back(
      runScanVariant<kBlockSize, kItemsPerThread, Layout::kStripedSharedPadded>(
          fixture));
}

template <int32_t kBlockSize>
void addScanBlockSize(
    const ScanFixture& fixture,
    std::vector<Measurement>& measurements) {
  measurements.push_back(
      runScanVariant<kBlockSize, 1, Layout::kBlockedVector>(fixture));
  addScanLayouts<kBlockSize, 2>(fixture, measurements);
  addScanLayouts<kBlockSize, 4>(fixture, measurements);
  addScanLayouts<kBlockSize, 8>(fixture, measurements);
}

// Device to device copy of the payload, the practical ceiling for anything
// that reads and writes the whole array once.
double copyFloorGbPerSecond(
    const ScanType* input,
    ScanType* output,
    int64_t numElements,
    int64_t movedBytes,
    std::vector<Measurement>& measurements) {
  const int64_t bytes = numElements * static_cast<int64_t>(sizeof(ScanType));
  const Timing timing = timeKernel([&]() {
    CUDA_CHECK(cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice));
  });
  Measurement measurement;
  measurement.variant = "copy";
  measurement.secondPass = timing;
  measurement.movedBytes = 2 * bytes;
  measurement.checked = false;
  measurements.push_back(measurement);
  return measurement.gbPerSecond();
}

void runScanBenchmark(int64_t numElements) {
  std::vector<ScanType> hostInput(numElements);
  std::mt19937_64 rng(1);
  for (int64_t i = 0; i < numElements; ++i) {
    // Small values keep the int64 sum exact and far from overflow.
    hostInput[i] = static_cast<ScanType>(rng() & 1023);
  }
  std::vector<ScanType> expected(numElements);
  ScanType running = 0;
  for (int64_t i = 0; i < numElements; ++i) {
    running += hostInput[i];
    expected[i] = running;
  }
  std::vector<ScanType> result(numElements);

  DeviceArray<ScanType> input(numElements);
  DeviceArray<ScanType> output(numElements);
  input.fromHost(hostInput.data());

  const int64_t movedBytes =
      2 * numElements * static_cast<int64_t>(sizeof(ScanType));
  const ScanFixture fixture{
      input.data(), output.data(), numElements, movedBytes, &expected, &result};

  std::vector<Measurement> measurements;
  addScanBlockSize<128>(fixture, measurements);
  addScanBlockSize<256>(fixture, measurements);

  {
    size_t tempBytes = 0;
    CUDA_CHECK(
        cub::DeviceScan::InclusiveSum(
            nullptr,
            tempBytes,
            input.data(),
            output.data(),
            static_cast<int32_t>(numElements)));
    DeviceArray<char> temp(static_cast<int64_t>(tempBytes));
    auto deviceScan = [&]() {
      CUDA_CHECK(
          cub::DeviceScan::InclusiveSum(
              temp.data(),
              tempBytes,
              input.data(),
              output.data(),
              static_cast<int32_t>(numElements)));
    };
    Measurement measurement;
    measurement.variant = "cub";
    measurement.movedBytes = movedBytes;
    measurement.checked = FLAGS_scan_check;
    if (FLAGS_scan_check) {
      output.fill(0xff);
      deviceScan();
      CUDA_CHECK(cudaDeviceSynchronize());
      output.toHost(result.data(), numElements);
      measurement.correct =
          std::equal(expected.begin(), expected.end(), result.begin());
    }
    measurement.secondPass = timeKernel(deviceScan);
    measurements.push_back(measurement);
  }

  const double floor = copyFloorGbPerSecond(
      input.data(), output.data(), numElements, movedBytes, measurements);

  // The two pass rows read the input twice, so they can reach at most two
  // thirds of the copy floor no matter how good the second pass is.
  std::cout << fmt::format(
      "\nInclusive prefix sum of {} int64 ({:.1f} MB). Rates are over the "
      "{:.1f} MB read plus written; peak {:.0f} GB/s, copy floor {:.0f} GB/s, "
      "two pass ceiling {:.0f}% of the floor.\n",
      numElements,
      numElements * sizeof(ScanType) / static_cast<double>(1 << 20),
      movedBytes / static_cast<double>(1 << 20),
      nominalGbPerSecond(),
      floor,
      100.0 * 2 / 3);
  printMeasurements("reduce", "scan", floor, measurements);
}

struct CompactFixture {
  const ScanType* values;
  const uint8_t* flags;
  ScanType* output;
  int64_t numElements;
  int64_t movedBytes;
  const std::vector<ScanType>* expected;
  std::vector<ScanType>* result;
};

template <int32_t kBlockSize, int32_t kItemsPerThread, Layout kLayout>
Measurement runCompactVariant(const CompactFixture& fixture) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  const int64_t numElements = fixture.numElements;
  const int64_t numTiles = (numElements + kTileSize - 1) / kTileSize;

  auto* kernel = compactKernel<kBlockSize, kItemsPerThread, kLayout>;
  auto* countKernel = countFlagsKernel<kBlockSize, kItemsPerThread, kLayout>;

  Measurement measurement;
  measurement.variant = layoutName(kLayout, kItemsPerThread);
  measurement.blockSize = kBlockSize;
  measurement.itemsPerThread = kItemsPerThread;
  measurement.movedBytes = fixture.movedBytes;
  const KernelResources resources = kernelResources(kernel, kBlockSize);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  const auto numBlocks = static_cast<int32_t>(std::min<int64_t>(
      static_cast<int64_t>(resources.blocksPerSm) * numMultiProcessors(),
      numTiles));
  const int64_t tilesPerBlock = (numTiles + numBlocks - 1) / numBlocks;

  DeviceArray<CountType> blockCounts(numBlocks);
  DeviceArray<CountType> blockSeeds(numBlocks);

  auto countPass = [&]() {
    countKernel<<<numBlocks, kBlockSize>>>(
        fixture.flags, numElements, tilesPerBlock, blockCounts.data());
  };
  auto seedPass = [&]() {
    exclusiveScanSeedsKernel<CountType, kBlockSize>
        <<<1, kBlockSize>>>(blockCounts.data(), blockSeeds.data(), numBlocks);
  };
  auto compactPass = [&]() {
    kernel<<<numBlocks, kBlockSize>>>(
        fixture.values,
        fixture.flags,
        numElements,
        tilesPerBlock,
        blockSeeds.data(),
        fixture.output);
  };

  measurement.checked = FLAGS_scan_check;
  if (FLAGS_scan_check) {
    CUDA_CHECK(cudaMemset(
        fixture.output, 0xff, fixture.expected->size() * sizeof(ScanType)));
    countPass();
    seedPass();
    compactPass();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        fixture.result->data(),
        fixture.output,
        fixture.expected->size() * sizeof(ScanType),
        cudaMemcpyDeviceToHost));
    measurement.correct = std::equal(
        fixture.expected->begin(),
        fixture.expected->end(),
        fixture.result->begin());
  }

  measurement.firstPassMs = timeKernel(countPass).min;
  measurement.seedMs = timeKernel(seedPass).min;
  measurement.secondPass = timeKernel(compactPass);
  CUDA_CHECK(cudaGetLastError());
  return measurement;
}

template <int32_t kBlockSize, int32_t kItemsPerThread>
void addCompactLayouts(
    const CompactFixture& fixture,
    std::vector<Measurement>& measurements) {
  measurements.push_back(
      runCompactVariant<kBlockSize, kItemsPerThread, Layout::kBlockedScalar>(
          fixture));
  measurements.push_back(
      runCompactVariant<kBlockSize, kItemsPerThread, Layout::kBlockedVector>(
          fixture));
  measurements.push_back(
      runCompactVariant<kBlockSize, kItemsPerThread, Layout::kStripedRegisters>(
          fixture));
  measurements.push_back(
      runCompactVariant<kBlockSize, kItemsPerThread, Layout::kStripedShared>(
          fixture));
  measurements.push_back(
      runCompactVariant<
          kBlockSize,
          kItemsPerThread,
          Layout::kStripedSharedPadded>(fixture));
}

template <int32_t kBlockSize>
void addCompactBlockSize(
    const CompactFixture& fixture,
    std::vector<Measurement>& measurements) {
  measurements.push_back(
      runCompactVariant<kBlockSize, 1, Layout::kBlockedVector>(fixture));
  addCompactLayouts<kBlockSize, 2>(fixture, measurements);
  addCompactLayouts<kBlockSize, 4>(fixture, measurements);
  addCompactLayouts<kBlockSize, 8>(fixture, measurements);
}

void runCompactionBenchmark(int64_t numElements) {
  std::vector<ScanType> hostValues(numElements);
  std::vector<uint8_t> hostFlags(numElements);
  std::mt19937_64 rng(2);
  const auto threshold = static_cast<uint64_t>(
      FLAGS_scan_selectivity *
      static_cast<double>(std::numeric_limits<uint64_t>::max()));
  for (int64_t i = 0; i < numElements; ++i) {
    hostValues[i] = static_cast<ScanType>(i);
    hostFlags[i] = rng() < threshold;
  }
  std::vector<ScanType> expected;
  expected.reserve(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    if (hostFlags[i]) {
      expected.push_back(hostValues[i]);
    }
  }
  std::vector<ScanType> result(expected.size());

  DeviceArray<ScanType> values(numElements);
  DeviceArray<uint8_t> flags(numElements);
  DeviceArray<ScanType> output(numElements);
  values.fromHost(hostValues.data());
  flags.fromHost(hostFlags.data());

  // Flags read once, values read once, selected values written once.
  const int64_t movedBytes = numElements +
      numElements * static_cast<int64_t>(sizeof(ScanType)) +
      static_cast<int64_t>(expected.size() * sizeof(ScanType));
  const CompactFixture fixture{
      values.data(),
      flags.data(),
      output.data(),
      numElements,
      movedBytes,
      &expected,
      &result};

  std::vector<Measurement> measurements;
  addCompactBlockSize<128>(fixture, measurements);
  addCompactBlockSize<256>(fixture, measurements);

  {
    DeviceArray<CountType> numSelected(1);
    size_t tempBytes = 0;
    CUDA_CHECK(
        cub::DeviceSelect::Flagged(
            nullptr,
            tempBytes,
            values.data(),
            flags.data(),
            output.data(),
            numSelected.data(),
            static_cast<int32_t>(numElements)));
    DeviceArray<char> temp(static_cast<int64_t>(tempBytes));
    auto deviceSelect = [&]() {
      CUDA_CHECK(
          cub::DeviceSelect::Flagged(
              temp.data(),
              tempBytes,
              values.data(),
              flags.data(),
              output.data(),
              numSelected.data(),
              static_cast<int32_t>(numElements)));
    };
    Measurement measurement;
    measurement.variant = "cub";
    measurement.movedBytes = movedBytes;
    measurement.checked = FLAGS_scan_check;
    if (FLAGS_scan_check) {
      output.fill(0xff);
      deviceSelect();
      CUDA_CHECK(cudaDeviceSynchronize());
      output.toHost(result.data(), expected.size());
      CountType hostNumSelected = 0;
      numSelected.toHost(&hostNumSelected, 1);
      measurement.correct =
          hostNumSelected == static_cast<CountType>(expected.size()) &&
          std::equal(expected.begin(), expected.end(), result.begin());
    }
    measurement.secondPass = timeKernel(deviceSelect);
    measurements.push_back(measurement);
  }

  const double floor = copyFloorGbPerSecond(
      values.data(), output.data(), numElements, movedBytes, measurements);

  // The count pass rereads the flags, which is the only traffic the rates do
  // not account for.
  const double twoPassCeiling = 100.0 * static_cast<double>(movedBytes) /
      static_cast<double>(movedBytes + numElements);
  std::cout << fmt::format(
      "\nStream compaction of {} int64 ({:.1f} MB) with one byte flags, "
      "{} selected ({:.1f}%). Rates are over the {:.1f} MB read plus "
      "written; peak {:.0f} GB/s, copy floor {:.0f} GB/s, two pass ceiling "
      "{:.0f}% of the floor.\n",
      numElements,
      numElements * sizeof(ScanType) / static_cast<double>(1 << 20),
      expected.size(),
      100.0 * expected.size() / numElements,
      movedBytes / static_cast<double>(1 << 20),
      nominalGbPerSecond(),
      floor,
      twoPassCeiling);
  printMeasurements("count", "compact", floor, measurements);
}

} // namespace
} // namespace facebook::velox::wave

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  using namespace facebook::velox::wave;

  int32_t device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << fmt::format(
      "{} sm_{}{} with {} SMs\n",
      properties.name,
      properties.major,
      properties.minor,
      properties.multiProcessorCount);

  const int64_t numElements = FLAGS_scan_bytes / sizeof(ScanType);
  if (numElements <= 0) {
    std::cout << "--scan_bytes must cover at least one int64\n";
    return 1;
  }
  runScanBenchmark(numElements);
  runCompactionBenchmark(numElements);
  return 0;
}

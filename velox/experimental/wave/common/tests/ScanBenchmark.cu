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

/// Measures the Wave block scan primitives (Scan.cuh inclusiveSum and
/// exclusiveSum) against cub::BlockScan at 1, 2, 4 and 8 elements per thread.
///
/// Two workloads are measured:
///  - A full array inclusive prefix sum over int64 data.
///  - A stream compaction where a value is written at the offset given by the
///    exclusive prefix sum of a one byte flag, with an int32 sum type.
///
/// Both use the same "reduce then scan" structure so that only the block level
/// primitive and the number of elements per thread differ between rows:
///  1. Each block reduces the contiguous partition of the input it owns.
///  2. One block exclusive scans the per block totals into per block seeds.
///  3. Each block scans its partition again, seeded with its own total.
/// The grid is sized for maximum occupancy of the scanning kernel, so a block
/// walks over many tiles of its partition rather than owning a single tile.
/// cub::DeviceScan and cub::DeviceSelect are reported as reference points.

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
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

DEFINE_int32(
    scan_repeats,
    5,
    "Number of timed repetitions. The shortest time is reported.");

DEFINE_bool(
    scan_check,
    true,
    "Verify every variant against a CPU reference before timing it.");

namespace facebook::velox::wave {
namespace {

constexpr int32_t kBlockSize = 256;
constexpr int32_t kNumWarpsPerBlock = kBlockSize / kWarpThreads;

// Widest vector type that loads 'kBytesPerThread' bytes in a single access.
// Specializations cover the sizes below 16 bytes, larger sizes use several
// 16 byte accesses.
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

// Loads kItemsPerThread consecutive elements per thread from a full tile. The
// tile is laid out blocked, so thread t owns elements [t * kItemsPerThread,
// (t + 1) * kItemsPerThread). The accesses are vectorized so that the whole
// tile is read with fully coalesced transactions.
template <typename T, int32_t kItemsPerThread>
__device__ inline void loadBlocked(const T* tile, T (&items)[kItemsPerThread]) {
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

template <typename T, int32_t kItemsPerThread>
__device__ inline void storeBlocked(
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

// Loads the last, partial tile. Elements at or after 'numValid' get 'fill',
// which is the identity of the sum, so the scan of a padded tile is the scan
// of the real elements.
template <typename T, int32_t kItemsPerThread>
__device__ inline void loadPartial(
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

template <typename T, int32_t kItemsPerThread>
__device__ inline void
storePartial(T* tile, int32_t numValid, const T (&items)[kItemsPerThread]) {
#pragma unroll
  for (int32_t i = 0; i < kItemsPerThread; ++i) {
    const int32_t index =
        static_cast<int32_t>(threadIdx.x) * kItemsPerThread + i;
    if (index < numValid) {
      tile[index] = items[i];
    }
  }
}

// Block wide scan of kItemsPerThread items per thread on top of the Wave
// primitives in Scan.cuh. For more than one item per thread this is the
// standard three step form: sum the thread's items, scan the per thread totals
// with a single call to the primitive, then expand serially in registers.
//
// Successive calls sharing the same TempStorage must be separated by a
// __syncthreads(): the primitives read their scratch after their last internal
// barrier, so a fast warp entering the next call would otherwise overwrite it.
template <typename T, int32_t kItemsPerThread>
struct WaveBlockScan {
  static constexpr const char* kName = "wave";

  struct TempStorage {
    T warpSums[kNumWarpsPerBlock];
    T total;
  };

  // Replaces 'items' with their block wide inclusive sum. Returns the sum of
  // all items in the block.
  static __device__ T
  inclusiveSumItems(T (&items)[kItemsPerThread], TempStorage& temp) {
    if constexpr (kItemsPerThread == 1) {
      items[0] =
          inclusiveSum<T, kBlockSize>(items[0], &temp.total, temp.warpSums);
    } else {
      T threadSum = 0;
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        threadSum += items[i];
      }
      T offset =
          exclusiveSum<T, kBlockSize>(threadSum, &temp.total, temp.warpSums);
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        offset += items[i];
        items[i] = offset;
      }
    }
    return temp.total;
  }

  // Replaces 'items' with their block wide exclusive sum. Returns the sum of
  // all items in the block.
  static __device__ T
  exclusiveSumItems(T (&items)[kItemsPerThread], TempStorage& temp) {
    if constexpr (kItemsPerThread == 1) {
      items[0] =
          exclusiveSum<T, kBlockSize>(items[0], &temp.total, temp.warpSums);
    } else {
      T threadSum = 0;
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        threadSum += items[i];
      }
      T offset =
          exclusiveSum<T, kBlockSize>(threadSum, &temp.total, temp.warpSums);
#pragma unroll
      for (int32_t i = 0; i < kItemsPerThread; ++i) {
        const T item = items[i];
        items[i] = offset;
        offset += item;
      }
    }
    return temp.total;
  }
};

// Same interface implemented with cub::BlockScan, which handles the multiple
// items per thread case natively.
template <
    typename T,
    int32_t kItemsPerThread,
    cub::BlockScanAlgorithm kAlgorithm>
struct CubBlockScan {
  using Scan = cub::BlockScan<T, kBlockSize, kAlgorithm>;
  using TempStorage = typename Scan::TempStorage;

  static __device__ T
  inclusiveSumItems(T (&items)[kItemsPerThread], TempStorage& temp) {
    T total;
    Scan(temp).InclusiveSum(items, items, total);
    return total;
  }

  static __device__ T
  exclusiveSumItems(T (&items)[kItemsPerThread], TempStorage& temp) {
    T total;
    Scan(temp).ExclusiveSum(items, items, total);
    return total;
  }
};

// cub's default. Rakes over a shared memory grid, which costs more scratch
// than the Wave primitives use.
template <typename T, int32_t kItemsPerThread>
struct CubRakingScan
    : CubBlockScan<T, kItemsPerThread, cub::BLOCK_SCAN_RAKING> {
  static constexpr const char* kName = "cub-rk";
};

// Structurally the same shuffle based algorithm as the Wave primitives, so
// this is the like for like comparison.
template <typename T, int32_t kItemsPerThread>
struct CubWarpScan
    : CubBlockScan<T, kItemsPerThread, cub::BLOCK_SCAN_WARP_SCANS> {
  static constexpr const char* kName = "cub-ws";
};

// Start and end of the partition of 'numElements' owned by this block. The
// partition is a whole number of tiles, so only the very last tile of the last
// non-empty partition can be partial.
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

// Exclusive scan of the per block totals into the per block seeds. Runs in a
// single block and is a negligible part of the total time, but is required for
// the two pass structure. Out of place so that repeating it for timing does
// not change its input.
template <typename T>
__global__ void
exclusiveScanSeedsKernel(const T* totals, T* seeds, int32_t numSeeds) {
  __shared__ T warpSums[kNumWarpsPerBlock];
  __shared__ T total;
  T carry = 0;
  for (int32_t base = 0; base < numSeeds; base += kBlockSize) {
    const auto index = base + threadIdx.x;
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

using ScanType = int64_t;

template <int32_t kItemsPerThread>
__global__ void reduceTilesKernel(
    const ScanType* input,
    int64_t numElements,
    int64_t tilesPerBlock,
    ScanType* blockSums) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  using Reduce = cub::BlockReduce<ScanType, kBlockSize>;
  __shared__ typename Reduce::TempStorage temp;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  ScanType threadSum = 0;
  for (int64_t base = start; base < end; base += kTileSize) {
    ScanType items[kItemsPerThread];
    const int64_t remaining = end - base;
    if (remaining >= kTileSize) {
      loadBlocked(input + base, items);
    } else {
      loadPartial(
          input + base, static_cast<int32_t>(remaining), ScanType(0), items);
    }
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      threadSum += items[i];
    }
  }
  const ScanType total = Reduce(temp).Sum(threadSum);
  if (threadIdx.x == 0) {
    blockSums[blockIdx.x] = total;
  }
}

template <typename Policy, int32_t kItemsPerThread>
__global__ void scanTilesKernel(
    const ScanType* input,
    ScanType* output,
    int64_t numElements,
    int64_t tilesPerBlock,
    const ScanType* blockSeeds) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  __shared__ typename Policy::TempStorage temp;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  ScanType running = blockSeeds[blockIdx.x];
  for (int64_t base = start; base < end; base += kTileSize) {
    ScanType items[kItemsPerThread];
    const int64_t remaining = end - base;
    const int32_t numValid =
        remaining >= kTileSize ? kTileSize : static_cast<int32_t>(remaining);
    if (numValid == kTileSize) {
      loadBlocked(input + base, items);
    } else {
      loadPartial(input + base, numValid, ScanType(0), items);
    }
    __syncthreads();
    const ScanType total = Policy::inclusiveSumItems(items, temp);
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      items[i] += running;
    }
    running += total;
    if (numValid == kTileSize) {
      storeBlocked(output + base, items);
    } else {
      storePartial(output + base, numValid, items);
    }
  }
}

// ---------------------------------------------------------------------------
// Stream compaction. int64 values, one byte flags, int32 sum type.
// ---------------------------------------------------------------------------

using CountType = int32_t;

template <int32_t kItemsPerThread>
__global__ void countFlagsKernel(
    const uint8_t* flags,
    int64_t numElements,
    int64_t tilesPerBlock,
    CountType* blockCounts) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  using Reduce = cub::BlockReduce<CountType, kBlockSize>;
  __shared__ typename Reduce::TempStorage temp;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  CountType threadCount = 0;
  for (int64_t base = start; base < end; base += kTileSize) {
    uint8_t items[kItemsPerThread];
    const int64_t remaining = end - base;
    if (remaining >= kTileSize) {
      loadBlocked(flags + base, items);
    } else {
      loadPartial(
          flags + base, static_cast<int32_t>(remaining), uint8_t(0), items);
    }
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      threadCount += items[i] != 0;
    }
  }
  const CountType total = Reduce(temp).Sum(threadCount);
  if (threadIdx.x == 0) {
    blockCounts[blockIdx.x] = total;
  }
}

template <typename Policy, int32_t kItemsPerThread>
__global__ void compactKernel(
    const ScanType* values,
    const uint8_t* flags,
    int64_t numElements,
    int64_t tilesPerBlock,
    const CountType* blockSeeds,
    ScanType* output) {
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  __shared__ typename Policy::TempStorage temp;
  int64_t start;
  int64_t end;
  partitionRange(numElements, tilesPerBlock, kTileSize, start, end);
  CountType running = blockSeeds[blockIdx.x];
  for (int64_t base = start; base < end; base += kTileSize) {
    uint8_t flagItems[kItemsPerThread];
    ScanType valueItems[kItemsPerThread];
    const int64_t remaining = end - base;
    if (remaining >= kTileSize) {
      loadBlocked(flags + base, flagItems);
      loadBlocked(values + base, valueItems);
    } else {
      const auto numValid = static_cast<int32_t>(remaining);
      loadPartial(flags + base, numValid, uint8_t(0), flagItems);
      loadPartial(values + base, numValid, ScanType(0), valueItems);
    }
    CountType offsets[kItemsPerThread];
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      offsets[i] = flagItems[i] != 0;
    }
    __syncthreads();
    const CountType total = Policy::exclusiveSumItems(offsets, temp);
    // Padding in a partial tile has a cleared flag, so it is never written.
#pragma unroll
    for (int32_t i = 0; i < kItemsPerThread; ++i) {
      if (flagItems[i] != 0) {
        output[running + offsets[i]] = valueItems[i];
      }
    }
    running += total;
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

// Runs 'body' once as a warmup and then FLAGS_scan_repeats times, returning
// the shortest elapsed time in milliseconds.
template <typename Body>
float timeBest(Body body) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  body();
  CUDA_CHECK(cudaDeviceSynchronize());
  float best = std::numeric_limits<float>::max();
  for (int32_t i = 0; i < FLAGS_scan_repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    body();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    best = std::min(best, milliseconds);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return best;
}

int32_t numMultiProcessors() {
  int32_t device;
  CUDA_CHECK(cudaGetDevice(&device));
  int32_t value;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
  return value;
}

struct KernelResources {
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
};

// Occupancy and per thread resource use of 'kernel' at kBlockSize threads.
template <typename Kernel>
KernelResources kernelResources(Kernel kernel) {
  KernelResources resources;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resources.blocksPerSm, kernel, kBlockSize, 0));
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  resources.numRegisters = attributes.numRegs;
  resources.sharedBytes = static_cast<int32_t>(attributes.sharedSizeBytes);
  return resources;
}

struct Measurement {
  std::string primitive;
  // 0 for the whole array cub reference, which does not expose this.
  int32_t itemsPerThread{0};
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
  int32_t numBlocks{0};
  float firstPassMs{0};
  float seedMs{0};
  float secondPassMs{0};
  bool correct{true};

  float totalMs() const {
    return firstPassMs + seedMs + secondPassMs;
  }
};

std::string optional(int32_t value) {
  return value == 0 ? std::string("-") : std::to_string(value);
}

std::string optional(float milliseconds) {
  return milliseconds == 0 ? std::string("-")
                           : fmt::format("{:.3f}", milliseconds);
}

// 'movedBytes' is what a single pass implementation must read and write. It is
// also exactly what the second pass of the two pass variants moves, so the two
// rates are, respectively, the rate of the whole operation and the rate of the
// kernel that contains the scan.
void printMeasurements(
    const std::string& firstPassName,
    const std::string& secondPassName,
    int64_t movedBytes,
    std::vector<Measurement> measurements) {
  // Fastest first. The order the variants were run in carries no information.
  std::sort(
      measurements.begin(),
      measurements.end(),
      [](const Measurement& left, const Measurement& right) {
        return left.totalMs() < right.totalMs();
      });
  std::cout << fmt::format(
      "{:>7} {:>6} {:>7} {:>7} {:>5} {:>5} {:>10} {:>8} {:>11} {:>9} {:>9} "
      "{:>9} {:>6}\n",
      "prim",
      "items",
      "blocks",
      "per SM",
      "regs",
      "smem",
      firstPassName,
      "seed ms",
      secondPassName,
      "total ms",
      "pass GB/s",
      "all GB/s",
      "check");
  for (const auto& measurement : measurements) {
    const double bytes = static_cast<double>(movedBytes);
    std::cout << fmt::format(
        "{:>7} {:>6} {:>7} {:>7} {:>5} {:>5} {:>10} {:>8} {:>11} {:>9.3f} "
        "{:>9.1f} {:>9.1f} {:>6}\n",
        measurement.primitive,
        optional(measurement.itemsPerThread),
        optional(measurement.numBlocks),
        optional(measurement.blocksPerSm),
        optional(measurement.numRegisters),
        optional(measurement.sharedBytes),
        optional(measurement.firstPassMs),
        optional(measurement.seedMs),
        optional(measurement.secondPassMs),
        measurement.totalMs(),
        bytes / (measurement.secondPassMs * 1e-3) / 1e9,
        bytes / (measurement.totalMs() * 1e-3) / 1e9,
        measurement.correct ? "ok" : "FAIL");
  }
}

template <template <typename, int32_t> class Policy, int32_t kItemsPerThread>
Measurement runScanVariant(
    const ScanType* input,
    ScanType* output,
    int64_t numElements,
    const std::vector<ScanType>& expected,
    std::vector<ScanType>& result) {
  using ScanPolicy = Policy<ScanType, kItemsPerThread>;
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  const int64_t numTiles = (numElements + kTileSize - 1) / kTileSize;

  auto* scanKernel = scanTilesKernel<ScanPolicy, kItemsPerThread>;
  auto* reduceKernel = reduceTilesKernel<kItemsPerThread>;

  Measurement measurement;
  measurement.primitive = ScanPolicy::kName;
  measurement.itemsPerThread = kItemsPerThread;
  const KernelResources resources = kernelResources(scanKernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = static_cast<int32_t>(std::min<int64_t>(
      static_cast<int64_t>(measurement.blocksPerSm) * numMultiProcessors(),
      numTiles));
  const int32_t numBlocks = measurement.numBlocks;
  const int64_t tilesPerBlock = (numTiles + numBlocks - 1) / numBlocks;

  DeviceArray<ScanType> blockSums(numBlocks);
  DeviceArray<ScanType> blockSeeds(numBlocks);

  auto reducePass = [&]() {
    reduceKernel<<<numBlocks, kBlockSize>>>(
        input, numElements, tilesPerBlock, blockSums.data());
  };
  auto seedPass = [&]() {
    exclusiveScanSeedsKernel<ScanType>
        <<<1, kBlockSize>>>(blockSums.data(), blockSeeds.data(), numBlocks);
  };
  auto scanPass = [&]() {
    scanKernel<<<numBlocks, kBlockSize>>>(
        input, output, numElements, tilesPerBlock, blockSeeds.data());
  };

  if (FLAGS_scan_check) {
    CUDA_CHECK(cudaMemset(output, 0xff, numElements * sizeof(ScanType)));
    reducePass();
    seedPass();
    scanPass();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        output,
        numElements * sizeof(ScanType),
        cudaMemcpyDeviceToHost));
    measurement.correct =
        std::equal(expected.begin(), expected.end(), result.begin());
  }

  measurement.firstPassMs = timeBest(reducePass);
  measurement.seedMs = timeBest(seedPass);
  measurement.secondPassMs = timeBest(scanPass);
  CUDA_CHECK(cudaGetLastError());
  return measurement;
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

  std::vector<Measurement> measurements;
  measurements.push_back(
      runScanVariant<WaveBlockScan, 1>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<WaveBlockScan, 2>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<WaveBlockScan, 4>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<WaveBlockScan, 8>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubWarpScan, 1>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubWarpScan, 2>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubWarpScan, 4>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubWarpScan, 8>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubRakingScan, 1>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubRakingScan, 2>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubRakingScan, 4>(
          input.data(), output.data(), numElements, expected, result));
  measurements.push_back(
      runScanVariant<CubRakingScan, 8>(
          input.data(), output.data(), numElements, expected, result));

  // cub::DeviceScan is a single pass decoupled look-back scan, so it reads the
  // input once where the two pass variants above read it twice.
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
    measurement.primitive = "device";
    if (FLAGS_scan_check) {
      output.fill(0xff);
      deviceScan();
      CUDA_CHECK(cudaDeviceSynchronize());
      output.toHost(result.data(), numElements);
      measurement.correct =
          std::equal(expected.begin(), expected.end(), result.begin());
    }
    measurement.secondPassMs = timeBest(deviceScan);
    measurements.push_back(measurement);
  }

  const int64_t movedBytes =
      2 * numElements * static_cast<int64_t>(sizeof(ScanType));
  std::cout << fmt::format(
      "\nInclusive prefix sum of {} int64 ({:.1f} MB). "
      "Rates are over the {:.1f} MB read plus written.\n",
      numElements,
      numElements * sizeof(ScanType) / static_cast<double>(1 << 20),
      movedBytes / static_cast<double>(1 << 20));
  printMeasurements("reduce ms", "scan ms", movedBytes, measurements);
}

template <template <typename, int32_t> class Policy, int32_t kItemsPerThread>
Measurement runCompactVariant(
    const ScanType* values,
    const uint8_t* flags,
    ScanType* output,
    int64_t numElements,
    const std::vector<ScanType>& expected,
    std::vector<ScanType>& result) {
  using ScanPolicy = Policy<CountType, kItemsPerThread>;
  constexpr int32_t kTileSize = kBlockSize * kItemsPerThread;
  const int64_t numTiles = (numElements + kTileSize - 1) / kTileSize;

  auto* kernel = compactKernel<ScanPolicy, kItemsPerThread>;
  auto* countKernel = countFlagsKernel<kItemsPerThread>;

  Measurement measurement;
  measurement.primitive = ScanPolicy::kName;
  measurement.itemsPerThread = kItemsPerThread;
  const KernelResources resources = kernelResources(kernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = static_cast<int32_t>(std::min<int64_t>(
      static_cast<int64_t>(measurement.blocksPerSm) * numMultiProcessors(),
      numTiles));
  const int32_t numBlocks = measurement.numBlocks;
  const int64_t tilesPerBlock = (numTiles + numBlocks - 1) / numBlocks;

  DeviceArray<CountType> blockCounts(numBlocks);
  DeviceArray<CountType> blockSeeds(numBlocks);

  auto countPass = [&]() {
    countKernel<<<numBlocks, kBlockSize>>>(
        flags, numElements, tilesPerBlock, blockCounts.data());
  };
  auto seedPass = [&]() {
    exclusiveScanSeedsKernel<CountType>
        <<<1, kBlockSize>>>(blockCounts.data(), blockSeeds.data(), numBlocks);
  };
  auto compactPass = [&]() {
    kernel<<<numBlocks, kBlockSize>>>(
        values, flags, numElements, tilesPerBlock, blockSeeds.data(), output);
  };

  if (FLAGS_scan_check) {
    CUDA_CHECK(cudaMemset(output, 0xff, expected.size() * sizeof(ScanType)));
    countPass();
    seedPass();
    compactPass();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        output,
        expected.size() * sizeof(ScanType),
        cudaMemcpyDeviceToHost));
    measurement.correct =
        std::equal(expected.begin(), expected.end(), result.begin());
  }

  measurement.firstPassMs = timeBest(countPass);
  measurement.seedMs = timeBest(seedPass);
  measurement.secondPassMs = timeBest(compactPass);
  CUDA_CHECK(cudaGetLastError());
  return measurement;
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

  std::vector<Measurement> measurements;
  measurements.push_back(
      runCompactVariant<WaveBlockScan, 1>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<WaveBlockScan, 2>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<WaveBlockScan, 4>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<WaveBlockScan, 8>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubWarpScan, 1>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubWarpScan, 2>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubWarpScan, 4>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubWarpScan, 8>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubRakingScan, 1>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubRakingScan, 2>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubRakingScan, 4>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));
  measurements.push_back(
      runCompactVariant<CubRakingScan, 8>(
          values.data(),
          flags.data(),
          output.data(),
          numElements,
          expected,
          result));

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
    measurement.primitive = "device";
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
    measurement.secondPassMs = timeBest(deviceSelect);
    measurements.push_back(measurement);
  }

  // Flags read once, values read once, selected values written once.
  const int64_t movedBytes = numElements +
      numElements * static_cast<int64_t>(sizeof(ScanType)) +
      static_cast<int64_t>(expected.size() * sizeof(ScanType));
  std::cout << fmt::format(
      "\nStream compaction of {} int64 ({:.1f} MB) with one byte flags, "
      "{} selected ({:.1f}%). Rates are over the {:.1f} MB read plus "
      "written.\n",
      numElements,
      numElements * sizeof(ScanType) / static_cast<double>(1 << 20),
      expected.size(),
      100.0 * expected.size() / numElements,
      movedBytes / static_cast<double>(1 << 20));
  printMeasurements("count ms", "compact ms", movedBytes, measurements);
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
      "{} sm_{}{} with {} SMs, block size {}\n",
      properties.name,
      properties.major,
      properties.minor,
      properties.multiProcessorCount,
      kBlockSize);

  const int64_t numElements = FLAGS_scan_bytes / sizeof(ScanType);
  if (numElements <= 0) {
    std::cout << "--scan_bytes must cover at least one int64\n";
    return 1;
  }
  runScanBenchmark(numElements);
  runCompactionBenchmark(numElements);
  return 0;
}

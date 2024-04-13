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

#include "velox/experimental/wave/common/Block.cuh"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/tests/CudaTest.h"

namespace facebook::velox::wave {

__global__ void
addOneKernel(int32_t* numbers, int32_t size, int32_t stride, int32_t repeats) {
  for (auto counter = 0; counter < repeats; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
         index += stride) {
      ++numbers[index];
    }
    __syncthreads();
  }
}

__global__ void
addOneSharedKernel(int32_t* numbers, int32_t size, int32_t stride, int32_t repeats) {
  extern __shared__ __align__(16) char smem[];
  int32_t* temp = reinterpret_cast<int32_t*>(smem);
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
       index += stride) {
    temp[threadIdx.x] = numbers[blockDim.x * blockIdx.x + threadIdx.x];
    for (auto counter = 0; counter < repeats; ++counter) {
      ++temp[index];
    }
    __syncthreads();
    numbers[blockDim.x * blockIdx.x + threadIdx.x] = temp[threadIdx.x];
  }
}

  
void TestStream::addOne(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void addOneWideKernel(WideParams params) {
  auto numbers = params.numbers;
  auto size = params.size;
  auto repeat = params.repeat;
  auto stride = params.stride;
  for (auto counter = 0; counter < repeat; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
         index += stride) {
      ++numbers[index];
    }
  }
}

void TestStream::addOneWide(
    int32_t* numbers,
    int32_t size,
    int32_t repeat,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  WideParams params;
  params.numbers = numbers;
  params.size = size;
  params.stride = stride;
  params.repeat = repeat;
  addOneWideKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

__device__ uint32_t scale32(uint32_t n, uint32_t scale) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(n)) * scale) >> 32;
}

__global__ void __launch_bounds__(1024) addOneRandomKernel(
    int32_t* numbers,
    const int32_t* lookup,
    uint32_t size,
    int32_t stride,
    int32_t repeats,
    bool emptyWarps,
    bool emptyThreads) {
  for (uint32_t counter = 0; counter < repeats; ++counter) {
    if (emptyWarps) {
      if (((threadIdx.x / 32) & 1) == 0) {
        for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
             index += stride) {
          auto rnd = scale32(index * (counter + 1) * kPrime32, size);
          numbers[index] += lookup[rnd];
          rnd = scale32((index + 32) * (counter + 1) * kPrime32, size);
          numbers[index + 32] += lookup[rnd];
        }
      }
    } else if (emptyThreads) {
      if ((threadIdx.x & 1) == 0) {
        for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
             index += stride) {
          auto rnd = scale32(index * (counter + 1) * kPrime32, size);
          numbers[index] += lookup[rnd];
          rnd = scale32((index + 1) * (counter + 1) * kPrime32, size);
          numbers[index + 1] += lookup[rnd];
        }
      }
    } else {
#pragma unroll
      for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
           index += stride) {
        auto rnd = scale32(index * (counter + 1) * kPrime32, size);
        numbers[index] += lookup[rnd];
      }
    }
    __syncthreads();
  }
  __syncthreads();
}

void TestStream::addOneRandom(
    int32_t* numbers,
    const int32_t* lookup,
    int32_t size,
    int32_t repeats,
    int32_t width,
    bool emptyWarps,
    bool emptyThreads) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneRandomKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, lookup, size, stride, repeats, emptyWarps, emptyThreads);
  CUDA_CHECK(cudaGetLastError());
}

__device__ inline uint64_t hashMix(const uint64_t upper, const uint64_t lower) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (lower ^ upper) * kMul;
  a ^= (a >> 47);
  uint64_t b = (upper ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}
void __global__ __launch_bounds__(1024) makeInputKernel(
    int32_t numRows,
    int32_t keyRange,
    int32_t powerOfTwo,
    int32_t startCount,
    uint64_t* hashes,
    uint8_t numColumns,
    int64_t** columns) {
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  if (powerOfTwo == 0) {
    columns[0][idx] = 0;
    return;
  }
  auto delta = startCount & (powerOfTwo - 1);
  auto previous = columns[0][idx];
  auto key = scale32((previous + delta + idx) * kPrime32, keyRange);
  columns[0][idx] = key;
  hashes[idx] = hashMix(1, key);
  for (auto i = 1; i < numColumns; ++i) {
    columns[i][idx] = i + (idx & 7);
  }
  __syncthreads();
  return;
}

void TestStream::makeInput(
    int32_t numRows,
    int32_t keyRange,
    int32_t powerOfTwo,
    int32_t startCount,
    uint64_t* hash,
    uint8_t numColumns,
    int64_t** columns) {
  auto numBlocks = roundUp(numRows, 256) / 256;
  makeInputKernel<<<256, numBlocks, 0, stream_->stream>>>(
      numRows, keyRange, powerOfTwo, startCount, hash, numColumns, columns);
  CUDA_CHECK(cudaGetLastError());
}

void __device__
updateAggs(int64_t* entry, uint16_t row, uint8_t numColumns, int64_t** args) {
  if (!entry) {
    *(long*)0 = 0;
  }
  if (false && entry[0] != args[0][row]) {
    *(long*)0 = 0;
		    }
for (auto i = 1; i < numColumns; ++i) {
    entry[i] += args[i][row];
  }
}

void __global__ __launch_bounds__(1024) partition8KKernel(
    int32_t numRows,
    uint8_t shift,
int64_t* keys,
    uint64_t* hash,
    uint16_t* partitions,
    uint16_t* rows) {
  auto base = blockIdx.x * 8192;
  auto end = base + 8192 < numRows ? base + 8192 : numRows;
  for (auto stride = 0; stride < 8192; stride += 1024) {
    auto idx = base + stride + threadIdx.x;
    if (idx < end) {
      rows[idx] = idx - base;
      partitions[idx] = (hash[idx] >> shift);
    } else {
      rows[idx] = 0xffff;
      partitions[idx] = 0xffff;
    }
  }
  __syncthreads();
  extern __shared__ __align__(16) char smem[];
  blockSort<1024, 8>(
      [&](auto i) { return partitions[base + i]; },
      [&](auto i) { return rows[base + i]; },
      partitions + base,
      rows + base,
      smem);
  __syncthreads();
}

int32_t TestStream::sort8KTempSize() {
  return blockSortSharedSize<1024, 8, uint16_t, uint16_t>();
}

void TestStream::partition8K(
    int32_t numRows,
    uint8_t shift,
int64_t* keys,
    uint64_t* hashes,
    uint16_t* partitions,
    uint16_t* rows) {
  auto tempBytes = sort8KTempSize();
  int32_t num8KBlocks = roundUp(numRows, 8192) / 8192;
  partition8KKernel<<<num8KBlocks, 1024, tempBytes, stream_->stream>>>(
      numRows, shift, keys, hashes, partitions, rows);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
template <typename T>
__device__ int findFirst(const T* data, int size, T target) {
  int lo = 0, hi = size;
  while (lo < hi) {
    int i = (lo + hi) / 2;
    if (data[i] < target) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return lo;
}

template <typename T>
__device__ int findLast(const T* data, int size, T target) {
  int lo = 0, hi = size;
  while (lo < hi) {
    int i = (lo + hi) / 2;
    if (data[i] <= target) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return lo;
}
} // namespace

void __device__ markSkew(MockProbe* probe, int32_t start, int32_t numRepeats) {}

/// Updates partitions of 'table' The input is divided into batches
/// of 8K entries, where the last can be under 8K, for a total of
/// 'numRows'. blockDim.x * gridDim.x must be 8192. Each TB takes
/// its fraction of a 64K range of partitions in 'partitions'. Each
/// 8K slice of 'partitions' is sorted. Each TB takes its range of
/// 64K, so if there are 8 blocks of 1024, the first handles
/// partitions [0, 8191], the second [8192, 16385] and so on. Each
/// TB begins by finding the index of the first and last item in
/// 'partitions' that falls in its range. Like this, the TBs are
/// guaranteed disjoint domains of the table. Each TB loops until
/// all input is processed. the TBs have a stride of 8192. So after
/// the first TB has done the first range of the first 8K, it looks
/// at the bounds of its range from the second 8K range of
/// 'partitions', until there is no next 8K piece in 'partitions'.
void __global__ __launch_bounds__(1024) update8KKernel(
    int32_t numRows,
    uint64_t* hash,
    uint16_t* partitions,
    uint16_t* rows,
    int64_t** args,
    MockProbe* probe,
    MockStatus* status,
    MockTable* table) {
  constexpr int32_t kBlockSize = 256;
  auto end = roundUp(numRows, 8192);
  for (auto batchStart = 0; batchStart < end; batchStart += 8192) {
    if (threadIdx.x == 0) {
      int32_t blockRangeSize = 0x10000 / gridDim.x;
      uint16_t firstPartition = blockRangeSize * blockIdx.x;
      uint16_t lastPartition = (blockIdx.x + 1) * blockRangeSize - 1;
      auto batchSize =
          batchStart + 8192 < numRows ? 8192 : numRows - batchStart;

      auto idx = findFirst(partitions + batchStart, batchSize, firstPartition);
      if (partitions[batchStart + idx] < firstPartition) {
        ++idx;
      }
      probe->begin[blockIdx.x] = idx;
      probe->end[blockIdx.x] =
          findLast(partitions + batchStart, batchSize, lastPartition);
    }
    probe->failFill[blockDim.x] = 0;
    __syncthreads();
      extern __shared__ __align__(16) char smem[];
      int32_t* starts = reinterpret_cast<int32_t*>(smem);

    int32_t partitionBegin = probe->begin[blockIdx.x] + batchStart;
    int32_t partitionEnd = probe->end[blockIdx.x] + batchStart;
    int64_t* keys = args[0];
    for (int32_t counter = partitionBegin; counter <= partitionEnd;
         counter += blockDim.x) {
      auto idx = counter + threadIdx.x;
      bool isLeader = false;
      probe->isOverflow[blockIdx.x * blockDim.x + threadIdx.x] = false;
      int32_t start;
      int32_t row;
      int32_t part;
      if (idx < partitionEnd) {
        // Indirection to access the keys, hashes, args.
        row = batchStart + rows[idx];
	if (row < batchStart || row > batchStart + 9191) {
	  *(long*)0 =0;
	}
        part = partitions[idx];
        isLeader = idx == counter || partitions[idx - 1] != part;
        bool hit = false;
        start = hash[row] & table->sizeMask;
	if (start == 0) {
	  *(long*)0 = 0;
	}
        int32_t nextPartition =
            (start & table->partitionMask) + table->partitionSize;
        auto firstProbe = start;
        for (;;) {
          auto entry = table->rows[start];
          if (!entry) {
            // The test is supposed to only look for existing keys.
            *(long*)0 = 0; // crash.
            break;
          }
          if (keys[row] == entry[0]) {
            hit = true;
            break;
          }
          start = start + 1;
          if (start >= nextPartition) {
            // Wrap around to the beginning of the partition. Mark as overflow
            // after exhausting the partition.
            start = firstProbe & table->partitionMask;
          }
          if (start == firstProbe) {
            probe->isOverflow[blockIdx.x + blockIdx.x * blockDim.x] = true;
            isLeader = false;
            break;
          }
        }

	if (start == 0) { *(long*)0 = 0; }
        probe->start[blockIdx.x * blockDim.x + threadIdx.x] = start;
	starts[threadIdx.x] = start;
        probe->isHit[blockIdx.x * blockDim.x + threadIdx.x] = hit;
      }
      __syncthreads();
      __threadfence();
      auto endThreadIdx = min(blockDim.x, partitionEnd - counter);
      if (threadIdx.x + 2< endThreadIdx) {
	if (        probe->start[blockIdx.x * blockDim.x + threadIdx.x] != start) {
	  *(long*)0 = 0;
	}
	if (starts[threadIdx.x + 2] != probe->start[blockDim.x * blockIdx.x + threadIdx.x + 2]) {
	  *(long*)0 = 0;
	}
      }
      if (isLeader) {
        auto idx = counter + threadIdx.x;
        auto endThreadIdx = min(blockDim.x, partitionEnd - counter);
        int32_t sameCnt = 0;
        // Leader updates all in the partition. Break at end of
        // partition or when next item has different partition. If the
        // same row repeats over kSkewMinRepeats, mark the skew.
        constexpr int kSkewMinRepeats = 5;
        int32_t nthUpdate = 0;

        for (;;) {
          updateAggs(table->rows[start], row, table->numColumns, args);
          ++nthUpdate;
          if (threadIdx.x + nthUpdate >= endThreadIdx ||
              partitions[idx + 1] != part) {
            break;
          }
          ++idx;
          auto newStart = starts[threadIdx.x + nthUpdate];
	  if (newStart != probe->start[blockIdx.x * blockDim.x + threadIdx.x + nthUpdate]) {
	    *(long*)0 = 0;
	  }
          if (newStart == start) {
            ++sameCnt;
          } else {
            if (sameCnt > kSkewMinRepeats) {
              markSkew(probe, start, sameCnt);
            }
            sameCnt = 0;
            start = newStart;
          }
          row = batchStart + rows[idx];
        }
        if (sameCnt > kSkewMinRepeats) {
          markSkew(probe, start, sameCnt);
        }
      }

      __syncthreads();
#if 0
      // We record the failed row, partition pairs.
      uint16_t failFill = probe->failFill[blockIdx.x];
      extern __shared__ __align__(16) char smem[];

      boolBlockToIndices<kBlockSize>(
          [&]() { return probe->isOverflow[threadIdx.x]; },
          failFill,
          &probe->failIdx[blockIdx.x * blockDim.x + failFill],
          smem,
          probe->failFill[blockIdx.x]);
      uint16_t newFailed = probe->failFill[blockIdx.x] - failFill;
      uint16_t rowTemp;
      uint16_t partTemp;
      if (threadIdx.x < newFailed) {
        auto source = batchStart + probe->begin[blockIdx.x] +
            probe->failIdx[blockDim.x * blockIdx.x + threadIdx.x + failFill];
        rowTemp = rows[source];
        partTemp = partitions[source];
      }
      __syncthreads();
      if (threadIdx.x < newFailed) {
	*(long*)0 = 0;
        int32_t destIdx =
            batchStart + probe->begin[blockIdx.x] + failFill + threadIdx.x;
        rows[destIdx] = rowTemp;
        partitions[destIdx] = partTemp;
      }
#endif
    }
#if 0
    // The partition is processed for one TB in one 8K batch.
    if (threadIdx.x == 0) {
      auto* tbStatus =
          status + blockIdx.x + (batchStart / 8192) * (8192 / kBlockSize);
      tbStatus->beginIn8K = probe->begin[blockIdx.x];
      tbStatus->endIn8K = probe->end[blockIdx.x];
      tbStatus->numFailed = probe->failFill[blockIdx.x];
      tbStatus->lastConsumed = 0;
    }
#endif
  }
  __syncthreads();
}

void TestStream::update8K(
    int32_t numRows,
    uint64_t* hash,
    uint16_t* partitions,
    uint16_t* rowNumbers,
    int64_t** args,
    MockProbe* probe,
    MockStatus* status,
    MockTable* table) {
  auto num8KBlocks = roundUp(numRows, 8192) / 8192;
  update8KKernel<<<
      8192 / kGroupBlockSize,
      kGroupBlockSize,
	1200, //boolToIndicesSharedSize<uint16_t, kGroupBlockSize>(),
      stream_->stream>>>(
      numRows, hash, partitions, rowNumbers, args, probe, status, table);
  CUDA_CHECK(cudaGetLastError());
}

REGISTER_KERNEL("addOne", addOneKernel);
REGISTER_KERNEL("addOneWide", addOneWideKernel);
REGISTER_KERNEL("addOneRandom", addOneRandomKernel);
REGISTER_KERNEL("makeInput", makeInputKernel);
REGISTER_KERNEL("partition8K", partition8KKernel);
REGISTER_KERNEL("update8K", update8KKernel);

} // namespace facebook::velox::wave

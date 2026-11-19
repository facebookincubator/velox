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

#include "velox/experimental/torchwave/DeviceGrid.h"

#include <fmt/format.h>
#include <folly/synchronization/CallOnce.h>
#include <glog/logging.h>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include "velox/experimental/torchwave/DeviceSymbols.h"
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

namespace {

// The kernel reads and writes these by offset.
static_assert(offsetof(BlockInfo, op) == 0);
static_assert(offsetof(BlockInfo, blockInOp) == 4);
static_assert(offsetof(BlockInfo, numBlocksInOp) == 8);
static_assert(offsetof(BlockInfo, params) == 16);
static_assert(offsetof(BlockInfo, debugInfo) == 24);
static_assert(offsetof(BlockInfo, start) == 32);
static_assert(offsetof(BlockInfo, barrierClocks) == 40);
static_assert(sizeof(BlockInfo) == 48);
static_assert(sizeof(DebugInfo) % 8 == 0);
static_assert(offsetof(Tensor, rank) == 8);
static_assert(offsetof(Tensor, dims) == 12);
static_assert(offsetof(Tensor, numEl) == 44);
static_assert(kMaxDims == 4);
static_assert(kDebugNoOp == -1);

constexpr int32_t kHeaderWords = 8;
constexpr int32_t kLaunchWords = 12;
constexpr int32_t kConstWords = 5;

// The kernel's thread count; also bounds nothing, the loops stride by it.
// kWarps warps in the kernel.
constexpr int32_t kGridThreads = 512;
// Threads of the kernel for steps of at most 32 launches: all load and
// write the block table, the first warp computes the grid.
constexpr int32_t kWarpGridThreads = 512;
// Shared memory per launch: clocks, count, cost, cap, blocks, first block,
// and the multiplier, cost, blocks and cap of the previous execution.
constexpr int32_t kSharedPerLaunch = 48;
// Blocks of the previous execution each thread reads.
constexpr int32_t kPrevPerThread = 4;
// Below the 48 KB a kernel gets without opting in, less the static shared of
// the grid and the fixups.
constexpr int32_t kMaxSharedBytes = 30 * 1024;
// Ints before the per-launch ints: the previous block count, the grid total,
// and spare words.
constexpr int32_t kStateWords = 16;

const char* const kGridSource = R"(
constexpr int kHeaderWords = 8;
constexpr int kLaunchWords = 12;
constexpr int kConstWords = 5;
constexpr int kBlockInfoBytes = 48;
constexpr int kWarps = 16;
constexpr int kRounds = 4;
constexpr int kStateWords = 16;
constexpr int kPrevPerThread = 4;
// Dynamic shared memory of a setup kernel: 48 KB less the static shared of
// the fixups and the grid.
constexpr int kMaxSetupShared = 30 * 1024;

// With TW_SETUP_CLOCKS, state words 2-8 get the cycles from the start of the
// kernel to each phase of the setup.
#ifdef TW_SETUP_CLOCKS
#define SETUP_START()                                                   \
  if (threadIdx.x == 0) {                                               \
    setupStart = clock64();                                             \
  }
#define SETUP_MARK(ints, k)                                             \
  if (threadIdx.x == 0) {                                               \
    (ints)[2 + (k)] = static_cast<int>(clock64() - setupStart);         \
  }
#else
#define SETUP_START()
#define SETUP_MARK(ints, k)
#endif
constexpr unsigned kFull = 0xffffffffu;

// The element count of the launch described at 'l', from its parameters.
__device__ long long launchCount(
    const int* desc, const int* l, const unsigned char* base) {
  const int kind = l[6];
  const int termBegin = l[7];
  const int numTerms = l[8];
  const int constBegin = l[9];
  const int numConsts = l[10];
  if (numTerms == 0 && numConsts == 0) {
    return 1;
  }
  const int* terms = desc + desc[4];
  const int* consts = desc + desc[5];
  long long count = 0;
  int dims[4] = {1, 1, 1, 1};
  int rank = 0;
  for (int t = 0; t < numTerms; ++t) {
    const unsigned char* term = base + terms[2 * (termBegin + t)];
    if (terms[2 * (termBegin + t) + 1]) {
      const long long value = *reinterpret_cast<const long long*>(term);
      count = value > count ? value : count;
      continue;
    }
    const int termRank = *reinterpret_cast<const signed char*>(term + 8);
    const int* termDims = reinterpret_cast<const int*>(term + 12);
    // Tensor::numEl is a uint32_t, as on the host.
    const long long numEl = *reinterpret_cast<const unsigned*>(term + 44);
    if (kind == 2) {
      count += numEl;
    } else if (kind == 1) {
      count = numEl > count ? numEl : count;
    } else {
      rank = termRank > rank ? termRank : rank;
      for (int d = 0; d < termRank; ++d) {
        int& dim = dims[4 - termRank + d];
        dim = termDims[d] > dim ? termDims[d] : dim;
      }
    }
  }
  if (kind == 3) {
    for (int c = 0; c < numConsts; ++c) {
      const int* shape = consts + kConstWords * (constBegin + c);
      const int constRank = shape[0];
      rank = constRank > rank ? constRank : rank;
      for (int d = 0; d < constRank; ++d) {
        int& dim = dims[4 - constRank + d];
        dim = shape[1 + d] > dim ? shape[1 + d] : dim;
      }
    }
    count = 1;
    for (int d = 4 - rank; d < 4; ++d) {
      count *= dims[d];
    }
  }
  return count;
}

// Blocks for 'cost' at completion time 1 / 'inverse', within [1, cap]. The
// host's assignBlocksEqualFinish computes the same.
__device__ int blocksAt(float cost, double inverse, int cap) {
  double wanted = inverse > 0 ? ceil(cost * inverse) : 1.0;
  if (wanted < 1) {
    wanted = 1;
  }
  return wanted > cap ? cap : static_cast<int>(wanted);
}

// Sum over the block; every thread gets the total.
__device__ double blockSum(double value, double* scratch) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFull, value, offset);
  }
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    scratch[threadIdx.x / 32] = value;
  }
  __syncthreads();
  double total = 0;
  for (int w = 0; w < kWarps; ++w) {
    total += scratch[w];
  }
  return total;
}

__device__ double blockMax(double value, double* scratch) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmax(value, __shfl_xor_sync(kFull, value, offset));
  }
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    scratch[threadIdx.x / 32] = value;
  }
  __syncthreads();
  double result = scratch[0];
  for (int w = 1; w < kWarps; ++w) {
    result = fmax(result, scratch[w]);
  }
  return result;
}

// Exclusive prefix sum over the block's threads; 'total' gets the sum.
__device__ int blockScan(int value, int* scratch, int& total) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  int inclusive = value;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(kFull, inclusive, offset);
    if (lane >= offset) {
      inclusive += other;
    }
  }
  __syncthreads();
  if (lane == 31) {
    scratch[warp] = inclusive;
  }
  __syncthreads();
  int before = 0;
  total = 0;
  for (int w = 0; w < kWarps; ++w) {
    before += w < warp ? scratch[w] : 0;
    total += scratch[w];
  }
  return before + inclusive - value;
}

// One block of kWarps warps. Shared memory holds per launch: clocks, count,
// cost, cap, blocks, first block, and the previous execution's multiplier,
// cost, blocks and cap; then a copy of the descriptor's 'numWords'.
// 'stepClocks' and 'stepWaits' are the step's per-block clocks and barrier
// clocks in the execution's clock buffer, for its 'launchBlocks' blocks. With
// 'previousClocks' they hold the previous execution's clocks of the step.
// 'maxPrevBlocks' bounds the previous execution's block count.
// Called by every thread of the block.
__device__ void gridBody(
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut) {
  float* multiplier = floats;
  float* prevCost = floats + n;
  int* state = ints;
  int* prevBlocks = ints + kStateWords;
  int* prevCap = prevBlocks + n;
  int* launchOf = prevCap + n;
  extern __shared__ long long shared[];
  long long* clocks = shared;
  long long* counts = clocks + n;
  float* cost = reinterpret_cast<float*>(counts + n);
  int* cap = reinterpret_cast<int*>(cost + n);
  int* blocks = cap + n;
  int* first = blocks + n;
  float* lastMultiplier = reinterpret_cast<float*>(first + n);
  float* lastCost = lastMultiplier + n;
  int* lastBlocks = reinterpret_cast<int*>(lastCost + n);
  int* lastCap = lastBlocks + n;
  int* desc = lastCap + n;
  __shared__ double scratch[kWarps];
  __shared__ int intScratch[kWarps];
  __shared__ int feasible[kWarps];
  __shared__ double times[kWarps];
  unsigned char* blockInfos = base + blockInfoOffset;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int numThreads = blockDim.x;

  // Every load that does not depend on another goes first, so the kernel
  // waits for memory about once. The host's empty flags are read before the
  // block table overwrites them. The previous execution's blocks are loaded
  // ahead of the test against its block count.
  for (int w = tid; w < numWords; w += numThreads) {
    desc[w] = deviceDesc[w];
  }
  for (int i = tid; i < n; i += numThreads) {
    first[i] = blockInfos[i];
    clocks[i] = 0;
    lastMultiplier[i] = multiplier[i];
    lastCost[i] = prevCost[i];
    lastBlocks[i] = prevBlocks[i];
    lastCap[i] = prevCap[i];
  }
  const int prevNum = state[0];
  int blockLaunch[kPrevPerThread];
  long long blockClocks[kPrevPerThread];
  long long blockWait[kPrevPerThread];
#pragma unroll
  for (int k = 0; k < kPrevPerThread; ++k) {
    const int b = k * numThreads + tid;
    blockLaunch[k] = -1;
    blockClocks[k] = 0;
    blockWait[k] = 0;
    if (b < maxPrevBlocks) {
      blockLaunch[k] = launchOf[b];
      if (previousClocks && b < launchBlocks) {
        blockClocks[k] = stepClocks[b];
        blockWait[k] = stepWaits[b];
      }
    }
  }
  __syncthreads();
  const int target = desc[1] < launchBlocks ? desc[1] : launchBlocks;
  const int blockSize = desc[2];
  const int countWait = desc[6];

  // Each launch's share of the previous execution's clocks against its share
  // of the estimated cost corrects its multiplier. A launch's blocks are
  // consecutive, so each warp sums the runs of one launch among its blocks
  // and adds each run once.
#pragma unroll
  for (int k = 0; k < kPrevPerThread; ++k) {
    const int b = k * numThreads + tid;
    const int launch = b < prevNum ? blockLaunch[k] : -1;
    long long sum = blockClocks[k] - (countWait ? 0 : blockWait[k]);
    sum = launch >= 0 && sum > 0 ? sum : 0;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const long long other = __shfl_up_sync(kFull, sum, offset);
      const int otherLaunch = __shfl_up_sync(kFull, launch, offset);
      if (lane >= offset && otherLaunch == launch) {
        sum += other;
      }
    }
    const int nextLaunch = __shfl_down_sync(kFull, launch, 1);
    if (launch >= 0 && (lane == 31 || nextLaunch != launch) && sum > 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(&clocks[launch]),
          static_cast<unsigned long long>(sum));
    }
  }
  __syncthreads();
  double localClocks = 0;
  double localCost = 0;
  for (int i = tid; i < n; i += numThreads) {
    localClocks += clocks[i];
    localCost += lastCost[i];
  }
  const double totalClocks = blockSum(localClocks, scratch);
  const double totalPrevCost = blockSum(localCost, scratch);
  const bool measured = prevNum > 0 && totalClocks > 0 && totalPrevCost > 0;

  // Counts, multipliers, estimated costs and block caps.
  double localMax = 0;
  for (int i = tid; i < n; i += numThreads) {
    const int* l = desc + kHeaderWords + kLaunchWords * i;
    float m = lastMultiplier[i];
    if (measured && lastCost[i] > 0 && lastBlocks[i] < lastCap[i]) {
      const double ratio =
          (clocks[i] / totalClocks) / (lastCost[i] / totalPrevCost);
      if (ratio > 1.1 || ratio < 1 / 1.1) {
        m *= static_cast<float>(ratio);
        multiplier[i] = m;
      }
    }
    const long long count = first[i] ? 0 : launchCount(desc, l, base);
    counts[i] = count;
    cost[i] = static_cast<float>(count) * __int_as_float(l[2]) * m;
    long long launchCap = l[5] ? 1 : (count + l[3] - 1) / l[3];
    if (launchCap > l[4]) {
      launchCap = l[4];
    }
    cap[i] = launchCap < 1 ? 1 : static_cast<int>(launchCap);
    localMax = fmax(localMax, static_cast<double>(cost[i]));
  }
  const double maxCost = blockMax(localMax, scratch);

  // The smallest common completion time T whose blocks fit the target,
  // searched in log T: each round, warp k tries the k-th of kWarps points
  // between a time known not to fit and one known to fit.
  const bool search = n < target && maxCost > 0;
  double fits = maxCost;
  if (search) {
    double low = log(maxCost / target);
    double high = log(maxCost);
    for (int round = 0; round < kRounds; ++round) {
      const double point = low + (high - low) * warp / kWarps;
      const double time = exp(point);
      long long sum = 0;
      for (int i = lane; i < n; i += 32) {
        sum += blocksAt(cost[i], 1.0 / time, cap[i]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(kFull, sum, offset);
      }
      if (lane == 0) {
        feasible[warp] = sum <= target;
        times[warp] = time;
      }
      __syncthreads();
      int smallest = kWarps;
      for (int w = kWarps - 1; w >= 0; --w) {
        smallest = feasible[w] ? w : smallest;
      }
      const double step = (high - low) / kWarps;
      const double oldLow = low;
      if (smallest < kWarps) {
        fits = times[smallest];
      }
      __syncthreads();
      if (smallest == 0) {
        break;
      }
      if (smallest < kWarps) {
        high = oldLow + step * smallest;
        low = oldLow + step * (smallest - 1);
      } else {
        low = oldLow + step * (kWarps - 1);
      }
    }
  }
  for (int i = tid; i < n; i += numThreads) {
    int assigned = search ? blocksAt(cost[i], 1.0 / fits, cap[i]) : 1;
    // All but the last block process a multiple of blockSize elements.
    if (assigned > 1) {
      const long long perBlock = (counts[i] + assigned - 1) / assigned;
      const long long aligned =
          (perBlock + blockSize - 1) / blockSize * blockSize;
      const long long fewer = (counts[i] + aligned - 1) / aligned;
      assigned = fewer < 1 ? 1 : static_cast<int>(fewer);
    }
    blocks[i] = assigned;
  }
  __syncthreads();

  // First block of each launch: each thread scans a run of launches.
  const int run = (n + numThreads - 1) / numThreads;
  const int runBegin = tid * run;
  const int runEnd = runBegin + run < n ? runBegin + run : n;
  int runSum = 0;
  for (int i = runBegin; i < runEnd; ++i) {
    runSum += blocks[i];
  }
  int sum = 0;
  int offset = blockScan(runSum, intScratch, sum);
  for (int i = runBegin; i < runEnd; ++i) {
    first[i] = offset;
    offset += blocks[i];
  }
  __syncthreads();

  // The block table, with blocks past the grid doing nothing.
  const int total = sum < launchBlocks ? sum : launchBlocks;
  for (int b = tid; b < launchBlocks; b += numThreads) {
    unsigned char* info =
        blockInfos + static_cast<long long>(b) * kBlockInfoBytes;
    unsigned char* blockDebug = debug + static_cast<long long>(b) * debugBytes;
    int launch = -1;
    if (b < total) {
      int low = 0;
      int high = n - 1;
      while (low < high) {
        const int middle = (low + high + 1) / 2;
        if (first[middle] <= b) {
          low = middle;
        } else {
          high = middle - 1;
        }
      }
      launch = low;
      const int* l = desc + kHeaderWords + kLaunchWords * launch;
      *reinterpret_cast<int*>(info) = l[0];
      *reinterpret_cast<int*>(info + 4) = b - first[launch];
      *reinterpret_cast<int*>(info + 8) = blocks[launch];
      *reinterpret_cast<unsigned char**>(info + 16) = base + l[1];
    } else {
      *reinterpret_cast<int*>(info) = -1;
      *reinterpret_cast<int*>(info + 4) = 0;
      *reinterpret_cast<int*>(info + 8) = 1;
      *reinterpret_cast<unsigned char**>(info + 16) = nullptr;
      // A block that does nothing leaves its DebugInfo and clocks as it
      // finds them. The host reads every launched block's DebugInfo for errors
      // and its clocks for the step's statistics.
      for (int w = 0; w < debugBytes / 8; ++w) {
        reinterpret_cast<long long*>(blockDebug)[w] = 0;
      }
      stepClocks[b] = 0;
      stepWaits[b] = 0;
    }
    *reinterpret_cast<unsigned char**>(info + 24) = blockDebug;
    *reinterpret_cast<long long*>(info + 32) = 0;
    *reinterpret_cast<long long*>(info + 40) = 0;
    launchOf[b] = launch;
  }
  for (int i = tid; i < n; i += numThreads) {
    prevCost[i] = cost[i];
    prevBlocks[i] = blocks[i];
    prevCap[i] = cap[i];
    countsOut[i] = counts[i];
  }
  if (tid == 0) {
    state[0] = total;
    state[1] = sum;
  }
}

extern "C" __global__ void deviceGrid(
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut) {
  SETUP_START();
  gridBody(
      deviceDesc,
      numWords,
      n,
      maxPrevBlocks,
      base,
      blockInfoOffset,
      launchBlocks,
      debug,
      debugBytes,
      stepClocks,
      stepWaits,
      previousClocks,
      floats,
      ints,
      countsOut);
}

// The step's symbol fixups, then its grid from the fixed-up parameters.
extern "C" __global__ void stepSetup(
    const int* records,
    const long long* code,
    int numRecords,
    const long long* symbols,
    int numSymbols,
    int numCode,
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut) {
  SETUP_START();
  extern __shared__ long long setupShared[];
  fixupBody(
      base,
      records,
      code,
      numRecords,
      symbols,
      numSymbols,
      numCode <= kSharedCode ? setupShared : nullptr,
      numCode,
      0,
      blockDim.x,
      1);
  __syncthreads();
  gridBody(
      deviceDesc,
      numWords,
      n,
      maxPrevBlocks,
      base,
      blockInfoOffset,
      launchBlocks,
      debug,
      debugBytes,
      stepClocks,
      stepWaits,
      previousClocks,
      floats,
      ints,
      countsOut);
}
__device__ double warpSum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFull, value, offset);
  }
  return value;
}

__device__ double warpMax(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmax(value, __shfl_xor_sync(kFull, value, offset));
  }
  return value;
}

// gridBody for a step of at most 32 launches. All warps of the block load
// the previous execution's blocks and write the block table; the first warp
// does the rest with lane i as launch i, shuffles replacing shared-memory
// reductions. Shared memory holds per lane the clocks, first block and
// block count, then per previous block its launch and clocks, then a copy of
// the descriptor.
// The bytes of dynamic shared memory gridWarpBody uses.
__host__ __device__ inline int warpGridBytes(int numWords, int maxPrevBlocks) {
  return 32 * 16 + 12 * maxPrevBlocks + 4 * numWords;
}

// Where stepSetupWarp puts the programs in its dynamic shared memory, in
// 8-byte words, or -1 when they do not fit in kMaxSetupShared.
__host__ __device__ inline int setupCodeOffset(
    int numWords,
    int maxPrevBlocks,
    int numCode) {
  const int offset = (warpGridBytes(numWords, maxPrevBlocks) + 7) / 8;
  return numCode <= kSharedCode && 8 * (offset + numCode) <= kMaxSetupShared
      ? offset
      : -1;
}

__device__ void gridWarpBody(
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut,
    int loadFirst,
    int loadThreads,
    int loadBarrier) {
  float* multiplier = floats;
  float* prevCost = floats + n;
  int* state = ints;
  int* prevBlocks = ints + kStateWords;
  int* prevCap = prevBlocks + n;
  int* launchOf = prevCap + n;
  extern __shared__ long long shared[];
  long long* clocks = shared;
  long long* blockClocks = clocks + 32;
  int* firstOf = reinterpret_cast<int*>(blockClocks + maxPrevBlocks);
  int* blocksOf = firstOf + 32;
  int* blockLaunch = blocksOf + 32;
  int* desc = blockLaunch + maxPrevBlocks;
  __shared__ int sharedTotal;
  __shared__ float costOf[32];
  __shared__ int capOf[32];
  __shared__ int fitsAt[kWarps];
  __shared__ double timeAt[kWarps];
  __shared__ double searchState[3];
  unsigned char* blockInfos = base + blockInfoOffset;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const bool leader = tid < 32;
  const int warp = tid / 32;
  const bool active = leader && lane < n;
  const int numThreads = blockDim.x;

  __shared__ int emptyOf[32];
  __shared__ float multiplierOf[32];
  __shared__ float lastCostOf[32];
  __shared__ int lastBlocksOf[32];
  __shared__ int lastCapOf[32];
  __shared__ int prevNumOf;

  // Every independent load first, by the threads from 'loadFirst' on, which
  // meet at named barrier 'loadBarrier'; the others may run the fixups
  // meanwhile. The previous blocks are loaded ahead of the test against
  // their count.
  const int loadTid = tid - loadFirst;
  if (loadTid >= 0 && loadTid < loadThreads) {
    for (int w = loadTid; w < numWords; w += loadThreads) {
      desc[w] = deviceDesc[w];
    }
    if (loadTid < 32) {
      clocks[loadTid] = 0;
      if (loadTid < n) {
        emptyOf[loadTid] = blockInfos[loadTid];
        multiplierOf[loadTid] = multiplier[loadTid];
        lastCostOf[loadTid] = prevCost[loadTid];
        lastBlocksOf[loadTid] = prevBlocks[loadTid];
        lastCapOf[loadTid] = prevCap[loadTid];
      }
    }
    const int prevNum = state[0];
    if (loadTid == 0) {
      prevNumOf = prevNum;
    }
    const int countWaitWord = 6;
    SETUP_MARK(ints, 1);
    for (int b = loadTid; b < maxPrevBlocks; b += loadThreads) {
      const int launch = launchOf[b];
      const bool measured = previousClocks && b < launchBlocks;
      const long long run = measured ? stepClocks[b] : 0;
      const long long wait = measured ? stepWaits[b] : 0;
      blockLaunch[b] = b < prevNum ? launch : -1;
      blockClocks[b] = deviceDesc[countWaitWord] ? run : run - wait;
    }
    barrier(loadBarrier, loadThreads);

    // The previous clocks per launch. A launch's blocks are consecutive, so
    // each warp adds each run among its blocks once.
    for (int runBase = (loadTid / 32) * 32; runBase < prevNum;
         runBase += loadThreads) {
      const int b = runBase + lane;
      const int launch = b < prevNum ? blockLaunch[b] : -1;
      long long sum = b < prevNum ? blockClocks[b] : 0;
      sum = launch >= 0 && sum > 0 ? sum : 0;
      for (int offset = 1; offset < 32; offset <<= 1) {
        const long long other = __shfl_up_sync(kFull, sum, offset);
        const int otherLaunch = __shfl_up_sync(kFull, launch, offset);
        if (lane >= offset && otherLaunch == launch) {
          sum += other;
        }
      }
      const int nextLaunch = __shfl_down_sync(kFull, launch, 1);
      if (launch >= 0 && launch < 32 &&
          (lane == 31 || nextLaunch != launch) && sum > 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&clocks[launch]),
            static_cast<unsigned long long>(sum));
      }
    }
  }
  __syncthreads();
  const int prevNum = prevNumOf;
  const int empty = active ? emptyOf[lane] : 0;
  float m = active ? multiplierOf[lane] : 1;
  const float lastCost = active ? lastCostOf[lane] : 0;
  const int lastBlocks = active ? lastBlocksOf[lane] : 0;
  const int lastCap = active ? lastCapOf[lane] : 0;

  int assigned = 0;
  long long count = 0;
  float cost = 0;
  int cap = 1;
  if (leader) {
    const int target = desc[1] < launchBlocks ? desc[1] : launchBlocks;
    SETUP_MARK(ints, 2);
    const long long myClocks = active ? clocks[lane] : 0;
    const double totalClocks = warpSum(static_cast<double>(myClocks));
    const double totalPrevCost = warpSum(lastCost);
    const bool measured =
        prevNum > 0 && totalClocks > 0 && totalPrevCost > 0;
    if (active) {
      if (measured && lastCost > 0 && lastBlocks < lastCap) {
        const double ratio =
            (myClocks / totalClocks) / (lastCost / totalPrevCost);
        if (ratio > 1.1 || ratio < 1 / 1.1) {
          m *= static_cast<float>(ratio);
          multiplier[lane] = m;
        }
      }
      const int* l = desc + kHeaderWords + kLaunchWords * lane;
      count = empty ? 0 : launchCount(desc, l, base);
      cost = static_cast<float>(count) * __int_as_float(l[2]) * m;
      long long launchCap = l[5] ? 1 : (count + l[3] - 1) / l[3];
      if (launchCap > l[4]) {
        launchCap = l[4];
      }
      cap = launchCap < 1 ? 1 : static_cast<int>(launchCap);
    }
    const double maxCost = warpMax(static_cast<double>(cost));
    SETUP_MARK(ints, 3);

    costOf[lane] = cost;
    capOf[lane] = cap;
    if (lane == 0) {
      searchState[0] = maxCost;
      searchState[1] = n < target && maxCost > 0 ? 1 : 0;
      searchState[2] = target;
    }
  }
  __syncthreads();

  // The search of gridBody: in each round, warp k tries point k, its lanes
  // the launches.
  const double maxCost = searchState[0];
  const bool search = searchState[1] != 0;
  const int target = static_cast<int>(searchState[2]);
  double fits = maxCost;
  if (search) {
    double low = log(maxCost / target);
    double high = log(maxCost);
    for (int round = 0; round < kRounds; ++round) {
      if (warp < kWarps) {
        const double point = low + (high - low) * warp / kWarps;
        const double time = exp(point);
        const double inverse = 1.0 / time;
        long long sum = lane < n ? blocksAt(costOf[lane], inverse, capOf[lane])
                                 : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
          sum += __shfl_xor_sync(kFull, sum, offset);
        }
        if (lane == 0) {
          fitsAt[warp] = sum <= target;
          timeAt[warp] = time;
        }
      }
      __syncthreads();
      int smallest = kWarps;
      for (int w = kWarps - 1; w >= 0; --w) {
        smallest = fitsAt[w] ? w : smallest;
      }
      const double step = (high - low) / kWarps;
      const double oldLow = low;
      if (smallest < kWarps) {
        fits = timeAt[smallest];
      }
      __syncthreads();
      if (smallest == 0) {
        break;
      }
      if (smallest < kWarps) {
        high = oldLow + step * smallest;
        low = oldLow + step * (smallest - 1);
      } else {
        low = oldLow + step * (kWarps - 1);
      }
    }
  }
  if (leader) {
    const int blockSize = desc[2];
    SETUP_MARK(ints, 4);
    if (active) {
      assigned = search ? blocksAt(cost, 1.0 / fits, cap) : 1;
      if (assigned > 1) {
        const long long perBlock = (count + assigned - 1) / assigned;
        const long long aligned =
            (perBlock + blockSize - 1) / blockSize * blockSize;
        const long long fewer = (count + aligned - 1) / aligned;
        assigned = fewer < 1 ? 1 : static_cast<int>(fewer);
      }
    }
    int inclusive = assigned;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(kFull, inclusive, offset);
      if (lane >= offset) {
        inclusive += other;
      }
    }
    firstOf[lane] = inclusive - assigned;
    blocksOf[lane] = assigned;
    if (lane == 31) {
      sharedTotal = inclusive;
    }
  }
  __syncthreads();

  SETUP_MARK(ints, 5);
  // The block table, with blocks past the grid doing nothing.
  const int sum = sharedTotal;
  const int total = sum < launchBlocks ? sum : launchBlocks;
  for (int b = tid; b < launchBlocks; b += numThreads) {
    unsigned char* info =
        blockInfos + static_cast<long long>(b) * kBlockInfoBytes;
    unsigned char* blockDebug = debug + static_cast<long long>(b) * debugBytes;
    int launch = -1;
    if (b < total) {
      int low = 0;
      int high = n - 1;
      while (low < high) {
        const int middle = (low + high + 1) / 2;
        if (firstOf[middle] <= b) {
          low = middle;
        } else {
          high = middle - 1;
        }
      }
      launch = low;
      const int* l = desc + kHeaderWords + kLaunchWords * launch;
      *reinterpret_cast<int*>(info) = l[0];
      *reinterpret_cast<int*>(info + 4) = b - firstOf[launch];
      *reinterpret_cast<int*>(info + 8) = blocksOf[launch];
      *reinterpret_cast<unsigned char**>(info + 16) = base + l[1];
    } else {
      *reinterpret_cast<int*>(info) = -1;
      *reinterpret_cast<int*>(info + 4) = 0;
      *reinterpret_cast<int*>(info + 8) = 1;
      *reinterpret_cast<unsigned char**>(info + 16) = nullptr;
      for (int w = 0; w < debugBytes / 8; ++w) {
        reinterpret_cast<long long*>(blockDebug)[w] = 0;
      }
      stepClocks[b] = 0;
      stepWaits[b] = 0;
    }
    *reinterpret_cast<unsigned char**>(info + 24) = blockDebug;
    *reinterpret_cast<long long*>(info + 32) = 0;
    *reinterpret_cast<long long*>(info + 40) = 0;
    launchOf[b] = launch;
  }
  if (active) {
    prevCost[lane] = cost;
    prevBlocks[lane] = assigned;
    prevCap[lane] = cap;
    countsOut[lane] = count;
  }
  if (tid == 0) {
    state[0] = total;
    state[1] = sum;
  }
  SETUP_MARK(ints, 6);
}

extern "C" __global__ void deviceGridWarp(
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut) {
  SETUP_START();
  gridWarpBody(
      deviceDesc,
      numWords,
      n,
      maxPrevBlocks,
      base,
      blockInfoOffset,
      launchBlocks,
      debug,
      debugBytes,
      stepClocks,
      stepWaits,
      previousClocks,
      floats,
      ints,
      countsOut,
      0,
      blockDim.x,
      1);
}

extern "C" __global__ void stepSetupWarp(
    const int* records,
    const long long* code,
    int numRecords,
    const long long* symbols,
    int numSymbols,
    int numCode,
    const int* deviceDesc,
    int numWords,
    int n,
    int maxPrevBlocks,
    unsigned char* base,
    long long blockInfoOffset,
    int launchBlocks,
    unsigned char* debug,
    int debugBytes,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    float* floats,
    int* ints,
    long long* countsOut) {
  SETUP_START();
  // The first half of the block runs the fixups while the second half loads
  // what the grid reads before the counts; the programs go after the grid's
  // shared memory.
  extern __shared__ long long setupShared[];
  const int half = blockDim.x / 2;
  const int codeOffset = setupCodeOffset(numWords, maxPrevBlocks, numCode);
  if (threadIdx.x < half) {
    fixupBody(
        base,
        records,
        code,
        numRecords,
        symbols,
        numSymbols,
        codeOffset >= 0 ? setupShared + codeOffset : nullptr,
        numCode,
        0,
        half,
        1);
    SETUP_MARK(ints, 0);
#ifdef TW_SETUP_CLOCKS
    if (threadIdx.x == 0) {
      ints[9] = fixupMark[0];
      ints[10] = fixupMark[1];
    }
#endif
  }
  gridWarpBody(
      deviceDesc,
      numWords,
      n,
      maxPrevBlocks,
      base,
      blockInfoOffset,
      launchBlocks,
      debug,
      debugBytes,
      stepClocks,
      stepWaits,
      previousClocks,
      floats,
      ints,
      countsOut,
      half,
      blockDim.x - half,
      2);
}
)";

facebook::velox::wave::CompiledKernel& gridKernel() {
  static std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel;
  static folly::once_flag flag;
  folly::call_once(flag, [] {
    const bool clocks = std::getenv("TW_SETUP_CLOCKS") != nullptr;
    kernel = facebook::velox::wave::CompiledKernel::getKernel(
        clocks ? "deviceGridClocks" : "deviceGrid", [clocks] {
          facebook::velox::wave::KernelSpec spec;
          spec.code = std::string(clocks ? "#define TW_SETUP_CLOCKS\n" : "") +
              DeviceSymbols::fixupSource() + kGridSource;
          spec.entryPoints = {
              "deviceGrid", "stepSetup", "deviceGridWarp", "stepSetupWarp"};
          return spec;
        });
  });
  return *kernel;
}

int32_t sharedBytes(int32_t numLaunches, int32_t numWords) {
  return kSharedPerLaunch * numLaunches + 4 * numWords;
}

at::Tensor toDevice(const void* data, int64_t numel, at::ScalarType type) {
  return at::from_blob(
             const_cast<void*>(data),
             {std::max<int64_t>(numel, 1)},
             at::TensorOptions().dtype(type))
      .to(at::TensorOptions().device(at::kCUDA).dtype(type));
}

} // namespace

bool packDeviceGrid(
    const GridDescriptor& grid,
    const std::vector<int32_t>& paramOffsets,
    const std::vector<float>& costAdjust,
    int32_t numSMs,
    DeviceGrid& out) {
  const auto numLaunches = static_cast<int32_t>(grid.launches.size());
  int32_t numTerms = 0;
  int32_t numConsts = 0;
  for (const auto& launch : grid.launches) {
    if (launch.count == GridLaunchDesc::Count::kHost) {
      return false;
    }
    numTerms += static_cast<int32_t>(launch.countOffsets.size());
    numConsts += static_cast<int32_t>(launch.constShapes.size());
  }
  if (grid.targetBlocks > kGridThreads * kPrevPerThread) {
    return false;
  }
  out = DeviceGrid{};
  out.numLaunches = numLaunches;
  out.launchBlocks = grid.targetBlocks;
  auto& words = out.words;
  const int32_t termBase = kHeaderWords + kLaunchWords * numLaunches;
  const int32_t constBase = termBase + 2 * numTerms;
  words.assign(constBase + kConstWords * numConsts, 0);
  if (sharedBytes(numLaunches, static_cast<int32_t>(words.size())) >
      kMaxSharedBytes) {
    return false;
  }
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  words[0] = numLaunches;
  words[1] = grid.targetBlocks;
  words[2] = grid.blockSize;
  words[3] = out.launchBlocks;
  words[4] = termBase;
  words[5] = constBase;
  int32_t term = 0;
  int32_t constShape = 0;
  out.multipliers.resize(numLaunches);
  for (int32_t i = 0; i < numLaunches; ++i) {
    const auto& launch = grid.launches[i];
    auto* launchWords = &words[kHeaderWords + kLaunchWords * i];
    const float adjust = costAdjust[i] > 0 ? costAdjust[i] : 1.0f;
    // The descriptor's cost includes the adjustment; the multiplier carries it.
    const float unitCost = launch.costPerInput / adjust;
    launchWords[0] = launch.opCode;
    launchWords[1] = paramOffsets[i];
    std::memcpy(&launchWords[2], &unitCost, sizeof(float));
    launchWords[3] = std::max(1, grid.blockSize * launch.elementsPerThread);
    launchWords[4] = launch.blocksPerSm > 0
        ? numSMs * launch.blocksPerSm
        : std::numeric_limits<int32_t>::max();
    launchWords[5] = launch.singleBlock ? 1 : 0;
    launchWords[6] = static_cast<int32_t>(launch.count);
    launchWords[7] = term;
    launchWords[8] = static_cast<int32_t>(launch.countOffsets.size());
    launchWords[9] = constShape;
    launchWords[10] = static_cast<int32_t>(launch.constShapes.size());
    for (size_t t = 0; t < launch.countOffsets.size(); ++t) {
      words[termBase + 2 * term] = launch.countOffsets[t];
      words[termBase + 2 * term + 1] = launch.countIsScalar[t];
      ++term;
    }
    for (const auto& shape : launch.constShapes) {
      if (shape.size() > static_cast<size_t>(kMaxDims)) {
        return false;
      }
      auto* shapeWords = &words[constBase + kConstWords * constShape];
      shapeWords[0] = static_cast<int32_t>(shape.size());
      std::copy(shape.begin(), shape.end(), shapeWords + 1);
      ++constShape;
    }
    out.multipliers[i] = adjust;
  }
  return true;
}

void runDeviceGrid(
    DeviceGrid& grid,
    uint8_t* deviceBase,
    int64_t blockInfoOffset,
    int32_t launchBlocks,
    DebugInfo* deviceDebugBase,
    int64_t* deviceClocks,
    bool previousClocks,
    const FixupArgs* fixups,
    facebook::velox::wave::Stream* stream) {
  TORCH_CHECK(deviceClocks != nullptr, "A device grid needs the step's clocks");
  const int32_t n = grid.numLaunches;
  if (!grid.device.defined()) {
    // One allocation: the packed descriptor, then per launch the multiplier
    // and previous cost, the ints, and the counts, 8-byte aligned.
    const auto numWords = static_cast<int64_t>(grid.words.size());
    grid.floatsOffset = numWords;
    grid.intsOffset = grid.floatsOffset + 2 * n;
    grid.longsOffset =
        (grid.intsOffset + kStateWords + 2 * n + grid.launchBlocks + 1) / 2 * 2;
    std::vector<int32_t> words(
        grid.longsOffset + 2 * static_cast<int64_t>(n), 0);
    std::copy(grid.words.begin(), grid.words.end(), words.begin());
    std::memcpy(
        &words[grid.floatsOffset], grid.multipliers.data(), n * sizeof(float));
    grid.device = toDevice(
        words.data(), static_cast<int64_t>(words.size()), at::ScalarType::Int);
  }
  auto* desc = grid.device.data_ptr<int32_t>();
  auto* floats = reinterpret_cast<float*>(desc + grid.floatsOffset);
  auto* ints = desc + grid.intsOffset;
  auto* longs = reinterpret_cast<int64_t*>(desc + grid.longsOffset);
  long long offset = blockInfoOffset;
  int numWords = static_cast<int>(grid.words.size());
  int numLaunches = n;
  int numBlocks = std::min(launchBlocks, grid.launchBlocks);
  int maxPrevBlocks = grid.launchBlocks;
  int previousClocksArg = previousClocks ? 1 : 0;
  int debugBytes = sizeof(DebugInfo);
  auto* debug = reinterpret_cast<unsigned char*>(deviceDebugBase);
  auto* stepClocks = reinterpret_cast<long long*>(deviceClocks);
  auto* stepWaits = stepClocks + launchBlocks;
  FixupArgs fixupArgs = fixups != nullptr ? *fixups : FixupArgs{};
  void* args[] = {
      &fixupArgs.records,
      &fixupArgs.code,
      &fixupArgs.numRecords,
      &fixupArgs.symbols,
      &fixupArgs.numSymbols,
      &fixupArgs.numCode,
      &desc,
      &numWords,
      &numLaunches,
      &maxPrevBlocks,
      &deviceBase,
      &offset,
      &numBlocks,
      &debug,
      &debugBytes,
      &stepClocks,
      &stepWaits,
      &previousClocksArg,
      &floats,
      &ints,
      &longs};
  // deviceGrid takes the arguments after the fixups'. A step of at most 32
  // launches runs on one warp.
  constexpr int32_t kNumFixupArgs = 6;
  // As warpGridBytes and setupCodeOffset in the kernel source.
  const int32_t warpBytes = 32 * 16 + 12 * maxPrevBlocks + 4 * numWords;
  const bool oneWarp = n <= 32 && warpBytes <= kMaxSharedBytes;
  const int32_t entry = (fixups != nullptr ? 1 : 0) + (oneWarp ? 2 : 0);
  const int32_t codeBytes = fixups != nullptr
      ? DeviceSymbols::fixupSharedBytes(fixupArgs.numCode)
      : 0;
  int32_t dynamicShared = 0;
  if (oneWarp) {
    // The programs go after the grid's arrays when both fit.
    const int32_t codeOffset = (warpBytes + 7) / 8 * 8;
    dynamicShared = codeBytes > 0 && codeOffset + codeBytes <= kMaxSharedBytes
        ? codeOffset + codeBytes
        : warpBytes;
  } else {
    dynamicShared = std::max(
        sharedBytes(n, static_cast<int32_t>(grid.words.size())), codeBytes);
  }
  gridKernel().launch(
      entry,
      1,
      oneWarp ? kWarpGridThreads : kGridThreads,
      dynamicShared,
      stream,
      fixups != nullptr ? args : args + kNumFixupArgs);
}

DeviceGridResult readDeviceGrid(const DeviceGrid& grid) {
  const int32_t n = grid.numLaunches;
  DeviceGridResult result;
  const auto words = grid.device.cpu();
  const auto* base = words.data_ptr<int32_t>();
  const auto* counts =
      reinterpret_cast<const int64_t*>(base + grid.longsOffset);
  result.counts.assign(counts, counts + n);
  const auto* multipliers =
      reinterpret_cast<const float*>(base + grid.floatsOffset);
  result.multipliers.assign(multipliers, multipliers + n);
  const auto* ints = base + grid.intsOffset;
  const auto* blocks = ints + kStateWords;
  result.blocks.assign(blocks, blocks + n);
  if (std::getenv("TW_SETUP_CLOCKS") != nullptr) {
    // Per phase: after the fixups, the loads, the previous clocks, the
    // counts, the search, the scan, the block table.
    static std::mutex mutex;
    static std::array<int64_t, 2> numSteps{};
    static std::array<std::array<int64_t, 9>, 2> sums{};
    std::lock_guard<std::mutex> lock(mutex);
    const int32_t fused = ints[2] > 0 ? 1 : 0;
    ++numSteps[fused];
    for (int32_t k = 0; k < 9; ++k) {
      sums[fused][k] += ints[2 + k];
    }
    if ((numSteps[0] + numSteps[1]) % 200 == 0) {
      for (int32_t f = 0; f < 2; ++f) {
        std::string text;
        for (int32_t k = 0; k < 9; ++k) {
          text += fmt::format(
              " {}", numSteps[f] > 0 ? sums[f][k] / numSteps[f] : 0);
        }
        LOG(INFO) << "Setup clocks " << (f ? "with" : "without") << " fixups, "
                  << numSteps[f] << " steps:" << text;
      }
    }
  }
  return result;
}

} // namespace torch::wave

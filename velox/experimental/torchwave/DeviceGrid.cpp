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
#include <algorithm>
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
#include "velox/experimental/torchwave/WaveConfig.h"
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

constexpr int32_t kHeaderWords = 9;
constexpr int32_t kLaunchWords = 12;
constexpr int32_t kConstWords = 5;

// The kernel's thread count; also bounds nothing, the loops stride by it.
// kWarps warps in the kernel.
constexpr int32_t kGridThreads = 512;
// Threads of the kernel for steps of at most 32 launches: all load and
// write the block table, the first warp computes the grid.
constexpr int32_t kWarpGridThreads = 512;
// Shared memory per launch: clocks, count, the last full grid's count,
// cost, cap, blocks, first block, and the multiplier, cost, blocks and cap of
// the previous execution.
constexpr int32_t kSharedPerLaunch = 56;
// Blocks of the previous execution each thread reads.
constexpr int32_t kPrevPerThread = 4;
// Below the 48 KB a kernel gets without opting in, less the static shared of
// the grid and the fixups.
constexpr int32_t kMaxSharedBytes = 22 * 1024;
// Ints before the per-launch ints: the previous block count, the grid total,
// the warm search's T, and the skip state from word 16 (kSinceFull on in the
// kernel source).
constexpr int32_t kStateWords = 26;
// State words as kSinceFull and kNumSkipped in the kernel source.
constexpr int32_t kSinceFullWord = 16;
constexpr int32_t kNumSkippedWord = 22;

const char* const kGridSource = R"(
constexpr int kHeaderWords = 9;
constexpr int kLaunchWords = 12;
constexpr int kConstWords = 5;
constexpr int kBlockInfoBytes = 48;
constexpr int kWarps = 16;
constexpr int kRounds = 4;
constexpr int kStateWords = 26;
constexpr int kPrevPerThread = 4;
// State words of a grid: executions since the last full grid, its
// launchBlocks, the step's parameter base and DebugInfo base it wrote the table
// for (two words each), the grids skipped so far, whether a full grid ran, the
// full grids so far, and whether the last one changed no multiplier.
constexpr int kSinceFull = 16;
constexpr int kFullLaunchBlocks = 17;
constexpr int kFullBase = 18;
constexpr int kFullDebug = 20;
constexpr int kNumSkipped = 22;
constexpr int kFullValid = 23;
constexpr int kNumFull = 24;
constexpr int kFullSettled = 25;
constexpr int kSkipWords = 10;
// Full grids after which the grid skips even if the multipliers still move.
constexpr int kMaxLearningGrids = 8;
// Bytes of a BlockInfo the host's empty flags can cover, in 8-byte words.
constexpr int kInfoLongs = kBlockInfoBytes / 8;
// Dynamic shared memory of a setup kernel: 48 KB less the static shared of
// the fixups and the grid.
constexpr int kMaxSetupShared = 22 * 1024;

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

// One dim of a broadcast: a 1 takes the other's extent, and an empty dim
// empties the result.
__device__ int broadcastDim(int dim, int term) {
  return term == 1 ? dim
      : dim == 1   ? term
      : dim == 0 || term == 0 ? 0
      : term > dim ? term
                   : dim;
}

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
  // Kinds as GridLaunchDesc::Count: 1 max, 2 sum, 3 broadcast, 4 product.
  long long count = kind == 4 ? 1 : 0;
  int dims[4] = {1, 1, 1, 1};
  int rank = 0;
  for (int t = 0; t < numTerms; ++t) {
    const unsigned char* term = base + terms[2 * (termBegin + t)];
    if (terms[2 * (termBegin + t) + 1]) {
      const long long value = *reinterpret_cast<const long long*>(term);
      count = kind == 4 ? count * value : value > count ? value : count;
      continue;
    }
    const int termRank = *reinterpret_cast<const signed char*>(term + 8);
    const int* termDims = reinterpret_cast<const int*>(term + 12);
    // Tensor::numEl is a uint32_t, as on the host.
    const long long numEl = *reinterpret_cast<const unsigned*>(term + 44);
    if (kind == 4) {
      count *= numEl;
    } else if (kind == 2) {
      count += numEl;
    } else if (kind == 1) {
      count = numEl > count ? numEl : count;
    } else {
      rank = termRank > rank ? termRank : rank;
      for (int d = 0; d < termRank; ++d) {
        int& dim = dims[4 - termRank + d];
        dim = broadcastDim(dim, termDims[d]);
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
        dim = broadcastDim(dim, shape[1 + d]);
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

// 'assigned' blocks for 'count' elements, fewer when the last would be empty:
// all but the last block process a multiple of 'blockSize' elements.
__device__ int alignedBlocks(long long count, int assigned, int blockSize) {
  if (assigned <= 1) {
    return assigned;
  }
  if (count < (1LL << 31)) {
    // The same in 32 bits, which divide much faster.
    const unsigned count32 = static_cast<unsigned>(count);
    const unsigned perBlock = (count32 + assigned - 1) / assigned;
    const unsigned aligned = (perBlock + blockSize - 1) / blockSize * blockSize;
    const unsigned fewer = (count32 + aligned - 1) / aligned;
    return fewer < 1 ? 1 : static_cast<int>(fewer);
  }
  const long long perBlock = (count + assigned - 1) / assigned;
  const long long aligned = (perBlock + blockSize - 1) / blockSize * blockSize;
  const long long fewer = (count + aligned - 1) / aligned;
  return fewer < 1 ? 1 : static_cast<int>(fewer);
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

// The element counts gridBody gives the launches: each launch's count read
// from the step's parameters. prepare() runs on every thread before the
// counts; count() gives launch i's.
struct ParamCounts {
  // Whether the T search starts from the previous execution's.
  static constexpr bool kWarmSearch = false;
  // Whether another block learns the multipliers; see learnMultipliers.
  static constexpr bool kLearned = false;
  __device__ int learnedFlags() const {
    return 0;
  }

  // Runs after gridBody's first loads.
  __device__ void afterLoads() {}

  __device__ void prepare(const int* desc, int n, int* ints) {}

  __device__ long long
  count(const int* desc, const int* l, const unsigned char* base, int i) {
    return launchCount(desc, l, base);
  }
};

// ParamCounts after the 'numFlags' blocks that write the parameters have set
// their flags to 'generation'.
struct FlagCounts {
  static constexpr bool kWarmSearch = false;
  static constexpr bool kLearned = false;
  __device__ int learnedFlags() const {
    return 0;
  }
  const int* flags;
  int numFlags;
  int generation;

  __device__ void afterLoads() {}

  __device__ void prepare(const int* desc, int n, int* ints) {
    for (int i = threadIdx.x; i < numFlags; i += blockDim.x) {
      waitFlag(flags + i, generation);
    }
    __threadfence();
    __syncthreads();
    WIDE_MARK(ints, 5);
  }

  __device__ long long
  count(const int* desc, const int* l, const unsigned char* base, int i) {
    return launchCount(desc, l, base);
  }
};

// Count terms per thread of a wide grid block, and the flags of a term table
// entry: an integer term; read from the parameters as the host wrote them;
// over symbols without capacity; loads what the placement block stores.
constexpr int kTermsPerThread = 4;
constexpr int kTermScalar = 1;
constexpr int kTermStatic = 2;
constexpr int kTermSized = 4;
constexpr int kTermWait = 8;

// The element counts of a wide setup's grid, which then does not wait for
// the other blocks: a thread per count term evaluates what the tensor blocks
// write to the parameter the term reads, from the symbols, then each launch
// combines its terms as launchCount does. A term table entry is four int32:
// the flags, with the slot entry's rank from bit 8; the term's offset in the
// parameters; the slot entry or, for an integer, the value code that sets it;
// and a spare word. begin() issues the loads that need nothing else, before
// gridBody's own, and afterLoads() those that need the first ones.
struct WideCounts {
  static constexpr bool kWarmSearch = true;
  static constexpr bool kLearned = true;
  // learnMultipliers' flag and result words.
  const int* learnFlag;
  // Waits for learnMultipliers; bit 0 measured, bit 1 a multiplier changed.
  __device__ int learnedFlags() const {
    waitFlag(learnFlag, generation);
    return loadAcquire(learnFlag + 1);
  }
  const int* plan;
  const int* terms;
  int numTerms;
  const long long* code;
  const long long* symbols;
  int numSymbols;
  const unsigned char* params;
  long long* values;
  long long* caps;
  long long* shapes;
  long long* termCount;
  int4* termDims;
  unsigned char* termScalar;
  long long* launchCounts;
  const int* placedFlag;
  int generation;
  WideSymbol symbol;
  int4 term[kTermsPerThread];
  // An integer's value, or the parameter's bytes 8-15, the rank first.
  long long hostWord[kTermsPerThread];
  int hostDims[kTermsPerThread][4];
  unsigned hostNumEl[kTermsPerThread];
  int dimCode[kTermsPerThread][4];

  __device__ void begin() {
    symbol.load(code + 1, plan[11]);
#pragma unroll
    for (int k = 0; k < kTermsPerThread; ++k) {
      const int t = threadIdx.x + k * blockDim.x;
      term[k] = t < numTerms ? reinterpret_cast<const int4*>(terms)[t]
                             : make_int4(0, 0, 0, 0);
    }
    copyShapes(plan, shapes);
  }

  __device__ void afterLoads() {
    symbol.locate(symbols, numSymbols);
#pragma unroll
    for (int k = 0; k < kTermsPerThread; ++k) {
      const int t = threadIdx.x + k * blockDim.x;
      if (t >= numTerms) {
        continue;
      }
      const int flags = term[k].x;
      const unsigned char* p = params + term[k].y;
      if (flags & kTermScalar) {
        hostWord[k] =
            (flags & kTermStatic) ? *reinterpret_cast<const long long*>(p) : 0;
        continue;
      }
      // Four 8-byte loads for the rank, dims and numEl: bytes 8-31 and
      // 40-47. A parameter is only 8-byte aligned.
      const auto* words = reinterpret_cast<const long long*>(p);
      const long long rankAndDim = words[1];
      const long long dims12 = words[2];
      const long long dims3 = words[3];
      const long long numElWord = words[5];
      hostWord[k] = rankAndDim;
      hostDims[k][0] = static_cast<int>(rankAndDim >> 32);
      hostDims[k][1] = static_cast<int>(dims12);
      hostDims[k][2] = static_cast<int>(dims12 >> 32);
      hostDims[k][3] = static_cast<int>(dims3);
      hostNumEl[k] = static_cast<unsigned>(numElWord >> 32);
      const int* entry = plan + plan[5] + kWideSlotWords * term[k].z;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        dimCode[k][d] = (flags & kTermStatic) ? -1 : entry[6 + d];
      }
    }
  }

  __device__ void prepare(const int* desc, int n, int* ints) {
    const long long* operands = code + plan[10];
    // The codes are in by now; their operands load while the symbols do.
#pragma unroll
    for (int k = 0; k < kTermsPerThread; ++k) {
      if (threadIdx.x + k * blockDim.x >= numTerms) {
        continue;
      }
      if (term[k].x & kTermScalar) {
        prefetchCode(term[k].z, operands);
        continue;
      }
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        prefetchCode(dimCode[k][d], operands);
      }
    }
    const int anyOver =
        symbol.finish(code + 1, plan[11], symbols, numSymbols, values, caps);
    const int over = __syncthreads_or(anyOver);
#pragma unroll
    for (int k = 0; k < kTermsPerThread; ++k) {
      const int t = threadIdx.x + k * blockDim.x;
      if (t >= numTerms) {
        continue;
      }
      const int flags = term[k].x;
      if (flags & kTermWait) {
        waitFlag(placedFlag, generation);
      }
      if (flags & kTermScalar) {
        termScalar[t] = 1;
        termCount[t] = (flags & kTermStatic) ? hostWord[k]
            : over                         ? 0
                   : evalCode(term[k].z, shapes, operands, values, nullptr, caps);
        continue;
      }
      termScalar[t] = 0;
      // As applySlot writes the parameter, and launchCount reads it.
      long long dim[4];
      unsigned numEl = hostNumEl[k];
      if (flags & kTermStatic) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          dim[d] = hostDims[k][d];
        }
      } else {
        const int rank = (flags >> 8) & 15;
        const int recordOver = (flags & kTermSized) ? 0 : over;
        long long product = 1;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          if (d < rank) {
            dim[d] = recordOver ? 0
                : dimCode[k][d] < 0
                ? hostDims[k][d]
                : evalCode(dimCode[k][d], shapes, operands, values, nullptr, caps);
            product *= dim[d];
          } else {
            dim[d] = hostDims[k][d];
          }
        }
        numEl = static_cast<unsigned>(product);
      }
      termCount[t] = numEl;
      // Right-aligned, a 1 where the term has no dim.
      const int hostRank = static_cast<signed char>(hostWord[k] & 0xff);
      int aligned[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int d = q - (4 - hostRank);
        aligned[q] = d < 0 ? 1
            : d == 0       ? static_cast<int>(dim[0])
            : d == 1       ? static_cast<int>(dim[1])
            : d == 2       ? static_cast<int>(dim[2])
                           : static_cast<int>(dim[3]);
      }
      termDims[t] = make_int4(aligned[0], aligned[1], aligned[2], aligned[3]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      launchCounts[i] = fold(desc, desc + kHeaderWords + kLaunchWords * i);
    }
    __syncthreads();
    WIDE_MARK(ints, 5);
  }

  // launchCount over the evaluated terms.
  __device__ long long fold(const int* desc, const int* l) const {
    const int kind = l[6];
    const int termBegin = l[7];
    const int numLaunchTerms = l[8];
    const int constBegin = l[9];
    const int numConsts = l[10];
    if (numLaunchTerms == 0 && numConsts == 0) {
      return 1;
    }
    long long count = kind == 4 ? 1 : 0;
    int dims[4] = {1, 1, 1, 1};
    for (int t = termBegin; t < termBegin + numLaunchTerms; ++t) {
      const long long value = termCount[t];
      if (termScalar[t]) {
        count = kind == 4 ? count * value : value > count ? value : count;
        continue;
      }
      if (kind == 4) {
        count *= value;
      } else if (kind == 2) {
        count += value;
      } else if (kind == 1) {
        count = value > count ? value : count;
      } else {
        const int4 term = termDims[t];
        dims[0] = broadcastDim(dims[0], term.x);
        dims[1] = broadcastDim(dims[1], term.y);
        dims[2] = broadcastDim(dims[2], term.z);
        dims[3] = broadcastDim(dims[3], term.w);
      }
    }
    if (kind == 3) {
      const int* consts = desc + desc[5];
      for (int c = 0; c < numConsts; ++c) {
        const int* shape = consts + kConstWords * (constBegin + c);
        const int constRank = shape[0];
        for (int d = 0; d < constRank; ++d) {
          int& dim = dims[4 - constRank + d];
          dim = broadcastDim(dim, shape[1 + d]);
        }
      }
      count = static_cast<long long>(dims[0]) * dims[1] * dims[2] * dims[3];
    }
    return count;
  }

  __device__ long long
  count(const int* desc, const int* l, const unsigned char* base, int i) {
    return launchCounts[i];
  }
};

// The multipliers a wide setup's grid uses, learned by the block before the
// grid block while the grid block loads and counts: they depend only on the
// previous execution's clocks and blocks, as in gridBody. Writes each to the
// third float array, then flag[1] (bit 0 measured, bit 1 one changed) and
// releases flag[0] at 'generation'. 'clocks' is shared, n long longs. The
// grid block writes what this reads only after it has waited for the flag;
// a grid that keeps its table does not read the result.
__device__ void learnMultipliers(
    int n,
    int maxPrevBlocks,
    int launchBlocks,
    const int* deviceDesc,
    const long long* stepClocks,
    const long long* stepWaits,
    int previousClocks,
    float* floats,
    const int* ints,
    int* flag,
    int generation,
    long long* clocks) {
  const float* multiplier = floats;
  const float* prevCost = floats + n;
  float* learned = floats + 2 * n;
  const int* prevBlocks = ints + kStateWords;
  const int* prevCap = prevBlocks + n;
  const int* launchOf = prevCap + n;
  __shared__ double learnScratch[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int numThreads = blockDim.x;
  for (int i = tid; i < n; i += numThreads) {
    clocks[i] = 0;
  }
  const int prevNum = ints[0];
  const int countWait = deviceDesc[6];
  __syncthreads();
  // As in gridBody: each warp sums the runs of one launch among its blocks.
  for (int first = 0; first < maxPrevBlocks; first += numThreads) {
    const int b = first + tid;
    const int launch = b < prevNum && b < maxPrevBlocks ? launchOf[b] : -1;
    long long sum = 0;
    if (launch >= 0 && previousClocks && b < launchBlocks) {
      sum = stepClocks[b] - (countWait ? 0 : stepWaits[b]);
    }
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
    localCost += prevCost[i];
  }
  const double totalClocks = blockSum(localClocks, learnScratch);
  const double totalPrevCost = blockSum(localCost, learnScratch);
  const bool measured = prevNum > 0 && totalClocks > 0 && totalPrevCost > 0;
  int changed = 0;
  for (int i = tid; i < n; i += numThreads) {
    float m = multiplier[i];
    if (measured && prevCost[i] > 0 && prevBlocks[i] < prevCap[i]) {
      const double ratio =
          (clocks[i] / totalClocks) / (prevCost[i] / totalPrevCost);
      if (ratio > 1.1 || ratio < 1 / 1.1) {
        m *= static_cast<float>(ratio);
        changed = 1;
      }
    }
    learned[i] = m;
  }
  changed = __syncthreads_or(changed);
  if (tid == 0) {
    flag[1] = (measured ? 1 : 0) | (changed ? 2 : 0);
    __threadfence();
    storeRelease(flag, generation);
  }
}

// One block of kWarps warps. Shared memory holds per launch: clocks, count,
// cost, cap, blocks, first block, and the previous execution's multiplier,
// cost, blocks and cap; then a copy of the descriptor's 'numWords'.
// 'stepClocks' and 'stepWaits' are the step's per-block clocks and barrier
// clocks in the execution's clock buffer, for its 'launchBlocks' blocks. With
// 'previousClocks' they hold the previous execution's clocks of the step. A
// negative 'numWords' leaves the descriptor in global memory, for a step whose
// descriptor does not fit. 'maxPrevBlocks' bounds the previous execution's
// block count. 'counter' gives each launch's element count; see
// ParamCounts. 'countsOut' holds per launch the counts of this execution, then
// those of the last full grid, then a copy of the block table's first bytes,
// which the host's empty flags overwrite.
// The grid is skipped, keeping the previous table, when descriptor word 7 asks
// for it (bits 0-15 a full grid at least every that many executions, bits
// 16-31 the tolerance in thousandths) and every launch's count is within the
// tolerance of its count at the last full grid. The step's kernels loop over
// their elements by numBlocksInOp, so any block count is correct; a launch
// keeps its blocks only where a full grid would give it no empty block. The
// multipliers learn only on the full grids, from the clocks of the execution
// before, which ran with the blocks the table has.
// Called by every thread of the block.
template <typename Counts>
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
    long long* countsOut,
    Counts& counter) {
  float* multiplier = floats;
  float* prevCost = floats + n;
  int* state = ints;
  int* prevBlocks = ints + kStateWords;
  int* prevCap = prevBlocks + n;
  int* launchOf = prevCap + n;
  long long* fullCounts = countsOut + n;
  long long* headCopy = fullCounts + n;
  const int headLongs = (n + kBlockInfoBytes - 1) / kBlockInfoBytes * kInfoLongs;
  extern __shared__ long long shared[];
  long long* clocks = shared;
  long long* counts = clocks + n;
  long long* lastCounts = counts + n;
  float* cost = reinterpret_cast<float*>(lastCounts + n);
  int* cap = reinterpret_cast<int*>(cost + n);
  int* blocks = cap + n;
  int* first = blocks + n;
  float* lastMultiplier = reinterpret_cast<float*>(first + n);
  float* lastCost = lastMultiplier + n;
  int* lastBlocks = reinterpret_cast<int*>(lastCost + n);
  int* lastCap = lastBlocks + n;
  int* sharedDesc = lastCap + n;
  const int* desc = numWords >= 0 ? sharedDesc : deviceDesc;
  __shared__ double scratch[kWarps];
  __shared__ int intScratch[kWarps];
  __shared__ unsigned fitMasks[3];
  __shared__ double points[2][kWarps];
  __shared__ float inverses[2][kWarps];
  __shared__ double logs[2];
  unsigned char* blockInfos = base + blockInfoOffset;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int numThreads = blockDim.x;

  // Every load that does not depend on another goes first, so the kernel
  // waits for memory about once. The host's empty flags are read before the
  // block table overwrites them. The previous execution's blocks are loaded
  // ahead of the test against its block count.
  // In batches, so that a thread's loads overlap.
  for (int begin = tid; begin < numWords; begin += 4 * numThreads) {
    int words[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int w = begin + k * numThreads;
      words[k] = w < numWords ? deviceDesc[w] : 0;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int w = begin + k * numThreads;
      if (w < numWords) {
        sharedDesc[w] = words[k];
      }
    }
  }
  if (tid < 3) {
    fitMasks[tid] = 0;
  }
  for (int i = tid; i < n; i += numThreads) {
    first[i] = blockInfos[i];
    clocks[i] = 0;
    lastMultiplier[i] = multiplier[i];
    lastCost[i] = prevCost[i];
    lastBlocks[i] = prevBlocks[i];
    lastCap[i] = prevCap[i];
    lastCounts[i] = fullCounts[i];
  }
  const int prevNum = state[0];
  const int skipWord = deviceDesc[7];
  const int sinceFull = state[kSinceFull];
  const int fullLaunchBlocks = state[kFullLaunchBlocks];
  // The ints are only 4-byte aligned.
  const unsigned fullBaseLow = state[kFullBase];
  const unsigned fullBaseHigh = state[kFullBase + 1];
  const unsigned fullDebugLow = state[kFullDebug];
  const unsigned fullDebugHigh = state[kFullDebug + 1];
  const int fullValid = state[kFullValid];
  const int numFull = state[kNumFull];
  const int fullSettled = state[kFullSettled];
  const long long headWord = tid < headLongs ? headCopy[tid] : 0;
  // The previous execution's T for a warm search.
  const int warmValid = state[14];
  const int warmLogT = state[15];
  int blockLaunch[kPrevPerThread];
  long long blockClocks[kPrevPerThread];
  long long blockWait[kPrevPerThread];
#pragma unroll
  for (int k = 0; k < kPrevPerThread; ++k) {
    const int b = k * numThreads + tid;
    blockLaunch[k] = -1;
    blockClocks[k] = 0;
    blockWait[k] = 0;
    if (!Counts::kLearned && b < maxPrevBlocks) {
      blockLaunch[k] = launchOf[b];
      if (previousClocks && b < launchBlocks) {
        blockClocks[k] = stepClocks[b];
        blockWait[k] = stepWaits[b];
      }
    }
  }
  __syncthreads();
  WIDE_MARK(ints, 7);
  counter.afterLoads();
  const int target = desc[1] < launchBlocks ? desc[1] : launchBlocks;
  const int blockSize = desc[2];
  const int countWait = desc[6];

  counter.prepare(desc, n, ints);

  // Counts and block caps, and whether each launch can keep its blocks.
  const int skipRefresh = skipWord & 0xffff;
  const long long skipTolerance = skipWord >> 16;
  // The table is kept only once the multipliers have settled: they learn on
  // full grids alone.
  const bool canSkip = skipRefresh > 0 && fullValid && prevNum > 0 &&
      (fullSettled || numFull >= kMaxLearningGrids) &&
      sinceFull + 1 < skipRefresh && fullLaunchBlocks == launchBlocks &&
      (static_cast<unsigned long long>(fullBaseHigh) << 32 | fullBaseLow) ==
          reinterpret_cast<unsigned long long>(base) &&
      (static_cast<unsigned long long>(fullDebugHigh) << 32 | fullDebugLow) ==
          reinterpret_cast<unsigned long long>(debug);
  int keeps = 1;
  for (int i = tid; i < n; i += numThreads) {
    const int* l = desc + kHeaderWords + kLaunchWords * i;
    const long long count = first[i] ? 0 : counter.count(desc, l, base, i);
    counts[i] = count;
    long long launchCap = l[5] ? 1 : (count + l[3] - 1) / l[3];
    if (launchCap > l[4]) {
      launchCap = l[4];
    }
    cap[i] = launchCap < 1 ? 1 : static_cast<int>(launchCap);
    const long long fullCount = lastCounts[i];
    const long long change =
        count > fullCount ? count - fullCount : fullCount - count;
    const int kept = lastBlocks[i];
    keeps &= (count == 0) == (fullCount == 0) &&
            change * 1000 <= skipTolerance * fullCount && kept <= cap[i] &&
            alignedBlocks(count, kept, blockSize) == kept
        ? 1
        : 0;
  }
  if (__syncthreads_and(keeps) && canSkip) {
    WIDE_MARK(ints, 8);
    for (int i = tid; i < n; i += numThreads) {
      const int* l = desc + kHeaderWords + kLaunchWords * i;
      prevCost[i] = static_cast<float>(counts[i]) * __int_as_float(l[2]) *
          lastMultiplier[i];
      prevCap[i] = cap[i];
      countsOut[i] = counts[i];
    }
    if (tid < headLongs) {
      reinterpret_cast<long long*>(blockInfos)[tid] = headWord;
    }
    // Blocks past the grid leave their DebugInfo and clocks as they find
    // them. Their DebugInfo, at the same address, is as the full grid left it.
    for (int b = prevNum + tid; b < launchBlocks; b += numThreads) {
      stepClocks[b] = 0;
      stepWaits[b] = 0;
    }
    if (tid == 0) {
      state[kSinceFull] = sinceFull + 1;
      ++state[kNumSkipped];
    }
    __syncthreads();
    WIDE_MARK(ints, 6);
    return;
  }

  double localMax = 0;
  int changed = 0;
  bool measured = false;
  if (Counts::kLearned) {
    // Another block learned them; see learnMultipliers.
    const int flags = counter.learnedFlags();
    measured = (flags & 1) != 0;
    changed = (flags & 2) != 0 ? 1 : 0;
    const float* learned = floats + 2 * n;
    for (int i = tid; i < n; i += numThreads) {
      const int* l = desc + kHeaderWords + kLaunchWords * i;
      const float m = learned[i];
      multiplier[i] = m;
      cost[i] = static_cast<float>(counts[i]) * __int_as_float(l[2]) * m;
      localMax = fmax(localMax, static_cast<double>(cost[i]));
    }
  } else {
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
    measured = prevNum > 0 && totalClocks > 0 && totalPrevCost > 0;

    // Multipliers and estimated costs.
    for (int i = tid; i < n; i += numThreads) {
      const int* l = desc + kHeaderWords + kLaunchWords * i;
      float m = lastMultiplier[i];
      if (measured && lastCost[i] > 0 && lastBlocks[i] < lastCap[i]) {
        const double ratio =
            (clocks[i] / totalClocks) / (lastCost[i] / totalPrevCost);
        if (ratio > 1.1 || ratio < 1 / 1.1) {
          m *= static_cast<float>(ratio);
          multiplier[i] = m;
          changed = 1;
        }
      }
      cost[i] = static_cast<float>(counts[i]) * __int_as_float(l[2]) * m;
      localMax = fmax(localMax, static_cast<double>(cost[i]));
    }
  }
  const double maxCost = blockMax(localMax, scratch);
  const bool settled = measured && !__syncthreads_or(changed);

  // The smallest common completion time T whose blocks fit the target,
  // searched in log T: each round, warp k tries the k-th of kWarps points
  // between a time known not to fit and one known to fit.
  const bool search = n < target && maxCost > 0;
  double fits = maxCost;
  // Each round tries one T per warp: lane k of warp 0 computes 1 / T of warp
  // k's point, and warp k sums the blocks at its T. Returns the smallest warp
  // whose blocks fit the target, or kWarps. Round r sets bit k of
  // fitMasks[r % 3] for a warp that fits and clears the mask of round r + 2,
  // which every thread has read before the barrier. The rounds judge in
  // FP32, which issues much faster than FP64; the blocks at the T they pick
  // are computed in FP64, as on the host.
  int numRounds = 0;
  double smallestPoint = 0;
  auto smallestFitting = [&](auto pointOf) {
    double* roundPoints = points[numRounds % 2];
    float* roundInverses = inverses[numRounds % 2];
    if (warp == 0 && lane < kWarps) {
      const double point = pointOf(lane);
      roundPoints[lane] = point;
      roundInverses[lane] = expf(-static_cast<float>(point));
    }
    __syncthreads();
    const float inverse = roundInverses[warp];
    // Each launch counts at most target + 1, so the sum stays in an int.
    int sum = 0;
    for (int i = lane; i < n; i += 32) {
      float wanted = ceilf(cost[i] * inverse);
      wanted = wanted < 1 ? 1 : wanted;
      const int assigned =
          wanted > cap[i] ? cap[i] : static_cast<int>(wanted);
      sum += assigned <= target ? assigned : target + 1;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(kFull, sum, offset);
    }
    unsigned* mask = fitMasks + numRounds % 3;
    if (lane == 0 && sum <= target) {
      atomicOr(mask, 1u << warp);
    }
    __syncthreads();
    const unsigned fitting = *mask;
    if (tid == 0) {
      fitMasks[(numRounds + 2) % 3] = 0;
    }
    ++numRounds;
    const int smallest = fitting != 0 ? __ffs(fitting) - 1 : kWarps;
    smallestPoint = smallest < kWarps ? roundPoints[smallest] : 0;
    return smallest;
  };
  bool found = false;
  double fitPoint = 0;
  if (search) {
    // On two warps at the same time: an FP64 log is a long chain.
    if (tid == 0) {
      logs[0] = log(maxCost / target);
    } else if (tid == 32) {
      logs[1] = log(maxCost);
    }
    __syncthreads();
    double low = logs[0];
    double high = logs[1];
    int firstRound = 0;
    if (Counts::kWarmSearch && warmValid == target) {
      // The cold search's points of round 3 around the previous execution's
      // T, one per warp. Where they bracket T, its last round finishes the
      // search; else it runs from the start.
      const double step = (high - low) / (kWarps * kWarps * kWarps);
      const double center = floor((__int_as_float(warmLogT) - low) / step);
      const int smallest = smallestFitting([&](int k) {
        const double point = low + step * (center - kWarps / 2 + k);
        return point < high ? point : high;
      });
      const double bracketLow =
          low + step * (center - kWarps / 2 + smallest - 1);
      if (smallest > 0 && smallest < kWarps && bracketLow >= low) {
        found = true;
        fitPoint = smallestPoint;
        high = bracketLow + step;
        low = bracketLow;
        firstRound = kRounds - 1;
      }
    }
    for (int round = firstRound; round < kRounds; ++round) {
      const int smallest = smallestFitting(
          [&](int k) { return low + (high - low) * k / kWarps; });
      const double step = (high - low) / kWarps;
      const double oldLow = low;
      if (smallest < kWarps) {
        found = true;
        fitPoint = smallestPoint;
      }
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
  if (found) {
    fits = exp(fitPoint);
  }
  if (Counts::kWarmSearch && tid == 0) {
    state[14] = search ? target : 0;
    // Only picks the next execution's first points, so a fast log does.
    state[15] = __float_as_int(__logf(static_cast<float>(fits)));
  }
  for (int i = tid; i < n; i += numThreads) {
    blocks[i] = alignedBlocks(
        counts[i],
        search ? blocksAt(cost[i], 1.0 / fits, cap[i]) : 1,
        blockSize);
  }
  // Test hook: overshoots the target as the FP64 blocks can.
  if (tid == 0 && desc[8] > 0) {
    blocks[n - 1] += desc[8];
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
  // The search judges the fit in FP32 and the blocks above are computed in
  // FP64, so they can add up to a few more than the target. A launch cut off
  // at the end of the grid would run fewer blocks than its numBlocksInOp:
  // its last elements would be skipped, and an op barrier would never fill.
  if (sum > target) {
    if (tid == 0) {
      int excess = sum - target;
      while (excess > 0) {
        int widest = 0;
        for (int i = 1; i < n; ++i) {
          widest = blocks[i] > blocks[widest] ? i : widest;
        }
        if (blocks[widest] <= 1) {
          break;
        }
        int fewer = blocks[widest] - 1;
        if (fewer > 1) {
          const long long perBlock = (counts[widest] + fewer - 1) / fewer;
          const long long aligned =
              (perBlock + blockSize - 1) / blockSize * blockSize;
          const long long fewest = (counts[widest] + aligned - 1) / aligned;
          fewer = fewest < 1 ? 1 : static_cast<int>(fewest);
        }
        excess -= blocks[widest] - fewer;
        blocks[widest] = fewer;
      }
    }
    __syncthreads();
    runSum = 0;
    for (int i = runBegin; i < runEnd; ++i) {
      runSum += blocks[i];
    }
    offset = blockScan(runSum, intScratch, sum);
  }
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
    }
    // Op, block in op, blocks in op, the padding, params.
    int4 head = make_int4(-1, 0, 1, 0);
    unsigned char* params = nullptr;
    if (launch >= 0) {
      const int* l = desc + kHeaderWords + kLaunchWords * launch;
      head = make_int4(l[0], b - first[launch], blocks[launch], 0);
      params = base + l[1];
    } else {
      // A block that does nothing leaves its DebugInfo and clocks as it
      // finds them. The host reads every launched block's DebugInfo for errors
      // and its clocks for the step's statistics.
      for (int w = 0; w < debugBytes / 8; ++w) {
        reinterpret_cast<long long*>(blockDebug)[w] = 0;
      }
      stepClocks[b] = 0;
      stepWaits[b] = 0;
    }
    // Three 16-byte stores, when aligned: the stores of the table are the
    // phase's cost.
    if ((reinterpret_cast<unsigned long long>(blockInfos) & 15) == 0) {
      auto* to = reinterpret_cast<int4*>(info);
      const auto paramsWord = reinterpret_cast<unsigned long long>(params);
      const auto debugWord = reinterpret_cast<unsigned long long>(blockDebug);
      to[0] = head;
      to[1] = make_int4(
          static_cast<int>(paramsWord),
          static_cast<int>(paramsWord >> 32),
          static_cast<int>(debugWord),
          static_cast<int>(debugWord >> 32));
      to[2] = make_int4(0, 0, 0, 0);
    } else {
      *reinterpret_cast<int*>(info) = head.x;
      *reinterpret_cast<int*>(info + 4) = head.y;
      *reinterpret_cast<int*>(info + 8) = head.z;
      *reinterpret_cast<unsigned char**>(info + 16) = params;
      *reinterpret_cast<unsigned char**>(info + 24) = blockDebug;
      *reinterpret_cast<long long*>(info + 32) = 0;
      *reinterpret_cast<long long*>(info + 40) = 0;
    }
    // The longs are only 8-byte aligned.
    if (b * kInfoLongs < headLongs) {
      long long* copy = headCopy + b * kInfoLongs;
      copy[0] = static_cast<long long>(static_cast<unsigned>(head.x)) |
          (static_cast<long long>(head.y) << 32);
      copy[1] = head.z;
      copy[2] = reinterpret_cast<long long>(params);
      copy[3] = reinterpret_cast<long long>(blockDebug);
      copy[4] = 0;
      copy[5] = 0;
    }
    launchOf[b] = launch;
  }
  for (int i = tid; i < n; i += numThreads) {
    prevCost[i] = cost[i];
    prevBlocks[i] = blocks[i];
    prevCap[i] = cap[i];
    countsOut[i] = counts[i];
    fullCounts[i] = counts[i];
  }
  if (tid == 0) {
    state[0] = total;
    state[1] = sum;
    state[kSinceFull] = 0;
    state[kFullLaunchBlocks] = launchBlocks;
    const auto baseWord = reinterpret_cast<unsigned long long>(base);
    const auto debugWord = reinterpret_cast<unsigned long long>(debug);
    state[kFullBase] = static_cast<int>(baseWord);
    state[kFullBase + 1] = static_cast<int>(baseWord >> 32);
    state[kFullDebug] = static_cast<int>(debugWord);
    state[kFullDebug + 1] = static_cast<int>(debugWord >> 32);
    state[kFullValid] = 1;
    state[kNumFull] = numFull + 1;
    state[kFullSettled] = settled ? 1 : 0;
  }
  __syncthreads();
  WIDE_MARK(ints, 6);
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
    long long* countsOut,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
  SETUP_START();
  ParamCounts paramCounts;
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
      countsOut,
      paramCounts);
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
    long long* countsOut,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
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
      1,
      failure,
      firstStep);
  __syncthreads();
  ParamCounts paramCounts;
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
      countsOut,
      paramCounts);
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
  long long* fullCounts = countsOut + n;
  long long* headCopy = fullCounts + n;
  const int headLongs = (n + kBlockInfoBytes - 1) / kBlockInfoBytes * kInfoLongs;
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
  __shared__ long long fullCountOf[32];
  __shared__ long long headOf[kInfoLongs];
  __shared__ int skipStateOf[kSkipWords];
  __shared__ int skipOf;

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
        fullCountOf[loadTid] = fullCounts[loadTid];
      }
      if (loadTid < headLongs) {
        headOf[loadTid] = headCopy[loadTid];
      }
      if (loadTid < kSkipWords) {
        skipStateOf[loadTid] = state[kSinceFull + loadTid];
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
  // Whether this grid changed no multiplier; in the first warp.
  bool settled = false;
  if (leader) {
    const int target = desc[1] < launchBlocks ? desc[1] : launchBlocks;
    SETUP_MARK(ints, 2);
    const int* l = desc + kHeaderWords + kLaunchWords * lane;
    if (active) {
      count = empty ? 0 : launchCount(desc, l, base);
      long long launchCap = l[5] ? 1 : (count + l[3] - 1) / l[3];
      if (launchCap > l[4]) {
        launchCap = l[4];
      }
      cap = launchCap < 1 ? 1 : static_cast<int>(launchCap);
    }
    // As in gridBody: keep the table when every count is within the
    // tolerance of the last full grid's.
    const int skipWord = desc[7];
    const int skipRefresh = skipWord & 0xffff;
    const long long skipTolerance = skipWord >> 16;
    const unsigned long long fullBase =
        static_cast<unsigned long long>(
            static_cast<unsigned>(skipStateOf[kFullBase - kSinceFull + 1]))
            << 32 |
        static_cast<unsigned>(skipStateOf[kFullBase - kSinceFull]);
    const unsigned long long fullDebug =
        static_cast<unsigned long long>(
            static_cast<unsigned>(skipStateOf[kFullDebug - kSinceFull + 1]))
            << 32 |
        static_cast<unsigned>(skipStateOf[kFullDebug - kSinceFull]);
    const int sinceFull = skipStateOf[0];
    const bool canSkip = skipRefresh > 0 &&
        skipStateOf[kFullValid - kSinceFull] && prevNum > 0 &&
        (skipStateOf[kFullSettled - kSinceFull] ||
         skipStateOf[kNumFull - kSinceFull] >= kMaxLearningGrids) &&
        sinceFull + 1 < skipRefresh &&
        skipStateOf[kFullLaunchBlocks - kSinceFull] == launchBlocks &&
        fullBase == reinterpret_cast<unsigned long long>(base) &&
        fullDebug == reinterpret_cast<unsigned long long>(debug);
    const long long fullCount = active ? fullCountOf[lane] : 0;
    const long long change =
        count > fullCount ? count - fullCount : fullCount - count;
    const bool keeps = !active ||
        ((count == 0) == (fullCount == 0) &&
         change * 1000 <= skipTolerance * fullCount && lastBlocks <= cap &&
         alignedBlocks(count, lastBlocks, desc[2]) == lastBlocks);
    const bool skip = __all_sync(kFull, keeps) && canSkip;
    if (skip) {
      if (active) {
        prevCost[lane] = static_cast<float>(count) * __int_as_float(l[2]) * m;
        prevCap[lane] = cap;
        countsOut[lane] = count;
      }
      if (lane < headLongs) {
        reinterpret_cast<long long*>(blockInfos)[lane] = headOf[lane];
      }
      if (lane == 0) {
        state[kSinceFull] = sinceFull + 1;
        ++state[kNumSkipped];
        skipOf = 1;
      }
    } else if (lane == 0) {
      skipOf = 0;
    }
    const long long myClocks = active ? clocks[lane] : 0;
    const double totalClocks = warpSum(static_cast<double>(myClocks));
    const double totalPrevCost = warpSum(lastCost);
    const bool measured =
        prevNum > 0 && totalClocks > 0 && totalPrevCost > 0;
    bool changed = false;
    if (active) {
      if (measured && lastCost > 0 && lastBlocks < lastCap && !skip) {
        const double ratio =
            (myClocks / totalClocks) / (lastCost / totalPrevCost);
        if (ratio > 1.1 || ratio < 1 / 1.1) {
          m *= static_cast<float>(ratio);
          multiplier[lane] = m;
          changed = true;
        }
      }
      cost = static_cast<float>(count) * __int_as_float(l[2]) * m;
    }
    settled = measured && !__any_sync(kFull, changed);
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
  if (skipOf) {
    // Blocks past the grid leave their clocks as they find them; their
    // DebugInfo is as the full grid left it.
    for (int b = prevNum + tid; b < launchBlocks; b += numThreads) {
      stepClocks[b] = 0;
      stepWaits[b] = 0;
    }
    SETUP_MARK(ints, 6);
    return;
  }

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
      if (lane == n - 1 && desc[8] > 0) {
        assigned += desc[8];
      }
    }
    // As in gridBody: give blocks back from the widest launches until the
    // grid fits the target.
    int excess = __reduce_add_sync(kFull, assigned) - target;
    while (excess > 0) {
      const int widest = __reduce_max_sync(kFull, assigned);
      if (widest <= 1) {
        break;
      }
      const unsigned ties = __ballot_sync(kFull, assigned == widest);
      if (lane == __ffs(ties) - 1) {
        int fewer = widest - 1;
        if (fewer > 1) {
          const long long perBlock = (count + fewer - 1) / fewer;
          const long long aligned =
              (perBlock + blockSize - 1) / blockSize * blockSize;
          const long long fewest = (count + aligned - 1) / aligned;
          fewer = fewest < 1 ? 1 : static_cast<int>(fewest);
        }
        assigned = fewer;
      }
      excess -= widest - __shfl_sync(kFull, assigned, __ffs(ties) - 1);
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
    if (b * kInfoLongs < headLongs) {
      long long* copy = headCopy + b * kInfoLongs;
      copy[0] = *reinterpret_cast<long long*>(info);
      copy[1] = *reinterpret_cast<long long*>(info + 8);
      copy[2] = *reinterpret_cast<long long*>(info + 16);
      copy[3] = reinterpret_cast<long long>(blockDebug);
      copy[4] = 0;
      copy[5] = 0;
    }
    launchOf[b] = launch;
  }
  if (active) {
    prevCost[lane] = cost;
    prevBlocks[lane] = assigned;
    prevCap[lane] = cap;
    countsOut[lane] = count;
    fullCounts[lane] = count;
  }
  if (tid == 0) {
    state[0] = total;
    state[1] = sum;
    state[kSinceFull] = 0;
    state[kFullLaunchBlocks] = launchBlocks;
    const auto baseWord = reinterpret_cast<unsigned long long>(base);
    const auto debugWord = reinterpret_cast<unsigned long long>(debug);
    state[kFullBase] = static_cast<int>(baseWord);
    state[kFullBase + 1] = static_cast<int>(baseWord >> 32);
    state[kFullDebug] = static_cast<int>(debugWord);
    state[kFullDebug + 1] = static_cast<int>(debugWord >> 32);
    state[kFullValid] = 1;
    ++state[kNumFull];
    state[kFullSettled] = settled ? 1 : 0;
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
    long long* countsOut,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
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
    long long* countsOut,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
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
        1,
        failure,
        firstStep);
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

// One step of a setup batch. As SetupMember in the host code.
struct SetupMember {
  const int* records;
  const long long* code;
  const long long* symbols;
  const int* desc;
  unsigned char* base;
  unsigned char* debug;
  long long* clocks;
  long long* waits;
  float* floats;
  int* ints;
  long long* countsOut;
  long long blockInfoOffset;
  int numRecords;
  int numSymbols;
  int numCode;
  int numWords;
  int n;
  int maxPrevBlocks;
  int launchBlocks;
  int debugBytes;
  int ordinal;
  int oneWarp;
  int waitsBegin;
  int numWaits;
  int signalAfterStores;
  int previousClocks;
};

// The setups of several steps, block i for member i: stepSetup or
// stepSetupWarp over the member's arguments, after the setups of the members
// it waits for have set their flags in 'done' to 'generation'. A member waits
// only for members before it. A member whose later readers load only words
// its store records write sets its flag after those, else at its end.
extern "C" __global__ void stepSetups(
    const SetupMember* members,
    const int* waits,
    int* done,
    int generation,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
  const SetupMember m = members[blockIdx.x];
  if (threadIdx.x == 0) {
    for (int i = 0; i < m.numWaits; ++i) {
      const int* flag = done + waits[m.waitsBegin + i];
      while (loadAcquire(flag) != generation) {
        __nanosleep(64);
      }
    }
    __threadfence();
  }
  __syncthreads();
  SETUP_START();
  extern __shared__ long long setupShared[];
  if (m.oneWarp) {
    const int half = blockDim.x / 2;
    const int codeOffset =
        setupCodeOffset(m.numWords, m.maxPrevBlocks, m.numCode);
    if (threadIdx.x < half) {
      fixupBody(
          m.base,
          m.records,
          m.code,
          m.numRecords,
          m.symbols,
          m.numSymbols,
          codeOffset >= 0 ? setupShared + codeOffset : nullptr,
          m.numCode,
          0,
          half,
          1,
          failure,
          m.ordinal,
          m.signalAfterStores ? done + blockIdx.x : nullptr,
          generation);
    }
    gridWarpBody(
        m.desc,
        m.numWords,
        m.n,
        m.maxPrevBlocks,
        m.base,
        m.blockInfoOffset,
        m.launchBlocks,
        m.debug,
        m.debugBytes,
        m.clocks,
        m.waits,
        m.previousClocks,
        m.floats,
        m.ints,
        m.countsOut,
        half,
        blockDim.x - half,
        2);
  } else {
    fixupBody(
        m.base,
        m.records,
        m.code,
        m.numRecords,
        m.symbols,
        m.numSymbols,
        m.numCode <= kSharedCode ? setupShared : nullptr,
        m.numCode,
        0,
        blockDim.x,
        1,
        failure,
        m.ordinal,
        m.signalAfterStores ? done + blockIdx.x : nullptr,
        generation);
    __syncthreads();
    ParamCounts paramCounts;
    gridBody(
        m.desc,
        m.numWords,
        m.n,
        m.maxPrevBlocks,
        m.base,
        m.blockInfoOffset,
        m.launchBlocks,
        m.debug,
        m.debugBytes,
        m.clocks,
        m.waits,
        m.previousClocks,
        m.floats,
        m.ints,
        m.countsOut,
        paramCounts);
  }
  __syncthreads();
  if (!m.signalAfterStores && threadIdx.x == 0) {
    __threadfence();
    storeRelease(done + blockIdx.x, generation);
  }
}

// A step's setup over several blocks of kWideThreads, from the plan at
// 'plan'; see buildWidePlan in the host code. Block 0 places the allocation
// group members when the plan has them, the next blocks update the tensor
// parameters and run the remaining records, and the last block runs the grid,
// whose counts wait for the others. Every block depends only on blocks before
// it, so the blocks need no co-residency beyond the order they start in.
// flags[b] is block b's flag, flags[gridDim.x] the setups run so far: a
// block sets its flag to that count + 1, and the grid block advances the
// count last. flags[gridDim.x + 1] and the word after are learnMultipliers'. The block's dynamic shared memory holds the grid's arrays or
// the temps, 'areaBytes' of them, then the symbols' values and capacities and
// the plan's shapes.
// One block per SM in the bounds: with two, ptxas holds the kernel to 64
// registers and spills on every role's critical path.
extern "C" __global__ void __launch_bounds__(kWideThreads, 1) wideSetup(
    const int* plan,
    const long long* code,
    const long long* symbols,
    int numSymbols,
    int* flags,
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
    const int* failure,
    int firstStep,
    int areaBytes,
    const int* terms,
    int numTerms) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
  SETUP_START();
  extern __shared__ long long setupShared[];
  const int gridBlock = gridDim.x - 1;
  const int generation = loadAcquire(flags + gridDim.x) + 1;
  long long* values = setupShared + areaBytes / 8;
  long long* caps = values + numSymbols;
  long long* shapes = caps + numSymbols;
  if (blockIdx.x == gridBlock && terms != nullptr) {
    WideCounts counter;
    counter.plan = plan;
    counter.terms = terms;
    counter.numTerms = numTerms;
    counter.code = code;
    counter.symbols = symbols;
    counter.numSymbols = numSymbols;
    counter.params = base;
    counter.values = values;
    counter.caps = caps;
    counter.shapes = shapes;
    counter.termCount = shapes + plan[8];
    counter.launchCounts = counter.termCount + numTerms;
    // 16-byte aligned.
    counter.termDims = reinterpret_cast<int4*>(
        counter.launchCounts + n + ((plan[8] + numTerms + n) & 1));
    counter.termScalar =
        reinterpret_cast<unsigned char*>(counter.termDims + numTerms);
    counter.placedFlag = flags;
    counter.learnFlag = flags + gridDim.x + 1;
    counter.generation = generation;
    counter.begin();
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
        countsOut,
        counter);
  } else if (blockIdx.x == gridBlock) {
    FlagCounts counter{flags, gridBlock, generation};
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
        countsOut,
        counter);
  }
  // The barrier orders the block's writes before thread 0's release.
  if (blockIdx.x == gridBlock) {
    __syncthreads();
    if (threadIdx.x == 0) {
      storeRelease(flags + gridDim.x, generation);
    }
    return;
  }
  // The grid block that evaluates its counts learns no multipliers itself.
  if (terms != nullptr && static_cast<int>(blockIdx.x) == gridBlock - 1) {
    learnMultipliers(
        n,
        maxPrevBlocks,
        launchBlocks,
        deviceDesc,
        stepClocks,
        stepWaits,
        previousClocks,
        floats,
        ints,
        flags + gridDim.x + 1,
        generation,
        setupShared);
    __syncthreads();
  }
  const int placementBlocks = plan[12];
  if (static_cast<int>(blockIdx.x) < placementBlocks) {
    widePlacement(
        plan,
        code,
        symbols,
        numSymbols,
        values,
        caps,
        shapes,
        setupShared,
        failure,
        firstStep,
        ints);
  } else {
    wideTensors(
        blockIdx.x - placementBlocks,
        plan,
        base,
        code,
        symbols,
        numSymbols,
        values,
        caps,
        shapes,
        flags,
        generation,
        ints);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    storeRelease(flags + blockIdx.x, generation);
  }
  if (blockIdx.x == 0) {
    WIDE_MARK(ints, placementBlocks ? 2 : 4);
  } else if (static_cast<int>(blockIdx.x) == gridBlock - 1) {
    WIDE_MARK(ints, 4);
  }
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
              "deviceGrid",
              "stepSetup",
              "deviceGridWarp",
              "stepSetupWarp",
              "stepSetups",
              "wideSetup"};
          return spec;
        });
  });
  return *kernel;
}

int32_t sharedBytes(int32_t numLaunches, int32_t numWords) {
  return kSharedPerLaunch * numLaunches + 4 * numWords;
}

// Whether the grid kernel keeps the descriptor in shared memory; else it reads
// it from global memory and needs only the per-launch arrays there.
bool descriptorInShared(int32_t numLaunches, int32_t numWords) {
  return sharedBytes(numLaunches, numWords) <= kMaxSharedBytes;
}

bool gridFitsShared(int32_t numLaunches) {
  return sharedBytes(numLaunches, 0) <= kMaxSharedBytes;
}

at::Tensor toDevice(const void* data, int64_t numel, at::ScalarType type) {
  return at::from_blob(
             const_cast<void*>(data),
             {std::max<int64_t>(numel, 1)},
             at::TensorOptions().dtype(type))
      .to(at::TensorOptions().device(at::kCUDA).dtype(type));
}

// As SetupMember in the kernel source.
struct SetupMember {
  const int32_t* records;
  const int64_t* code;
  const int64_t* symbols;
  const int32_t* desc;
  uint8_t* base;
  uint8_t* debug;
  int64_t* clocks;
  int64_t* waits;
  float* floats;
  int32_t* ints;
  int64_t* countsOut;
  int64_t blockInfoOffset;
  int32_t numRecords;
  int32_t numSymbols;
  int32_t numCode;
  int32_t numWords;
  int32_t n;
  int32_t maxPrevBlocks;
  int32_t launchBlocks;
  int32_t debugBytes;
  int32_t ordinal;
  int32_t oneWarp;
  int32_t waitsBegin;
  int32_t numWaits;
  int32_t signalAfterStores;
  int32_t previousClocks;
};
static_assert(sizeof(SetupMember) == 152);

// Makes the device allocation of 'grid' on first use: the packed descriptor,
// then per launch the multiplier and previous cost, the ints, and per launch
// the counts and the counts of the last full grid, 8-byte aligned, then the
// copy of the block table's first BlockInfos.
void ensureDeviceGrid(DeviceGrid& grid) {
  if (grid.device.defined()) {
    return;
  }
  const int64_t n = grid.numLaunches;
  const auto numWords = static_cast<int64_t>(grid.words.size());
  grid.floatsOffset = numWords;
  grid.intsOffset = grid.floatsOffset + 3 * n;
  grid.longsOffset =
      (grid.intsOffset + kStateWords + 2 * n + grid.launchBlocks + 1) / 2 * 2;
  const int64_t headLongs = (n + static_cast<int64_t>(sizeof(BlockInfo)) - 1) /
      static_cast<int64_t>(sizeof(BlockInfo)) *
      static_cast<int64_t>(sizeof(BlockInfo) / 8);
  std::vector<int32_t> words(grid.longsOffset + 2 * (2 * n + headLongs), 0);
  std::copy(grid.words.begin(), grid.words.end(), words.begin());
  std::memcpy(
      &words[grid.floatsOffset], grid.multipliers.data(), n * sizeof(float));
  grid.device = toDevice(
      words.data(), static_cast<int64_t>(words.size()), at::ScalarType::Int);
}

// The launch of a step's setup: the kernel entry, its dynamic shared memory,
// whether it runs the grid on one warp, and the descriptor words, negative
// when the descriptor stays in global memory.
struct SetupShape {
  int32_t entry;
  int32_t dynamicShared;
  bool oneWarp;
  int32_t numWords;
};

SetupShape setupShape(const DeviceGrid& grid, const FixupArgs* fixups) {
  const int32_t n = grid.numLaunches;
  int32_t numWords = static_cast<int32_t>(grid.words.size());
  // As warpGridBytes and setupCodeOffset in the kernel source.
  const int32_t warpBytes = 32 * 16 + 12 * grid.launchBlocks + 4 * numWords;
  const bool oneWarp = n <= 32 && warpBytes <= kMaxSharedBytes;
  const int32_t entry = (fixups != nullptr ? 1 : 0) + (oneWarp ? 2 : 0);
  const int32_t codeBytes =
      fixups != nullptr ? DeviceSymbols::fixupSharedBytes(fixups->numCode) : 0;
  int32_t dynamicShared = 0;
  if (oneWarp) {
    // The programs go after the grid's arrays when both fit.
    const int32_t codeOffset = (warpBytes + 7) / 8 * 8;
    dynamicShared = codeBytes > 0 && codeOffset + codeBytes <= kMaxSharedBytes
        ? codeOffset + codeBytes
        : warpBytes;
  } else {
    const bool inShared = descriptorInShared(n, numWords);
    dynamicShared =
        std::max(sharedBytes(n, inShared ? numWords : 0), codeBytes);
    if (!inShared) {
      numWords = -numWords;
    }
  }
  return {entry, dynamicShared, oneWarp, numWords};
}

// Launches the wide setup of 'fixups': its plan's blocks, then the grid
// block. The arguments are runDeviceGrid's. Returns false, launching nothing,
// when its shared memory does not fit the device.
bool runWideSetup(
    DeviceGrid& grid,
    const FixupArgs& fixups,
    uint8_t* deviceBase,
    long long blockInfoOffset,
    int numBlocks,
    unsigned char* debug,
    long long* stepClocks,
    long long* stepWaits,
    int previousClocks,
    facebook::velox::wave::Stream* stream,
    const int32_t* failure,
    int32_t firstStep) {
  auto* desc = grid.device.data_ptr<int32_t>();
  auto* floats = reinterpret_cast<float*>(desc + grid.floatsOffset);
  auto* ints = desc + grid.intsOffset;
  auto* longs = reinterpret_cast<int64_t*>(desc + grid.longsOffset);
  int numLaunches = grid.numLaunches;
  // The descriptor goes to shared memory whatever its size: a wide setup
  // opts in to more than kMaxSharedBytes.
  int numWords = static_cast<int32_t>(grid.words.size());
  const int32_t gridBytes = sharedBytes(numLaunches, numWords);
  if (grid.wideTermsPlan != fixups.wideHost) {
    grid.wideTermsPlan = fixups.wideHost;
    const auto terms =
        DeviceSymbols::wideCountTerms(*fixups.wideHost, grid.words);
    grid.wideTerms = terms.empty() ? at::Tensor()
                                   : toDevice(
                                         terms.data(),
                                         static_cast<int64_t>(terms.size()),
                                         at::ScalarType::Int);
  }
  const int32_t* terms =
      grid.wideTerms.defined() ? grid.wideTerms.data_ptr<int32_t>() : nullptr;
  int numTerms = grid.wideTerms.defined()
      ? static_cast<int32_t>(grid.wideTerms.numel() / 4)
      : 0;
  // As WideCounts in the kernel source.
  constexpr int32_t kTermsPerThread = 4;
  if (numTerms > kTermsPerThread * kGridThreads) {
    terms = nullptr;
    numTerms = 0;
  }
  // 16-byte aligned, so that the term dims after the longs are.
  int areaBytes =
      (std::max(gridBytes, DeviceSymbols::kWideAreaBytes) + 15) / 16 * 16;
  const int32_t dynamicShared = areaBytes +
      static_cast<int32_t>(sizeof(int64_t)) *
          (2 * fixups.numSymbols + fixups.wideShapeWords + numTerms +
           numLaunches + 1) +
      17 * numTerms;
  if (grid.wideSharedBytes != dynamicShared) {
    grid.wideSharedBytes = dynamicShared;
    grid.wideFits = gridKernel().occupancy(5, kGridThreads, dynamicShared) > 0;
  }
  if (!grid.wideFits) {
    return false;
  }
  int maxPrevBlocks = grid.launchBlocks;
  int debugBytes = sizeof(DebugInfo);
  const int32_t* plan = fixups.wide;
  const int64_t* code = fixups.code;
  const int64_t* symbols = fixups.symbols;
  int numSymbols = fixups.numSymbols;
  int32_t* flags = fixups.wideFlags;
  void* args[] = {
      &plan,
      &code,
      &symbols,
      &numSymbols,
      &flags,
      &desc,
      &numWords,
      &numLaunches,
      &maxPrevBlocks,
      &deviceBase,
      &blockInfoOffset,
      &numBlocks,
      &debug,
      &debugBytes,
      &stepClocks,
      &stepWaits,
      &previousClocks,
      &floats,
      &ints,
      &longs,
      &failure,
      &firstStep,
      &areaBytes,
      &terms,
      &numTerms};
  // The sixth entry point.
  gridKernel().launch(
      5, fixups.numWideBlocks, kGridThreads, dynamicShared, stream, args);
  return true;
}

} // namespace

bool deviceGridFits(const GridDescriptor& grid, int32_t numLaunches) {
  if (grid.launches.size() == static_cast<size_t>(numLaunches)) {
    for (const auto& launch : grid.launches) {
      if (launch.count == GridLaunchDesc::Count::kHost) {
        return false;
      }
    }
  }
  return gridFitsShared(numLaunches);
}

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
  if (!gridFitsShared(numLaunches)) {
    return false;
  }
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  words[0] = numLaunches;
  words[1] = grid.targetBlocks;
  words[2] = grid.blockSize;
  words[3] = out.launchBlocks;
  words[4] = termBase;
  words[5] = constBase;
  const auto& config = WaveConfig::get();
  words[8] = config.deviceGridTestExtraBlocks;
  if (config.gridSkipRefresh > 1) {
    words[7] = std::min(config.gridSkipRefresh, 0xffff) |
        (static_cast<int32_t>(
             std::clamp(config.gridSkipTolerance, 0.0f, 1.0f) * 1000)
         << 16);
  }
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
    facebook::velox::wave::Stream* stream,
    const int32_t* failure,
    int32_t firstStep) {
  TORCH_CHECK(deviceClocks != nullptr, "A device grid needs the step's clocks");
  const int32_t n = grid.numLaunches;
  ensureDeviceGrid(grid);
  auto* desc = grid.device.data_ptr<int32_t>();
  auto* floats = reinterpret_cast<float*>(desc + grid.floatsOffset);
  auto* ints = desc + grid.intsOffset;
  auto* longs = reinterpret_cast<int64_t*>(desc + grid.longsOffset);
  long long offset = blockInfoOffset;
  const auto shape = setupShape(grid, fixups);
  int numWords = shape.numWords;
  int numLaunches = n;
  int numBlocks = std::min(launchBlocks, grid.launchBlocks);
  int maxPrevBlocks = grid.launchBlocks;
  int previousClocksArg =
      previousClocks && WaveConfig::get().deviceGridLearn ? 1 : 0;
  int debugBytes = sizeof(DebugInfo);
  auto* debug = reinterpret_cast<unsigned char*>(deviceDebugBase);
  auto* stepClocks = reinterpret_cast<long long*>(deviceClocks);
  auto* stepWaits = stepClocks + launchBlocks;
  // A step the one-warp grid takes sets up faster in its one block: the wide
  // grid block's serial phases cost more than they divide.
  if (fixups != nullptr && fixups->wide != nullptr && !shape.oneWarp &&
      runWideSetup(
          grid,
          *fixups,
          deviceBase,
          blockInfoOffset,
          numBlocks,
          debug,
          stepClocks,
          stepWaits,
          previousClocksArg,
          stream,
          failure,
          firstStep)) {
    return;
  }
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
      &longs,
      &failure,
      &firstStep};
  // deviceGrid takes the arguments after the fixups'. A step of at most 32
  // launches runs on one warp.
  constexpr int32_t kNumFixupArgs = 6;
  gridKernel().launch(
      shape.entry,
      1,
      shape.oneWarp ? kWarpGridThreads : kGridThreads,
      shape.dynamicShared,
      stream,
      fixups != nullptr ? args : args + kNumFixupArgs);
}

void runSetupBatch(
    SetupBatchDevice& batch,
    const std::vector<SetupMemberArgs>& members,
    facebook::velox::wave::Stream* stream,
    const int32_t* failure,
    int32_t firstStep) {
  static_assert(kGridThreads == kWarpGridThreads);
  const auto numMembers = static_cast<int32_t>(members.size());
  std::vector<SetupMember> packed(numMembers);
  std::vector<int32_t> waits;
  int32_t dynamicShared = 0;
  for (int32_t i = 0; i < numMembers; ++i) {
    const auto& member = members[i];
    auto& grid = *member.grid;
    ensureDeviceGrid(grid);
    const auto shape = setupShape(grid, member.fixups);
    dynamicShared = std::max(dynamicShared, shape.dynamicShared);
    auto* desc = grid.device.data_ptr<int32_t>();
    const auto& fixups = *member.fixups;
    auto& m = packed[i];
    m.records = fixups.records;
    m.code = fixups.code;
    m.symbols = fixups.symbols;
    m.desc = desc;
    m.base = member.deviceBase;
    m.debug = reinterpret_cast<uint8_t*>(member.deviceDebugBase);
    TORCH_CHECK(
        member.deviceClocks != nullptr,
        "A setup batch member needs the step's clocks");
    m.clocks = member.deviceClocks;
    m.waits = member.deviceClocks + member.launchBlocks;
    m.floats = reinterpret_cast<float*>(desc + grid.floatsOffset);
    m.ints = desc + grid.intsOffset;
    m.countsOut = reinterpret_cast<int64_t*>(desc + grid.longsOffset);
    m.blockInfoOffset = member.blockInfoOffset;
    m.numRecords = fixups.numRecords;
    m.numSymbols = fixups.numSymbols;
    m.numCode = fixups.numCode;
    m.numWords = shape.numWords;
    m.n = grid.numLaunches;
    m.maxPrevBlocks = grid.launchBlocks;
    m.launchBlocks = std::min(member.launchBlocks, grid.launchBlocks);
    m.debugBytes = sizeof(DebugInfo);
    m.ordinal = member.ordinal;
    m.oneWarp = shape.oneWarp ? 1 : 0;
    m.waitsBegin = static_cast<int32_t>(waits.size());
    m.numWaits = static_cast<int32_t>(member.waitsFor.size());
    m.signalAfterStores = member.signalAfterStores ? 1 : 0;
    m.previousClocks = member.previousClocks ? 1 : 0;
    for (auto wait : member.waitsFor) {
      TORCH_CHECK(wait >= 0 && wait < i, "A setup waits for a later one");
      waits.push_back(wait);
    }
  }
  const auto memberBytes =
      static_cast<int64_t>(packed.size() * sizeof(SetupMember));
  std::vector<uint8_t> words(
      memberBytes + std::max<size_t>(waits.size(), 1) * sizeof(int32_t));
  std::memcpy(words.data(), packed.data(), memberBytes);
  std::memcpy(
      words.data() + memberBytes, waits.data(), waits.size() * sizeof(int32_t));
  // The members change only when a step's buffers or the symbol table move.
  if (words != batch.words) {
    batch.words = std::move(words);
    batch.device = toDevice(
        batch.words.data(),
        static_cast<int64_t>(batch.words.size()),
        at::ScalarType::Byte);
  }
  if (!batch.done.defined() || batch.done.numel() < numMembers) {
    batch.done = at::zeros(
        {numMembers}, at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    batch.generation = 0;
  }
  ++batch.generation;
  auto* membersArg = batch.device.data_ptr<uint8_t>();
  auto* waitsArg = membersArg + memberBytes;
  auto* doneArg = batch.done.data_ptr<int32_t>();
  int generation = batch.generation;
  int firstStepArg = firstStep;
  void* args[] = {
      &membersArg, &waitsArg, &doneArg, &generation, &failure, &firstStepArg};
  // The fifth entry point.
  gridKernel().launch(4, numMembers, kGridThreads, dynamicShared, stream, args);
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
  result.sinceFullGrid = ints[kSinceFullWord];
  result.numSkipped = ints[kNumSkippedWord];
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

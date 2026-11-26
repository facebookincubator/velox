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

/// Device side GPU decode templates, instantiated by run-time generated reader
/// kernels (see ReaderKernel.h). Must only include headers that are
/// registered for NVRTC (see jit/make_headers.sh).

#include "velox/experimental/wave/common/Bits.cuh"
#include "velox/experimental/wave/common/Scan.cuh"
#include "velox/experimental/wave/dwio/decode/DecodeOp.h"

namespace facebook::velox::wave {

__host__ __device__ inline bool isSet(const uint8_t* bits, int32_t idx) {
  return bits[idx / 8] & (1 << (idx & 7));
}

__host__ __device__ inline void setBit(uint8_t* bits, uint32_t idx) {
  bits[idx / 8] |= (1 << (idx % 8));
}

__host__ __device__ inline void clearBit(uint8_t* bits, uint32_t idx) {
  bits[idx / 8] &= ~(1 << (idx % 8));
}

__host__ __device__ inline void
setBit(uint8_t* bits, uint32_t idx, bool value) {
  value ? setBit(bits, idx) : clearBit(bits, idx);
}

namespace detail {

template <typename T, typename U>
struct IsSame {
  static constexpr bool value = false;
};

template <typename T>
struct IsSame<T, T> {
  static constexpr bool value = true;
};

/// Shared memory for a TB wide scan or reduce: one 8 byte slot per warp plus
/// 2 slots for the total. Placed at the start of dynamic shared memory.
template <int32_t kBlockSize>
constexpr int32_t kScanSmemSlots = kBlockSize / kWarpThreads + 2;

template <typename T, int32_t kBlockSize>
inline __device__ T* scanTemp(char* smem) {
  return reinterpret_cast<T*>(smem);
}

template <typename T, int32_t kBlockSize>
inline __device__ T* scanTotal(char* smem) {
  return reinterpret_cast<T*>(
      smem + (kScanSmemSlots<kBlockSize> - 1) * sizeof(int64_t));
}

/// Sum of 'value' over 'kWidth' consecutive lanes. The result is valid in
/// the first lane of each group of 'kWidth' lanes.
template <typename T, int32_t kWidth>
inline __device__ T segmentSum(T value) {
#pragma unroll
  for (int32_t offset = kWidth / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, kWidth);
  }
  return value;
}

/// TB wide sum of 'value'. The result is returned to all threads. 'smem' has
/// kScanSmemSlots<kBlockSize> 8 byte slots. Ends with __syncthreads.
template <typename T, int32_t kBlockSize>
inline __device__ T blockSum(T value, char* smem) {
  constexpr int32_t kNumWarps = kBlockSize / kWarpThreads;
  auto* temp = scanTemp<T, kBlockSize>(smem);
  value = segmentSum<T, kWarpThreads>(value);
  if (kNumWarps == 1) {
    return __shfl_sync(0xffffffff, value, 0);
  }
  if ((threadIdx.x & (kWarpThreads - 1)) == 0) {
    temp[threadIdx.x / kWarpThreads] = value;
  }
  __syncthreads();
  if (threadIdx.x < kWarpThreads) {
    T warpTotal = threadIdx.x < kNumWarps ? temp[threadIdx.x] : 0;
    warpTotal = segmentSum<T, kWarpThreads>(warpTotal);
    if (threadIdx.x == 0) {
      *scanTotal<T, kBlockSize>(smem) = warpTotal;
    }
  }
  __syncthreads();
  T result = *scanTotal<T, kBlockSize>(smem);
  __syncthreads();
  return result;
}
template <typename T>
__device__ void decodeTrivial(GpuDecode::Trivial& op) {
  auto address = reinterpret_cast<uint64_t>(op.input);
  constexpr uint64_t kMask = sizeof(T) - 1;
  int32_t lowShift = (address & kMask) * 8;
  int32_t highShift = sizeof(T) * 8 - lowShift;
  auto source = reinterpret_cast<T*>(address & ~kMask);
  int32_t end = op.end;
  T* result = reinterpret_cast<T*>(op.result);
  auto scatter = op.scatter;
  if (scatter) {
    for (auto i = op.begin + threadIdx.x; i < end; i += blockDim.x) {
      result[scatter[i]] =
          (source[i] >> lowShift) | (source[i + 1] << highShift);
    }
  } else {
    for (auto i = op.begin + threadIdx.x; i < end; i += blockDim.x) {
      result[i] = (source[i] >> lowShift) | (source[i + 1] << highShift);
    }
  }
}

enum class ResultOp { kDict, kBaseline, kDictScatter, kBaselineScatter };

template <typename T, ResultOp op>
__device__ void storeResult(
    int32_t i,
    uint64_t bitfield,
    T baseline,
    const T* dict,
    const int32_t* scatter,
    T* result) {
  if constexpr (op == ResultOp::kBaseline) {
    result[i] = bitfield + baseline;
  } else if constexpr (op == ResultOp::kBaselineScatter) {
    result[scatter[i]] = bitfield + baseline;
  } else if constexpr (op == ResultOp::kDict) {
    result[i] = dict[bitfield + baseline];
  } else if constexpr (op == ResultOp::kDictScatter) {
    result[scatter[i]] = dict[bitfield + baseline];
  }
}

template <typename T, ResultOp resultOp>
__device__ void decodeDictionaryOnBitpack(GpuDecode::DictionaryOnBitpack& op) {
  auto i = op.begin + threadIdx.x;
  auto end = op.end;
  auto address = reinterpret_cast<uint64_t>(op.indices);
  int32_t alignOffset = (address & 7) * 8;
  address &= ~7UL;
  auto words = reinterpret_cast<uint64_t*>(address);
  const T* dict = reinterpret_cast<const T*>(op.alphabet);
  auto scatter = op.scatter;
  auto baseline = op.baseline;
  auto bitWidth = op.bitWidth;
  uint64_t mask = (1LU << bitWidth) - 1;
  auto result = reinterpret_cast<T*>(op.result);
  for (; i < end; i += blockDim.x) {
    int32_t bitIndex = i * bitWidth + alignOffset;
    int32_t wordIndex = bitIndex >> 6;
    int32_t bit = bitIndex & 63;
    uint64_t word = words[wordIndex];
    uint64_t index = word >> bit;
    if (bitWidth + bit > 64) {
      uint64_t nextWord = words[wordIndex + 1];
      index |= nextWord << (64 - bit);
    }
    index &= mask;
    storeResult<T, resultOp>(i, index, baseline, dict, scatter, result);
  }
}

template <typename T>
__device__ void decodeDictionaryOnBitpack(GpuDecode::DictionaryOnBitpack& op) {
  if (!op.scatter) {
    if (op.alphabet) {
      decodeDictionaryOnBitpack<T, ResultOp::kDict>(op);
    } else {
      decodeDictionaryOnBitpack<T, ResultOp::kBaseline>(op);
    }
  } else {
    if (op.alphabet) {
      decodeDictionaryOnBitpack<T, ResultOp::kDictScatter>(op);
    } else {
      decodeDictionaryOnBitpack<T, ResultOp::kBaselineScatter>(op);
    }
  }
}

template <int32_t kBlockSize, typename T>
__device__ int scatterIndices(
    const T* values,
    T value,
    int32_t begin,
    int32_t end,
    int32_t* indices) {
  extern __shared__ char smem[];
  int numMatch;
  bool match;
  int32_t k = 0;
  for (auto j = begin; j < end; j += kBlockSize) {
    auto jt = j + threadIdx.x;
    match = (jt < end && values[jt] == value);
    int subtotal;
    numMatch = exclusiveSum<int32_t, kBlockSize>(
        match,
        scanTotal<int32_t, kBlockSize>(smem),
        scanTemp<int32_t, kBlockSize>(smem));
    subtotal = *scanTotal<int32_t, kBlockSize>(smem);
    __syncthreads();
    if (match) {
      indices[k + numMatch] = jt - begin;
    }
    k += subtotal;
  }
  return k;
}

template <int kBlockSize>
__device__ int scatterIndices(
    const uint8_t* bits,
    bool value,
    int32_t begin,
    int32_t end,
    int32_t* indices) {
  extern __shared__ char smem[];
  constexpr int kPerThread = 8;
  int numMatch[kPerThread];
  bool match[kPerThread];
  int32_t k = 0;
  constexpr auto kBitsPerBlock = kBlockSize * kPerThread;
  for (auto j = begin; j < end; j += kBitsPerBlock) {
    auto jt = j + threadIdx.x * kPerThread;
    int32_t threadCount = 0;
    for (auto i = 0; i < kPerThread; ++i) {
      match[i] = jt + i < end && isSet(bits, jt + i) == value;
      numMatch[i] = threadCount;
      threadCount += match[i];
    }
    int32_t threadBase = exclusiveSum<int32_t, kBlockSize>(
        threadCount,
        scanTotal<int32_t, kBlockSize>(smem),
        scanTemp<int32_t, kBlockSize>(smem));
    int subtotal = *scanTotal<int32_t, kBlockSize>(smem);
    for (auto i = 0; i < kPerThread; ++i) {
      numMatch[i] += threadBase;
    }
    __syncthreads();
    for (auto i = 0; i < kPerThread; ++i) {
      if (match[i]) {
        indices[k + numMatch[i]] = jt + i - begin;
      }
    }
    k += subtotal;
  }
  return k;
}

__device__ inline void
bitpackBools(const bool* input, int count, uint8_t* out) {
  int nbytes = count / 8;
  for (auto i = threadIdx.x; i < nbytes; i += blockDim.x) {
    uint8_t value = 0;
    for (int j = 0; j < 8; ++j) {
      value |= input[8 * i + j] << j;
    }
    out[i] = value;
  }
  if (threadIdx.x == 0 && count % 8 != 0) {
    auto extra = count % 8;
    for (int j = count - extra; j < count; ++j) {
      setBit(out, j, input[j]);
    }
  }
}

__device__ inline void decodeSparseBool(GpuDecode::SparseBool& op) {
  for (auto i = threadIdx.x; i < op.totalCount; i += blockDim.x) {
    op.bools[i] = !op.sparseValue;
  }
  __syncthreads();
  for (auto i = threadIdx.x; i < op.sparseCount; i += blockDim.x) {
    op.bools[op.sparseIndices[i]] = op.sparseValue;
  }
  __syncthreads();
  bitpackBools(op.bools, op.totalCount, op.result);
}

__device__ inline uint32_t readVarint32(const char** pos) {
  uint32_t value = (**pos) & 127;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 7;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 14;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 21;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (*((*pos)++) & 127) << 28;
  return value;
}

__device__ inline uint64_t readVarint64(const char** pos) {
  uint64_t value = (**pos) & 127;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 7;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 14;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= (**pos & 127) << 21;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(**pos & 127) << 28;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(**pos & 127) << 35;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(**pos & 127) << 42;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(**pos & 127) << 49;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(**pos & 127) << 56;
  if (!(*((*pos)++) & 128)) {
    return value;
  }
  value |= static_cast<uint64_t>(*((*pos)++) & 127) << 63;
  return value;
}

template <int kBlockSize, typename T>
__device__ int decodeVarint(
    const char* input,
    int size,
    bool* ends,
    int32_t* endPos,
    T* output) {
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    ends[i] = ~input[i] & 0x80;
  }
  auto numOut = scatterIndices<kBlockSize>(ends, true, 0, size, endPos);
  for (auto i = threadIdx.x; i < numOut; i += blockDim.x) {
    auto* pos = input + (i == 0 ? 0 : (endPos[i - 1] + 1));
    if constexpr (sizeof(T) == 4) {
      output[i] = readVarint32(&pos);
    } else {
      static_assert(sizeof(T) == 8);
      output[i] = readVarint64(&pos);
    }
  }
  return numOut;
}

template <int kBlockSize, typename T>
__device__ void decodeVarintOp(GpuDecode::Varint& op) {
  int resultSize = decodeVarint<kBlockSize, T>(
      op.input, op.size, op.ends, op.endPos, (T*)op.result);
  if (threadIdx.x == 0) {
    op.resultSize = resultSize;
  }
}

template <int kBlockSize, typename T>
__device__ void decodeMainlyConstant(GpuDecode::MainlyConstant& op) {
  auto otherCount = scatterIndices<kBlockSize>(
      op.isCommon, false, 0, op.count, op.otherIndices);
  auto commonValue = *(const T*)op.commonValue;
  auto* otherValues = (const T*)op.otherValues;
  auto* result = (T*)op.result;
  for (auto i = threadIdx.x; i < op.count; i += blockDim.x) {
    result[i] = commonValue;
  }
  __syncthreads();
  for (auto i = threadIdx.x; i < otherCount; i += blockDim.x) {
    result[op.otherIndices[i]] = otherValues[i];
  }
  if (threadIdx.x == 0 && op.otherCount) {
    *op.otherCount = otherCount;
  }
}

template <int kBlockSize, typename T, typename U>
__device__ T sum(const U* values, int size) {
  extern __shared__ char smem[];
  T total = 0;
  for (int i = 0; i < size; i += kBlockSize) {
    auto numValid = min(size - i, kBlockSize);
    T value = 0;
    if (threadIdx.x < numValid) {
      value = values[i + threadIdx.x];
    }
    total += blockSum<T, kBlockSize>(value, smem);
  }
  return total;
}

template <int kBlockSize>
__device__ void rleTotalLength(GpuDecode::RleTotalLength& op) {
  auto result = sum<kBlockSize, int64_t>(op.input, op.count);
  if (threadIdx.x == 0) {
    *op.result = result;
  }
}

template <typename T>
__device__ int upperBound(const T* data, int size, T target) {
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

template <int kBlockSize, typename T>
__device__ void decodeRle(GpuDecode::Rle& op) {
  extern __shared__ char smem[];
  // The scan temp is followed by kBlockSize offsets.
  auto* offsets = reinterpret_cast<int32_t*>(
      smem + kScanSmemSlots<kBlockSize> * sizeof(int64_t));
  auto* values = (const T*)op.values;
  auto* result = (T*)op.result;
  int total = 0;
  for (int i = 0; i < op.count; i += blockDim.x) {
    auto ti = threadIdx.x + i;
    auto len = ti < op.count ? op.lengths[ti] : 0;
    int32_t offset, subtotal;
    __syncthreads();
    offset = inclusiveSum<int32_t, kBlockSize>(
        len,
        scanTotal<int32_t, kBlockSize>(smem),
        scanTemp<int32_t, kBlockSize>(smem));
    subtotal = *scanTotal<int32_t, kBlockSize>(smem);
    __syncthreads();
    offsets[threadIdx.x] = offset;
    __syncthreads();
    for (auto j = threadIdx.x; j < subtotal; j += blockDim.x) {
      result[total + j] = values
          [i +
           upperBound(
               offsets, static_cast<int>(blockDim.x), static_cast<int32_t>(j))];
    }
    total += subtotal;
  }
}

template <int kBlockSize>
__device__ void makeScatterIndices(GpuDecode::MakeScatterIndices& op) {
  auto indicesCount = scatterIndices<kBlockSize>(
      op.bits, op.findSetBits, op.begin, op.end, op.indices);
  if (threadIdx.x == 0 && op.indicesCount) {
    *op.indicesCount = indicesCount;
  }
}

template <typename T>
inline __device__ T loadBits(
    void const* p,
    uint32_t idx,
    uint32_t width = 8 * sizeof(T),
    int64_t baseline = 0) {
  if (sizeof(T) == 4 || width <= 32) {
    return loadBits32(p, idx * width, width) + baseline;
  } else {
    return loadBits64(p, idx * width, width) + baseline;
  }
}

template <typename T, DecodeStep kEncoding>
inline __device__ T randomAccessDecode(const GpuDecode* op, int32_t idx) {
  switch (kEncoding) {
    case DecodeStep::kDictionaryOnBitpack: {
      const auto& d = op->data.dictionaryOnBitpack;
      auto width = d.bitWidth;
      if (sizeof(T) == 4 || width <= 32) {
        return loadBits32(d.indices, idx * width, width) + d.baseline;
      } else {
        return loadBits64(d.indices, idx * width, width) + d.baseline;
      }
    }
  }
}

template <typename T, WaveFilterKind kFilterKind, bool kFixedFilter = true>
__device__ bool testFilter(const GpuDecode* op, T data) {
  switch (kFixedFilter ? kFilterKind : op->filterKind) {
    case WaveFilterKind::kDictFilter: {
      return (op->filterBitmap[data >> 5] & (1 << (data & 31))) != 0;
    }
    case WaveFilterKind::kBigintRange: {
      long2 bounds = *reinterpret_cast<const long2*>(&op->filter);
      return data >= bounds.x && data <= bounds.y;
    }
    default:
      return true;
  }
}

// Returns true if the warp of 'row' is below or contains the warp of 'maxRow'.
inline __device__ bool warpInRange(int32_t row, int32_t maxRow) {
  return (row & ~31U) < maxRow;
}

template <
    typename T,
    typename IndexT,
    int32_t kBlockSize,
    bool kHasFilter,
    bool kHasResult,
    bool kHasNulls,
    DictMode kDictMode>
__device__ void makeResult(
    const GpuDecode* op,
    T data,
    int32_t row,
    bool filterPass,
    int32_t nthLoop,
    uint8_t nullFlag,
    int32_t* temp) {
  auto base = nthLoop * kBlockSize;
  if (kHasFilter && kDictMode == DictMode::kRecordFilter &&
      warpInRange(op->baseRow + base + threadIdx.x, op->maxRow)) {
    if (op->filterBitmap == nullptr) {
      // Never executes. But is needed for -O3 not to die by bad compilation.
      printf("bad %d", __LINE__); // assert(false);
    }
    uint32_t filters = __ballot_sync(0xffffffff, filterPass);
    if ((threadIdx.x & 31) == 0) {
      op->filterBitmap[(threadIdx.x + base) / 32] = filters;
    }
    if (filterPass) {
      reinterpret_cast<T*>(op->result)[base + threadIdx.x] = data;
    }
    return;
  }

  constexpr bool kAlwaysDict = !IsSame<T, IndexT>::value;
  if (kHasFilter) {
    int32_t resultIdx = exclusiveSum<int16_t, kBlockSize>(
        static_cast<int16_t>(filterPass),
        nullptr,
        reinterpret_cast<int16_t*>(temp));
    if (threadIdx.x == kBlockSize - 1) {
      op->blockStatus[nthLoop].numRows = resultIdx + filterPass;
      if (op->filterRowCount) {
        op->filterRowCount[nthLoop] = resultIdx + filterPass;
      }
    }
    if (filterPass) {
      auto* alphabet =
          reinterpret_cast<const T*>(op->data.dictionaryOnBitpack.alphabet);
      resultIdx += base;
      op->resultRows[resultIdx] = row;
      if (kHasResult) {
        reinterpret_cast<T*>(op->result)[resultIdx] =
            (kAlwaysDict || alphabet) ? alphabet[data] : data;
        if (kHasNulls && op->resultNulls) {
          op->resultNulls[resultIdx] = nullFlag;
        }
      }
    }
  } else {
    if (!filterPass) {
      // In the no filter case, filterPass is false for lanes that are after the
      // last row.
      return;
    }
    auto* alphabet =
        reinterpret_cast<const T*>(op->data.dictionaryOnBitpack.alphabet);
    auto resultIdx = base + threadIdx.x;
    reinterpret_cast<T*>(op->result)[resultIdx] =
        (kAlwaysDict || alphabet) ? alphabet[data] : data;
    if (kHasNulls && op->resultNulls) {
      op->resultNulls[resultIdx] = nullFlag;
    }
  }
}

template <
    typename T,
    int32_t kBlockSize,
    DecodeStep kEncoding,
    WaveFilterKind kFilterKind,
    bool kHasResult,
    typename IndexT,
    NullMode kNullMode,
    DictMode kDictMode>
__device__ void decodeSelective(GpuDecode* op) {
  int32_t nthLoop = 0;
  switch (kNullMode) {
    case NullMode::kDenseNonNull: {
      if (kFilterKind == WaveFilterKind::kAlwaysTrue) {
        //  No-filter case with everything inlined.
        auto base = op->baseRow;
        auto i = threadIdx.x;
        auto& d = op->data.dictionaryOnBitpack;
        auto end = op->maxRow - op->baseRow;
        auto bitWidth = d.bitWidth;
        auto address = reinterpret_cast<uint64_t>(d.indices);
        auto baseline = d.baseline;
        if (bitWidth < 32) {
          int32_t alignOffset = (address & 3) * 8;
          address &= ~3UL;
          auto words = reinterpret_cast<uint32_t*>(address);
          uint32_t mask = (1L << bitWidth) - 1;
          auto* result = reinterpret_cast<T*>(op->result);
          auto* alphabet =
              reinterpret_cast<const T*>(op->data.dictionaryOnBitpack.alphabet);
          for (; i < end; i += blockDim.x) {
            int32_t bitIndex = (i + base) * bitWidth + alignOffset;
            int32_t wordIndex = bitIndex >> 5;
            int32_t bit = bitIndex & 31;
            uint32_t word = words[wordIndex];
            uint32_t index = __funnelshift_r(
                word, (bitWidth + bit > 32 ? words[wordIndex + 1] : 0), bit);
            index &= mask;
            result[i] =
                alphabet ? alphabet[index + baseline] : index + baseline;
          }
        } else {
          int32_t alignOffset = (address & 7) * 8;
          address &= ~7UL;
          auto words = reinterpret_cast<uint64_t*>(address);
          uint64_t mask = (1LU << bitWidth) - 1;
          auto* result = reinterpret_cast<T*>(op->result);
          for (; i < end; i += blockDim.x) {
            int32_t bitIndex = (i + base) * bitWidth + alignOffset;
            int32_t wordIndex = bitIndex >> 6;
            int32_t bit = bitIndex & 63;
            uint64_t word = words[wordIndex];
            uint64_t index = word >> bit;
            if (bitWidth + bit > 64) {
              uint64_t nextWord = words[wordIndex + 1];
              index |= nextWord << (64 - bit);
            }
            index &= mask;
            result[i] = index + baseline;
          }
        }
      } else {
        do {
          auto row = threadIdx.x + op->baseRow + nthLoop * kBlockSize;
          bool filterPass = false;
          IndexT data{};
          if (row < op->maxRow) {
            data = randomAccessDecode<IndexT, kEncoding>(op, row);
            filterPass = testFilter<IndexT, kFilterKind, true>(op, data);
          }
          makeResult<
              T,
              IndexT,
              kBlockSize,
              kFilterKind != WaveFilterKind::kAlwaysTrue,
              kHasResult,
              false,
              kDictMode>(
              op, data, row, filterPass, nthLoop, kNotNull, op->temp);
        } while (++nthLoop < op->numRowsPerThread);
      }
      break;
    }
    case NullMode::kSparseNonNull:
      do {
        int32_t numRows = op->blockStatus[nthLoop].numRows;
        bool filterPass = false;
        IndexT data{};
        int32_t row = 0;
        if (threadIdx.x < numRows) {
          row = op->rows[threadIdx.x + nthLoop * kBlockSize];
          data = randomAccessDecode<IndexT, kEncoding>(op, row);
          filterPass = testFilter<IndexT, kFilterKind, true>(op, data);
        }
        makeResult<
            T,
            IndexT,
            kBlockSize,
            kFilterKind != WaveFilterKind::kAlwaysTrue,
            kHasResult,
            false,
            kDictMode>(op, data, row, filterPass, nthLoop, kNotNull, op->temp);
      } while (++nthLoop < op->numRowsPerThread);
      break;
    case NullMode::kDenseNullable: {
      int32_t maxRow = op->maxRow;
      int32_t dataIdx = 0;
      auto* state = reinterpret_cast<NonNullState*>(op->temp);
      if (threadIdx.x == 0 && op->isNullsBitmap) {
        state->nonNullsBelow = op->nthBlock == 0
            ? 0
            : op->nonNullBases
                  [op->nthBlock *
                       (op->gridNumRowsPerThread / (1024 / kBlockSize)) -
                   1];
        state->nonNullsBelowRow =
            op->gridNumRowsPerThread * op->nthBlock * kBlockSize;
      }
      __syncthreads();
      do {
        int32_t base = op->baseRow + nthLoop * kBlockSize;
        bool filterPass = false;
        int32_t dataIdx;
        IndexT data{};
        if (base < maxRow) {
          if (op->isNullsBitmap) {
            dataIdx = nonNullIndex256(
                op->nulls, base, min(kBlockSize, maxRow - base), state);
          } else {
            dataIdx = base + threadIdx.x < maxRow
                ? (op->nulls[base + threadIdx.x] ? base + threadIdx.x : -1)
                : -1;
          }
          filterPass = base + threadIdx.x < maxRow;
          if (filterPass) {
            if (dataIdx == -1) {
              if (!op->nullsAllowed) {
                filterPass = false;
              }
            } else {
              data = randomAccessDecode<IndexT, kEncoding>(op, dataIdx);
              filterPass = testFilter<IndexT, kFilterKind, true>(op, data);
            }
          }
        }
        // nonNullIndex256() has just been read out of state->temp, which
        // makeResult() reuses as scan storage.
        __syncthreads();
        makeResult<
            T,
            IndexT,
            kBlockSize,
            kFilterKind != WaveFilterKind::kAlwaysTrue,
            kHasResult,
            true,
            kDictMode>(
            op,
            data,
            base + threadIdx.x,
            filterPass,
            nthLoop,
            dataIdx == -1 ? kNull : kNotNull,
            state->temp);
      } while (++nthLoop < op->numRowsPerThread);
      break;
    }
    case NullMode::kSparseNullable: {
      auto state = reinterpret_cast<NonNullState*>(op->temp);
      if (threadIdx.x == 0 && op->isNullsBitmap) {
        state->nonNullsBelow = op->nthBlock == 0
            ? 0
            : op->nonNullBases
                  [op->nthBlock *
                       (op->gridNumRowsPerThread / (1024 / kBlockSize)) -
                   1];
        state->nonNullsBelowRow =
            op->gridNumRowsPerThread * op->nthBlock * kBlockSize;
      }
      __syncthreads();
      do {
        int32_t base = kBlockSize * nthLoop;
        int32_t numRows = op->blockStatus[nthLoop].numRows;
        if (numRows == 0) {
        } else {
          bool filterPass = true;
          IndexT data{};
          int32_t dataIdx;
          if (op->isNullsBitmap) {
            dataIdx = nonNullIndex256Sparse(
                op->nulls, op->rows + base, numRows, state);
          } else {
            dataIdx = threadIdx.x < numRows
                ? (op->nulls[op->rows[base + threadIdx.x]]
                       ? op->rows[base + threadIdx.x]
                       : -1)
                : -1;
          }
          filterPass = threadIdx.x < numRows;
          if (filterPass) {
            if (dataIdx == -1) {
              if (!op->nullsAllowed) {
                filterPass = false;
              }
            } else {
              data = randomAccessDecode<IndexT, kEncoding>(op, dataIdx);
              filterPass = testFilter<IndexT, kFilterKind, true>(op, data);
            }
          }
          // state->temp is shared between nonNullIndex256Sparse() and the
          // scan in makeResult(), in both directions across iterations.
          __syncthreads();
          makeResult<
              T,
              IndexT,
              kBlockSize,
              kFilterKind != WaveFilterKind::kAlwaysTrue,
              kHasResult,
              true,
              kDictMode>(
              op,
              data,
              op->rows[base + threadIdx.x],
              filterPass,
              nthLoop,
              dataIdx == -1 ? kNull : kNotNull,
              state->temp);
          __syncthreads();
        }
      } while (++nthLoop < op->numRowsPerThread);
      break;
    }
  }
  __syncthreads();
}

// Returns the position of 'target' in 'data' to 'data + size'. Not finding the
// value is an error and the values are expected to be unique.
inline __device__ int
findRow(const int32_t* rows, int32_t size, int32_t row, GpuDecode* op) {
  int lo = 0, hi = size;
  while (lo < hi) {
    int i = (lo + hi) / 2;
    if (rows[i] == row) {
      return i;
    }
    if (rows[i] < row) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  printf("Expecting to find  row %d in findRow() size %d %p\n", row, size, op);
  assert(false);
}

template <typename T, int32_t kBlockSize>
__device__ void compactValues(GpuDecode* op) {
  auto& compact = op->data.compact;
  int32_t nthLoop = 0;
  do {
    auto numRows = op->blockStatus[nthLoop].numRows;
    T sourceValue;
    uint8_t sourceNull;
    int32_t base;
    if (threadIdx.x < numRows) {
      base = nthLoop * kBlockSize;
      auto row = compact.finalRows[base + threadIdx.x];
      auto numSource = compact.sourceNumRows[nthLoop];
      auto sourceRow = findRow(compact.sourceRows + base, numSource, row, op);
      sourceValue =
          reinterpret_cast<const T*>(compact.source)[base + sourceRow];
      if (compact.sourceNull) {
        sourceNull = compact.sourceNull[base + sourceRow];
      }
    }
    __syncthreads();
    if (threadIdx.x < numRows) {
      reinterpret_cast<T*>(compact.source)[base + threadIdx.x] = sourceValue;
      if (compact.sourceNull) {
        compact.sourceNull[base + threadIdx.x] = sourceNull;
      }
    }
  } while (++nthLoop < op->numRowsPerThread);
}

template <int kBlockSize>
__device__ void setRowCountNoFilter(GpuDecode::RowCountNoFilter& op) {
  auto numRows = op.numRows;
  auto* status = op.status;
  auto numBlocks = roundUp(numRows, kBlockSize) / kBlockSize;
  if (op.gridStatusSize > 0) {
    auto grid = roundUp(
        reinterpret_cast<uintptr_t>(status) + numBlocks * sizeof(BlockStatus),
        8);
    int64_t* statusEnd = reinterpret_cast<int64_t*>(grid + op.gridStatusSize);
    auto ptr = reinterpret_cast<int64_t*>(grid) + threadIdx.x;
    for (; ptr < statusEnd; ptr += kBlockSize) {
      *ptr = 0;
    }
    if (op.gridOnly) {
      return;
    }
  }
  for (auto base = 0; base < numBlocks; base += kBlockSize) {
    auto idx = threadIdx.x + base;
    if (idx < numBlocks) {
      // Every thread writes a row count for kBlockSize rows.  all row
      // counts except the last are kBlockSize. The next kernel sets
      // lane status to active for rows below rowCount and to inactive
      // for others.
      status[idx].numRows =
          idx < numBlocks - 1 ? kBlockSize : numRows - idx * kBlockSize;
    }
  }
}

template <int32_t kBlockSize, int32_t kWidth>
inline __device__ void reduceCase(
    int32_t cnt,
    int32_t nthLoop,
    int32_t numResults,
    int32_t* results,
    int32_t* temp) {
  static_assert(kWidth == 4 || kWidth == 8 || kWidth == 16 || kWidth == 32);
  auto sum = segmentSum<int32_t, kWidth>(cnt);
  constexpr int32_t kResultsPerLoop = kBlockSize / kWidth;

  if ((threadIdx.x & (kWidth - 1)) == 0) {
    temp[threadIdx.x / kWidth] = sum;
  }
  __syncthreads();
  // Add up the temps.
  sum = threadIdx.x < kResultsPerLoop ? temp[threadIdx.x] : 0;
  if (threadIdx.x == 0 && nthLoop > 0) {
    sum += results[nthLoop * kResultsPerLoop - 1];
  }
  auto result = inclusiveSum<int32_t, kBlockSize / kWidth>(
      threadIdx.x < kResultsPerLoop ? sum : 0, nullptr, temp);
  auto resultIdx = threadIdx.x + nthLoop * kResultsPerLoop;
  if (resultIdx < numResults) {
    results[resultIdx] = result;
  }
}

/// Counts set bits in runs of kStride bits. kStride is 256, 512, 1024 or 2048.
template <int kBlockSize, int32_t kStride>
__device__ void countBits(GpuDecode& step) {
  auto& op = step.data.countBits;
  auto numBits = op.numBits;
  bool aligned = (reinterpret_cast<uintptr_t>(op.bits) & 7) == 0;
  int32_t numWords = roundUp(op.numBits, 64) / 64;
  int32_t numResults = (numBits - 1) / kStride;
  auto* bits = reinterpret_cast<const uint64_t*>(op.bits);
  for (auto i = 0; i < numBits; i += 64 * kBlockSize) {
    auto idx = threadIdx.x + (i / 64);
    int32_t cnt = 0;
    if (idx < numWords) {
      if (aligned) {
        cnt = __popcll(bits[idx]);
      } else {
        cnt = __popcll(unalignedLoad64(bits + idx));
      }
    }
    reduceCase<kBlockSize, kStride / 64>(
        cnt,
        i / (64 * kBlockSize),
        numResults,
        reinterpret_cast<int32_t*>(step.result),
        step.temp);
  }
}

__device__ inline int32_t
rowExists(int32_t row, const int32_t* rows, int32_t numRows) {
  int lo = 0, hi = numRows;
  while (lo < hi) {
    int i = (lo + hi) / 2;
    if (rows[i] == row) {
      return i;
    }
    if (rows[i] < row) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return -1;
}

// returns the index of the matching row in the rows array if exists . -1
// if not exists.
template <int32_t kBlockSize>
__device__ int32_t rowExists(int32_t nthLoop, GpuDecode* op) {
  auto row = threadIdx.x + op->data.selectiveChunked.chunkStart + op->baseRow +
      nthLoop * kBlockSize; // global row id
  auto tileSize = kBlockSize * op->numRowsPerThread;
  auto resultBlockId = row / tileSize;
  auto loopId = (row % tileSize) / kBlockSize;
  auto resultStart = resultBlockId * tileSize + loopId * kBlockSize;
  auto numRows =
      op->filterRowCount[resultBlockId * op->numRowsPerThread + loopId];
  auto match = rowExists(row, op->rows + resultStart, numRows);
  return match == -1 ? match : resultStart + match;
}

template <typename T, int32_t kBlockSize, bool kHasNulls>
__device__ void selectiveFilterChunked(GpuDecode* op) {
  int32_t nthLoop = 0;
  if constexpr (kHasNulls) {
    int32_t maxRow = op->maxRow;
    auto* state = reinterpret_cast<NonNullState*>(op->temp);
    if (threadIdx.x == 0) {
      state->nonNullsBelow = op->nthBlock == 0
          ? 0
          : op->nonNullBases
                [op->nthBlock *
                     (op->gridNumRowsPerThread / (1024 / kBlockSize)) -
                 1];
      state->nonNullsBelowRow =
          op->gridNumRowsPerThread * op->nthBlock * kBlockSize;
    }
    __syncthreads();
    do {
      auto base = op->baseRow + nthLoop * kBlockSize;
      int32_t dataIdx = nonNullIndex256(
          op->nulls, base, min(kBlockSize, maxRow - base), state);

      auto row = threadIdx.x + base; // local row id wrt the chunk
      auto resultIdx =
          row < op->maxRow ? rowExists<kBlockSize>(nthLoop, op) : -1;
      if (resultIdx != -1) {
        if (dataIdx != -1) {
          auto data = loadBits<T>(op->data.selectiveChunked.input, dataIdx);
          reinterpret_cast<T*>(op->result)[resultIdx] = data;
        }
        op->resultNulls[resultIdx] = (dataIdx == -1 ? kNull : kNotNull);
      }
    } while (++nthLoop < op->numRowsPerThread);
  } else {
    do {
      auto row = threadIdx.x + op->baseRow +
          nthLoop * kBlockSize; // local row id wrt the chunk

      auto resultIdx =
          row < op->maxRow ? rowExists<kBlockSize>(nthLoop, op) : -1;
      if (resultIdx != -1) {
        auto data = loadBits<T>(op->data.selectiveChunked.input, row);
        reinterpret_cast<T*>(op->result)[resultIdx] = data;
      }
    } while (++nthLoop < op->numRowsPerThread);
  }
  __syncthreads();
}

} // namespace detail
} // namespace facebook::velox::wave

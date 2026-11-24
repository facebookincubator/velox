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

/// Device side decoding of columns that does not depend on the file format:
/// the rank bitmap of null flags, the mapping of top level rows to column rows
/// through enclosing structs and arrays, the result writes of selective
/// decoding, and the element ranges of arrays and maps. A file format provides
/// accessors for its encodings and a format type for decodeColumn(). Must only
/// include headers that are registered for NVRTC (see jit/make_headers.sh).

#include "velox/experimental/wave/common/StringView.h"
#include "velox/experimental/wave/dwio/decode/DecodeOps.cuh"

namespace facebook::velox::wave::detail {

/// Rows per entry in the rank counts of a RankBitmap.
constexpr int32_t kRankWindow = 256;

/// Layout of a bitmap with rank counts: an int32 count of set bits before
/// each kRankWindow rows, then the bitmap.
struct RankBitmap {
  __device__ RankBitmap(const char* aux, int32_t numRows)
      : counts(reinterpret_cast<const int32_t*>(aux)),
        bits(reinterpret_cast<const uint64_t*>(aux + countBytes(numRows))) {}

  static __host__ __device__ int32_t numWindows(int32_t numRows) {
    return (numRows + kRankWindow - 1) / kRankWindow;
  }

  static __host__ __device__ int32_t countBytes(int32_t numRows) {
    return roundUp(numWindows(numRows) * sizeof(int32_t), sizeof(uint64_t));
  }

  /// Size of the aux data for 'numRows' rows.
  static __host__ __device__ int32_t size(int32_t numRows) {
    return countBytes(numRows) + numWindows(numRows) * (kRankWindow / 8);
  }

  __device__ bool isSet(int32_t row) const {
    return (bits[row >> 6] >> (row & 63)) & 1;
  }

  /// Returns the number of set bits below 'row'.
  __device__ int32_t rank(int32_t row) const {
    int32_t result = counts[row / kRankWindow];
    for (auto word = (row / kRankWindow) * (kRankWindow / 64); word < row >> 6;
         ++word) {
      result += __popcll(bits[word]);
    }
    return result + __popcll(bits[row >> 6] & lowMask<uint64_t>(row & 63));
  }

  const int32_t* counts;
  const uint64_t* bits;
};

/// Returns a StringView of 'size' bytes at 'data'. Strings of up to 12 bytes
/// are inline. A longer string points to 'data', which must stay live. See
/// common/StringView.h.
inline __device__ StringView makeStringView(const char* data, int32_t size) {
  StringView result;
  result.init(data, size);
  return result;
}

/// Tests the filter of 'op' on 'value'. Strings have no filters.
template <typename R, WaveFilterKind kFilterKind>
struct ColumnFilter {
  static __device__ bool test(const GpuDecode* op, R value) {
    return testFilter<R, kFilterKind, true>(op, value);
  }
};

template <WaveFilterKind kFilterKind>
struct ColumnFilter<StringView, kFilterKind> {
  static __device__ bool test(const GpuDecode* /*op*/, StringView /*value*/) {
    return true;
  }
};

/// Returns the chunk of column row 'row'. The chunks of the op cover 'row'.
inline __device__ const ChunkDesc& findChunk(
    const ColumnAccess& column,
    int32_t row) {
  int32_t low = 0;
  int32_t high = column.numChunks - 1;
  while (low < high) {
    const int32_t middle = (low + high + 1) / 2;
    if (column.chunks[middle].firstRow <= row) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return column.chunks[low];
}

/// Maps a top level row to the row of the column through the nulls of the
/// enclosing structs. Returns false if an enclosing struct is null.
inline __device__ bool levelRow(const ColumnAccess& column, int32_t& row) {
  for (auto i = 0; i < column.numLevels; ++i) {
    RankBitmap nonNulls(column.levelAux[i], column.levelRows[i]);
    if (!nonNulls.isSet(row)) {
      return false;
    }
    row = nonNulls.rank(row);
  }
  return true;
}

/// Writes the value, null flag and row of a filter result or the value and
/// null flag of a non-filter result for the 'nthLoop'th kBlockSize rows of
/// the TB.
template <typename T, int32_t kBlockSize, bool kHasFilter, bool kHasResult>
__device__ void writeResult(
    const GpuDecode* op,
    T data,
    bool nonNull,
    int32_t row,
    bool pass,
    int32_t nthLoop) {
  const int32_t base = nthLoop * kBlockSize;
  auto resultIdx = base + threadIdx.x;
  if (kHasFilter) {
    resultIdx = exclusiveSum<int16_t, kBlockSize>(
        static_cast<int16_t>(pass),
        nullptr,
        reinterpret_cast<int16_t*>(op->temp));
    if (threadIdx.x == kBlockSize - 1) {
      op->blockStatus[nthLoop].numRows = resultIdx + pass;
      if (op->filterRowCount) {
        op->filterRowCount[nthLoop] = resultIdx + pass;
      }
    }
    resultIdx += base;
    if (pass) {
      op->resultRows[resultIdx] = row;
    }
  }
  // Without a filter, lanes after the last row have 'pass' false.
  if (pass && kHasResult) {
    reinterpret_cast<T*>(op->result)[resultIdx] = data;
    if (op->resultNulls) {
      op->resultNulls[resultIdx] = nonNull ? kNotNull : kNull;
    }
  }
}

/// Returns the top level row of the 'nthLoop'th kBlockSize rows of the TB for
/// this thread and sets 'valid' to false for a thread without a row. Dense
/// rows run from 'baseRow' to 'maxRow'. Sparse rows are the rows passing the
/// previous filter in 'rows', with the row counts in 'blockStatus'.
template <int32_t kBlockSize>
__device__ int32_t
opRow(const GpuDecode* op, bool sparse, int32_t nthLoop, bool& valid) {
  if (sparse) {
    valid = threadIdx.x < op->blockStatus[nthLoop].numRows;
    return valid ? op->rows[threadIdx.x + nthLoop * kBlockSize] : 0;
  }
  // Row numbers are int32_t.
  int32_t row;
  row = op->baseRow + nthLoop * kBlockSize + threadIdx.x;
  valid = row < op->maxRow;
  return row;
}

/// Selective decode of the values of a column. Each row is read from the
/// chunk that contains it. 'Format' provides
///   static Context context(const GpuDecode*)
///   template <typename R, typename... Accessors>
///   static bool read(int32_t shape, const Context&, int32_t root, int32_t
///   row, R& value)
///   template <typename R, WaveFilterKind, typename... Accessors>
///   static bool readFiltered(int32_t shape, const Context&, int32_t root,
///   int32_t row, const GpuDecode*, R& value, bool& nonNull)
/// where 'shape' selects the accessor of the chunk's encoding tree from
/// 'Accessors'. read() returns false for a null. readFiltered() returns true
/// if the row passes the filter.
template <
    typename Format,
    typename T,
    int32_t kBlockSize,
    WaveFilterKind kFilterKind,
    bool kHasResult,
    bool kSparse,
    typename... Accessors>
__device__ void decodeColumn(GpuDecode* op) {
  constexpr bool kHasFilter = kFilterKind != WaveFilterKind::kAlwaysTrue;
  const auto& column = op->column();
  const auto context = Format::context(op);
  int32_t nthLoop = 0;
  do {
    bool valid;
    const int32_t row = opRow<kBlockSize>(op, kSparse, nthLoop, valid);
    T data{};
    bool pass = false;
    bool nonNull = true;
    int32_t columnRow =
        valid && column.elementRows ? column.elementRows[row] : row;
    if (valid && !levelRow(column, columnRow)) {
      // An enclosing struct is null.
      valid = false;
      nonNull = false;
      pass = kHasFilter ? op->nullsAllowed : true;
    }
    if (valid) {
      const auto& chunk = findChunk(column, columnRow);
      if (kHasFilter) {
        pass = Format::template readFiltered<T, kFilterKind, Accessors...>(
            chunk.shape,
            context,
            chunk.root,
            columnRow - chunk.firstRow,
            op,
            data,
            nonNull);
      } else {
        nonNull = Format::template read<T, Accessors...>(
            chunk.shape, context, chunk.root, columnRow - chunk.firstRow, data);
        pass = true;
      }
    }
    writeResult<T, kBlockSize, kHasFilter, kHasResult>(
        op, data, nonNull, row, pass, nthLoop);
  } while (++nthLoop < op->numRowsPerThread);
  __syncthreads();
}

/// Sets bits [begin, end) of 'bits' to 'value'. Other threads may set bits in
/// the same words.
inline __device__ void
setBitRange(uint64_t* bits, int32_t begin, int32_t end, bool value) {
  while (begin < end) {
    const int32_t word = begin >> 6;
    const int32_t wordEnd = min(end, (word + 1) * 64);
    uint64_t mask =
        lowMask<uint64_t>(wordEnd - word * 64) & ~lowMask<uint64_t>(begin & 63);
    auto* address = reinterpret_cast<unsigned long long*>(&bits[word]);
    if (value) {
      atomicOr(address, mask);
    } else {
      atomicAnd(address, ~mask);
    }
    begin = wordEnd;
  }
}

/// Sets bits [first, first + numRows) of 'bits' from 'value' for the words
/// of the range, one word per thread. 'value(lo)' returns 64 bits starting at
/// row 'lo' of the range. Other words of 'bits' are not changed.
template <int32_t kBlockSize, typename Value>
__device__ void
setBitsFrom(uint64_t* bits, int32_t first, int32_t numRows, Value value) {
  const int32_t end = first + numRows;
  for (int32_t word = (first >> 6) + threadIdx.x; word < (end + 63) >> 6;
       word += kBlockSize) {
    const int32_t lo = max(first, word * 64);
    const int32_t hi = min(end, word * 64 + 64);
    const int32_t shift = lo - word * 64;
    const uint64_t mask = lowMask<uint64_t>(hi - lo) << shift;
    const uint64_t data = (value(lo - first) << shift) & mask;
    if (mask == ~0ULL) {
      bits[word] = data;
    } else {
      auto* address = reinterpret_cast<unsigned long long*>(&bits[word]);
      atomicAnd(address, ~mask);
      atomicOr(address, data);
    }
  }
}

/// Makes the count of set bits before each kRankWindow rows of a RankBitmap.
template <int32_t kBlockSize>
__device__ void rankCounts(
    int32_t numRows,
    const uint64_t* bits,
    int32_t* counts,
    int32_t* temp) {
  __shared__ int32_t total;
  constexpr int32_t kWords = kRankWindow / 64;
  const int32_t numWindows = RankBitmap::numWindows(numRows);
  int32_t start = 0;
  for (int32_t first = 0; first < numWindows; first += kBlockSize) {
    const auto window = first + threadIdx.x;
    int32_t count = 0;
    if (window < numWindows) {
      for (auto i = 0; i < kWords; ++i) {
        count += __popcll(bits[window * kWords + i]);
      }
    }
    const int32_t below =
        start + exclusiveSum<int32_t, kBlockSize>(count, &total, temp);
    if (window < numWindows) {
      counts[window] = below;
    }
    start += total;
    __syncthreads();
  }
}

/// Reads the element start, size and null flag of arrays or maps, or the
/// null flag of structs, for the rows of the op from the aux data made by the
/// prepare ops of the lengths or nulls column. For lengths, the aux data is
/// the start of the elements of each column row, the end of the last row,
/// then a null flag per row. For struct nulls, it is a RankBitmap of non-null
/// rows. Top level rows are dense from 'baseRow' to 'maxRow' or sparse after a
/// filter. Rows below an array or map are dense and map to column rows
/// through 'elementRows'. Writes the total size of the TB's rows to
/// 'totals[nthBlock]'.
template <int32_t kBlockSize>
__device__ void nestedRows(GpuDecode* op) {
  const auto& column = op->column();
  const bool sparse = op->nullMode == NullMode::kSparseNonNull;
  auto* starts = reinterpret_cast<const int32_t*>(column.aux);
  auto* streamNulls =
      reinterpret_cast<const uint8_t*>(starts + column.streamRows + 1);
  __shared__ int32_t loopTotal;
  int32_t tbTotal = 0;
  int32_t nthLoop = 0;
  do {
    bool valid;
    const int32_t row = opRow<kBlockSize>(op, sparse, nthLoop, valid);
    int32_t columnRow =
        valid && column.elementRows ? column.elementRows[row] : row;
    bool nonNull = valid && levelRow(column, columnRow);
    int32_t start = 0;
    int32_t size = 0;
    if (column.isStruct) {
      if (nonNull) {
        RankBitmap nonNulls(column.aux, column.streamRows);
        nonNull = nonNulls.isSet(columnRow);
      }
    } else if (nonNull) {
      nonNull = streamNulls[columnRow] == kNotNull;
      start = starts[columnRow];
      size = starts[columnRow + 1] - start;
    }
    const auto resultIdx = nthLoop * kBlockSize + threadIdx.x;
    if (resultIdx < column.numResults) {
      if (!column.isStruct) {
        column.offsets[resultIdx] = start;
        column.sizes[resultIdx] = size;
      }
      if (op->resultNulls && valid) {
        op->resultNulls[resultIdx] = nonNull ? kNotNull : kNull;
      }
    }
    if (!column.isStruct) {
      inclusiveSum<int32_t, kBlockSize>(size, &loopTotal, op->temp);
      tbTotal += loopTotal;
      __syncthreads();
    }
  } while (++nthLoop < op->numRowsPerThread);
  if (!column.isStruct && threadIdx.x == 0) {
    column.totals[op->nthBlock] = tbTotal;
  }
  __syncthreads();
}

/// Replaces the element starts in 'offsets' with offsets in the element
/// vector from 'base' and writes the element column row of each element.
template <int32_t kBlockSize>
__device__ void expandLengths(GpuDecode* op) {
  const auto& expand = op->data.expandLengths;
  __shared__ int32_t loopTotal;
  int32_t base = expand.base;
  for (int32_t first = 0; first < expand.numRows; first += kBlockSize) {
    const auto idx = first + threadIdx.x;
    const int32_t size = idx < expand.numRows ? expand.sizes[idx] : 0;
    const int32_t offset =
        base + exclusiveSum<int32_t, kBlockSize>(size, &loopTotal, op->temp);
    if (idx < expand.numRows) {
      const int32_t start = expand.offsets[idx];
      expand.offsets[idx] = offset;
      for (auto i = 0; i < size; ++i) {
        expand.rows[offset + i] = start + i;
      }
    }
    base += loopTotal;
    __syncthreads();
  }
}

} // namespace facebook::velox::wave::detail

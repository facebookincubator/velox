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

/// Device side access to Nimble encoding trees. Each encoding is an accessor
/// type with
///   template <typename R> static R get(context, node, row)
/// that returns the value of 'row' of the encoding at node 'node' as R. A
/// chunk's tree is the nested accessor type, e.g.
/// NDict<NFbw<uint32_t>, NFbw<uint64_t>>. Nulls and filters of the top of a
/// tree go through NTop. Must only include headers that are registered for
/// NVRTC (see jit/make_headers.sh).

#include "velox/experimental/wave/dwio/decode/ColumnOps.cuh"

namespace facebook::velox::wave::detail {

/// Nimble EncodingType values.
struct NimbleEncoding {
  static constexpr uint8_t kTrivial = 0;
  static constexpr uint8_t kRle = 1;
  static constexpr uint8_t kDictionary = 2;
  static constexpr uint8_t kFixedBitWidth = 3;
  static constexpr uint8_t kNullable = 5;
  static constexpr uint8_t kSparseBool = 6;
  static constexpr uint8_t kConstant = 9;
  static constexpr uint8_t kMainlyConstant = 10;
};

/// Nimble DataType values.
struct NimbleDataType {
  static constexpr uint8_t kInt32 = 5;
  static constexpr uint8_t kUint32 = 6;
  static constexpr uint8_t kInt64 = 7;
  static constexpr uint8_t kUint64 = 8;
  static constexpr uint8_t kString = 12;
};

struct NimbleContext {
  const NimbleNode* nodes;
  const char* base;
  const char* aux;
};

/// Trivial and FixedBitWidth: 'bitWidth' bit values plus a baseline. T is the
/// physical type, uint32_t or uint64_t. Trivial has the full width and a 0
/// baseline.
template <typename T>
struct NFbw {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    return static_cast<R>(
        loadBits<T>(c.base + n.dataOffset, row, n.bitWidth, n.value));
  }
};

struct NConstant {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t) {
    return static_cast<R>(c.nodes[node].value);
  }
};

template <typename Indices, typename Alphabet>
struct NDict {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    auto index = Indices::template get<uint32_t>(c, n.children[1], row);
    return Alphabet::template get<R>(c, n.children[0], index);
  }
};

/// Values of runs. The run of a row is found by binary search in the run
/// ends made by the prepare op.
template <typename Values>
struct NRle {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    auto* ends = reinterpret_cast<const int32_t*>(c.aux + n.aux);
    int32_t low = 0;
    int32_t high = c.nodes[n.children[0]].numRows - 1;
    while (low < high) {
      const int32_t middle = (low + high) / 2;
      if (ends[middle] < row) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    return Values::template get<R>(c, n.children[1], low);
  }
};

/// The common value or the next of the other values. The aux bitmap has the
/// rows with other values.
template <typename Others>
struct NMainlyConstant {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    RankBitmap others(c.aux + n.aux, n.numRows);
    if (!others.isSet(row)) {
      return static_cast<R>(n.value);
    }
    return Others::template get<R>(c, n.children[1], others.rank(row));
  }
};

/// Non-null values and nulls. The aux bitmap has the non-null rows.
template <typename Values>
struct NNullable {};

/// Trivial strings: the bytes of all strings and an encoding of their lengths.
/// The aux data is the offset of each string and the end of the last one.
/// Values are StringViews.
struct NString {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    auto* offsets = reinterpret_cast<const int32_t*>(c.aux + n.aux);
    const int32_t begin = offsets[row];
    return static_cast<R>(makeStringView(
        c.base + n.dataOffset + begin, offsets[row + 1] - begin));
  }
};

/// A constant string. 'value' is its size and 'dataOffset' its position.
struct NConstantString {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t) {
    const auto& n = c.nodes[node];
    return static_cast<R>(
        makeStringView(c.base + n.dataOffset, static_cast<int32_t>(n.value)));
  }
};

/// MainlyConstant of strings. The common string is like NConstantString.
template <typename Others>
struct NMainlyConstantString {
  template <typename R>
  static __device__ R get(const NimbleContext& c, int32_t node, int32_t row) {
    const auto& n = c.nodes[node];
    RankBitmap others(c.aux + n.aux, n.numRows);
    if (!others.isSet(row)) {
      return static_cast<R>(
          makeStringView(c.base + n.dataOffset, static_cast<int32_t>(n.value)));
    }
    return Others::template get<R>(c, n.children[1], others.rank(row));
  }
};

/// Reads the top of a chunk's tree, with nulls and filters. 'read' returns
/// false for a null. 'readFiltered' returns true if the row passes the filter
/// and sets 'nonNull'.
template <typename Accessor>
struct NTop {
  template <typename R>
  static __device__ bool
  read(const NimbleContext& c, int32_t node, int32_t row, R& value) {
    value = Accessor::template get<R>(c, node, row);
    return true;
  }

  template <typename R, WaveFilterKind kFilterKind>
  static __device__ bool readFiltered(
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    value = Accessor::template get<R>(c, node, row);
    nonNull = true;
    return ColumnFilter<R, kFilterKind>::test(op, value);
  }
};

template <typename Values>
struct NTop<NNullable<Values>> {
  template <typename R>
  static __device__ bool
  read(const NimbleContext& c, int32_t node, int32_t row, R& value) {
    const auto& n = c.nodes[node];
    RankBitmap nonNulls(c.aux + n.aux, n.numRows);
    if (!nonNulls.isSet(row)) {
      return false;
    }
    return NTop<Values>::template read<R>(
        c, n.children[0], nonNulls.rank(row), value);
  }

  template <typename R, WaveFilterKind kFilterKind>
  static __device__ bool readFiltered(
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    const auto& n = c.nodes[node];
    RankBitmap nonNulls(c.aux + n.aux, n.numRows);
    if (!nonNulls.isSet(row)) {
      nonNull = false;
      return op->nullsAllowed;
    }
    return NTop<Values>::template readFiltered<R, kFilterKind>(
        c, n.children[0], nonNulls.rank(row), op, value, nonNull);
  }
};

/// Tests the filter on the alphabet bitmap made by the prepare op and reads
/// the value only for passing rows.
template <typename Indices, typename Alphabet>
struct NTop<NDict<Indices, Alphabet>> {
  template <typename R>
  static __device__ bool
  read(const NimbleContext& c, int32_t node, int32_t row, R& value) {
    value = NDict<Indices, Alphabet>::template get<R>(c, node, row);
    return true;
  }

  template <typename R, WaveFilterKind kFilterKind>
  static __device__ bool readFiltered(
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    const auto& n = c.nodes[node];
    nonNull = true;
    if (n.aux < 0) {
      value = NDict<Indices, Alphabet>::template get<R>(c, node, row);
      return ColumnFilter<R, kFilterKind>::test(op, value);
    }
    auto index = Indices::template get<uint32_t>(c, n.children[1], row);
    auto* passing = reinterpret_cast<const uint64_t*>(c.aux + n.aux);
    if (((passing[index >> 6] >> (index & 63)) & 1) == 0) {
      return false;
    }
    value = Alphabet::template get<R>(c, n.children[0], index);
    return true;
  }
};

template <typename R, WaveFilterKind kFilterKind>
__device__ bool
testWithVerdict(const NimbleNode& n, const GpuDecode* op, R value) {
  if (n.flags & NimbleNodeFlags::kHasVerdict) {
    return (n.flags & NimbleNodeFlags::kPasses) != 0;
  }
  return ColumnFilter<R, kFilterKind>::test(op, value);
}

template <>
struct NTop<NConstant> {
  template <typename R>
  static __device__ bool
  read(const NimbleContext& c, int32_t node, int32_t row, R& value) {
    value = NConstant::template get<R>(c, node, row);
    return true;
  }

  template <typename R, WaveFilterKind kFilterKind>
  static __device__ bool readFiltered(
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    nonNull = true;
    value = NConstant::template get<R>(c, node, row);
    return testWithVerdict<R, kFilterKind>(c.nodes[node], op, value);
  }
};

template <typename Others>
struct NTop<NMainlyConstant<Others>> {
  template <typename R>
  static __device__ bool
  read(const NimbleContext& c, int32_t node, int32_t row, R& value) {
    value = NMainlyConstant<Others>::template get<R>(c, node, row);
    return true;
  }

  template <typename R, WaveFilterKind kFilterKind>
  static __device__ bool readFiltered(
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    const auto& n = c.nodes[node];
    nonNull = true;
    RankBitmap others(c.aux + n.aux, n.numRows);
    if (!others.isSet(row)) {
      value = static_cast<R>(n.value);
      return testWithVerdict<R, kFilterKind>(n, op, value);
    }
    value = Others::template get<R>(c, n.children[1], others.rank(row));
    return ColumnFilter<R, kFilterKind>::test(op, value);
  }
};

/// Reads the top of the tree of accessor 'shape' of Accessors.
template <typename R, typename First, typename... Rest>
__device__ bool nimbleRead(
    int32_t shape,
    const NimbleContext& c,
    int32_t node,
    int32_t row,
    R& value) {
  if constexpr (sizeof...(Rest) == 0) {
    return NTop<First>::template read<R>(c, node, row, value);
  } else {
    if (shape == 0) {
      return NTop<First>::template read<R>(c, node, row, value);
    }
    return nimbleRead<R, Rest...>(shape - 1, c, node, row, value);
  }
}

template <
    typename R,
    WaveFilterKind kFilterKind,
    typename First,
    typename... Rest>
__device__ bool nimbleReadFiltered(
    int32_t shape,
    const NimbleContext& c,
    int32_t node,
    int32_t row,
    const GpuDecode* op,
    R& value,
    bool& nonNull) {
  if constexpr (sizeof...(Rest) == 0) {
    return NTop<First>::template readFiltered<R, kFilterKind>(
        c, node, row, op, value, nonNull);
  } else {
    if (shape == 0) {
      return NTop<First>::template readFiltered<R, kFilterKind>(
          c, node, row, op, value, nonNull);
    }
    return nimbleReadFiltered<R, kFilterKind, Rest...>(
        shape - 1, c, node, row, op, value, nonNull);
  }
}

/// The format type of decodeColumn() for Nimble encoding trees.
struct NimbleFormat {
  static __device__ NimbleContext context(const GpuDecode* op) {
    const auto& nimble = op->data.nimble;
    return {nimble.nodes, nimble.column.base, nimble.column.aux};
  }

  template <typename R, typename... Accessors>
  static __device__ bool read(
      int32_t shape,
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      R& value) {
    return nimbleRead<R, Accessors...>(shape, c, node, row, value);
  }

  template <typename R, WaveFilterKind kFilterKind, typename... Accessors>
  static __device__ bool readFiltered(
      int32_t shape,
      const NimbleContext& c,
      int32_t node,
      int32_t row,
      const GpuDecode* op,
      R& value,
      bool& nonNull) {
    return nimbleReadFiltered<R, kFilterKind, Accessors...>(
        shape, c, node, row, op, value, nonNull);
  }
};
/// Returns value 'row' of a Trivial, FixedBitWidth or Constant node of
/// unsigned 32 or 64 bit values. Used by prepare ops, where the encoding of
/// a child is not a template parameter.
inline __device__ uint64_t
nimbleScalar(const NimbleContext& c, int32_t node, int32_t row) {
  const auto& n = c.nodes[node];
  if (n.encoding == NimbleEncoding::kConstant) {
    return n.value;
  }
  return loadBits<uint64_t>(c.base + n.dataOffset, row, n.bitWidth, n.value);
}

/// Returns value 'row' of an unsigned integer node of any encoding except
/// Nullable. The aux data of RLE and MainlyConstant nodes must be made.
/// Dictionaries nest up to kDepth levels.
template <int32_t kDepth = 2>
__device__ uint64_t
nimbleUint(const NimbleContext& c, int32_t node, int32_t row) {
  for (;;) {
    const auto& n = c.nodes[node];
    switch (n.encoding) {
      case NimbleEncoding::kDictionary:
        if constexpr (kDepth > 0) {
          row = static_cast<int32_t>(
              nimbleUint<kDepth - 1>(c, n.children[1], row));
        } else {
          row = static_cast<int32_t>(nimbleScalar(c, n.children[1], row));
        }
        node = n.children[0];
        continue;
      case NimbleEncoding::kRle: {
        auto* ends = reinterpret_cast<const int32_t*>(c.aux + n.aux);
        int32_t low = 0;
        int32_t high = c.nodes[n.children[0]].numRows - 1;
        while (low < high) {
          const int32_t middle = (low + high) / 2;
          if (ends[middle] < row) {
            low = middle + 1;
          } else {
            high = middle;
          }
        }
        node = n.children[1];
        row = low;
        continue;
      }
      case NimbleEncoding::kMainlyConstant: {
        RankBitmap others(c.aux + n.aux, n.numRows);
        if (!others.isSet(row)) {
          return n.value;
        }
        row = others.rank(row);
        node = n.children[1];
        continue;
      }
      default:
        return nimbleScalar(c, node, row);
    }
  }
}

/// Like nimbleUint() but 'node' may be Nullable. Returns false for a null.
inline __device__ bool nimbleNullableUint(
    const NimbleContext& c,
    int32_t node,
    int32_t row,
    uint64_t& value) {
  const auto& n = c.nodes[node];
  if (n.encoding == NimbleEncoding::kNullable) {
    RankBitmap nonNulls(c.aux + n.aux, n.numRows);
    if (!nonNulls.isSet(row)) {
      return false;
    }
    node = n.children[0];
    row = nonNulls.rank(row);
  }
  value = nimbleUint(c, node, row);
  return true;
}

/// Returns 'bits' as the signed or unsigned integer of the node's data type.
inline __device__ int64_t nimbleInteger(uint8_t dataType, uint64_t bits) {
  switch (dataType) {
    case NimbleDataType::kInt32:
      return static_cast<int32_t>(bits);
    case NimbleDataType::kUint32:
      return static_cast<uint32_t>(bits);
    default:
      return static_cast<int64_t>(bits);
  }
}

/// Sets bits [first, first + numRows) of 'bits' from a bool node. A bit is
/// set for a row whose value is 'polarity'.
template <int32_t kBlockSize>
__device__ void nimbleBoolBits(
    const NimbleContext& c,
    int32_t node,
    int32_t first,
    int32_t numRows,
    bool polarity,
    uint64_t* bits,
    int32_t* temp) {
  const auto& n = c.nodes[node];
  switch (n.encoding) {
    case NimbleEncoding::kTrivial:
      setBitsFrom<kBlockSize>(bits, first, numRows, [&](int32_t row) {
        auto value = loadBits64(c.base + n.dataOffset, row, 64);
        return polarity ? value : ~value;
      });
      break;
    case NimbleEncoding::kConstant: {
      const uint64_t value = (n.value != 0) == polarity ? ~0ULL : 0;
      setBitsFrom<kBlockSize>(
          bits, first, numRows, [&](int32_t) { return value; });
      break;
    }
    case NimbleEncoding::kSparseBool: {
      // Rows at the positions have 'value', the others the opposite.
      const bool sparse = n.value != 0;
      const uint64_t others = sparse == polarity ? 0 : ~0ULL;
      setBitsFrom<kBlockSize>(
          bits, first, numRows, [&](int32_t) { return others; });
      __syncthreads();
      const int32_t numPositions = c.nodes[n.children[0]].numRows;
      for (auto i = threadIdx.x; i < numPositions; i += kBlockSize) {
        const auto position =
            static_cast<int32_t>(nimbleScalar(c, n.children[0], i));
        if (position < numRows) {
          setBitRange(
              bits, first + position, first + position + 1, sparse == polarity);
        }
      }
      break;
    }
    case NimbleEncoding::kRle: {
      // Runs alternate starting with the initial value in 'value'.
      __shared__ int32_t total;
      const int32_t numRuns = c.nodes[n.children[0]].numRows;
      const bool initial = n.value != 0;
      int32_t start = 0;
      for (int32_t firstRun = 0; firstRun < numRuns; firstRun += kBlockSize) {
        const auto runIdx = firstRun + threadIdx.x;
        const int32_t length = runIdx < numRuns
            ? static_cast<int32_t>(nimbleScalar(c, n.children[0], runIdx))
            : 0;
        const int32_t offset =
            start + exclusiveSum<int32_t, kBlockSize>(length, &total, temp);
        if (runIdx < numRuns) {
          const bool value = (initial != ((runIdx & 1) != 0)) == polarity;
          setBitRange(bits, first + offset, first + offset + length, value);
        }
        start += total;
        __syncthreads();
      }
      break;
    }
  }
  __syncthreads();
}

/// Makes the element starts and null flags of the rows of the lengths stream
/// of an array or map in 'offsets': the exclusive sum of the lengths of the
/// stream rows before each row, then the end of the last row, then a null
/// flag per row. Runs in one TB after the prepare ops of the stream's nodes.
template <int32_t kBlockSize>
__device__ void prepareNimbleLengths(GpuDecode* op) {
  const auto& nimble = op->data.nimble;
  const NimbleContext c{nimble.nodes, nimble.column.base, nimble.column.aux};
  auto* offsets = nimble.column.offsets;
  auto* nulls =
      reinterpret_cast<uint8_t*>(offsets + nimble.column.streamRows + 1);
  __shared__ int32_t total;
  int32_t start = 0;
  for (auto i = 0; i < nimble.column.numChunks; ++i) {
    const auto& chunk = nimble.column.chunks[i];
    const int32_t numRows = c.nodes[chunk.root].numRows;
    for (int32_t first = 0; first < numRows; first += kBlockSize) {
      const auto row = first + threadIdx.x;
      uint64_t length = 0;
      bool nonNull = false;
      if (row < numRows) {
        nonNull = nimbleNullableUint(c, chunk.root, row, length);
        nulls[chunk.firstRow + row] = nonNull ? kNotNull : kNull;
      }
      const int32_t offset =
          start +
          exclusiveSum<int32_t, kBlockSize>(
              nonNull ? static_cast<int32_t>(length) : 0, &total, op->temp);
      if (row < numRows) {
        offsets[chunk.firstRow + row] = offset;
      }
      start += total;
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    offsets[nimble.column.streamRows] = start;
  }
  __syncthreads();
}

/// Makes the aux data of node 'op->data.nimble.node'. Runs in one TB.
template <int32_t kBlockSize>
__device__ void prepareNimbleNode(GpuDecode* op) {
  const auto& nimble = op->data.nimble;
  const NimbleContext c{nimble.nodes, nimble.column.base, nimble.column.aux};
  if (nimble.node == GpuDecode::Nimble::kLengths) {
    prepareNimbleLengths<kBlockSize>(op);
    return;
  }
  if (nimble.node == GpuDecode::Nimble::kStructNulls) {
    // The chunks are the bool encodings of the nulls of a struct. The bitmap
    // of non-null rows is at the start of the aux data.
    const auto& last = nimble.column.chunks[nimble.column.numChunks - 1];
    const int32_t numRows = last.firstRow + c.nodes[last.root].numRows;
    auto* bits = reinterpret_cast<uint64_t*>(
        nimble.column.aux + RankBitmap::countBytes(numRows));
    for (auto i = 0; i < nimble.column.numChunks; ++i) {
      const auto& chunk = nimble.column.chunks[i];
      nimbleBoolBits<kBlockSize>(
          c,
          chunk.root,
          chunk.firstRow,
          c.nodes[chunk.root].numRows,
          true,
          bits,
          op->temp);
    }
    rankCounts<kBlockSize>(
        numRows, bits, reinterpret_cast<int32_t*>(nimble.column.aux), op->temp);
    return;
  }
  const auto& n = c.nodes[nimble.node];
  char* aux = nimble.column.aux + n.aux;
  switch (n.encoding) {
    case NimbleEncoding::kTrivial: {
      // Strings. The offset of each string and the end of the last.
      auto* offsets = reinterpret_cast<int32_t*>(aux);
      __shared__ int32_t total;
      int32_t start = 0;
      for (int32_t first = 0; first < n.numRows; first += kBlockSize) {
        const auto row = first + threadIdx.x;
        const int32_t length = row < n.numRows
            ? static_cast<int32_t>(nimbleUint(c, n.children[0], row))
            : 0;
        const int32_t offset =
            start + exclusiveSum<int32_t, kBlockSize>(length, &total, op->temp);
        if (row <= n.numRows) {
          offsets[row] = offset;
        }
        start += total;
        __syncthreads();
      }
      if (n.numRows % kBlockSize == 0 && threadIdx.x == 0) {
        offsets[n.numRows] = start;
      }
      break;
    }
    case NimbleEncoding::kNullable:
    case NimbleEncoding::kMainlyConstant: {
      const bool isNullable = n.encoding == NimbleEncoding::kNullable;
      auto* counts = reinterpret_cast<int32_t*>(aux);
      auto* bits =
          reinterpret_cast<uint64_t*>(aux + RankBitmap::countBytes(n.numRows));
      // Nullable sets non-null rows, MainlyConstant sets rows that are not
      // common.
      nimbleBoolBits<kBlockSize>(
          c,
          n.children[isNullable ? 1 : 0],
          0,
          n.numRows,
          isNullable,
          bits,
          op->temp);
      rankCounts<kBlockSize>(n.numRows, bits, counts, op->temp);
      break;
    }
    case NimbleEncoding::kRle: {
      __shared__ int32_t total;
      auto* ends = reinterpret_cast<int32_t*>(aux);
      const int32_t numRuns = c.nodes[n.children[0]].numRows;
      int32_t start = 0;
      for (int32_t first = 0; first < numRuns; first += kBlockSize) {
        const auto runIdx = first + threadIdx.x;
        const int32_t length = runIdx < numRuns
            ? static_cast<int32_t>(nimbleScalar(c, n.children[0], runIdx))
            : 0;
        const int32_t end =
            start + inclusiveSum<int32_t, kBlockSize>(length, &total, op->temp);
        if (runIdx < numRuns) {
          ends[runIdx] = end - 1;
        }
        start += total;
        __syncthreads();
      }
      break;
    }
    case NimbleEncoding::kDictionary: {
      const auto& alphabet = c.nodes[n.children[0]];
      auto* passing = reinterpret_cast<uint64_t*>(aux);
      for (int32_t first = 0; first < alphabet.numRows; first += kBlockSize) {
        const auto i = first + threadIdx.x;
        bool pass = false;
        if (i < alphabet.numRows) {
          const auto value = nimbleInteger(
              alphabet.dataType, nimbleScalar(c, n.children[0], i));
          pass = testFilter<int64_t, WaveFilterKind::kBigintRange, true>(
              op, value);
        }
        const uint32_t ballot = __ballot_sync(0xffffffff, pass);
        if ((threadIdx.x & 31) == 0) {
          reinterpret_cast<uint32_t*>(passing)[i / 32] = ballot;
        }
      }
      break;
    }
  }
  __syncthreads();
}

} // namespace facebook::velox::wave::detail

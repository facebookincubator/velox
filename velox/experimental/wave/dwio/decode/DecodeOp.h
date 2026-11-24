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

/// Device-visible description of GPU decode ops. Included by host code and by
/// run-time generated reader kernels, so no host-only includes here.

#include "velox/experimental/wave/common/CompilerDefines.h"
#include "velox/experimental/wave/vector/Operand.h"

namespace facebook::velox::wave {

#ifndef __CUDACC_RTC__
class ColumnReader;
class Stream;
#endif

enum class NullMode : uint8_t {
  kDenseNonNull,
  kSparseNonNull,
  kDenseNullable,
  kSparseNullable
};

enum class WaveFilterKind : uint8_t {
  kAlwaysTrue,
  kNotNull,
  kNull,
  kBigintRange,
  kDoubleRange,
  kFloatRange,
  kBigintValues,
  kDictFilter
};

struct alignas(16) WaveFilterBase {
  union {
    int64_t int64Range[2];
    float floatRange[2];
    double doubleRange[2];
    struct {
      int32_t size;
      void* table;
    } values;
  } _;
  // flags for float/double range.
  bool lowerUnbounded;
  bool upperUnbounded;
  bool lowerExclusive;
  bool upperExclusive;
};

/// Node of a Nimble encoding tree in device memory. Offsets are relative to
/// the start of the stream so that the descriptors have no pointers.
struct NimbleNode {
  /// Baseline of FixedBitWidth, value of Constant. Physical bits, zero
  /// extended.
  int64_t value;
  /// Offset of the payload from the start of the stream.
  int64_t dataOffset;
  int32_t numRows;
  /// Indices of child nodes in the node array of the stream, -1 if none.
  /// Dictionary: alphabet, indices. Nullable: values, nulls. MainlyConstant:
  /// isCommon, other values. RLE: run lengths, run values. SparseBool:
  /// positions.
  int32_t children[2];
  /// Offset of data made by a prepare op in the aux area of the column, -1 if
  /// none. Nullable and MainlyConstant: non-null (other value) count before
  /// each 256 rows followed by a bitmap of non-null (other value) rows. RLE:
  /// inclusive end row of each run. Dictionary with a filter: bitmap of
  /// alphabet entries that pass.
  int32_t aux;
  uint8_t bitWidth;
  /// Nimble EncodingType.
  uint8_t encoding;
  /// Nimble DataType.
  uint8_t dataType;
  /// Bits of NimbleNodeFlags.
  uint8_t flags;
};

/// Flags of a NimbleNode.
struct NimbleNodeFlags {
  /// The filter of the column was evaluated on the host for 'value' of a
  /// Constant or the common value of a MainlyConstant.
  static constexpr uint8_t kHasVerdict = 1;
  /// 'value' passes the filter. Valid with kHasVerdict.
  static constexpr uint8_t kPasses = 2;
};

/// Maximum number of nullable structs enclosing a column.
constexpr int32_t kMaxLevels = 3;

/// A separately encoded range of rows of a column in device memory, e.g. a
/// chunk of a Nimble stream.
struct ChunkDesc {
  /// First row of the chunk in the column.
  int32_t firstRow;
  /// Index of the root node of the chunk's encoding tree in the node array of
  /// the column.
  int32_t root;
  /// Position of the accessor of the chunk's tree in the accessor list of the
  /// op.
  int32_t shape;
  int32_t padding;
};

/// The part of a decode op of a column that does not depend on the file
/// format: where the encoded rows are, how top level rows map to rows of the
/// column, and the results of arrays, maps and structs below the top level.
/// It is the first member of the op data of each format, see
/// GpuDecode::column().
struct ColumnAccess {
  /// Start of the encoded data in device memory.
  const char* base;
  /// The chunks that overlap the rows of the TB, in row order.
  const ChunkDesc* chunks;
  /// Data made by prepare ops.
  char* aux;
  /// Aux data of the nulls of each enclosing nullable struct, outermost
  /// first. A row of a struct maps to the row of its child by rank in the
  /// bitmap. See RankBitmap.
  const char* levelAux[kMaxLevels];
  /// Column rows of the rows of an op below an array or map. Row i of the op
  /// reads column row elementRows[i]. nullptr for top level rows.
  const int32_t* elementRows;
  /// kNested of an array or map: the start of the elements of each row in the
  /// element column, the number of elements of each row and the total number
  /// of elements of the TB at 'totals[nthBlock]'.
  int32_t* offsets;
  int32_t* sizes;
  int32_t* totals;
  int32_t numChunks;
  /// Host side id of the accessor list of the op. See accessorList().
  int32_t accessorsId;
  /// Number of enclosing nullable structs of the column.
  int32_t numLevels;
  /// Number of rows of each enclosing struct.
  int32_t levelRows[kMaxLevels];
  /// Rows of the lengths or struct nulls column.
  int32_t streamRows;
  /// kNested: Number of rows of the result vectors after the first row of the
  /// op.
  int32_t numResults;
  /// kNested: true for the nulls of a struct, false for the lengths of an
  /// array or map.
  bool isStruct;
};

/// Instructions for GPU decode.  This can be decoding,
/// or pre/post processing other than decoding.
enum class DecodeStep {
  kSelective32,
  kCompact64,
  kSelective64,
  kSelective32Chunked,
  kSelective64Chunked,
  kConstant32,
  kConstant64,
  kConstantChar,
  kConstantBool,
  kConstantBytes,
  kTrivial,
  kTrivialNoOp,
  kMainlyConstant,
  kBitpack32,
  kBitpack64,
  kRleTotalLength,
  kRleBool,
  kRle,
  kDictionary,
  kDictionaryOnBitpack,
  kVarint,
  kNullable,
  kSentinel,
  kSparseBool,
  kMakeScatterIndices,
  kScatter32,
  kScatter64,
  kLengthToOffset,
  kMissing,
  kStruct,
  kArray,
  kMap,
  kFlatMap,
  kFlatMapNode,
  kRowCountNoFilter,
  kCountBits,
  /// Encoding of a selective op over Nimble encoding trees. See
  /// GpuDecode::Nimble.
  kNimble,
  /// Makes the aux data of one node of a Nimble stream. See NimbleNode::aux.
  kNimblePrepare,
  /// Reads the element range and null flag of arrays and maps or the null
  /// flag of structs at the rows of the op from the aux data made by the
  /// prepare ops. Format independent. See ColumnAccess.
  kNested,
  /// Turns the element starts and sizes of arrays into offsets in the element
  /// vector and writes the element row of each element. See
  /// GpuDecode::ExpandLengths.
  kExpandLengths,
  kUnsupported,
};

enum class DictMode {
  // Decoded values are returned as is
  kNone,
  // Decoded values are indices into a dictionary.
  kDict,
  // Decoded values are indices into a dictionary and into a bitmap where 1
  // means filter passed.
  kDictFilter,
  // Decoded values are raw values and the filter result is to be stored into a
  // filter pass bitmap.
  kRecordFilter
};

/// Describes a decoding loop's input and result disposition.
struct alignas(16) GpuDecode {
  /// Constant in 'numRows' to signify the number comes from 'blockstatus'.
  static constexpr int32_t kFilterHits = -1;

  // The operation to perform. Decides which branch of the union to use.
  DecodeStep step{};
  DecodeStep encoding{};

  WaveTypeKind dataType{};

  /// If false, implies a not null filter. If there is a filter, specifies
  /// whether nulls pass.
  bool nullsAllowed{true};

  WaveFilterKind filterKind{WaveFilterKind::kAlwaysTrue};

  NullMode nullMode{};

  /// Specifies use of dictionary.
  DictMode dictMode{DictMode::kNone};

  /// Number of chunks (e.g. Parquet pages). If > 1, different rows row ranges
  /// have different encodings. The first chunk's encoding is in 'data'. The
  /// next chunk's encoding is in the next GpuDecode's 'data'. Each chunk has
  /// its own 'nulls'. The input row numbers and output data/row numbers are
  /// given by the first GpuDecode.
  uint8_t numChunks{1};

  /// Index of this op's instantiation in the compiled reader kernel. Set at
  /// launch time from the normalized set of signatures of the launch.
  uint8_t opcode{0};

  // Ordinal number of TB in TBs working on the same column. Each TB does a
  // multiple of TB width rows. The TBs for different ranges of rows are
  // launched in the same grid but are independent. The ordinal for non-first
  // TBs gets the base index for values.
  uint16_t nthBlock{0};
  /// Number of rows to process per thread of this block. This is equal across
  /// the grid, except for last block.
  uint16_t numRowsPerThread{1};

  /// Number of rows per thread in the grid, same for all blocks including the
  /// last one.
  uint16_t gridNumRowsPerThread{1};

  /// Number of rows to decode. if kFilterHits, the previous GpuDecode gives
  /// this number in BlockStatus. If 'rows' is set, this is the number of valid
  /// elements in 'rows'. If 'rows' is not set, the start is ''baseRow'
  int32_t maxRow{0};

  // If rows are densely decoded, this is the first row in terms of nullable
  // rows to decode in this TB.
  int32_t baseRow{0};

  /// Row count from filter. If a filter precedes this, the row count
  /// is read from this and 'rows' is set to the result row
  /// numbers. If this is a filter, the row count is written to
  /// 'blockStatus' and the row numbers are written to
  /// 'resultRows'. There is one BlockStatus for each kBlockSize rows
  /// in the grid. One TB covers 'numRowsPerThread' consecutive
  /// BlockStatuses. Because the previous filter for the range of rows
  /// is in the same TB read and write of row count is ordered by
  /// syncthreads at the end of each filter. The subscript is
  /// 'nthBlock * numRowsPerThread + <nth loop>'.
  BlockStatus* blockStatus{nullptr};

  /// Extra copy of result row count for each of kBlockSize rows if
  /// this is a filter. This is needed for non-last filters that
  /// extract values. The values have to be aligned after the final
  /// filtering result is known. This is no longer available in
  /// blockStatus.
  int32_t* filterRowCount{nullptr};

  // If multiple TBs on the same column and there are nulls, this is the start
  // offset of the TB's range of rows in non-null values. nullptr if no nulls.
  int32_t* nonNullBases{nullptr};

  /// If there are multiple chunks, this is an array of starts of non-first
  /// chunks.
  int32_t* chunkBounds{nullptr};

  /// If results will be scattered because of nulls, this is the bitmap with a 0
  /// for null. Subscripted with row number in encoding unit.
  char* nulls{nullptr};

  // If rows are sparsely decoded, this is the array of row numbers to extract.
  // The numbers are in terms of nullable rows.
  int32_t* rows{nullptr};

  // Data for pushed down filter. Interpretation depends on 'filterKind'.
  WaveFilterBase filter{};

  // Temp storage. Requires 2 + (kBlockSize / kWarpThreads) ints for each TB.
  int32_t* temp{nullptr};

  /// Row numbers that pass 'filter'. nullptr if no filter.
  int32_t* resultRows{nullptr};

  /// Result nulls.
  uint8_t* resultNulls{nullptr};

  /// Result array. nullptr if filter only.
  void* result{nullptr};

  // Bitmap of filter pass flags indexed by dictionary index. Filled in if
  // 'dictMode' is kDictFilter or kRecordFilter.
  uint32_t* filterBitmap{nullptr};

  struct Trivial {
    // Type of the input and result data.
    WaveTypeKind dataType;
    // Input data.
    const void* input;
    // Begin position for input, scatter and result.
    int begin;
    // End position (exclusive) for input, scatter and result.
    int end;
    // If not null, contains the output position relative to result pointer.
    const int32_t* scatter;
    // Starting address of the result.
    void* result;
  };

  struct MainlyConstant {
    // Type of the values and result.
    WaveTypeKind dataType;
    // Number of total values that should be written to result.
    int count;
    // Common value that is repeated.
    const void* commonValue;
    // Sparse values that will be picked up when isCommon is false.
    const void* otherValues;
    // Bitmask indicating whether a values is common or not.
    const uint8_t* isCommon;
    // Temporary storage to keep non-common value indices.  Should be
    // preallocated to at least count large.
    int32_t* otherIndices;
    // Starting address of the result.
    void* result;
    // If non-null, the count of non-common elements will be written to this
    // address.
    int32_t* otherCount;
  };

  struct DictionaryOnBitpack {
    // Type of the alphabet and result.
    WaveTypeKind dataType;
    // Dictionary alphabet.
    const void* alphabet;
    // Indices into the alphabet.
    const uint64_t* indices;
    // Begin position for indices, scatter and result.
    int begin;
    // End position (exclusive) for indices, scatter and result.
    int end;
    // Bit width of each index.
    int bitWidth;
    // If not null, contains the output position relative to result pointer.
    const int32_t* scatter;
    // All indices should be offseted by this baseline.
    int64_t baseline;
    // Starting address of the result.
    void* result;
  };

  struct SelectiveChunked {
    int32_t chunkStart;
    const void* input;
  };

  struct Varint {
    // Address of the input data.
    const char* input;
    // Byte size of the input data.
    int size;
    // Temporary storage to keep whether each byte is a terminal for a number.
    // Should be allocated at least "size" large.
    bool* ends;
    // Temporary storage to keep the location of end position of each number.
    // Should be allocated at least "size" large.
    int32_t* endPos;
    // Type of the result number.
    WaveTypeKind resultType;
    // Starting address of the result.
    void* result;
    // Count of the numbers in the result.
    int resultSize;
  };

  struct SparseBool {
    // Number of bits in the result.
    int totalCount;
    // Sparse value; common value is the opposite of this.
    bool sparseValue;
    // Bit indices where sparse value should be stored.
    const int32_t* sparseIndices;
    // Number of sparse values.
    int sparseCount;
    // Temporary storage to keep bool representation of bits.  Should be at
    // least totalCount large.
    bool* bools;
    // Address of the result bits.  We do not support unaligned result because
    // different blocks could access the same byte and cause racing condition.
    uint8_t* result;
  };

  struct RleTotalLength {
    // Input data to be summed.
    const int32_t* input;
    // Number of input data.
    int count;
    // The sum of input data.
    int64_t* result;
  };

  struct Rle {
    // Type of values and result.
    WaveTypeKind valueType;
    // Values that will be repeated.
    const void* values;
    // Length of each value.
    const int32_t* lengths;
    // Number of values and lengths.
    int count;
    // Starting address of the result.
    const void* result;
  };

  struct MakeScatterIndices {
    // Input bits.
    const uint8_t* bits;
    // Whether set or unset bits should be found in the input.
    bool findSetBits;
    // Begin bit position of the input.
    int begin;
    // End bit position (exclusive) of the input.
    int end;
    // Address of the result indices.
    int32_t* indices;
    // If non-null, store the number of indices being written.
    int32_t* indicesCount;
  };

  struct RowCountNoFilter {
    int32_t numRows;
    BlockStatus* status;
    int32_t gridStatusSize;
    bool gridOnly;
  };

  struct CountBits {
    const uint8_t* bits;
    // Number of bits. A count is produced for each run of 'resultStride' bits
    // fully included in 'numBits'. If numBits is 600 and resultStride is 256
    // then result[0] is the count of ones in the first 256, then result[1] is
    // the count of ones in the first 512 and result[2] is unset.
    int32_t numBits;
    // 256/512/1024/2048.
    int32_t resultStride;
    // One int per warp (blockDim.x/32).
  };

  struct CompactValues {
    // Selected row numbers from the source filtered column.
    int32_t* sourceRows;
    // Row count for each kBlockSize rows of the source.
    int32_t* sourceNumRows;
    /// The rows selected by the last filter. blockStatus has the count.
    int32_t* finalRows;
    /// The results produced by the first filtered column.
    void* source;
    /// Null flags. nullptr if not nullable.
    uint8_t* sourceNull;
  };

  /// Selective decode of a Nimble stream. Each row is read from the chunk
  /// that contains it, with the accessor of the chunk's encoding tree.
  struct Nimble {
    ColumnAccess column;
    /// All nodes of the stream.
    const NimbleNode* nodes;
    /// The node to prepare in a kNimblePrepare op. kStructNulls for the nulls
    /// stream of a struct and kLengths for the lengths stream of an array or
    /// map, whose chunks are in 'column.chunks'.
    int32_t node;

    static constexpr int32_t kStructNulls = -1;
    static constexpr int32_t kLengths = -2;
  };

  /// Sizes of arrays to offsets in the element vector and element rows. The
  /// TB covers the rows of one kNested TB.
  struct ExpandLengths {
    /// Element starts in the element stream on input, offsets in the element
    /// vector on output.
    int32_t* offsets;
    const int32_t* sizes;
    /// Element stream row of each element.
    int32_t* rows;
    /// Offset of the first element of the TB in the element vector.
    int32_t base;
    int32_t numRows;
  };

  union {
    Nimble nimble;
    ExpandLengths expandLengths;
    Trivial trivial;
    MainlyConstant mainlyConstant;
    DictionaryOnBitpack dictionaryOnBitpack;
    Varint varint;
    SparseBool sparseBool;
    RleTotalLength rleTotalLength;
    Rle rle;
    MakeScatterIndices makeScatterIndices;
    RowCountNoFilter rowCountNoFilter;
    CountBits countBits;
    CompactValues compact;
    SelectiveChunked selectiveChunked;
  } data{};

  /// The format independent part of the data of a column op. It is the first
  /// member of the op data of each file format.
  WAVE_DEVICE_HOST ColumnAccess& column() {
    return data.nimble.column;
  }
  WAVE_DEVICE_HOST const ColumnAccess& column() const {
    return data.nimble.column;
  }

  /// True if 'nulls' is a bitmap of nulls. False if 'nulls' an array of uint8_t
  bool isNullsBitmap{true};

#ifndef __CUDACC_RTC__
  /// Returns the amount of int aligned global memory per TB needed in 'temp'
  /// for standard size TB.
  int32_t tempSize() const;

  /// Returns the amount of shared memory for standard size thread block for
  /// 'step'.
  int32_t sharedMemorySize() const;

  /// Sets the pushed down filter from ScanSpec of 'reader'. Uses 'stream' to
  /// setup device-side data like hash tables.
  void setFilter(ColumnReader* reader, Stream* stream);
#endif
};

/// Describes multiple sequences of decode ops. Each TB executes a sequence of
/// decode steps. The data area starts with a range of instruction numbers for
/// each thread block. The first TB runs from 0 to ends[0]. The nth runs from
/// ends[nth-1] to ends[nth]. After gridDim.x ends, we round to a 4 int aligned
/// offset and have an array of GpuDecodes.
struct alignas(16) GpuDecodeParams {
  // If need to represent more than this many ops, use a dynamically allocated
  // external array in 'external'.
  static constexpr int32_t kMaxInlineOps = 15;

  // Pointer to standalone description of work. If nullptr, the description of
  // work fits inline in 'this'.
  GpuDecodeParams* external{nullptr};
  void* padding{nullptr};
  // The end of each decode program. The first starts at 0. The end is
  // ends[blockIdx.x].
  int32_t ends
      [kMaxInlineOps * (sizeof(GpuDecode) + sizeof(int32_t)) /
       sizeof(int32_t)] = {};
};

} // namespace facebook::velox::wave

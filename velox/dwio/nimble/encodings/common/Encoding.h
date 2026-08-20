/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/buffer/BufferPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

#include <memory>
#include <string_view>
#include <type_traits>

/// The Encoding class defines an interface for interacting with encodings
/// (aka vectors, aka arrays) of encoded data. The API is tailored for
/// typical usage patterns within query engines, and is designed to be
/// easily extensible.
///
/// Some general notes:
///  1. Output bitmaps are compatible with the FixedBitArray class. In
///     particular this means you must use FixedBitArray::bufferSize to
///     determine the space needed for the bitmaps you pass to, e.g.,
///     the output buffer of Encoding::Equals.
///  2. When passing types into functions via the void* interfaces,
///     and when materializing data, numeric types are input/output via
///     their native types, while string are passed via std::string_view
///     -- NOT std::string!
///  3. We refer to the 'row pointer' in various places. Think of this as
///     the iterator over the encoding's data. Encoding::Reset() is like
///     rp = vec.begin(), advancing N rows is like rp = rp + N, etc.
///  4. An Encoding doesn't own the underlying data. If you want to create two
///     copies for some reason on the same data go ahead, they are totally
///     independent.
///  5. Each subclass of Encoding will provide its own serialization methods.
///     The resulting string should be passed to the subclass's constructor
///     at read time.
///  6. Although serialized Encodings can be stored independently if desired
///     (as they are just strings), when written to disk they are normally
///     stored together in a Tablet (see tablet.h).
///  7. We consistently use 4 byte unsigned integers for things like offsets
///     and row counts. Encodings, and indeed entire Tablets, are required to
///     fit within a uint32_t in size. This is enforced in the normal
///     ingestion pathways. If you directly use the serialization methods,
///     be careful that this remains true!
///  8. See the notes in the Encoding class about array encodings. These
///  encodings
///     have slightly different semantics than other encodings.

namespace facebook::velox::dwio::common {
struct DecodingStats;
} // namespace facebook::velox::dwio::common

namespace facebook::nimble {

class EncodingBufferPool;
class SharedDictionaryResolver;

template <typename T, typename Filter, typename ExtractValues, bool kIsDense>
using DecoderVisitor =
    velox::dwio::common::ColumnVisitor<T, Filter, ExtractValues, kIsDense>;

using vector_size_t = velox::vector_size_t;

/// Extra parameters that need to be persisted/used during a single call of
/// readWithVisitor at column reader level, which might span multiple calls of
/// readWithVisitor (one per chunk) in decoders.
struct ReadWithVisitorParams {
  // Create the reader nulls buffer if not already exists and return pointer to
  // the raw buffer.  When it is created, it is created with the full length
  // across potential multiple chunks.
  std::function<uint64_t*()> makeReaderNulls;

  // Resolves which buffer `SelectiveColumnReader::resultNulls()' returns for
  // this read by setting the `returnReaderNulls_' flag (and `anyNulls_').
  // Resolves the flag only; allocates no buffer.  Must be called after decoding
  // nulls in `NullableEncoding'.
  std::function<void()> setReturnNullsMode;

  // Create the result nulls if not already exists.  Similar to
  // `makeReaderNulls', we create one single buffer for all the results nulls
  // across potential multiple chunks during one read.
  std::function<void()> prepareResultNulls;

  // Number of rows scanned so far.  Contains rows scanned in previous chunks
  // during this read call as well.
  vector_size_t numScanned;
};

class Encoding {
 public:
  /// Options that control encoding behavior during encode and decode.
  /// Includes format-level settings (e.g., varint row count from file version)
  /// and per-encoding configuration (e.g., block size from serde params).
  struct Options {
    /// When true, rowCount in the encoding prefix is varint-encoded instead of
    /// fixed 4-byte uint32. Determined from file format version or serializer
    /// version.
    bool useVarintRowCount = false;

    /// Optional buffer pool for reusing scratch buffers across encoding
    /// lifetimes. When set, encodings return their scratch buffers to the pool
    /// on destruction and grab pre-allocated buffers on construction, avoiding
    /// MemoryPool alloc/free overhead.
    velox::BufferPool* bufferPool = nullptr;

    /// Optional pool for Nimble Buffer scratch arenas used while encoding
    /// nested child streams.
    EncodingBufferPool* encodingBufferPool = nullptr;

    /// FrequencyPartitionEncoding index type (cast to FreqPartIndexType).
    /// 0 = NoIndex (default, backward-compatible), 1 = PerTierBitmaps,
    /// 2 = TierTagArray, 3 = EliasFano.
    uint8_t frequencyPartitionIndex = 0;

    /// Block size for BlockBitPacking encoding. Determines how many rows
    /// are packed per block. Written to the stream header; the reader
    /// reads it back from the stream (self-describing).
    uint16_t blockBitPackingBlockSize = kBlockBitPackingBlockSize;

    /// Number of rows per independently encoded DeltaBlock block.
    ///
    /// Each block stores one base value plus fixed-width deltas between
    /// adjacent sorted values inside that block. Smaller blocks adapt better
    /// when local gap widths change, but pay more per-block metadata. Larger
    /// blocks reduce metadata, but one large gap can widen every delta in the
    /// block. The value is serialized in the stream header, so readers are
    /// self-describing.
    uint16_t deltaBlockSize{256};

    /// When true, FOR-family payloads use the exact required bit width. When
    /// false, FixedBitWidth and PFOR round to byte or bucket boundaries.
    bool fixedBitWidthUseExactBits{false};

    /// EXPERIMENTATION: Allows ALP to participate in nested floating-point
    /// encoding selection. False by default; do not enable for production
    /// until ALP is production-ready.
    bool allowNestedAlpSelection{false};

    /// Per-column decoding statistics for timing decompression.
    velox::dwio::common::DecodingStats* decodingStats = nullptr;

    /// Resolves alphabets referenced by SharedDictionary encodings. The
    /// resolver is bound to the current file and stripe decode context.
    std::shared_ptr<const SharedDictionaryResolver> sharedDictionaryResolver{};

    velox::io::IoCounter* decompressCounter() const {
      return decodingStats != nullptr ? &decodingStats->decompressCPUTimeNanos
                                      : nullptr;
    }

    /// FSST is kept only when its final encoded size is at most this fraction
    /// of the original string bytes.
    double fsstCompressionTargetRatio = 0.6;
  };

  static constexpr int kEncodingTypeOffset =
      EncodingPrefix::kEncodingTypeOffset;
  static constexpr int kDataTypeOffset = EncodingPrefix::kDataTypeOffset;
  static constexpr int kRowCountOffset = EncodingPrefix::kRowCountOffset;
  static constexpr int kPrefixSize = EncodingPrefix::kFixedPrefixSize;

  Encoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Options& options);
  virtual ~Encoding() = default;

  EncodingType encodingType() const {
    return encodingType_;
  }

  DataType dataType() const {
    return dataType_;
  }

  uint32_t rowCount() const {
    return rowCount_;
  }

  /// Returns the byte offset where encoding-specific data begins, right after
  /// the common prefix (EncodingType + DataType + rowCount).
  uint32_t dataOffset() const {
    return prefixSize_;
  }

  static void copyIOBuf(char* pos, const folly::IOBuf& buf);

  virtual uint32_t nullCount() const {
    return 0;
  }

  /// Resets the internal state (e.g. row pointer) to newly constructed form.
  virtual void reset() = 0;

  /// Advances the row pointer N rows. Note that we don't provide a 'backwards'
  /// iterator; if you need to move your row pointer back, reset() and skip().
  virtual void skip(uint32_t rowCount) = 0;

  /// Materializes the next |rowCount| rows into buffer. Advances
  /// the row pointer |rowCount|.
  ///
  /// Remember that buffer is interpreted as a physicalType*, with
  /// std::string_view* for strings and bool* (not a bitmap) for bool. For
  /// non-POD types, namely string data, note that the materialized values are
  /// only guaranteed valid until the next non-const call to *this.
  ///
  /// If this->isNullable(), null rows are materialized as physicalType(). To be
  /// able to distinguish between nulls and non-null physicalType() values, use
  /// MaterializeNullable.
  virtual void materialize(uint32_t rowCount, void* buffer) = 0;

  /// Nullable method.
  /// Like Materialize, but also sets the ith bit of bitmap to reflect whether
  /// ith row was null or not. 0 means null, 1 means not null. Null rows are
  /// left untouched instead of being filled with default values.
  ///
  /// If |scatterOutputBitmap| is provided, |rowCount| still indicates how many
  /// items to be read from the encoding. However, instead of placing them
  /// sequentially in the output |buffer|, the items are scattered. This means
  /// that item will be place into the slot where the corresponding positional
  /// bit is set to 1 in |scatterOutputBitmap|, (note that the value being read
  /// may be null). For every positional scatter bit set to 0, it will fill a
  /// null in the poisition in the output |buffer|. |rowCount| should match the
  /// number of bits set to 1 in |scatterOutputBitmap|. For scattered reads,
  /// |buffer| and
  /// |getOutputNulls()| should have enough space to accommodate
  /// |scatterOutputBitmap.size()| items. When |offset| is specified, use the
  /// |scatterOutputBitmap| starting from |offset| and scatter to |buffer| and
  /// |getOutputNulls()| starting from |offset|.
  ///
  /// Returns number of items that are not null. In the case when all values are
  /// non null, |getOutputNulls()| will not be filled with all 1s. It's
  /// expected that caller explicitly checks for that condition.
  virtual uint32_t materializeNullable(
      uint32_t rowCount,
      void* buffer,
      std::function<void*()> getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr,
      uint32_t offset = 0) = 0;

  /// Read bool values to output buffer as bits.
  virtual void materializeBoolsAsBits(
      uint32_t /*rowCount*/,
      uint64_t* /*buffer*/,
      int /*begin*/) {
    NIMBLE_NOT_IMPLEMENTED(typeid(*this).name());
  }

  /// Whether this encoding is nullable, i.e. contains any nulls. This property
  /// modifies how engines need to interpret many of the function results, and a
  /// number of functions are only callable if isNullable() returns true.
  virtual bool isNullable() const {
    return false;
  }

  /// Returns true if this encoding stores dictionary-encoded data.
  /// When true, dictionarySize(), dictionaryEntry(), and dictionaryEntries()
  /// are valid to call.
  virtual bool dictionaryEnabled() const {
    return false;
  }

  /// Returns the number of unique values in the dictionary.
  virtual uint32_t dictionarySize() const {
    NIMBLE_UNREACHABLE("dictionarySize on non-dictionary encoding");
  }

  /// Returns a pointer to the value at the given dictionary index.
  /// The index must be in [0, dictionarySize()). The pointer is valid for
  /// the lifetime of the encoding.
  virtual const void* dictionaryEntry(uint32_t /* index */) const {
    NIMBLE_UNREACHABLE("dictionaryEntry on non-dictionary encoding");
  }

  /// Returns a pointer to the contiguous dictionary alphabet data.
  /// The pointer is valid for the lifetime of the encoding. The caller
  /// must cast to the appropriate type (e.g., `const std::string_view*`).
  /// For NullableEncoding, unwraps and delegates to the inner encoding.
  /// Future stacks should extend to recursively find the inner
  /// DictionaryEncoding through MainlyConstant and RLE wrappers.
  virtual const void* dictionaryEntries() const {
    NIMBLE_UNREACHABLE("dictionaryEntries on non-dictionary encoding");
  }

  /// Materializes raw dictionary indices for rowCount dense consecutive
  /// non-null rows. Does not touch any visitor/reader state.
  /// Overridden by DictionaryEncoding and MainlyConstantEncoding.
  virtual void materializeIndices(uint32_t /*rowCount*/, uint32_t* /*buffer*/) {
    NIMBLE_UNREACHABLE("materializeIndices on non-dictionary encoding");
  }

  // A string for debugging/iteration that gives details about *this.
  // Offset adds that many spaces before the msg (useful for children
  // encodings).
  virtual std::string debugString(int offset = 0) const;

 protected:
  static void serializePrefix(
      EncodingType encodingType,
      DataType dataType,
      uint32_t rowCount,
      bool useVarint,
      char*& pos) {
    EncodingPrefix::serialize(encodingType, dataType, rowCount, useVarint, pos);
  }

  static uint32_t serializePrefixSize(uint32_t rowCount, bool useVarint) {
    return EncodingPrefix::serializedSize(rowCount, useVarint);
  }

  static DataType readDataType(std::string_view data) {
    return EncodingPrefix::readDataType(data);
  }

  static uint32_t readRowCount(std::string_view data, bool useVarint) {
    return EncodingPrefix::readRowCount(data, useVarint);
  }

  static uint32_t prefixSize(std::string_view data, bool useVarint) {
    return EncodingPrefix::prefixSize(data, useVarint);
  }

  void releaseBuffer(velox::BufferPtr& buffer) {
    if (auto* bufferPool = options_.bufferPool) {
      bufferPool->release(std::move(buffer));
    } else {
      buffer.reset();
    }
  }

  /// Acquires a raw BufferPtr with at least bytes capacity. Uses the buffer
  /// pool when it has a large enough buffer, otherwise allocates a new one.
  velox::BufferPtr getBuffer(size_t bytes) {
    if (auto* bufferPool = options_.bufferPool) {
      if (auto buffer = bufferPool->get(bytes)) {
        return buffer;
      }
    }
    return velox::AlignedBuffer::allocate<char>(bytes, pool_);
  }

  /// Acquires a Vector backed by a buffer pool allocation if available.
  /// Falls back to a fresh empty Vector from the memory pool.
  template <typename V>
  Vector<V> getVectorBuffer() {
    if (auto* bufferPool = options_.bufferPool) {
      if (auto buf = bufferPool->get()) {
        return Vector<V>(std::move(buf));
      }
    }
    return Vector<V>(pool_);
  }

  /// Releases a Vector's underlying buffer back to the buffer pool.
  void releaseVectorBuffer(auto& vector) {
    if (auto* bufferPool = options_.bufferPool) {
      bufferPool->release(vector.releaseBuffer());
    }
  }

  velox::memory::MemoryPool* const pool_;
  const std::string_view data_;
  const EncodingType encodingType_;
  const Options options_;
  const DataType dataType_;
  const uint32_t rowCount_;
  const uint32_t prefixSize_;
};

// The TypedEncoding<physicalType> class exposes the same interface as the base
// Encoding class but provides common typed implementation for some apis exposed
// by the Encoding class. T means semantic data type, physicalType means data
// type used for encoding
template <typename T, typename physicalType>
class TypedEncoding : public Encoding {
  static_assert(
      std::is_same_v<physicalType, typename TypeTraits<T>::physicalType>);

 public:
  TypedEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Options& options = {})
      : Encoding{pool, data, options} {}

  /// Similar to materialize(), but scatters values to output buffer according
  /// to scatterOutputBitmap. When scatterOutputBitmap is nullptr or all 1's,
  /// the output nullBitmap will not be set. It's expected that caller
  /// explicitly checks against the return value and handle such all 1's cases
  /// properly.
  uint32_t materializeNullable(
      uint32_t rowCount,
      void* buffer,
      std::function<void*()> getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap,
      uint32_t offset) override {
    // 1. Read X items from the encoding.
    // 2. Spread the items read in #1 into positions in |buffer| where
    // |scatterOutputBitmap| has a matching positional bit set to 1.
    if (offset > 0) {
      buffer = static_cast<physicalType*>(buffer) + offset;
    }

    materialize(rowCount, buffer);

    // TODO: check rowCount matches the number of set bits in
    // scatterOutputBitmap.
    auto scatterCount =
        scatterOutputBitmap ? scatterOutputBitmap->size() - offset : rowCount;
    if (scatterCount == rowCount) {
      // No need to scatter. Avoid setting nullBitmap since caller is expected
      // to explicitly handle such non-null cases.
      return rowCount;
    }

    NIMBLE_CHECK_LT(rowCount, scatterCount, "Unexpected count");

    void* nullBitmap = getOutputNulls();

    velox::bits::BitmapBuilder nullBits{nullBitmap, offset + scatterCount};
    nullBits.copy(*scatterOutputBitmap, offset, offset + scatterCount);

    // Scatter backward
    uint32_t pos = offset + scatterCount - 1;
    physicalType* output =
        static_cast<physicalType*>(buffer) + scatterCount - 1;
    const physicalType* source =
        static_cast<physicalType*>(buffer) + rowCount - 1;
    while (output != source) {
      if (scatterOutputBitmap->test(pos)) {
        *output = *source;
        --source;
      }
      --output;
      --pos;
    }

    return rowCount;
  }
};

// Dispatch the visitor and params according to the encoding type and data type.
// We need to do this to mimic a virtual method call on Encoding with templates.
template <typename DecoderVisitor>
void callReadWithVisitor(
    Encoding& encoding,
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params);

namespace detail {

template <typename T, typename PhysicalType>
T castFromPhysicalType(const PhysicalType& value) {
  if constexpr (isFloatingPointType<T>()) {
    static_assert(sizeof(T) == sizeof(PhysicalType));
    return reinterpret_cast<const T&>(value);
  } else {
    return value;
  }
}

// Materializes dictionary indices into the reader's rawValues_ and scatters
// for null gaps. Used by DictionaryEncoding and MainlyConstantEncoding
// readIndicesWithVisitor as the dense no-filter fast path.
// Updates visitor state (addNumValues/setRowIndex) to mark all rows consumed.
//
// @param encoding The encoding to call materializeIndices on.
// @param visitor The visitor (used for outerNonNullRows scratch buffer).
// @param params Read params; provides numScanned (the read offset) and, on the
//        filtered path, the result-nulls buffer allocation.
// @param rawNulls The null bitmap from nullsInReadRange (may be nullptr).
// @param numReadRows Total rows in the read range (including nulls).
// @param numNonNulls Count of non-null rows in the range.
template <typename V>
void readDenseMaterializedIndices(
    Encoding& encoding,
    V& visitor,
    ReadWithVisitorParams& params,
    const uint64_t* rawNulls,
    uint32_t numReadRows,
    uint32_t numNonNulls) {
  const auto readOffset = params.numScanned;
  auto* rawOutputValues =
      reinterpret_cast<int32_t*>(visitor.reader().rawValues());
  const auto valueOutputOffset = visitor.reader().numValues();
  encoding.materializeIndices(
      numNonNulls,
      reinterpret_cast<uint32_t*>(rawOutputValues + valueOutputOffset));
  params.prepareResultNulls();
  if (numNonNulls == numReadRows) {
    visitor.addNumValues(numReadRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }
  auto& outerNonNullRows = visitor.outerNonNullRows();
  outerNonNullRows.resize(numNonNulls);
  auto* rawOuterNonNullRows = outerNonNullRows.data();
  velox::simd::indicesOfSetBits(
      rawNulls, readOffset, readOffset + numReadRows, rawOuterNonNullRows);
  for (auto& row : outerNonNullRows) {
    row = row - readOffset + valueOutputOffset;
  }
  velox::dwio::common::scatterNonNulls(
      /*targetBegin=*/0,
      numNonNulls,
      /*sourceBegin=*/valueOutputOffset,
      rawOuterNonNullRows,
      rawOutputValues);
  visitor.addNumValues(numReadRows);
  visitor.setRowIndex(visitor.numRows());
}

// Gathers dictionary indices from a pre-materialized buffer into rawValues_
// for sparse (non-dense) row sets. The buffer contains dense indices for all
// non-null rows in the range [numScanned, numScanned + numReadRows).
// The visitor's row set determines which positions to gather.
// Updates visitor state (addNumValues/setRowIndex) to mark all rows consumed.
//
// Walks non-null positions and the visitor's row set in lockstep so the
// encoding-space index is tracked incrementally — O(numReadRows)
// instead of O(numVisitorRows * numReadRows).
//
// @param encoding The encoding to call materializeIndices on.
// @param visitor The visitor with the sparse row set.
// @param readOffset Bit offset into rawNulls for the current read range.
// @param prepareResultNulls Callback to allocate resultNulls_ on first null.
// @param rawNulls The null bitmap from nullsInReadRange (may be nullptr).
// @param numReadRows Total rows in the read range.
// @param numNonNulls Count of non-null rows in the range.
// @param indicesBuffer Scratch buffer with at least numNonNulls entries.
template <typename V>
void readSparseMaterializedIndices(
    Encoding& encoding,
    V& visitor,
    vector_size_t readOffset,
    const std::function<void()>& prepareResultNulls,
    const uint64_t* rawNulls,
    uint32_t numReadRows,
    uint32_t numNonNulls,
    uint32_t* indicesBuffer) {
  encoding.materializeIndices(numNonNulls, indicesBuffer);

  auto* rawOutputValues =
      reinterpret_cast<int32_t*>(visitor.reader().rawValues());
  const auto valueOutputOffset = visitor.reader().numValues();

  // Allocate and clear resultNulls_ before the loop rather than on the first
  // emitted null: the loop only writes null positions, so a read whose
  // surviving rows are all non-null would leave a reused buffer untouched and
  // leak stale null bits into the output. Called unconditionally: the callback
  // passes the read-range null state to prepareNulls, which self-short-circuits
  // (no allocate/clear) when there are no nulls.
  prepareResultNulls();

  const auto startRowIndex = visitor.rowIndex();
  uint32_t indexOffset = 0;
  auto readRowIndex = startRowIndex;
  bool anyNull = false;
  for (uint32_t row = 0; row < numReadRows && readRowIndex < visitor.numRows();
       ++row) {
    const bool readCurrentRow =
        row == static_cast<uint32_t>(visitor.rowAt(readRowIndex) - readOffset);
    const bool isNull = rawNulls != nullptr &&
        !velox::bits::isBitSet(rawNulls, readOffset + row);
    if (isNull) {
      if (readCurrentRow) {
        anyNull = true;
        // In the sparse case, returnReaderNulls_ is false
        // (setReturnNullsMode only sets it for dense rows). resultNulls()
        // then returns resultNulls_ instead of nullsInReadRange_. We must
        // explicitly write null bits to resultNulls_ so
        // DictionaryVector::validate() knows to skip these positions. Also
        // write 0 as a safe placeholder index.
        const auto outputPos = valueOutputOffset + readRowIndex - startRowIndex;
        rawOutputValues[outputPos] = 0;
        velox::bits::setNull(visitor.reader().rawResultNulls(), outputPos);
        ++readRowIndex;
      }
      continue;
    }
    if (readCurrentRow) {
      rawOutputValues[valueOutputOffset + readRowIndex - startRowIndex] =
          static_cast<int32_t>(indicesBuffer[indexOffset]);
      ++readRowIndex;
    }
    ++indexOffset;
  }
  if (anyNull) {
    visitor.reader().setHasNulls();
  }
  visitor.addNumValues(visitor.numRows() - visitor.rowIndex());
  visitor.setRowIndex(visitor.numRows());
}

template <typename DecoderVisitor, typename Skip, typename F>
void readWithVisitorSlow(
    DecoderVisitor& visitor,
    const ReadWithVisitorParams& params,
    Skip skip,
    F decodeOne) {
  using T = typename DecoderVisitor::DataType;
  constexpr bool kExtractToReader = std::is_same_v<
      typename DecoderVisitor::Extract,
      velox::dwio::common::ExtractToReader>;
  auto* nulls = visitor.reader().rawNullsInReadRange();
  if constexpr (kExtractToReader) {
    params.prepareResultNulls();
  }
  auto numScanned = params.numScanned;
  bool atEnd = false;
  while (!atEnd) {
    if constexpr (!std::is_null_pointer_v<Skip>) {
      auto numNonNulls = visitor.currentRow() - numScanned;
      if (nulls) {
        numNonNulls -=
            velox::bits::countNulls(nulls, numScanned, visitor.currentRow());
      }
      if (numNonNulls > 0) {
        skip(numNonNulls);
      }
      numScanned = visitor.currentRow() + 1;
    }
    if (nulls && velox::bits::isBitNull(nulls, visitor.currentRow())) {
      if (!visitor.allowNulls()) {
        visitor.addRowIndex(1);
        atEnd = visitor.atEnd();
      } else if (kExtractToReader && visitor.reader().returnReaderNulls()) {
        visitor.addRowIndex(1);
        visitor.addNumValues(1);
        atEnd = visitor.atEnd();
      } else {
        visitor.processNull(atEnd);
      }
    } else {
      visitor.process(castFromPhysicalType<T>(decodeOne()), atEnd);
    }
  }
}

template <typename TEncoding, typename V>
void readWithVisitorFast(
    TEncoding& encoding,
    V& visitor,
    ReadWithVisitorParams& params,
    const uint64_t* nulls) {
  constexpr bool kOutputNulls = !V::kHasFilter && !V::kHasHook;
  const auto numRows = visitor.numRows() - visitor.rowIndex();
  auto& outerRows = visitor.outerNonNullRows();
  if (!nulls) {
    encoding.template bulkScan<false>(
        visitor,
        params.numScanned,
        visitor.rows() + visitor.rowIndex(),
        numRows,
        velox::iota(visitor.numRows(), outerRows) + visitor.rowIndex());
    return;
  }
  // TODO: Store last non null index and num non-nulls so far in decoder to
  // accelerate multi-chunk decoding.
  const auto numNonNullsSoFar =
      velox::bits::countNonNulls(nulls, 0, params.numScanned);
  if constexpr (V::dense) {
    NIMBLE_DCHECK(
        !visitor.reader().hasNulls() || visitor.reader().returnReaderNulls());
    outerRows.resize(numRows);
    auto numNonNulls = velox::simd::indicesOfSetBits(
        nulls, visitor.rowIndex(), visitor.numRows(), outerRows.data());
    outerRows.resize(numNonNulls);
    if constexpr (kOutputNulls) {
      if (numNonNulls != numRows) {
        if (!visitor.reader().returnReaderNulls()) {
          params.prepareResultNulls();
          velox::bits::copyBits(
              nulls,
              visitor.rowIndex(),
              visitor.reader().rawResultNulls(),
              visitor.rowIndex(),
              numRows);
        }
        visitor.setHasNulls();
      }
    }
    if (outerRows.empty()) {
      if constexpr (kOutputNulls) {
        visitor.addNumValues(numRows);
      }
      visitor.addRowIndex(numRows);
    } else {
      encoding.template bulkScan<true>(
          visitor,
          numNonNullsSoFar,
          visitor.rows() + numNonNullsSoFar,
          numNonNulls,
          outerRows.data());
    }
    return;
  }
  auto& innerRows = visitor.innerNonNullRows();
  int32_t tailSkip = -1;
  uint64_t* resultNulls = nullptr;
  uint8_t* chunkResultNulls = nullptr;
  if constexpr (kOutputNulls) {
    params.prepareResultNulls();
    resultNulls = visitor.reader().rawResultNulls();
    chunkResultNulls = reinterpret_cast<uint8_t*>(resultNulls) +
        velox::bits::nbytes(visitor.rowIndex());
  }
  bool anyNulls =
      velox::dwio::common::nonNullRowsFromSparse<V::kHasFilter, kOutputNulls>(
          nulls,
          velox::RowSet(visitor.rows() + visitor.rowIndex(), numRows),
          innerRows,
          outerRows,
          chunkResultNulls,
          tailSkip);
  if constexpr (kOutputNulls) {
    if (auto remainderBitCount = numRows % 8) {
      // `nonNullRowsFromSparse' resets the bits in last byte after `numRows'.
      // We need to set them back to ensure rows in next chunk is not
      // incorrectly nullified.
      chunkResultNulls[(numRows - 1) / 8] |= static_cast<uint8_t>(-1)
          << remainderBitCount;
    }
  }
  if (anyNulls) {
    visitor.setHasNulls();
  }
  if (kOutputNulls && visitor.rowIndex() % 8 != 0) {
    auto chunkResultStart = velox::bits::roundUp(visitor.rowIndex(), 8);
    velox::bits::copyBits(
        resultNulls,
        chunkResultStart,
        resultNulls,
        visitor.rowIndex(),
        numRows);
    // Since chunk result is copied over, any trailing nulls in the original
    // positions need to be set back to non nulls, otherwise we get incorrect
    // nulls at beginning of next chunk.
    velox::bits::fillBits(
        resultNulls,
        visitor.rowIndex() + numRows,
        chunkResultStart + numRows,
        true);
  }
  if (!V::kHasFilter && visitor.rowIndex() > 0) {
    for (auto& row : outerRows) {
      row += visitor.rowIndex();
    }
  }
  if (innerRows.empty()) {
    if constexpr (kOutputNulls) {
      visitor.addNumValues(numRows);
    }
    visitor.addRowIndex(numRows);
    encoding.skip(tailSkip - numNonNullsSoFar);
  } else {
    encoding.template bulkScan<true>(
        visitor,
        numNonNullsSoFar,
        innerRows.data(),
        innerRows.size(),
        outerRows.data());
    encoding.skip(tailSkip);
  }
}

// DataType is the type of DecoderVisitor::DataType.  The corresponding
// ValueType is the type we store in values buffer of selective column reader.
template <typename DataType>
using ValueType = std::conditional_t<
    std::is_same_v<DataType, std::string_view>,
    velox::StringView,
    DataType>;

template <typename V, typename DataType>
ValueType<DataType> dataToValue(const V& visitor, DataType data) {
  if constexpr (std::is_same_v<DataType, std::string_view>) {
    return visitor.reader().copyStringValueIfNeed(data);
  } else {
    return data;
  }
}

template <typename T, typename V>
T* mutableValues(const V& visitor, vector_size_t size) {
  T* values = visitor.reader().template mutableValues<T>(size);
  if constexpr (V::kHasHook) {
    // Use end of the region to avoid overwrite values in previous chunk
    // with dictionary indices.
    values += visitor.reader().valuesCapacity() / sizeof(T) - size -
        visitor.reader().numValues();
  }
  return values;
}

} // namespace detail

} // namespace facebook::nimble

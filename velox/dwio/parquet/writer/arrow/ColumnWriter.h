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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>

#include "velox/dwio/parquet/common/RleEncodingInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/util/Compression.h"

namespace arrow {
class Array;
} // namespace arrow

namespace facebook::velox::parquet {
class BitWriter;
class RleEncoder;
} // namespace facebook::velox::parquet

namespace facebook::velox::parquet::arrow {

namespace util {
class CodecOptions;
} // namespace util

struct ArrowWriteContext;
class ColumnChunkMetaDataBuilder;
class ColumnDescriptor;
class ColumnIndexBuilder;
class DataPage;
class DictionaryPage;
class Encryptor;
class OffsetIndexBuilder;
class WriterProperties;

using ArrowOutputStream = ::arrow::io::OutputStream;

class PARQUET_EXPORT LevelEncoder {
 public:
  LevelEncoder();
  ~LevelEncoder();

  static int maxBufferSize(
      Encoding::type encoding,
      int16_t maxLevel,
      int numBufferedValues);

  // Initialize the LevelEncoder.
  void init(
      Encoding::type encoding,
      int16_t maxLevel,
      int numBufferedValues,
      uint8_t* data,
      int dataSize);

  // Encodes a batch of levels from an array and returns the number of levels.
  // encoded.
  int encode(int batchSize, const int16_t* levels);

  int32_t len() {
    if (encoding_ != Encoding::kRle) {
      throw ParquetException("Only implemented for RLE encoding");
    }
    return rleLength_;
  }

 private:
  int bitWidth_;
  int rleLength_;
  Encoding::type encoding_;
  std::unique_ptr<RleEncoder> rleEncoder_;
  std::unique_ptr<BitWriter> bitPackedEncoder_;
};

class PARQUET_EXPORT PageWriter {
 public:
  virtual ~PageWriter() {}

  static std::unique_ptr<PageWriter> open(
      std::shared_ptr<ArrowOutputStream> sink,
      Compression::type codec,
      ColumnChunkMetaDataBuilder* metadata,
      int16_t rowGroupOrdinal = -1,
      int16_t columnChunkOrdinal = -1,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      bool bufferedRowGroup = false,
      std::shared_ptr<Encryptor> headerEncryptor = NULLPTR,
      std::shared_ptr<Encryptor> dataEncryptor = NULLPTR,
      bool pageWriteChecksumEnabled = false,
      // Column_index_builder MUST outlive the PageWriter.
      ColumnIndexBuilder* columnIndexBuilder = NULLPTR,
      // Offset_index_builder MUST outlive the PageWriter.
      OffsetIndexBuilder* offsetIndexBuilder = NULLPTR,
      const util::CodecOptions& codecOptions = util::CodecOptions{});

  // TODO: remove this and port to new signature.
  // ARROW_DEPRECATED(.
  //    "Deprecated in 13.0.0. Use codecOptions-taking overload instead.")
  static std::unique_ptr<PageWriter> open(
      std::shared_ptr<ArrowOutputStream> sink,
      Compression::type codec,
      int compressionLevel,
      ColumnChunkMetaDataBuilder* metadata,
      int16_t rowGroupOrdinal = -1,
      int16_t columnChunkOrdinal = -1,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      bool bufferedRowGroup = false,
      std::shared_ptr<Encryptor> headerEncryptor = NULLPTR,
      std::shared_ptr<Encryptor> dataEncryptor = NULLPTR,
      bool pageWriteChecksumEnabled = false,
      // Column_index_builder MUST outlive the PageWriter.
      ColumnIndexBuilder* columnIndexBuilder = NULLPTR,
      // Offset_index_builder MUST outlive the PageWriter.
      OffsetIndexBuilder* offsetIndexBuilder = NULLPTR);

  // The Column Writer decides if dictionary encoding is used if set and.
  // If the dictionary encoding has fallen back to default encoding on reaching.
  // dictionary page limit.
  virtual void close(bool hasDictionary, bool fallback) = 0;

  // Return the number of uncompressed bytes written (including header size)
  virtual int64_t writeDataPage(const DataPage& page) = 0;

  // Return the number of uncompressed bytes written (including header size)
  virtual int64_t writeDictionaryPage(const DictionaryPage& page) = 0;

  /// \brief The total number of bytes written as serialized data and.
  /// Dictionary pages to the sink so far.
  virtual int64_t totalCompressedBytesWritten() const = 0;

  virtual bool hasCompressor() = 0;

  virtual void compress(
      const ::arrow::Buffer& srcBuffer,
      ::arrow::ResizableBuffer* destBuffer) = 0;
};

class PARQUET_EXPORT ColumnWriter {
 public:
  virtual ~ColumnWriter() = default;

  static std::shared_ptr<ColumnWriter> make(
      ColumnChunkMetaDataBuilder*,
      std::unique_ptr<PageWriter>,
      const WriterProperties* properties);

  /// \brief Closes the ColumnWriter, commits any buffered values to pages.
  /// \return Total size of the column in bytes.
  virtual int64_t close() = 0;

  /// \brief The physical Parquet type of the column.
  virtual Type::type type() const = 0;

  /// \brief The schema for the column.
  virtual const ColumnDescriptor* descr() const = 0;

  /// \brief The number of rows written so far.
  virtual int64_t rowsWritten() const = 0;

  /// \brief The total size of the compressed pages + page headers. Values.
  /// Are still buffered and not written to a pager yet.
  ///
  /// So in un-buffered mode, it always returns 0.
  virtual int64_t totalCompressedBytes() const = 0;

  /// \brief The total number of bytes written as serialized data and.
  /// Dictionary pages to the ColumnChunk so far.
  /// These bytes are uncompressed bytes.
  virtual int64_t totalBytesWritten() const = 0;

  /// \brief The total number of bytes written as serialized data and.
  /// Dictionary pages to the ColumnChunk so far.
  /// If the column is uncompressed, the value would be equal to.
  /// Total_bytes_written().
  virtual int64_t totalCompressedBytesWritten() const = 0;

  /// \brief The file-level writer properties.
  virtual const WriterProperties* properties() = 0;

  /// \brief Write Apache Arrow columnar data directly to ColumnWriter. Returns
  /// error status if the array data type is not compatible with the concrete
  /// writer type.
  ///
  /// Leaf_array is always a primitive (possibly dictionary encoded type).
  /// Leaf_field_nullable indicates whether the leaf array is considered.
  /// nullable according to its schema in a Table or its parent array.
  virtual ::arrow::Status writeArrow(
      const int16_t* defLevels,
      const int16_t* repLevels,
      int64_t numLevels,
      const ::arrow::Array& leafArray,
      ArrowWriteContext* ctx,
      bool leafFieldNullable) = 0;

  /// \brief Estimated size of the values that are not written to a page yet.
  virtual int64_t estimatedBufferedValueBytes() const = 0;
};

// API to write values to a single column. This is the main client facing API.
template <typename DType>
class TypedColumnWriter : public ColumnWriter {
 public:
  using T = typename DType::CType;

  // Write a batch of repetition levels, definition levels, and values to the
  // column.
  // `num_values` is the number of logical leaf values.
  // `def_levels` (resp. `rep_levels`) can be null if the column's max
  // definition level (resp. max repetition level) is 0. If not null, each of
  // `def_levels` and `rep_levels` must have at least `num_values`.
  //
  // The number of physical values written (taken from `values`) is returned.
  // It can be smaller than `num_values` is there are some undefined values.
  virtual int64_t writeBatch(
      int64_t numValues,
      const int16_t* defLevels,
      const int16_t* repLevels,
      const T* values) = 0;

  /// Write a batch of repetition levels, definition levels, and values to the
  /// column.
  ///
  /// In comparison to WriteBatch the length of repetition and definition levels
  /// is the same as of the number of values read for max_definition_level == 1.
  /// In the case of max_definition_level > 1, the repetition and definition
  /// levels are larger than the values but the values include the null entries
  /// with definition_level == (max_definition_level - 1). Thus we have to
  /// differentiate in the parameters of this function if the input has the
  /// length of num_values or the _number of rows in the lowest nesting level_.
  ///
  /// In the case that the most inner node in the Parquet is required, the
  /// _number of rows in the lowest nesting level_ is equal to the number of
  /// non-null values. If the inner-most schema node is optional, the _number of
  /// rows in the lowest nesting level_ also includes all values with
  /// definition_level == (max_definition_level - 1).
  ///
  /// @param num_values number of levels to write.
  /// @param def_levels The Parquet definition levels, length is num_values.
  /// @param rep_levels The Parquet repetition levels, length is num_values.
  /// @param valid_bits Bitmap that indicates if the row is null on the lowest.
  /// Nesting.
  ///   Level. The length is number of rows in the lowest nesting level.
  /// @param valid_bits_offset The offset in bits of the valid_bits where the.
  ///   First relevant bit resides.
  /// @param values The values in the lowest nested level including.
  ///   Spacing for nulls on the lowest levels; input has the length.
  ///   Of the number of rows on the lowest nesting level.
  virtual void writeBatchSpaced(
      int64_t numValues,
      const int16_t* defLevels,
      const int16_t* repLevels,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      const T* values) = 0;

  // Estimated size of the values that are not written to a page yet.
  virtual int64_t estimatedBufferedValueBytes() const = 0;
};

using BoolWriter = TypedColumnWriter<BooleanType>;
using Int32Writer = TypedColumnWriter<Int32Type>;
using Int64Writer = TypedColumnWriter<Int64Type>;
using Int96Writer = TypedColumnWriter<Int96Type>;
using FloatWriter = TypedColumnWriter<FloatType>;
using DoubleWriter = TypedColumnWriter<DoubleType>;
using ByteArrayWriter = TypedColumnWriter<ByteArrayType>;
using FixedLenByteArrayWriter = TypedColumnWriter<FLBAType>;

namespace internal {

// Timestamp conversion constants
constexpr int64_t kJulianEpochOffsetDays = INT64_C(2440588);

template <int64_t UnitPerDay, int64_t NanosecondsPerUnit>
inline void arrowTimestampToImpalaTimestamp(
    const int64_t Time,
    Int96* impalaTimestamp) {
  int64_t julianDays = (Time / UnitPerDay) + kJulianEpochOffsetDays;
  (*impalaTimestamp).value[2] = (uint32_t)julianDays;

  int64_t lastDayUnits = Time % UnitPerDay;
  auto lastDayNanos = lastDayUnits * NanosecondsPerUnit;
  // Impala_timestamp will be unaligned every other entry so do memcpy instead.
  // Of assign and reinterpret cast to avoid undefined behavior.
  std::memcpy(impalaTimestamp, &lastDayNanos, sizeof(int64_t));
}

constexpr int64_t kSecondsInNanos = INT64_C(1000000000);

inline void secondsToImpalaTimestamp(
    const int64_t seconds,
    Int96* impalaTimestamp) {
  arrowTimestampToImpalaTimestamp<kSecondsPerDay, kSecondsInNanos>(
      seconds, impalaTimestamp);
}

constexpr int64_t kMillisecondsInNanos = kSecondsInNanos / INT64_C(1000);

inline void millisecondsToImpalaTimestamp(
    const int64_t milliseconds,
    Int96* impalaTimestamp) {
  arrowTimestampToImpalaTimestamp<kMillisecondsPerDay, kMillisecondsInNanos>(
      milliseconds, impalaTimestamp);
}

constexpr int64_t kMicrosecondsInNanos = kMillisecondsInNanos / INT64_C(1000);

inline void microsecondsToImpalaTimestamp(
    const int64_t microseconds,
    Int96* impalaTimestamp) {
  arrowTimestampToImpalaTimestamp<kMicrosecondsPerDay, kMicrosecondsInNanos>(
      microseconds, impalaTimestamp);
}

constexpr int64_t kNanosecondsInNanos = INT64_C(1);

inline void nanosecondsToImpalaTimestamp(
    const int64_t nanoseconds,
    Int96* impalaTimestamp) {
  arrowTimestampToImpalaTimestamp<kNanosecondsPerDay, kNanosecondsInNanos>(
      nanoseconds, impalaTimestamp);
}

} // namespace internal
} // namespace facebook::velox::parquet::arrow

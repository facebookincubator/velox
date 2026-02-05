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
#include <memory>
#include <utility>
#include <vector>

#include "velox/dwio/parquet/common/LevelConversion.h"
#include "velox/dwio/parquet/common/RleEncodingInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace arrow {

class Array;
class ChunkedArray;

namespace bit_util {
class BitReader;
} // namespace bit_util

namespace util {
class RleDecoder;
} // namespace util

} // namespace arrow

namespace facebook::velox::parquet::arrow {

class Decryptor;
class Page;

// 16 MB is the default maximum page header size.
static constexpr uint32_t kDefaultMaxPageHeaderSize = 16 * 1024 * 1024;

// 16 KB is the default expected page header size.
static constexpr uint32_t kDefaultPageHeaderSize = 16 * 1024;

// \brief DataPageStats stores encoded statistics and number of values/rows for.
// A page.
struct PARQUET_EXPORT dataPageStats {
  dataPageStats(
      const EncodedStatistics* EncodedStatistics,
      int32_t numValues,
      std::optional<int32_t> numRows)
      : EncodedStatistics(EncodedStatistics),
        numValues(numValues),
        numRows(numRows) {}

  // Encoded statistics extracted from the page header.
  // Nullptr if there are no statistics in the page header.
  const EncodedStatistics* EncodedStatistics;
  // Number of values stored in the page. Filled for both V1 and V2 data pages.
  // For repeated fields, this can be greater than number of rows. For.
  // Non-repeated fields, this will be the same as the number of rows.
  int32_t numValues;
  // Number of rows stored in the page. std::nullopt if not available.
  std::optional<int32_t> numRows;
};

class PARQUET_EXPORT LevelDecoder {
 public:
  LevelDecoder();
  ~LevelDecoder();

  // Initialize the LevelDecoder state with new data.
  // And return the number of bytes consumed.
  int setData(
      Encoding::type encoding,
      int16_t maxLevel,
      int numBufferedValues,
      const uint8_t* data,
      int32_t dataSize);

  void setDataV2(
      int32_t numBytes,
      int16_t maxLevel,
      int numBufferedValues,
      const uint8_t* data);

  // Decodes a batch of levels into an array and returns the number of levels.
  // Decoded.
  int decode(int batchSize, int16_t* levels);

 private:
  int bitWidth_;
  int numValuesRemaining_;
  Encoding::type encoding_;
  std::unique_ptr<RleDecoder> rleDecoder_;
  std::unique_ptr<BitReader> bitPackedDecoder_;
  int16_t maxLevel_;
};

struct CryptoContext {
  CryptoContext(
      bool startWithDictionaryPage,
      int16_t rgOrdinal,
      int16_t colOrdinal,
      std::shared_ptr<Decryptor> meta,
      std::shared_ptr<Decryptor> data)
      : startDecryptWithDictionaryPage(startWithDictionaryPage),
        rowGroupOrdinal(rgOrdinal),
        columnOrdinal(colOrdinal),
        metaDecryptor(std::move(meta)),
        dataDecryptor(std::move(data)) {}
  CryptoContext() {}

  bool startDecryptWithDictionaryPage = false;
  int16_t rowGroupOrdinal = -1;
  int16_t columnOrdinal = -1;
  std::shared_ptr<Decryptor> metaDecryptor;
  std::shared_ptr<Decryptor> dataDecryptor;
};

// Abstract page iterator interface. This way, we can feed column pages to the.
// ColumnReader through whatever mechanism we choose.
class PARQUET_EXPORT PageReader {
  using DataPageFilter = std::function<bool(const dataPageStats&)>;

 public:
  virtual ~PageReader() = default;

  static std::unique_ptr<PageReader> open(
      std::shared_ptr<ArrowInputStream> stream,
      int64_t totalNumValues,
      Compression::type Codec,
      bool alwaysCompressed = false,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      const CryptoContext* ctx = NULLPTR);
  static std::unique_ptr<PageReader> open(
      std::shared_ptr<ArrowInputStream> stream,
      int64_t totalNumValues,
      Compression::type Codec,
      const ReaderProperties& properties,
      bool alwaysCompressed = false,
      const CryptoContext* ctx = NULLPTR);

  // If data_page_filter is present (not null), NextPage() will call the.
  // Callback function exactly once per page in the order the pages appear in.
  // The column. If the callback function returns true the page will be.
  // Skipped. The callback will be called only if the page type is DATA_PAGE or.
  // DATA_PAGE_V2. Dictionary pages will not be skipped.
  // Caller is responsible for checking that statistics are correct using.
  // ApplicationVersion::HasCorrectStatistics().
  // \note API EXPERIMENTAL.
  void setDataPageFilter(DataPageFilter dataPageFilter) {
    dataPageFilter_ = std::move(dataPageFilter);
  }

  // @returns: shared_ptr<Page>(nullptr) on EOS, std::shared_ptr<Page>.
  // Containing new Page otherwise.
  //
  // The returned Page may contain references that aren't guaranteed to live.
  // Beyond the next call to NextPage().
  virtual std::shared_ptr<Page> nextPage() = 0;

  virtual void setMaxPageHeaderSize(uint32_t size) = 0;

 protected:
  // Callback that decides if we should skip a page or not.
  DataPageFilter dataPageFilter_;
};

class PARQUET_EXPORT ColumnReader {
 public:
  virtual ~ColumnReader() = default;

  static std::shared_ptr<ColumnReader> make(
      const ColumnDescriptor* descr,
      std::unique_ptr<PageReader> pager,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  // Returns true if there are still values in this column.
  virtual bool hasNext() = 0;

  virtual Type::type type() const = 0;

  virtual const ColumnDescriptor* descr() const = 0;

  // Get the encoding that can be exposed by this reader. If it returns.
  // Dictionary encoding, then ReadBatchWithDictionary can be used to read data.
  //
  // \note API EXPERIMENTAL.
  virtual ExposedEncoding getExposedEncoding() = 0;

 protected:
  friend class RowGroupReader;
  // Set the encoding that can be exposed by this reader.
  //
  // \note API EXPERIMENTAL.
  virtual void setExposedEncoding(ExposedEncoding encoding) = 0;
};

// API to read values from a single column. This is a main client facing API.
template <typename DType>
class TypedColumnReader : public ColumnReader {
 public:
  typedef typename DType::CType T;

  // Read a batch of repetition levels, definition levels, and values from the.
  // Column.
  //
  // Since null values are not stored in the values, the number of values read.
  // May be less than the number of repetition and definition levels. With.
  // Nested data this is almost certainly true.
  //
  // Set def_levels or rep_levels to nullptr if you want to skip reading them.
  // This is only safe if you know through some other source that there are no.
  // Undefined values.
  //
  // To fully exhaust a row group, you must read batches until the number of.
  // Values read reaches the number of stored values according to the metadata.
  //
  // This API is the same for both V1 and V2 of the DataPage.
  //
  // @returns: actual number of levels read (see values_read for number of.
  // values read)
  virtual int64_t readBatch(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      T* values,
      int64_t* valuesRead) = 0;

  /// Read a batch of repetition levels, definition levels, and values from the.
  /// Column and leave spaces for null entries on the lowest level in the
  /// values. Buffer.
  ///
  /// In comparison to ReadBatch the length of repetition and definition levels.
  /// Is the same as of the number of values read for max_definition_level == 1.
  /// In the case of max_definition_level > 1, the repetition and definition.
  /// Levels are larger than the values but the values include the null entries.
  /// With definition_level == (max_definition_level - 1).
  ///
  /// To fully exhaust a row group, you must read batches until the number of.
  /// Values read reaches the number of stored values according to the metadata.
  ///
  /// @param batch_size the number of levels to read.
  /// @param[out] def_levels The Parquet definition levels, output has.
  ///   The length levels_read.
  /// @param[out] rep_levels The Parquet repetition levels, output has.
  ///   The length levels_read.
  /// @param[out] values The values in the lowest nested level including.
  ///   Spacing for nulls on the lowest levels; output has the length.
  ///   Values_read.
  /// @param[out] valid_bits Memory allocated for a bitmap that indicates if.
  ///   The row is null or on the maximum definition level. For performance.
  ///   Reasons the underlying buffer should be able to store 1 bit more than.
  ///   Required. If this requires an additional byte, this byte is only read.
  ///   But never written to.
  /// @param valid_bits_offset The offset in bits of the valid_bits where the.
  ///   First relevant bit resides.
  /// @param[out] levels_read The number of repetition/definition levels that.
  /// Were read.
  /// @param[out] values_read The number of values read, this includes all.
  ///   Non-null entries as well as all null-entries on the lowest level.
  ///   (i.e. definition_level == max_definition_level - 1)
  /// @param[out] null_count The number of nulls on the lowest levels.
  ///   (i.e. (values_read - null_count) is total number of non-null entries)
  ///
  /// \deprecated Since 4.0.0.
  ARROW_DEPRECATED(
      "Doesn't handle nesting correctly and unused outside of unit tests.")
  virtual int64_t readBatchSpaced(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      T* values,
      uint8_t* validBits,
      int64_t validBitsOffset,
      int64_t* levelsRead,
      int64_t* valuesRead,
      int64_t* nullCount) = 0;

  // Skip reading values. This method will work for both repeated and.
  // Non-repeated fields. Note that this method is skipping values and not.
  // Records. This distinction is important for repeated fields, meaning that.
  // We are not skipping over the values to the next record. For example,.
  // Consider the following two consecutive records containing one repeated.
  // Field:
  // {[1, 2, 3]}, {[4, 5]}. If we Skip(2), our next read value will be 3, which.
  // Is inside the first record.
  // Returns the number of values skipped.
  virtual int64_t skip(int64_t numValuesToSkip) = 0;

  // Read a batch of repetition levels, definition levels, and indices from the.
  // Column. And read the dictionary if a dictionary page is encountered during.
  // Reading pages. This API is similar to ReadBatch(), with ability to read.
  // Dictionary and indices. It is only valid to call this method  when the.
  // Reader can expose dictionary encoding. (i.e., the reader's.
  // GetExposedEncoding() returns DICTIONARY).
  //
  // The dictionary is read along with the data page. When there's no data
  // page,. The dictionary won't be returned.
  //
  // @param batch_size The batch size to read.
  // @param[out] def_levels The Parquet definition levels.
  // @param[out] rep_levels The Parquet repetition levels.
  // @param[out] indices The dictionary indices.
  // @param[out] indices_read The number of indices read.
  // @param[out] dict The pointer to dictionary values. It will return nullptr.
  // If there's no data page. Each column chunk only has one dictionary page.
  // The dictionary is owned by the reader, so the caller is responsible for.
  // Copying the dictionary values before the reader gets destroyed.
  // @param[out] dict_len The dictionary length. It will return 0 if there's no.
  // Data page.
  // @returns: actual number of levels read (see indices_read for number of.
  // Indices read.
  //
  // \note API EXPERIMENTAL.
  virtual int64_t readBatchWithDictionary(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      int32_t* indices,
      int64_t* indicesRead,
      const T** dict,
      int32_t* dictLen) = 0;
};

namespace internal {

/// \brief Stateful column reader that delimits semantic records for both flat.
/// And nested columns.
///
/// \note API EXPERIMENTAL.
/// \since 1.3.0.
class PARQUET_EXPORT RecordReader {
 public:
  /// \brief Creates a record reader.
  /// @param descr Column descriptor.
  /// @param leaf_info Level info, used to determine if a column is nullable or.
  /// Not.
  /// @param pool Memory pool to use for buffering values and rep/def levels.
  /// @param read_dictionary True if reading directly as Arrow.
  /// Dictionary-encoded.
  /// @param read_dense_for_nullable True if reading dense and not leaving
  /// space. For null values.
  static std::shared_ptr<RecordReader> make(
      const ColumnDescriptor* descr,
      LevelInfo leafInfo,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      bool readDictionary = false,
      bool readDenseForNullable = false);

  virtual ~RecordReader() = default;

  /// \brief Attempt to read indicated number of records from column chunk.
  /// Note that for repeated fields, a record may have more than one value.
  /// And all of them are read. If read_dense_for_nullable() it will.
  /// Not leave any space for null values. Otherwise, it will read spaced.
  /// \return number of records read.
  virtual int64_t readRecords(int64_t numRecords) = 0;

  /// \brief Attempt to skip indicated number of records from column chunk.
  /// Note that for repeated fields, a record may have more than one value.
  /// And all of them are skipped.
  /// \return number of records skipped.
  virtual int64_t skipRecords(int64_t numRecords) = 0;

  /// \brief Pre-allocate space for data. Results in better flat read.
  /// Performance.
  virtual void reserve(int64_t numValues) = 0;

  /// \brief Clear consumed values and repetition/definition levels as the.
  /// Result of calling ReadRecords.
  /// For FLBA and ByteArray types, call GetBuilderChunks() to reset them.
  virtual void reset() = 0;

  /// \brief Transfer filled values buffer to caller. A new one will be.
  /// Allocated in subsequent ReadRecords calls.
  virtual std::shared_ptr<ResizableBuffer> releaseValues() = 0;

  /// \brief Transfer filled validity bitmap buffer to caller. A new one will.
  /// Be allocated in subsequent ReadRecords calls.
  virtual std::shared_ptr<ResizableBuffer> releaseIsValid() = 0;

  /// \brief Return true if the record reader has more internal data yet to.
  /// Process.
  virtual bool hasMoreData() const = 0;

  /// \brief Advance record reader to the next row group. Must be set before.
  /// Any records could be read/skipped.
  /// \param[in] reader obtained from RowGroupReader::GetColumnPageReader.
  virtual void setPageReader(std::unique_ptr<PageReader> reader) = 0;

  /// \brief Returns the underlying column reader's descriptor.
  virtual const ColumnDescriptor* descr() const = 0;

  virtual void debugPrintState() = 0;

  /// \brief Decoded definition levels.
  int16_t* defLevels() const {
    return reinterpret_cast<int16_t*>(defLevels_->mutable_data());
  }

  /// \brief Decoded repetition levels.
  int16_t* repLevels() const {
    return reinterpret_cast<int16_t*>(repLevels_->mutable_data());
  }

  /// \brief Decoded values, including nulls, if any.
  /// FLBA and ByteArray types do not use this array and read into their own.
  /// Builders.
  uint8_t* values() const {
    return values_->mutable_data();
  }

  /// \brief Number of values written, including space left for nulls if any.
  /// If this Reader was constructed with read_dense_for_nullable(), there is
  /// no. Space for nulls and null_count() will be 0. There is no.
  /// Read-ahead/buffering for values. For FLBA and ByteArray types this value.
  /// Reflects the values written with the last ReadRecords call since those.
  /// Readers will reset the values after each call.
  int64_t valuesWritten() const {
    return valuesWritten_;
  }

  /// \brief Number of definition / repetition levels (from those that have.
  /// Been decoded) that have been consumed inside the reader.
  int64_t levelsPosition() const {
    return levelsPosition_;
  }

  /// \brief Number of definition / repetition levels that have been written.
  /// Internally in the reader. This may be larger than values_written()
  /// because. For repeated fields we need to look at the levels in advance to
  /// figure out. The record boundaries.
  int64_t levelsWritten() const {
    return levelsWritten_;
  }

  /// \brief Number of nulls in the leaf that we have read so far into the.
  /// Values vector. This is only valid when !read_dense_for_nullable(). When.
  /// Read_dense_for_nullable() it will always be 0.
  int64_t nullCount() const {
    return nullCount_;
  }

  /// \brief True if the leaf values are nullable.
  bool nullableValues() const {
    return nullableValues_;
  }

  /// \brief True if reading directly as Arrow dictionary-encoded.
  bool readDictionary() const {
    return readDictionary_;
  }

  /// \brief True if reading dense for nullable columns.
  bool readDenseForNullable() const {
    return readDenseForNullable_;
  }

 protected:
  /// \brief Indicates if we can have nullable values. Note that repeated
  /// fields. May or may not be nullable.
  bool nullableValues_;

  bool atRecordStart_;
  int64_t recordsRead_;

  /// \brief Stores values. These values are populated based on each
  /// ReadRecords. Call. No extra values are buffered for the next call.
  /// SkipRecords will not. Add any value to this buffer.
  std::shared_ptr<::arrow::ResizableBuffer> values_;
  /// \brief False for BYTE_ARRAY, in which case we don't allocate the values.
  /// Buffer and we directly read into builder classes.
  bool usesValues_;

  /// \brief Values that we have read into 'values_' + 'null_count_'.
  int64_t valuesWritten_;
  int64_t valuesCapacity_;
  int64_t nullCount_;

  /// \brief Each bit corresponds to one element in 'values_' and specifies if.
  /// It is null or not null. Not set if read_dense_for_nullable_ is true.
  std::shared_ptr<::arrow::ResizableBuffer> validBits_;

  /// \brief Buffer for definition levels. May contain more levels than.
  /// Is actually read. This is because we read levels ahead to.
  /// Figure out record boundaries for repeated fields.
  /// For flat required fields, 'def_levels_' and 'rep_levels_' are not.
  ///  Populated. For non-repeated fields 'rep_levels_' is not populated.
  /// 'Def_levels_' and 'rep_levels_' must be of the same size if present.
  std::shared_ptr<::arrow::ResizableBuffer> defLevels_;
  /// \brief Buffer for repetition levels. Only populated for repeated.
  /// Fields.
  std::shared_ptr<::arrow::ResizableBuffer> repLevels_;

  /// \brief Number of definition / repetition levels that have been written.
  /// Internally in the reader. This may be larger than values_written() since.
  /// For repeated fields we need to look at the levels in advance to figure
  /// out. The record boundaries.
  int64_t levelsWritten_;
  /// \brief Position of the next level that should be consumed.
  int64_t levelsPosition_;
  int64_t levelsCapacity_;

  bool readDictionary_ = false;
  // If true, we will not leave any space for the null values in the values_.
  // Vector.
  bool readDenseForNullable_ = false;
};

class BinaryRecordReader : virtual public RecordReader {
 public:
  virtual std::vector<std::shared_ptr<::arrow::Array>> getBuilderChunks() = 0;
};

/// \brief Read records directly to dictionary-encoded Arrow form (int32.
/// Indices). Only valid for BYTE_ARRAY columns.
class DictionaryRecordReader : virtual public RecordReader {
 public:
  virtual std::shared_ptr<::arrow::ChunkedArray> getResult() = 0;
};

} // namespace internal

using BoolReader = TypedColumnReader<BooleanType>;
using Int32Reader = TypedColumnReader<Int32Type>;
using Int64Reader = TypedColumnReader<Int64Type>;
using Int96Reader = TypedColumnReader<Int96Type>;
using FloatReader = TypedColumnReader<FloatType>;
using DoubleReader = TypedColumnReader<DoubleType>;
using ByteArrayReader = TypedColumnReader<ByteArrayType>;
using FixedLenByteArrayReader = TypedColumnReader<FLBAType>;

} // namespace facebook::velox::parquet::arrow

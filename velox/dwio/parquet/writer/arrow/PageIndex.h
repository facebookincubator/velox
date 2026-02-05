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

#include "arrow/io/interfaces.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

#include <optional>
#include <vector>

namespace facebook::velox::parquet::arrow {

class ColumnDescriptor;
class EncodedStatistics;
class FileMetaData;
class InternalFileDecryptor;
struct PageIndexLocation;
class ReaderProperties;
class RowGroupMetaData;
class RowGroupPageIndexReader;
class SchemaDescriptor;

/// \brief ColumnIndex is a proxy around.
/// Facebook::velox::parquet::thrift::ColumnIndex.
class PARQUET_EXPORT ColumnIndex {
 public:
  /// \brief Create a ColumnIndex from a serialized thrift message.
  static std::unique_ptr<ColumnIndex> make(
      const ColumnDescriptor& descr,
      const void* serializedIndex,
      uint32_t indexLen,
      const ReaderProperties& properties);

  virtual ~ColumnIndex() = default;

  /// \brief A bitmap with a bit set for each data page that has only null.
  /// Values.
  ///
  /// The length of this vector is equal to the number of data pages in the.
  /// Column.
  virtual const std::vector<bool>& nullPages() const = 0;

  /// \brief A vector of encoded lower bounds for each data page in this column.
  ///
  /// `Null_pages` should be inspected first, as only pages with non-null
  /// values. May have their lower bounds populated.
  virtual const std::vector<std::string>& encodedMinValues() const = 0;

  /// \brief A vector of encoded upper bounds for each data page in this column.
  ///
  /// `Null_pages` should be inspected first, as only pages with non-null
  /// values. May have their upper bounds populated.
  virtual const std::vector<std::string>& encodedMaxValues() const = 0;

  /// \brief The ordering of lower and upper bounds.
  ///
  /// The boundary order applies across all lower bounds, and all upper bounds,.
  /// Respectively. However, the order between lower bounds and upper bounds.
  /// Cannot be derived from this.
  virtual BoundaryOrder::type boundaryOrder() const = 0;

  /// \brief Whether per-page null count information is available.
  virtual bool hasNullCounts() const = 0;

  /// \brief An optional vector with the number of null values in each data.
  /// Page.
  ///
  /// `Has_null_counts` should be called first to determine if this information.
  /// Is available.
  virtual const std::vector<int64_t>& nullCounts() const = 0;

  /// \brief A vector of page indices for non-null pages.
  virtual const std::vector<int32_t>& nonNullPageIndices() const = 0;
};

/// \brief Typed implementation of ColumnIndex.
template <typename DType>
class PARQUET_EXPORT TypedColumnIndex : public ColumnIndex {
 public:
  using T = typename DType::CType;

  /// \brief A vector of lower bounds for each data page in this column.
  ///
  /// This is like `encoded_min_values`, but with the values decoded according.
  /// To the column's physical type. `min_values` and `max_values` can be used.
  /// Together with `boundary_order` in order to prune some data pages when.
  /// Searching for specific values.
  virtual const std::vector<T>& minValues() const = 0;

  /// \brief A vector of upper bounds for each data page in this column.
  ///
  /// Just like `min_values`, but for upper bounds instead of lower bounds.
  virtual const std::vector<T>& maxValues() const = 0;
};

using BoolColumnIndex = TypedColumnIndex<BooleanType>;
using Int32ColumnIndex = TypedColumnIndex<Int32Type>;
using Int64ColumnIndex = TypedColumnIndex<Int64Type>;
using FloatColumnIndex = TypedColumnIndex<FloatType>;
using DoubleColumnIndex = TypedColumnIndex<DoubleType>;
using ByteArrayColumnIndex = TypedColumnIndex<ByteArrayType>;
using FLBAColumnIndex = TypedColumnIndex<FLBAType>;

/// \brief PageLocation is a proxy around.
/// Facebook::velox::parquet::thrift::PageLocation.
struct PARQUET_EXPORT PageLocation {
  /// File offset of the data page.
  int64_t offset;
  /// Total compressed size of the data page and header.
  int32_t compressedPageSize;
  /// Row id of the first row in the page within the row group.
  int64_t firstRowIndex;
};

/// \brief OffsetIndex is a proxy around.
/// Facebook::velox::parquet::thrift::OffsetIndex.
class PARQUET_EXPORT OffsetIndex {
 public:
  /// \brief Create a OffsetIndex from a serialized thrift message.
  static std::unique_ptr<OffsetIndex> make(
      const void* serializedIndex,
      uint32_t indexLen,
      const ReaderProperties& properties);

  virtual ~OffsetIndex() = default;

  /// \brief A vector of locations for each data page in this column.
  virtual const std::vector<PageLocation>& pageLocations() const = 0;
};

/// \brief Interface for reading the page index for a Parquet row group.
class PARQUET_EXPORT RowGroupPageIndexReader {
 public:
  virtual ~RowGroupPageIndexReader() = default;

  /// \brief Read column index of a column chunk.
  ///
  /// \param[in] i column ordinal of the column chunk.
  /// \returns column index of the column or nullptr if it does not exist.
  /// \throws ParquetException if the index is out of bound.
  virtual std::shared_ptr<ColumnIndex> getColumnIndex(int32_t i) = 0;

  /// \brief Read offset index of a column chunk.
  ///
  /// \param[in] i column ordinal of the column chunk.
  /// \returns offset index of the column or nullptr if it does not exist.
  /// \throws ParquetException if the index is out of bound.
  virtual std::shared_ptr<OffsetIndex> getOffsetIndex(int32_t i) = 0;
};

struct PageIndexSelection {
  /// Specifies whether to read the column index.
  bool columnIndex = false;
  /// Specifies whether to read the offset index.
  bool offsetIndex = false;
};

PARQUET_EXPORT
std::ostream& operator<<(std::ostream& out, const PageIndexSelection& params);

struct RowGroupIndexReadRange {
  /// Base start and total size of column index of all column chunks in a row.
  /// Group. If none of the column chunks have column index, it is set to.
  /// Std::nullopt.
  std::optional<::arrow::io::ReadRange> columnIndex = std::nullopt;
  /// Base start and total size of offset index of all column chunks in a row.
  /// Group. If none of the column chunks have offset index, it is set to.
  /// Std::nullopt.
  std::optional<::arrow::io::ReadRange> offsetIndex = std::nullopt;
};

/// \brief Interface for reading the page index for a Parquet file.
class PARQUET_EXPORT PageIndexReader {
 public:
  virtual ~PageIndexReader() = default;

  /// \brief Create a PageIndexReader instance.
  /// \returns a PageIndexReader instance.
  /// WARNING: The returned PageIndexReader references to all the input.
  /// Parameters, so it must not outlive all of the input parameters. Usually.
  /// These input parameters come from the same ParquetFileReader object, so it.
  /// Must not outlive the reader that creates this PageIndexReader.
  static std::shared_ptr<PageIndexReader> make(
      ::arrow::io::RandomAccessFile* input,
      std::shared_ptr<FileMetaData> fileMetadata,
      const ReaderProperties& properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  /// \brief Get the page index reader of a specific row group.
  /// \param[in] i row group ordinal to get page index reader.
  /// \returns RowGroupPageIndexReader of the specified row group. A nullptr
  /// may. Or may.
  ///          Not be returned if the page index for the row group is.
  ///          Unavailable. It is the caller's responsibility to check the.
  ///          Return value of follow-up calls to the RowGroupPageIndexReader.
  /// \throws ParquetException if the index is out of bound.
  virtual std::shared_ptr<RowGroupPageIndexReader> rowGroup(int i) = 0;

  /// \brief Advise the reader which part of page index will be read later.
  ///
  /// The PageIndexReader can optionally prefetch and cache page index that.
  /// May be read later to get better performance.
  ///
  /// The contract of this function is as below:
  /// 1) If WillNeed() has not been called for a specific row group and the
  /// page. Index.
  ///    Exists, follow-up calls to get column index or offset index of all.
  ///    Columns in this row group SHOULD NOT FAIL, but the performance may not.
  ///    Be optimal.
  /// 2) If WillNeed() has been called for a specific row group, follow-up
  /// calls. To get.
  ///    Page index are limited to columns and index type requested by.
  ///    WillNeed(). So it MAY FAIL if columns that are not requested by.
  ///    WillNeed() are requested.
  /// 3) Later calls to WillNeed() MAY OVERRIDE previous calls of same row.
  /// Groups. For example, 1) If WillNeed() is not called for row group 0, then.
  /// Follow-up calls to read.
  ///    Column index and/or offset index of all columns of row group 0 should.
  ///    Not fail if its page index exists.
  /// 2) If WillNeed() is called for columns 0 and 1 for row group 0, then.
  /// Follow-up.
  ///    Call to read page index of column 2 for row group 0 MAY FAIL even if.
  ///    Its page index exists.
  /// 3) If WillNeed() is called for row group 0 with offset index only, then.
  ///    Follow-up call to read column index of row group 0 MAY FAIL even if.
  ///    The column index of this column exists.
  /// 4) If WillNeed() is called for columns 0 and 1 for row group 0, then
  /// later.
  ///    Call to WillNeed() for columns 1 and 2 for row group 0. The later one.
  ///    Overrides previous call and only columns 1 and 2 of row group 0 are.
  ///    Allowed to access.
  ///
  /// \param[in] row_group_indices list of row group ordinal to read page index.
  /// Later. \param[in] column_indices list of column ordinal to read page
  /// index. Later. If it is.
  ///            Empty, it means all columns in the row group will be read.
  /// \param[in] selection which kind of page index is required later.
  virtual void willNeed(
      const std::vector<int32_t>& rowGroupIndices,
      const std::vector<int32_t>& columnIndices,
      const PageIndexSelection& selection) = 0;

  /// \brief Advise the reader page index of these row groups will not be read.
  /// Any more.
  ///
  /// The PageIndexReader implementation has the opportunity to cancel any.
  /// Prefetch or release resource that are related to these row groups.
  ///
  /// \param[in] row_group_indices list of row group ordinal that whose page.
  /// Index will not be accessed any more.
  virtual void willNotNeed(const std::vector<int32_t>& rowGroupIndices) = 0;

  /// \brief Determine the column index and offset index ranges for the given.
  /// Row group.
  ///
  /// \param[in] row_group_metadata row group metadata to get column chunk.
  /// Metadata. \param[in] columns list of column ordinals to get page index.
  /// If. The list is empty,.
  ///            It means all columns in the row group.
  /// \returns RowGroupIndexReadRange of the specified row group. Throws.
  /// ParquetException.
  ///          If the selected column ordinal is out of bound or metadata of.
  ///          Page index is corrupted.
  static RowGroupIndexReadRange determinePageIndexRangesInRowGroup(
      const RowGroupMetaData& rowGroupMetadata,
      const std::vector<int32_t>& columns);
};

/// \brief Interface for collecting column index of data pages in a column.
/// Chunk.
class PARQUET_EXPORT ColumnIndexBuilder {
 public:
  /// \brief API convenience to create a ColumnIndexBuilder.
  static std::unique_ptr<ColumnIndexBuilder> make(
      const ColumnDescriptor* descr);

  virtual ~ColumnIndexBuilder() = default;

  /// \brief Add statistics of a data page.
  ///
  /// If the ColumnIndexBuilder has seen any corrupted statistics, it will.
  /// Not update statistics any more.
  ///
  /// \param stats Page statistics in the encoded form.
  virtual void addPage(const EncodedStatistics& stats) = 0;

  /// \brief Complete the column index.
  ///
  /// Once called, AddPage() can no longer be called.
  /// WriteTo() and Build() can only called after Finish() has been called.
  virtual void finish() = 0;

  /// \brief Serialize the column index thrift message.
  ///
  /// If the ColumnIndexBuilder has seen any corrupted statistics, it will.
  /// Not write any data to the sink.
  ///
  /// \param[out] sink output stream to write the serialized message.
  virtual void writeTo(::arrow::io::OutputStream* sink) const = 0;

  /// \brief Create a ColumnIndex directly.
  ///
  /// \return If the ColumnIndexBuilder has seen any corrupted statistics, it.
  /// Simply returns nullptr. Otherwise the column index is built and returned.
  virtual std::unique_ptr<ColumnIndex> build() const = 0;
};

/// \brief Interface for collecting offset index of data pages in a column.
/// Chunk.
class PARQUET_EXPORT OffsetIndexBuilder {
 public:
  /// \brief API convenience to create a OffsetIndexBuilder.
  static std::unique_ptr<OffsetIndexBuilder> make();

  virtual ~OffsetIndexBuilder() = default;

  /// \brief Add page location of a data page.
  virtual void addPage(
      int64_t offset,
      int32_t compressedPageSize,
      int64_t firstRowIndex) = 0;

  /// \brief Add page location of a data page.
  void addPage(const PageLocation& pageLocation) {
    addPage(
        pageLocation.offset,
        pageLocation.compressedPageSize,
        pageLocation.firstRowIndex);
  }

  /// \brief Complete the offset index.
  ///
  /// In the buffered row group mode, data pages are flushed into memory.
  /// Sink and the OffsetIndexBuilder has only collected the relative offset.
  /// Which requires adjustment once they are flushed to the file.
  ///
  /// \param final_position Final stream offset to add for page offset.
  /// Adjustment.
  virtual void finish(int64_t finalPosition) = 0;

  /// \brief Serialize the offset index thrift message.
  ///
  /// \param[out] sink output stream to write the serialized message.
  virtual void writeTo(::arrow::io::OutputStream* sink) const = 0;

  /// \brief Create an OffsetIndex directly.
  virtual std::unique_ptr<OffsetIndex> build() const = 0;
};

/// \brief Interface for collecting page index of a parquet file.
class PARQUET_EXPORT PageIndexBuilder {
 public:
  /// \brief API convenience to create a PageIndexBuilder.
  static std::unique_ptr<PageIndexBuilder> make(const SchemaDescriptor* schema);

  virtual ~PageIndexBuilder() = default;

  /// \brief Start a new row group.
  virtual void appendRowGroup() = 0;

  /// \brief Get the ColumnIndexBuilder from column ordinal.
  ///
  /// \param i Column ordinal.
  /// \return ColumnIndexBuilder for the column and its memory ownership
  /// belongs. To the PageIndexBuilder.
  virtual ColumnIndexBuilder* getColumnIndexBuilder(int32_t i) = 0;

  /// \brief Get the OffsetIndexBuilder from column ordinal.
  ///
  /// \param i Column ordinal.
  /// \return OffsetIndexBuilder for the column and its memory ownership
  /// belongs. To the PageIndexBuilder.
  virtual OffsetIndexBuilder* getOffsetIndexBuilder(int32_t i) = 0;

  /// \brief Complete the page index builder and no more write is allowed.
  virtual void finish() = 0;

  /// \brief Serialize the page index thrift message.
  ///
  /// Only valid column indexes and offset indexes are serialized and their.
  /// Locations are set.
  ///
  /// \param[out] sink The output stream to write the page index.
  /// \param[out] location The location of all page index to the start of sink.
  virtual void writeTo(
      ::arrow::io::OutputStream* sink,
      PageIndexLocation* location) const = 0;
};

} // namespace facebook::velox::parquet::arrow

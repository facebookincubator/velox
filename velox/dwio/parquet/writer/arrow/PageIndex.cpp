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

#include "velox/dwio/parquet/writer/arrow/PageIndex.h"

#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/util/OverflowUtilInternal.h"

#include "arrow/util/unreachable.h"

#include <limits>
#include <numeric>

namespace facebook::velox::parquet::arrow {

namespace {

template <typename DType>
void decode(
    std::unique_ptr<typename EncodingTraits<DType>::Decoder>& decoder,
    const std::string& input,
    std::vector<typename DType::CType>* output,
    size_t outputIndex) {
  if (ARROW_PREDICT_FALSE(outputIndex >= output->size())) {
    throw ParquetException("Index out of bound");
  }

  decoder->setData(
      /*num_values=*/1,
      reinterpret_cast<const uint8_t*>(input.c_str()),
      static_cast<int>(input.size()));
  const auto numValues =
      decoder->decode(&output->at(outputIndex), /*max_values=*/1);
  if (ARROW_PREDICT_FALSE(numValues != 1)) {
    throw ParquetException("Could not decode statistics value");
  }
}

template <>
void decode<BooleanType>(
    std::unique_ptr<BooleanDecoder>& decoder,
    const std::string& input,
    std::vector<bool>* output,
    size_t outputIndex) {
  if (ARROW_PREDICT_FALSE(outputIndex >= output->size())) {
    throw ParquetException("Index out of bound");
  }

  bool value;
  decoder->setData(
      /*num_values=*/1,
      reinterpret_cast<const uint8_t*>(input.c_str()),
      static_cast<int>(input.size()));
  const auto numValues = decoder->decode(&value, /*max_values=*/1);
  if (ARROW_PREDICT_FALSE(numValues != 1)) {
    throw ParquetException("Could not decode statistics value");
  }
  output->at(outputIndex) = value;
}

template <>
void decode<ByteArrayType>(
    std::unique_ptr<ByteArrayDecoder>&,
    const std::string& input,
    std::vector<ByteArray>* output,
    size_t outputIndex) {
  if (ARROW_PREDICT_FALSE(outputIndex >= output->size())) {
    throw ParquetException("Index out of bound");
  }

  if (ARROW_PREDICT_FALSE(
          input.size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()))) {
    throw ParquetException("Invalid encoded byte array length");
  }

  output->at(outputIndex) = {
      /*len=*/static_cast<uint32_t>(input.size()),
      /*ptr=*/reinterpret_cast<const uint8_t*>(input.data())};
}

template <typename DType>
class TypedColumnIndexImpl : public TypedColumnIndex<DType> {
 public:
  using T = typename DType::CType;

  TypedColumnIndexImpl(
      const ColumnDescriptor& descr,
      facebook::velox::parquet::thrift::ColumnIndex columnIndex)
      : columnIndex_(std::move(columnIndex)) {
    // Make sure the number of pages is valid and it does not overflow to.
    // Int32_t.
    const size_t numPages = columnIndex_.null_pages.size();
    if (numPages >= static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        columnIndex_.min_values.size() != numPages ||
        columnIndex_.max_values.size() != numPages ||
        (columnIndex_.__isset.null_counts &&
         columnIndex_.null_counts.size() != numPages)) {
      throw ParquetException("Invalid column index");
    }

    const size_t numNonNullPages = static_cast<size_t>(std::accumulate(
        columnIndex_.null_pages.cbegin(),
        columnIndex_.null_pages.cend(),
        0,
        [](int32_t numNonNullPages, bool nullPage) {
          return numNonNullPages + (nullPage ? 0 : 1);
        }));
    VELOX_DCHECK_LE(numNonNullPages, numPages);

    // Allocate slots for decoded values.
    minValues_.resize(numPages);
    maxValues_.resize(numPages);
    nonNullPageIndices_.reserve(numNonNullPages);

    // Decode min and max values according to the physical type.
    // Note that null page are skipped.
    auto plainDecoder = makeTypedDecoder<DType>(Encoding::kPlain, &descr);
    for (size_t i = 0; i < numPages; ++i) {
      if (!columnIndex_.null_pages[i]) {
        // The check on `num_pages` has guaranteed the cast below is safe.
        nonNullPageIndices_.emplace_back(static_cast<int32_t>(i));
        decode<DType>(plainDecoder, columnIndex_.min_values[i], &minValues_, i);
        decode<DType>(plainDecoder, columnIndex_.max_values[i], &maxValues_, i);
      }
    }
    VELOX_DCHECK_EQ(numNonNullPages, nonNullPageIndices_.size());
  }

  const std::vector<bool>& nullPages() const override {
    return columnIndex_.null_pages;
  }

  const std::vector<std::string>& encodedMinValues() const override {
    return columnIndex_.min_values;
  }

  const std::vector<std::string>& encodedMaxValues() const override {
    return columnIndex_.max_values;
  }

  BoundaryOrder::type boundaryOrder() const override {
    return loadenumSafe(&columnIndex_.boundary_order);
  }

  bool hasNullCounts() const override {
    return columnIndex_.__isset.null_counts;
  }

  const std::vector<int64_t>& nullCounts() const override {
    return columnIndex_.null_counts;
  }

  const std::vector<int32_t>& nonNullPageIndices() const override {
    return nonNullPageIndices_;
  }

  const std::vector<T>& minValues() const override {
    return minValues_;
  }

  const std::vector<T>& maxValues() const override {
    return maxValues_;
  }

 private:
  /// Wrapped thrift column index.
  const facebook::velox::parquet::thrift::ColumnIndex columnIndex_;
  /// Decoded typed min/max values. Undefined for null pages.
  std::vector<T> minValues_;
  std::vector<T> maxValues_;
  /// A list of page indices for non-null pages.
  std::vector<int32_t> nonNullPageIndices_;
};

class OffsetIndexImpl : public OffsetIndex {
 public:
  explicit OffsetIndexImpl(
      const facebook::velox::parquet::thrift::OffsetIndex& offsetIndex) {
    pageLocations_.reserve(offsetIndex.page_locations.size());
    for (const auto& pageLocation : offsetIndex.page_locations) {
      pageLocations_.emplace_back(
          PageLocation{
              pageLocation.offset,
              pageLocation.compressed_page_size,
              pageLocation.first_row_index});
    }
  }

  const std::vector<PageLocation>& pageLocations() const override {
    return pageLocations_;
  }

 private:
  std::vector<PageLocation> pageLocations_;
};

class RowGroupPageIndexReaderImpl : public RowGroupPageIndexReader {
 public:
  RowGroupPageIndexReaderImpl(
      ::arrow::io::RandomAccessFile* input,
      std::shared_ptr<RowGroupMetaData> rowGroupMetadata,
      const ReaderProperties& properties,
      int32_t rowGroupOrdinal,
      const RowGroupIndexReadRange& indexReadRange,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor)
      : input_(input),
        rowGroupMetadata_(std::move(rowGroupMetadata)),
        properties_(properties),
        rowGroupOrdinal_(rowGroupOrdinal),
        indexReadRange_(indexReadRange),
        fileDecryptor_(std::move(fileDecryptor)) {}

  /// Read column index of a column chunk.
  std::shared_ptr<ColumnIndex> getColumnIndex(int32_t i) override {
    if (i < 0 || i >= rowGroupMetadata_->numColumns()) {
      throw ParquetException("Invalid column index at column ordinal ", i);
    }

    auto colChunk = rowGroupMetadata_->columnChunk(i);
    std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata =
        colChunk->cryptoMetadata();
    if (cryptoMetadata != nullptr) {
      ParquetException::NYI("Cannot read encrypted column index yet");
    }

    auto columnIndexLocation = colChunk->getColumnIndexLocation();
    if (!columnIndexLocation.has_value()) {
      return nullptr;
    }

    checkReadRangeOrThrow(
        *columnIndexLocation, indexReadRange_.columnIndex, rowGroupOrdinal_);

    if (columnIndexBuffer_ == nullptr) {
      PARQUET_ASSIGN_OR_THROW(
          columnIndexBuffer_,
          input_->ReadAt(
              indexReadRange_.columnIndex->offset,
              indexReadRange_.columnIndex->length));
    }

    int64_t bufferoffset =
        columnIndexLocation->offset - indexReadRange_.columnIndex->offset;
    // ColumnIndex::Make() requires the type of serialized thrift message to be.
    // Uint32_t.
    uint32_t length = static_cast<uint32_t>(columnIndexLocation->length);
    auto descr = rowGroupMetadata_->schema()->column(i);
    return ColumnIndex::make(
        *descr, columnIndexBuffer_->data() + bufferoffset, length, properties_);
  }

  /// Read offset index of a column chunk.
  std::shared_ptr<OffsetIndex> getOffsetIndex(int32_t i) override {
    if (i < 0 || i >= rowGroupMetadata_->numColumns()) {
      throw ParquetException("Invalid offset index at column ordinal ", i);
    }

    auto colChunk = rowGroupMetadata_->columnChunk(i);
    std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata =
        colChunk->cryptoMetadata();
    if (cryptoMetadata != nullptr) {
      ParquetException::NYI("Cannot read encrypted offset index yet");
    }

    auto offsetIndexLocation = colChunk->getOffsetIndexLocation();
    if (!offsetIndexLocation.has_value()) {
      return nullptr;
    }

    checkReadRangeOrThrow(
        *offsetIndexLocation, indexReadRange_.offsetIndex, rowGroupOrdinal_);

    if (offsetIndexBuffer_ == nullptr) {
      PARQUET_ASSIGN_OR_THROW(
          offsetIndexBuffer_,
          input_->ReadAt(
              indexReadRange_.offsetIndex->offset,
              indexReadRange_.offsetIndex->length));
    }

    int64_t bufferoffset =
        offsetIndexLocation->offset - indexReadRange_.offsetIndex->offset;
    // OffsetIndex::Make() requires the type of serialized thrift message to be.
    // Uint32_t.
    uint32_t length = static_cast<uint32_t>(offsetIndexLocation->length);
    return OffsetIndex::make(
        offsetIndexBuffer_->data() + bufferoffset, length, properties_);
  }

 private:
  static void checkReadRangeOrThrow(
      const IndexLocation& indexLocation,
      const std::optional<::arrow::io::ReadRange>& indexReadRange,
      int32_t rowGroupOrdinal) {
    if (!indexReadRange.has_value()) {
      throw ParquetException(
          "Missing page index read range of row group ",
          rowGroupOrdinal,
          ", it may not exist or has not been requested");
    }

    /// The coalesced read range is invalid.
    if (indexReadRange->offset < 0 || indexReadRange->length <= 0) {
      throw ParquetException(
          "Invalid page index read range: offset ",
          indexReadRange->offset,
          " length ",
          indexReadRange->length);
    }

    /// The location to page index itself is corrupted.
    if (indexLocation.offset < 0 || indexLocation.length <= 0) {
      throw ParquetException(
          "Invalid page index location: offset ",
          indexLocation.offset,
          " length ",
          indexLocation.length);
    }

    /// Page index location must be within the range of the read range.
    if (indexLocation.offset < indexReadRange->offset ||
        indexLocation.offset + indexLocation.length >
            indexReadRange->offset + indexReadRange->length) {
      throw ParquetException(
          "Page index location [offset:",
          indexLocation.offset,
          ",length:",
          indexLocation.length,
          "] is out of range from previous WillNeed request [offset:",
          indexReadRange->offset,
          ",length:",
          indexReadRange->length,
          "], row group: ",
          rowGroupOrdinal);
    }
  }

 private:
  /// The input stream that can perform random access read.
  ::arrow::io::RandomAccessFile* input_;

  /// The row group metadata to get column chunk metadata.
  std::shared_ptr<RowGroupMetaData> rowGroupMetadata_;

  /// Reader properties used to deserialize thrift object.
  const ReaderProperties& properties_;

  /// The ordinal of the row group in the file.
  int32_t rowGroupOrdinal_;

  /// File offsets and sizes of the page Index of all column chunks in the row.
  /// Group.
  RowGroupIndexReadRange indexReadRange_;

  /// File-level decryptor.
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;

  /// Buffer to hold the raw bytes of the page index.
  /// Will be set lazily when the corresponding page index is accessed for the.
  /// 1St time.
  std::shared_ptr<::arrow::Buffer> columnIndexBuffer_;
  std::shared_ptr<::arrow::Buffer> offsetIndexBuffer_;
};

class PageIndexReaderImpl : public PageIndexReader {
 public:
  PageIndexReaderImpl(
      ::arrow::io::RandomAccessFile* input,
      std::shared_ptr<FileMetaData> fileMetadata,
      const ReaderProperties& properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor)
      : input_(input),
        fileMetadata_(std::move(fileMetadata)),
        properties_(properties),
        fileDecryptor_(std::move(fileDecryptor)) {}

  std::shared_ptr<RowGroupPageIndexReader> rowGroup(int i) override {
    if (i < 0 || i >= fileMetadata_->numRowGroups()) {
      throw ParquetException("Invalid row group ordinal: ", i);
    }

    auto rowGroupMetadata = fileMetadata_->rowGroup(i);

    // Find the read range of the page index of the row group if provided by.
    // WillNeed()
    RowGroupIndexReadRange indexReadRange;
    auto iter = indexReadRanges_.find(i);
    if (iter != indexReadRanges_.cend()) {
      /// This row group has been requested by WillNeed(). Only column index.
      /// And/or offset index of requested columns can be read.
      indexReadRange = iter->second;
    } else {
      /// If the row group has not been requested by WillNeed(), by default
      /// both. Column index and offset index of all column chunks for the row
      /// group. Can be read.
      indexReadRange = PageIndexReader::determinePageIndexRangesInRowGroup(
          *rowGroupMetadata, {});
    }

    if (indexReadRange.columnIndex.has_value() ||
        indexReadRange.offsetIndex.has_value()) {
      return std::make_shared<RowGroupPageIndexReaderImpl>(
          input_,
          std::move(rowGroupMetadata),
          properties_,
          i,
          indexReadRange,
          fileDecryptor_);
    }

    /// The row group does not has page index or has not been requested by.
    /// WillNeed(). Simply returns nullptr.
    return nullptr;
  }

  void willNeed(
      const std::vector<int32_t>& rowGroupIndices,
      const std::vector<int32_t>& columnIndices,
      const PageIndexSelection& selection) override {
    std::vector<::arrow::io::ReadRange> readRanges;
    for (int32_t rowGroupOrdinal : rowGroupIndices) {
      auto readRange = PageIndexReader::determinePageIndexRangesInRowGroup(
          *fileMetadata_->rowGroup(rowGroupOrdinal), columnIndices);
      if (selection.columnIndex && readRange.columnIndex.has_value()) {
        readRanges.push_back(*readRange.columnIndex);
      } else {
        // Mark the column index as not requested.
        readRange.columnIndex = std::nullopt;
      }
      if (selection.offsetIndex && readRange.offsetIndex.has_value()) {
        readRanges.push_back(*readRange.offsetIndex);
      } else {
        // Mark the offset index as not requested.
        readRange.offsetIndex = std::nullopt;
      }
      indexReadRanges_.emplace(rowGroupOrdinal, std::move(readRange));
    }
    PARQUET_THROW_NOT_OK(input_->WillNeed(readRanges));
  }

  void willNotNeed(const std::vector<int32_t>& rowGroupIndices) override {
    for (int32_t rowGroupOrdinal : rowGroupIndices) {
      indexReadRanges_.erase(rowGroupOrdinal);
    }
  }

 private:
  /// The input stream that can perform random read.
  ::arrow::io::RandomAccessFile* input_;

  /// The file metadata to get row group metadata.
  std::shared_ptr<FileMetaData> fileMetadata_;

  /// Reader properties used to deserialize thrift object.
  const ReaderProperties& properties_;

  /// File-level decrypter.
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;

  /// Coalesced read ranges of page index of row groups that have been
  /// suggested. By WillNeed(). Key is the row group ordinal.
  std::unordered_map<int32_t, RowGroupIndexReadRange> indexReadRanges_;
};

/// \brief Internal state of page index builder.
enum class BuilderState {
  /// Created but not yet write any data.
  kCreated,
  /// Some data are written but not yet finished.
  kStarted,
  /// All data are written and no more write is allowed.
  kFinished,
  /// The builder has corrupted data or empty data and therefore discarded.
  kDiscarded
};

template <typename DType>
class ColumnIndexBuilderImpl final : public ColumnIndexBuilder {
 public:
  using T = typename DType::CType;

  explicit ColumnIndexBuilderImpl(const ColumnDescriptor* descr)
      : descr_(descr) {
    /// Initialize the null_counts vector as set. Invalid null_counts vector.
    /// From any page will invalidate the null_counts vector of the column.
    /// Index.
    columnIndex_.__isset.null_counts = true;
    columnIndex_.boundary_order =
        facebook::velox::parquet::thrift::BoundaryOrder::UNORDERED;
  }

  void addPage(const EncodedStatistics& stats) override {
    if (state_ == BuilderState::kFinished) {
      throw ParquetException("Cannot add page to finished ColumnIndexBuilder.");
    } else if (state_ == BuilderState::kDiscarded) {
      /// The offset index is discarded. Do nothing.
      return;
    }

    state_ = BuilderState::kStarted;

    if (stats.allNullValue) {
      columnIndex_.null_pages.emplace_back(true);
      columnIndex_.min_values.emplace_back("");
      columnIndex_.max_values.emplace_back("");
    } else if (stats.hasMin && stats.hasMax) {
      const size_t pageOrdinal = columnIndex_.null_pages.size();
      nonNullPageIndices_.emplace_back(pageOrdinal);
      columnIndex_.min_values.emplace_back(stats.min());
      columnIndex_.max_values.emplace_back(stats.max());
      columnIndex_.null_pages.emplace_back(false);
    } else {
      /// This is a non-null page but it lacks of meaningful min/max values.
      /// Discard the column index.
      state_ = BuilderState::kDiscarded;
      return;
    }

    if (columnIndex_.__isset.null_counts && stats.hasNullCount) {
      columnIndex_.null_counts.emplace_back(stats.nullCount);
    } else {
      columnIndex_.__isset.null_counts = false;
      columnIndex_.null_counts.clear();
    }
  }

  void finish() override {
    switch (state_) {
      case BuilderState::kCreated: {
        /// No page is added. Discard the column index.
        state_ = BuilderState::kDiscarded;
        return;
      }
      case BuilderState::kFinished:
        throw ParquetException("ColumnIndexBuilder is already finished.");
      case BuilderState::kDiscarded:
        // The column index is discarded. Do nothing.
        return;
      case BuilderState::kStarted:
        break;
    }

    state_ = BuilderState::kFinished;

    /// Clear null_counts vector because at least one page does not provide it.
    if (!columnIndex_.__isset.null_counts) {
      columnIndex_.null_counts.clear();
    }

    /// Decode min/max values according to the data type.
    const size_t nonNullPageCount = nonNullPageIndices_.size();
    std::vector<T> min_values, max_values;
    min_values.resize(nonNullPageCount);
    max_values.resize(nonNullPageCount);
    auto decoder = makeTypedDecoder<DType>(Encoding::kPlain, descr_);
    for (size_t i = 0; i < nonNullPageCount; ++i) {
      auto pageOrdinal = nonNullPageIndices_.at(i);
      decode<DType>(
          decoder, columnIndex_.min_values.at(pageOrdinal), &min_values, i);
      decode<DType>(
          decoder, columnIndex_.max_values.at(pageOrdinal), &max_values, i);
    }

    /// Decide the boundary order from decoded min/max values.
    auto boundary_order = determineBoundaryOrder(min_values, max_values);
    columnIndex_.__set_boundary_order(toThrift(boundary_order));
  }

  void writeTo(::arrow::io::OutputStream* sink) const override {
    if (state_ == BuilderState::kFinished) {
      ThriftSerializer{}.serialize(&columnIndex_, sink);
    }
  }

  std::unique_ptr<ColumnIndex> build() const override {
    if (state_ == BuilderState::kFinished) {
      return std::make_unique<TypedColumnIndexImpl<DType>>(
          *descr_, columnIndex_);
    }
    return nullptr;
  }

 private:
  BoundaryOrder::type determineBoundaryOrder(
      const std::vector<T>& min_values,
      const std::vector<T>& max_values) const {
    VELOX_DCHECK_EQ(min_values.size(), max_values.size());
    if (min_values.empty()) {
      return BoundaryOrder::Unordered;
    }

    std::shared_ptr<TypedComparator<DType>> Comparator;
    try {
      Comparator = makeComparator<DType>(descr_);
    } catch (const ParquetException&) {
      /// Simply return unordered for unsupported Comparator.
      return BoundaryOrder::Unordered;
    }

    /// Check if both min_values and max_values are in ascending order.
    bool isAscending = true;
    for (size_t i = 1; i < min_values.size(); ++i) {
      if (Comparator->compare(min_values[i], min_values[i - 1]) ||
          Comparator->compare(max_values[i], max_values[i - 1])) {
        isAscending = false;
        break;
      }
    }
    if (isAscending) {
      return BoundaryOrder::Ascending;
    }

    /// Check if both min_values and max_values are in descending order.
    bool isDescending = true;
    for (size_t i = 1; i < min_values.size(); ++i) {
      if (Comparator->compare(min_values[i - 1], min_values[i]) ||
          Comparator->compare(max_values[i - 1], max_values[i])) {
        isDescending = false;
        break;
      }
    }
    if (isDescending) {
      return BoundaryOrder::Descending;
    }

    /// Neither ascending nor descending is detected.
    return BoundaryOrder::Unordered;
  }

  const ColumnDescriptor* descr_;
  facebook::velox::parquet::thrift::ColumnIndex columnIndex_;
  std::vector<size_t> nonNullPageIndices_;
  BuilderState state_ = BuilderState::kCreated;
};

class OffsetIndexBuilderImpl final : public OffsetIndexBuilder {
 public:
  OffsetIndexBuilderImpl() = default;

  void addPage(
      int64_t offset,
      int32_t compressedPageSize,
      int64_t firstRowIndex) override {
    if (state_ == BuilderState::kFinished) {
      throw ParquetException("Cannot add page to finished OffsetIndexBuilder.");
    } else if (state_ == BuilderState::kDiscarded) {
      /// The offset index is discarded. Do nothing.
      return;
    }

    state_ = BuilderState::kStarted;

    facebook::velox::parquet::thrift::PageLocation pageLocation;
    pageLocation.__set_offset(offset);
    pageLocation.__set_compressed_page_size(compressedPageSize);
    pageLocation.__set_first_row_index(firstRowIndex);
    offsetIndex_.page_locations.emplace_back(std::move(pageLocation));
  }

  void finish(int64_t finalPosition) override {
    switch (state_) {
      case BuilderState::kCreated: {
        /// No pages are added. Simply discard the offset index.
        state_ = BuilderState::kDiscarded;
        break;
      }
      case BuilderState::kStarted: {
        /// Adjust page offsets according the final position.
        if (finalPosition > 0) {
          for (auto& pageLocation : offsetIndex_.page_locations) {
            pageLocation.__set_offset(pageLocation.offset + finalPosition);
          }
        }
        state_ = BuilderState::kFinished;
        break;
      }
      case BuilderState::kFinished:
      case BuilderState::kDiscarded:
        throw ParquetException("OffsetIndexBuilder is already finished");
    }
  }

  void writeTo(::arrow::io::OutputStream* sink) const override {
    if (state_ == BuilderState::kFinished) {
      ThriftSerializer{}.serialize(&offsetIndex_, sink);
    }
  }

  std::unique_ptr<OffsetIndex> build() const override {
    if (state_ == BuilderState::kFinished) {
      return std::make_unique<OffsetIndexImpl>(offsetIndex_);
    }
    return nullptr;
  }

 private:
  facebook::velox::parquet::thrift::OffsetIndex offsetIndex_;
  BuilderState state_ = BuilderState::kCreated;
};

class PageIndexBuilderImpl final : public PageIndexBuilder {
 public:
  explicit PageIndexBuilderImpl(const SchemaDescriptor* schema)
      : schema_(schema) {}

  void appendRowGroup() override {
    if (finished_) {
      throw ParquetException(
          "Cannot call AppendRowGroup() to finished PageIndexBuilder.");
    }

    // Append new builders of next row group.
    const auto numColumns = static_cast<size_t>(schema_->numColumns());
    columnIndexBuilders_.emplace_back();
    offsetIndexBuilders_.emplace_back();
    columnIndexBuilders_.back().resize(numColumns);
    offsetIndexBuilders_.back().resize(numColumns);

    VELOX_DCHECK_EQ(columnIndexBuilders_.size(), offsetIndexBuilders_.size());
    VELOX_DCHECK_EQ(columnIndexBuilders_.back().size(), numColumns);
    VELOX_DCHECK_EQ(offsetIndexBuilders_.back().size(), numColumns);
  }

  ColumnIndexBuilder* getColumnIndexBuilder(int32_t i) override {
    checkState(i);
    std::unique_ptr<ColumnIndexBuilder>& builder =
        columnIndexBuilders_.back()[i];
    if (builder == nullptr) {
      builder = ColumnIndexBuilder::make(schema_->column(i));
    }
    return builder.get();
  }

  OffsetIndexBuilder* getOffsetIndexBuilder(int32_t i) override {
    checkState(i);
    std::unique_ptr<OffsetIndexBuilder>& builder =
        offsetIndexBuilders_.back()[i];
    if (builder == nullptr) {
      builder = OffsetIndexBuilder::make();
    }
    return builder.get();
  }

  void finish() override {
    finished_ = true;
  }

  void writeTo(::arrow::io::OutputStream* sink, PageIndexLocation* location)
      const override {
    if (!finished_) {
      throw ParquetException(
          "Cannot call WriteTo() to unfinished PageIndexBuilder.");
    }

    location->columnIndexLocation.clear();
    location->offsetIndexLocation.clear();

    /// Serialize column index ordered by row group ordinal and then column.
    /// Ordinal.
    serializeIndex(columnIndexBuilders_, sink, &location->columnIndexLocation);

    /// Serialize offset index ordered by row group ordinal and then column.
    /// Ordinal.
    serializeIndex(offsetIndexBuilders_, sink, &location->offsetIndexLocation);
  }

 private:
  /// Make sure column ordinal is not out of bound and the builder is in good.
  /// State.
  void checkState(int32_t columnOrdinal) const {
    if (finished_) {
      throw ParquetException("PageIndexBuilder is already finished.");
    }
    if (columnOrdinal < 0 || columnOrdinal >= schema_->numColumns()) {
      throw ParquetException("Invalid column ordinal: ", columnOrdinal);
    }
    if (offsetIndexBuilders_.empty() || columnIndexBuilders_.empty()) {
      throw ParquetException("No row group appended to PageIndexBuilder.");
    }
  }

  template <typename Builder>
  void serializeIndex(
      const std::vector<std::vector<std::unique_ptr<Builder>>>&
          pageIndexBuilders,
      ::arrow::io::OutputStream* sink,
      std::map<size_t, std::vector<std::optional<IndexLocation>>>* location)
      const {
    const auto numColumns = static_cast<size_t>(schema_->numColumns());

    /// Serialize the same kind of page index row group by row group.
    for (size_t rowGroup = 0; rowGroup < pageIndexBuilders.size(); ++rowGroup) {
      const auto& rowGroupPageIndexBuilders = pageIndexBuilders[rowGroup];
      VELOX_DCHECK_EQ(rowGroupPageIndexBuilders.size(), numColumns);

      bool hasValidIndex = false;
      std::vector<std::optional<IndexLocation>> locations(
          numColumns, std::nullopt);

      /// In the same row group, serialize the same kind of page index column
      /// by. Column.
      for (size_t column = 0; column < numColumns; ++column) {
        const auto& columnPageIndexBuilder = rowGroupPageIndexBuilders[column];
        if (columnPageIndexBuilder != nullptr) {
          /// Try serializing the page index.
          PARQUET_ASSIGN_OR_THROW(int64_t posBeforeWrite, sink->Tell());
          columnPageIndexBuilder->writeTo(sink);
          PARQUET_ASSIGN_OR_THROW(int64_t posAfterWrite, sink->Tell());
          int64_t len = posAfterWrite - posBeforeWrite;

          /// The page index is not serialized and skip reporting its location.
          if (len == 0) {
            continue;
          }

          if (len > std::numeric_limits<int32_t>::max()) {
            throw ParquetException("Page index size overflows to INT32_MAX");
          }
          locations[column] = {posBeforeWrite, static_cast<int32_t>(len)};
          hasValidIndex = true;
        }
      }

      if (hasValidIndex) {
        location->emplace(rowGroup, std::move(locations));
      }
    }
  }

  const SchemaDescriptor* schema_;
  std::vector<std::vector<std::unique_ptr<ColumnIndexBuilder>>>
      columnIndexBuilders_;
  std::vector<std::vector<std::unique_ptr<OffsetIndexBuilder>>>
      offsetIndexBuilders_;
  bool finished_ = false;
};

} // namespace

RowGroupIndexReadRange PageIndexReader::determinePageIndexRangesInRowGroup(
    const RowGroupMetaData& rowGroupMetadata,
    const std::vector<int32_t>& columns) {
  int64_t ciStart = std::numeric_limits<int64_t>::max();
  int64_t oiStart = std::numeric_limits<int64_t>::max();
  int64_t ciEnd = -1;
  int64_t oiEnd = -1;

  auto mergeRange = [](const std::optional<IndexLocation>& indexLocation,
                       int64_t* start,
                       int64_t* end) {
    if (indexLocation.has_value()) {
      int64_t indexEnd = 0;
      if (indexLocation->offset < 0 || indexLocation->length <= 0 ||
          ::arrow::internal::addWithOverflow(
              indexLocation->offset, indexLocation->length, &indexEnd)) {
        throw ParquetException(
            "Invalid page index location: offset ",
            indexLocation->offset,
            " length ",
            indexLocation->length);
      }
      *start = std::min(*start, indexLocation->offset);
      *end = std::max(*end, indexEnd);
    }
  };

  if (columns.empty()) {
    for (int32_t i = 0; i < rowGroupMetadata.numColumns(); ++i) {
      auto colChunk = rowGroupMetadata.columnChunk(i);
      mergeRange(colChunk->getColumnIndexLocation(), &ciStart, &ciEnd);
      mergeRange(colChunk->getOffsetIndexLocation(), &oiStart, &oiEnd);
    }
  } else {
    for (int32_t i : columns) {
      if (i < 0 || i >= rowGroupMetadata.numColumns()) {
        throw ParquetException("Invalid column ordinal ", i);
      }
      auto colChunk = rowGroupMetadata.columnChunk(i);
      mergeRange(colChunk->getColumnIndexLocation(), &ciStart, &ciEnd);
      mergeRange(colChunk->getOffsetIndexLocation(), &oiStart, &oiEnd);
    }
  }

  RowGroupIndexReadRange readRange;
  if (ciEnd != -1) {
    readRange.columnIndex = {ciStart, ciEnd - ciStart};
  }
  if (oiEnd != -1) {
    readRange.offsetIndex = {oiStart, oiEnd - oiStart};
  }
  return readRange;
}

// ----------------------------------------------------------------------.
// Public factory functions.

std::unique_ptr<ColumnIndex> ColumnIndex::make(
    const ColumnDescriptor& descr,
    const void* serializedIndex,
    uint32_t indexLen,
    const ReaderProperties& properties) {
  facebook::velox::parquet::thrift::ColumnIndex columnIndex;
  ThriftDeserializer deserializer(properties);
  deserializer.deserializeMessage(
      reinterpret_cast<const uint8_t*>(serializedIndex),
      &indexLen,
      &columnIndex);
  switch (descr.physicalType()) {
    case Type::kBoolean:
      return std::make_unique<TypedColumnIndexImpl<BooleanType>>(
          descr, std::move(columnIndex));
    case Type::kInt32:
      return std::make_unique<TypedColumnIndexImpl<Int32Type>>(
          descr, std::move(columnIndex));
    case Type::kInt64:
      return std::make_unique<TypedColumnIndexImpl<Int64Type>>(
          descr, std::move(columnIndex));
    case Type::kInt96:
      return std::make_unique<TypedColumnIndexImpl<Int96Type>>(
          descr, std::move(columnIndex));
    case Type::kFloat:
      return std::make_unique<TypedColumnIndexImpl<FloatType>>(
          descr, std::move(columnIndex));
    case Type::kDouble:
      return std::make_unique<TypedColumnIndexImpl<DoubleType>>(
          descr, std::move(columnIndex));
    case Type::kByteArray:
      return std::make_unique<TypedColumnIndexImpl<ByteArrayType>>(
          descr, std::move(columnIndex));
    case Type::kFixedLenByteArray:
      return std::make_unique<TypedColumnIndexImpl<FLBAType>>(
          descr, std::move(columnIndex));
    case Type::kUndefined:
      return nullptr;
  }
  ::arrow::Unreachable("Cannot make ColumnIndex of an unknown type");
  return nullptr;
}

std::unique_ptr<OffsetIndex> OffsetIndex::make(
    const void* serializedIndex,
    uint32_t indexLen,
    const ReaderProperties& properties) {
  facebook::velox::parquet::thrift::OffsetIndex offsetIndex;
  ThriftDeserializer deserializer(properties);
  deserializer.deserializeMessage(
      reinterpret_cast<const uint8_t*>(serializedIndex),
      &indexLen,
      &offsetIndex);
  return std::make_unique<OffsetIndexImpl>(offsetIndex);
}

std::shared_ptr<PageIndexReader> PageIndexReader::make(
    ::arrow::io::RandomAccessFile* input,
    std::shared_ptr<FileMetaData> fileMetadata,
    const ReaderProperties& properties,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::make_shared<PageIndexReaderImpl>(
      input, std::move(fileMetadata), properties, std::move(fileDecryptor));
}

std::unique_ptr<ColumnIndexBuilder> ColumnIndexBuilder::make(
    const ColumnDescriptor* descr) {
  switch (descr->physicalType()) {
    case Type::kBoolean:
      return std::make_unique<ColumnIndexBuilderImpl<BooleanType>>(descr);
    case Type::kInt32:
      return std::make_unique<ColumnIndexBuilderImpl<Int32Type>>(descr);
    case Type::kInt64:
      return std::make_unique<ColumnIndexBuilderImpl<Int64Type>>(descr);
    case Type::kInt96:
      return std::make_unique<ColumnIndexBuilderImpl<Int96Type>>(descr);
    case Type::kFloat:
      return std::make_unique<ColumnIndexBuilderImpl<FloatType>>(descr);
    case Type::kDouble:
      return std::make_unique<ColumnIndexBuilderImpl<DoubleType>>(descr);
    case Type::kByteArray:
      return std::make_unique<ColumnIndexBuilderImpl<ByteArrayType>>(descr);
    case Type::kFixedLenByteArray:
      return std::make_unique<ColumnIndexBuilderImpl<FLBAType>>(descr);
    case Type::kUndefined:
      return nullptr;
  }
  ::arrow::Unreachable("Cannot make ColumnIndexBuilder of an unknown type");
  return nullptr;
}

std::unique_ptr<OffsetIndexBuilder> OffsetIndexBuilder::make() {
  return std::make_unique<OffsetIndexBuilderImpl>();
}

std::unique_ptr<PageIndexBuilder> PageIndexBuilder::make(
    const SchemaDescriptor* schema) {
  return std::make_unique<PageIndexBuilderImpl>(schema);
}

std::ostream& operator<<(
    std::ostream& out,
    const PageIndexSelection& selection) {
  out << "PageIndexSelection{column_index = " << selection.columnIndex
      << ", offset_index = " << selection.offsetIndex << "}";
  return out;
}

} // namespace facebook::velox::parquet::arrow

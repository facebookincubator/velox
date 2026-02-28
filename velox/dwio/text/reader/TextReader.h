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

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "folly/CppAttributes.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/compression/Compression.h"

namespace facebook::velox::text {

using common::CompressionKind;
using common::ScanSpec;
using dwio::common::BufferedInput;
using dwio::common::ColumnStatistics;
using dwio::common::Mutation;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;
using dwio::common::SerDeOptions;
using dwio::common::TypeWithId;
using memory::MemoryPool;

using RejectedRow = dwio::common::RejectedRow;
using OnRowReject = dwio::common::OnRowReject;

// Shared state for a file between TextReader and TextRowReader
struct FileContents {
  FileContents(MemoryPool& pool, const std::shared_ptr<const RowType>& t);

  const std::shared_ptr<const RowType> schema;

  std::unique_ptr<BufferedInput> input;
  std::unique_ptr<dwio::common::SeekableInputStream> inputStream;
  std::unique_ptr<dwio::common::SeekableInputStream> decompressedInputStream;
  MemoryPool& pool;
  uint64_t fileLength;
  CompressionKind compression;
  dwio::common::compression::CompressionOptions compressionOptions;
  SerDeOptions serDeOptions;

  OnRowReject onRowReject;
};

using DelimType = uint8_t;
constexpr DelimType DelimTypeNone = 0;
constexpr DelimType DelimTypeEOR = 1;
constexpr DelimType DelimTypeEOE = 2;

class TextReader : public dwio::common::Reader {
 public:
  TextReader(ReaderOptions options, std::unique_ptr<BufferedInput> input);

  std::optional<uint64_t> numberOfRows() const final;

  std::unique_ptr<ColumnStatistics> columnStatistics(
      uint32_t index) const final;

  const RowTypePtr& rowType() const final;

  const std::shared_ptr<const TypeWithId>& typeWithId() const final;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const RowReaderOptions& options) const final;

 private:
  ReaderOptions options_;
  mutable std::shared_ptr<const TypeWithId> typeWithId_;
  std::shared_ptr<FileContents> contents_;
};

class TextRowReader : public dwio::common::RowReader {
 public:
  TextRowReader(
      std::shared_ptr<FileContents> fileContents,
      const RowReaderOptions& options);

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const Mutation* mutation = nullptr) final;

  int64_t nextRowNumber() final {
    return atEOF_ ? -1 : static_cast<int64_t>(currentRow_) + 1;
  }

  int64_t nextReadSize(uint64_t size) final {
    return static_cast<int64_t>(std::min(fileLength_ - currentRow_, size));
  }

  void updateRuntimeStats(dwio::common::RuntimeStatistics& stats) const final {}

  void resetFilterCaches() final {}

  std::optional<size_t> estimatedRowSize() const final {
    return std::nullopt;
  }

  uint64_t seekToRow(uint64_t rowNumber);

 private:
  const RowType& getFileType() const;

  uint64_t getLength();

  uint64_t getStreamLength() const;

  void setEOF();

  void incrementDepth();

  void decrementDepth(DelimType& delim);

  void setEOE(DelimType& delim);

  void resetEOE(DelimType& delim);

  void setEOR(DelimType& delim);

  bool isEOR(DelimType delim);

  bool isOuterEOR(DelimType delim);

  bool isEOEorEOR(DelimType delim);

  void setNone(DelimType& delim);

  bool isNone(DelimType delim);

  DelimType getDelimType(uint8_t v);

  template <bool skipLF = false>
  char getByteUncheckedOptimized(DelimType& delim);

  uint8_t getByteOptimized(DelimType& delim);

  bool getEOR(DelimType& delim, bool& isNull);

  bool skipLine();

  void resetLine();

  static std::string_view
  getString(TextRowReader& th, bool& isNull, DelimType& delim);

  template <typename T>
  static T getNumeric(TextRowReader& th, bool& isNull, DelimType& delim);

  static bool getBoolean(TextRowReader& th, bool& isNull, DelimType& delim);

  void readElement(
      const std::shared_ptr<const Type>& t,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim);

  template <typename T, typename Filter, typename F>
  bool putValue(
      const F& f,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);

  template <typename T, typename Filter>
  bool setValueFromString(
      std::string_view str,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      std::function<std::optional<T>(std::string_view)> convert,
      const velox::common::Filter* filter);

  std::string_view ownedStringView() const {
    return std::string_view{ownedString_.data(), ownedString_.size()};
  }

  const std::shared_ptr<FileContents> contents_;
  const std::shared_ptr<velox::common::ScanSpec>& scanSpec_;

  RowReaderOptions options_;
  uint64_t currentRow_ = 0;
  uint64_t pos_;
  bool atEOL_;
  bool atEOF_;
  bool atSOL_;
  bool atPhysicalEOF_;
  uint8_t depth_;
  std::string unreadData_;
  std::string_view preLoadedUnreadData_;
  int unreadIdx_;
  uint64_t limit_; // lowest offset not in the range
  uint64_t fileLength_;
  std::vector<char> ownedString_;
  std::shared_ptr<dwio::common::DataBuffer<char>> varBinBuf_;
  bool rowHasError_ = false;
  std::string_view errorValue_;

  // true -> ok, else skip
  using ColumnReaderFunc = bool (TextRowReader::*)(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);

  static constexpr int64_t kNotProjected = -1;

  struct FileColumnDesc {
    int64_t resultVectorIdx = kNotProjected;
    ColumnReaderFunc reader = nullptr;
    const velox::common::Filter* filter = nullptr;
  };
  std::vector<FileColumnDesc> fileColumns_;

  void initializeColumnReaders();

  // Specialized readers for each type, templatized on filter type.
  // Return true if value passes the filter, false if filtered out.
  template <typename Filter>
  bool readInteger(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readDate(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readBigInt(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readBigIntDecimal(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readSmallInt(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readTinyInt(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readBoolean(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readVarChar(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readVarBinary(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readReal(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readDouble(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readTimestamp(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readHugeInt(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readHugeIntDecimal(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readArray(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readMap(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
  template <typename Filter>
  bool readRow(
      const Type& type,
      BaseVector* FOLLY_NULLABLE data,
      vector_size_t insertionRow,
      DelimType& delim,
      const velox::common::Filter* filter);
};

} // namespace facebook::velox::text

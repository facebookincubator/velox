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
#include <functional>
#include <limits>
#include <string>

#include "folly/CppAttributes.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/compression/Compression.h"

namespace facebook::velox::text {

using common::CompressionKind;
using common::ScanSpec;
using dwio::common::BufferedInput;

static constexpr uint64_t kTextBlockSize = 1024 * 1024;
using dwio::common::ColumnSelector;
using dwio::common::ColumnStatistics;
using dwio::common::Mutation;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;
using dwio::common::SerDeOptions;
using dwio::common::TypeWithId;
using memory::MemoryPool;

struct FileContents {
  FileContents(MemoryPool& pool, const std::shared_ptr<const RowType>& t);

  const size_t COLUMN_POSITION_INVALID = std::numeric_limits<size_t>::max();
  const std::shared_ptr<const RowType> schema;

  std::unique_ptr<BufferedInput> input;
  std::unique_ptr<dwio::common::SeekableInputStream> inputStream;
  std::unique_ptr<dwio::common::SeekableInputStream> decompressedInputStream;
  MemoryPool& pool;
  uint64_t fileLength;
  CompressionKind compression;
  dwio::common::compression::CompressionOptions compressionOptions;
  SerDeOptions serDeOptions;
  std::array<bool, 128> needsEscape;
};

class TextReader : public dwio::common::Reader {
 public:
  TextReader(
      const ReaderOptions& options,
      std::unique_ptr<BufferedInput> input);

  std::optional<uint64_t> numberOfRows() const override;

  std::unique_ptr<ColumnStatistics> columnStatistics(
      uint32_t index) const override;

  const RowTypePtr& rowType() const override;

  CompressionKind getCompression() const;

  const std::shared_ptr<const TypeWithId>& typeWithId() const override;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const RowReaderOptions& options) const override;

  uint64_t getFileLength() const;

 private:
  ReaderOptions options_;
  mutable std::shared_ptr<const TypeWithId> typeWithId_;
  std::shared_ptr<FileContents> contents_;
  std::shared_ptr<const TypeWithId> schemaWithId_;
  std::shared_ptr<const RowType> internalSchema_;
};

class TextRowReader : public dwio::common::RowReader {
 public:
  // Precompiled setter function - avoids runtime type dispatch
  using SetterFunction =
      std::function<void(BaseVector*, vector_size_t, std::string_view)>;

  TextRowReader(
      std::shared_ptr<FileContents> fileContents,
      const RowReaderOptions& options);

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const Mutation* mutation = nullptr) override;

  int64_t nextRowNumber() override;

  int64_t nextReadSize(uint64_t size) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override;

  void resetFilterCaches() override;

  std::optional<size_t> estimatedRowSize() const override;

  const ColumnSelector& getColumnSelector() const;

  std::shared_ptr<const TypeWithId> getSelectedType() const;

  uint64_t getRowNumber() const;

  uint64_t seekToRow(uint64_t rowNumber);

 private:
  const RowReaderOptions& getDefaultOpts();

  const std::shared_ptr<const RowType>& getType() const;

  bool isSelectedField(const std::shared_ptr<const TypeWithId>& t);

  const char* getStreamNameData() const;

  uint64_t getStreamLength() const;

  // Load next chunk into buffer, returns false if EOF
  bool loadBuffer();

  // Find '\n' in stream buffer starting from streamPos_
  // Returns {lineEnd position, line content} or {npos, {}} if not found
  std::pair<size_t, std::string_view> findLine();

  // Skip to end of current line (for split boundaries)
  void skipToNextLine();

  // Find next delimiter, handling escape characters if needed
  size_t findDelimiter(std::string_view str, char delim, size_t start = 0)
      const;

  // Process a complete line into the result vector
  void processLine(RowVector* result, vector_size_t row, std::string_view line);

  // Parse array with precompiled element setter
  void writeArrayWithSetter(
      ArrayVector* arrayVec,
      vector_size_t row,
      std::string_view value,
      char elemDelim,
      const SetterFunction& elementSetter);

  // Parse map with precompiled key/value setters
  void writeMapWithSetters(
      MapVector* mapVec,
      vector_size_t row,
      std::string_view value,
      char pairDelim,
      char kvDelim,
      const SetterFunction& keySetter,
      const SetterFunction& valueSetter);

  // Parse row with precompiled field setters
  void writeRowWithSetters(
      RowVector* rowVec,
      vector_size_t row,
      std::string_view value,
      char fieldDelim,
      const std::vector<SetterFunction>& fieldSetters);

  // Check if value represents null
  bool isNullValue(std::string_view value) const;

  // Unescape a string value if escaping is enabled
  std::string unescapeValue(std::string_view value) const;

  // Initialize precompiled setters for all columns
  void initializeColumnSetters();

  // Create precompiled setter for (fileType -> reqType) with coercion support
  // depth: delimiter depth (1 for top-level columns, increases for nesting)
  SetterFunction
  makeSetter(const TypePtr& fileType, const TypePtr& reqType, int depth = 1);

  const std::shared_ptr<FileContents> contents_;
  const std::shared_ptr<const TypeWithId> schemaWithId_;
  const std::shared_ptr<velox::common::ScanSpec>
      scanSpec_; // Copy, not reference!

  mutable std::shared_ptr<const TypeWithId> selectedSchema_;

  RowReaderOptions options_;
  ColumnSelector columnSelector_;
  uint64_t currentRow_{0};
  uint64_t pos_{0};
  bool atEOF_{false};
  bool atPhysicalEOF_{false};
  uint64_t limit_;
  std::shared_ptr<dwio::common::DataBuffer<char>> varBinBuf_;

  // - streamData_/streamSize_: Points directly to stream's buffer
  // - leftover_: Only used when a line spans chunk boundaries (rare)
  const char* streamData_{nullptr};
  int streamSize_{0};
  size_t streamPos_{0};
  std::string leftover_; // Partial line from previous chunk

  // Deferred skip handling (done in next() instead of constructor)
  bool skipPartialLine_{false}; // Skip to next \n at start of split
  uint64_t rowsToSkip_{0}; // Number of header rows to skip

  // Precompiled setters for each file column
  // columnSetters_[fileColIndex] = {outputColIndex, setter} or nullopt if not
  // selected
  struct ColumnSetter {
    size_t outputIndex;
    SetterFunction setter;
  };
  std::vector<std::optional<ColumnSetter>> columnSetters_;

  char fieldDelim_{'\1'};
  std::string_view nullString_;
  bool isEscaped_{false};
};

} // namespace facebook::velox::text

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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "velox/dwio/text/reader/ReaderDecompressor.h"

using velox::dwio::text::compression::ReaderDecompressor;

namespace facebook::velox::text {

using dwio::common::ColumnSelector;
using dwio::common::RowReaderOptions;
using dwio::common::TypeWithId;

struct FileContents {
  const size_t COLUMN_POSITION_INVALID = std::numeric_limits<size_t>::max();

  const std::shared_ptr<const RowType> schema;

  /// TODO: mising member stream
  // std::unique_ptr<PreloadableReader> stream;

  memory::MemoryPool& pool;
  uint64_t fileLength;
  common::CompressionKind compression;

  std::unique_ptr<ReaderDecompressor> decompressedStream;
  dwio::common::SerDeOptions serDeOptions;
  std::array<bool, 128> needsEscape;

  FileContents(
      memory::MemoryPool& pool,
      const std::shared_ptr<const RowType>& t);
};

using DelimType = uint8_t;

constexpr DelimType DelimTypeNone = 0;
constexpr DelimType DelimTypeEOR = 1;
constexpr DelimType DelimTypeEOE = 2;

class RowReaderImpl : public dwio::common::RowReader {
 private:
  const RowReaderOptions& getDefaultOpts();

  const std::shared_ptr<FileContents> contents_;
  RowReaderOptions options_;

  ColumnSelector columnSelector_;

  const std::shared_ptr<const TypeWithId> schemaWithId_;
  mutable std::shared_ptr<const TypeWithId> selectedSchema_;

  uint64_t currentRow_;

  uint64_t pos_;
  bool atEOL_;
  bool atEOF_;
  bool atSOL_;
  uint8_t depth_;
  std::string unreadData_;
  uint64_t limit_;
  uint64_t fileLength_;
  const bool prestoTextReader_;

  const std::shared_ptr<const RowType>& getType() const;

  bool isSelectedField(const std::shared_ptr<const TypeWithId>& t);

  const char* getStreamNameData() const;

  uint64_t getLength();

  uint64_t getStreamLength();

  void setEOF();

  void incrementDepth();

  void decrementDepth(DelimType& delim);

  void setEOE(DelimType& delim);

  void resetEOE(DelimType& delim);

  bool isEOE(DelimType delim);

  void setEOR(DelimType& delim);

  bool isEOR(DelimType delim);

  bool isOuterEOR(DelimType delim);

  bool isEOEorEOR(DelimType delim);

  void setNone(DelimType& delim);

  bool isNone(DelimType delim);

  DelimType getDelimType(uint8_t v);

  void read(
      void* buf,
      uint64_t length,
      uint64_t offset,
      velox::dwio::common::LogType);

  /// TODO: Add implementation
  template <bool skipLF = false>
  uint8_t getByteUnchecked(DelimType& delim) {
    return '\n';
  }

  uint8_t getByte(DelimType& delim);

  bool getEOR(DelimType& delim, bool& isNull);

  bool skipLine();

  void resetLine();

  static std::string
  getString(RowReaderImpl& th, bool& isNull, DelimType& delim);

  template <typename T>
  static T getInteger(RowReaderImpl& th, bool& isNull, DelimType& delim);

  static bool getBoolean(RowReaderImpl& th, bool& isNull, DelimType& delim);

  static float getFloat(RowReaderImpl& th, bool& isNull, DelimType& delim);

  static double getDouble(RowReaderImpl& th, bool& isNull, DelimType& delim);

  /// TODO: Add implementation
  static void trim(std::string& s) {}

  void readElement(
      const std::shared_ptr<const Type>& t,
      const std::shared_ptr<const Type>& reqT,
      BaseVector* FOLLY_NULLABLE data,
      DelimType& delim);

  uint64_t incrementNumElements(BaseVector& data);

  /// TODO: Add implementation
  template <class T>
  T checkCast(BaseVector* FOLLY_NULLABLE data) {
    return nullptr;
  }

  template <class T, class reqT>
  void putValue(
      std::function<T(RowReaderImpl& th, bool& isNull, DelimType& delim)> f,
      BaseVector* FOLLY_NULLABLE data,
      DelimType& delim);

 public:
  RowReaderImpl(
      std::shared_ptr<FileContents> fileContents,
      const RowReaderOptions& options,
      bool prestoTextReader = false);

  void updateSelected();
  const ColumnSelector& getColumnSelector() const;

  std::shared_ptr<const TypeWithId> getSelectedType() const;

  std::unique_ptr<BaseVector> createRowBatch(uint64_t size) const override;

  bool next(BaseVector& data) override;

  uint64_t getRowNumber() const;

  uint64_t seekToRow(uint64_t rowNumber);

  uint64_t skipRows(uint64_t numberOfRowsToSkip);

  /// TODO: Add implementation
  bool isPrestoTextReader() const {
    return false;
  }
};

} // namespace facebook::velox::text

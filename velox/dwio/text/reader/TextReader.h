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

#include <common/file/Utils.h>
#include "velox/common/compression/Compression.h"
#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::text {

class ReaderBase {
 public:
  ReaderBase(
      dwio::common::ReaderOptions options,
      std::unique_ptr<dwio::common::BufferedInput> input);

  const RowTypePtr& schema() const {
    return schema_;
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId() const {
    return typeWithId_;
  }

  uint64_t fileLength() const {
    return input_->getReadFile()->size();
  }

  const dwio::common::SerDeOptions& serdeOptions() {
    return options_.serDeOptions();
  }

  std::unique_ptr<dwio::common::SeekableInputStream> loadBlock(
      common::Region region) const;

 private:
  const dwio::common::ReaderOptions options_;
  const std::unique_ptr<dwio::common::BufferedInput> input_;
  const RowTypePtr schema_;
  std::shared_ptr<const dwio::common::TypeWithId> typeWithId_;
  memory::MemoryPool* memoryPool_;
};

class TextRowReader : public dwio::common::RowReader {
 public:
  ~TextRowReader() override = default;
  TextRowReader(
      const std::shared_ptr<ReaderBase>& reader,
      const dwio::common::RowReaderOptions& options);

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const dwio::common::Mutation* mutation) override;

  int64_t nextRowNumber() override {
    return row_;
  }

  int64_t nextReadSize(uint64_t size) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override {}

  void resetFilterCaches() override {}

  std::optional<size_t> estimatedRowSize() const override {
    return std::nullopt;
  }

 private:
  using SetterFunction =
      std::function<void(VectorPtr&, vector_size_t, std::string_view)>;

  void processLine(RowVector* result, int32_t row, std::string_view line);

  SetterFunction makeSetter(const TypePtr& type);

  template <TypeKind Kind>
  SetterFunction makePrimitiveSetter();

  void writeArrayValue(
      const SetterFunction& setter,
      VectorPtr& columnVector,
      int32_t row,
      std::string_view value) const;

  void writeRowValue(
      const std::vector<SetterFunction>& childSetters,
      VectorPtr& vector,
      vector_size_t row,
      std::string_view value) const;

  void writeMapValue(
      const SetterFunction& keySetter,
      const SetterFunction& valueSetter,
      VectorPtr& vector,
      vector_size_t row,
      std::string_view value) const;

  template <TypeKind KIND>
  typename TypeTraits<KIND>::NativeType castFromString(
      const std::string_view& value);

  static int32_t castFromDateString(const std::string_view& value);

  static constexpr uint64_t kBlockSize = 1024 * 1024; // 1MB
  static constexpr uint64_t kEstimatedRowSize = 1024; // 1KB

  std::shared_ptr<ReaderBase> readerBase_;
  std::unique_ptr<dwio::common::SeekableInputStream> stream_;

  RowTypePtr fileSchema_;

  std::unordered_map<uint32_t, std::pair<uint32_t, SetterFunction>>
      fileIndexToSetters_;
  std::unordered_map<uint32_t, VectorPtr> constantColumnVectors_;

  uint8_t fieldDelim_;
  uint8_t collectionDelim_;
  uint8_t mapKeyDelim_;

  uint64_t row_;
  uint64_t skipRows_;

  uint64_t fileLength_;
  uint64_t dataOffset_;
  uint64_t dataEndOffset_;
  uint64_t blockEndOffset_;
  std::string leftover_;

  const char* bufferPtr_;
  int32_t bufferSize_;
  int32_t bufferOffset_;

  bool skippedPartialStartLine_;
};

class TextReader : public dwio::common::Reader {
 public:
  TextReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options);

  ~TextReader() override = default;

  std::optional<uint64_t> numberOfRows() const override {
    return std::nullopt;
  }

  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t index) const override {
    return nullptr;
  }

  const RowTypePtr& rowType() const override {
    return readerBase_->schema();
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override {
    return readerBase_->typeWithId();
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options) const override {
    return std::make_unique<TextRowReader>(readerBase_, options);
  }

 private:
  std::shared_ptr<ReaderBase> readerBase_;
  const std::shared_ptr<const dwio::common::TypeWithId> typeWithId_;
};

class TextReaderFactory : public dwio::common::ReaderFactory {
 public:
  TextReaderFactory() : ReaderFactory(dwio::common::FileFormat::TEXT) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options) override {
    return std::make_unique<TextReader>(std::move(input), options);
  }
};

} // namespace facebook::velox::text

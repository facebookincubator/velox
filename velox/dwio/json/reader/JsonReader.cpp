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

#include "velox/dwio/json/reader/JsonReader.h"

#include "velox/dwio/common/exception/Exceptions.h"

namespace facebook::velox::json {

FileContents::FileContents(
    memory::MemoryPool& pool,
    std::shared_ptr<const RowType> schema,
    dwio::common::JsonSerDeOptions serDeOptions)
    : pool{pool},
      schema{std::move(schema)},
      serDeOptions{std::move(serDeOptions)},
      input{nullptr} {}

JsonReader::JsonReader(
    const dwio::common::ReaderOptions& options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : options_{options} {
  auto schema = options_.fileSchema();
  VELOX_USER_CHECK_NOT_NULL(schema, "File schema for JSON must be set.");
  VELOX_USER_CHECK(schema->isRow(), "File schema for JSON must be a ROW type.");

  contents_ = std::make_shared<FileContents>(
      options_.memoryPool(),
      std::move(schema),
      dwio::common::JsonSerDeOptions{});
  contents_->input = std::move(input);
}

std::optional<uint64_t> JsonReader::numberOfRows() const {
  return std::nullopt;
}

std::unique_ptr<dwio::common::ColumnStatistics> JsonReader::columnStatistics(
    uint32_t /*index*/) const {
  return nullptr;
}

const RowTypePtr& JsonReader::rowType() const {
  return contents_->schema;
}

const std::shared_ptr<const dwio::common::TypeWithId>& JsonReader::typeWithId()
    const {
  if (typeWithId_ == nullptr) {
    typeWithId_ = dwio::common::TypeWithId::create(rowType());
  }
  return typeWithId_;
}

std::unique_ptr<dwio::common::RowReader> JsonReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<JsonRowReader>(contents_, options);
}

JsonRowReader::JsonRowReader(
    std::shared_ptr<FileContents> contents,
    const dwio::common::RowReaderOptions& options)
    : contents_{std::move(contents)}, options_{options} {}

uint64_t JsonRowReader::next(
    uint64_t /*size*/,
    VectorPtr& /*result*/,
    const dwio::common::Mutation* /*mutation*/) {
  // Phase 1 stub: no parsing yet.
  return 0;
}

int64_t JsonRowReader::nextRowNumber() {
  return kAtEnd;
}

int64_t JsonRowReader::nextReadSize(uint64_t /*size*/) {
  return kAtEnd;
}

void JsonRowReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& /*stats*/) const {
  // The JSON reader produces no runtime statistics. There is no
  // parse-error counter because there is no lenient mode: every parse
  // error throws.
}

void JsonRowReader::resetFilterCaches() {
  // No filter caches; the JSON reader does not use ScanSpec filters.
}

std::optional<size_t> JsonRowReader::estimatedRowSize() const {
  // JSON Lines has no cheap size estimate without a sampling pass.
  return std::nullopt;
}

} // namespace facebook::velox::json

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

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/TypeWithId.h"

namespace facebook::velox::json {

/// Shared state for a JSON file between JsonReader and the JsonRowReader
/// instances it spawns. Holds the input stream, schema, and SerDe options
/// so they are not duplicated across readers.
struct FileContents {
  FileContents(
      memory::MemoryPool& pool,
      std::shared_ptr<const RowType> schema,
      dwio::common::JsonSerDeOptions serDeOptions);

  /// Memory pool used for vector allocations during reads.
  memory::MemoryPool& pool;

  /// Top-level row schema requested by the caller. JSON has no embedded
  /// schema; this comes from the connector.
  const std::shared_ptr<const RowType> schema;

  /// SerDe options controlling parse behavior. Empty in this phase; later
  /// phases add date/timestamp format strings.
  dwio::common::JsonSerDeOptions serDeOptions;

  /// Decompressed byte stream for the file. Owned here so JsonRowReader
  /// can read from it without taking ownership.
  std::unique_ptr<dwio::common::BufferedInput> input;
};

/// Reader for the JSON file format (JSON Lines, matching Hive
/// org.apache.hive.hcatalog.data.JsonSerDe). Constructs JsonRowReader
/// instances that parse records out of the underlying stream.
class JsonReader : public dwio::common::Reader {
 public:
  JsonReader(
      const dwio::common::ReaderOptions& options,
      std::unique_ptr<dwio::common::BufferedInput> input);

  /// JSON Lines has no record count metadata; always returns nullopt.
  std::optional<uint64_t> numberOfRows() const override;

  /// JSON Lines has no per-column statistics; always returns nullptr.
  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t index) const override;

  /// Returns the schema supplied via ReaderOptions::fileSchema().
  const RowTypePtr& rowType() const override;

  /// Returns the schema with node identifiers attached.
  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override;

  /// Creates a row reader for the given range and column selection.
  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options) const override;

 private:
  // Reader-level options (memory pool, file schema, SerDe options).
  dwio::common::ReaderOptions options_;

  // Lazily computed schema with node identifiers.
  mutable std::shared_ptr<const dwio::common::TypeWithId> typeWithId_;

  // Per-file shared state passed to every row reader this reader creates.
  std::shared_ptr<FileContents> contents_;
};

/// Row reader for the JSON file format. Phase 1 stub: implements the
/// full RowReader interface but produces no rows.
class JsonRowReader : public dwio::common::RowReader {
 public:
  JsonRowReader(
      std::shared_ptr<FileContents> contents,
      const dwio::common::RowReaderOptions& options);

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const dwio::common::Mutation* mutation = nullptr) override;

  int64_t nextRowNumber() override;

  int64_t nextReadSize(uint64_t size) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override;

  void resetFilterCaches() override;

  std::optional<size_t> estimatedRowSize() const override;

 private:
  // Per-file shared state (input stream, schema, options).
  const std::shared_ptr<FileContents> contents_;

  // Caller-supplied row reader options (range, selector, scan spec).
  dwio::common::RowReaderOptions options_;
};

} // namespace facebook::velox::json

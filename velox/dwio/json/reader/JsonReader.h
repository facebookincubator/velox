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

#include <string>
#include <unordered_map>

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/functions/lib/DateTimeFormatter.h"

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

  /// SerDe options controlling parse behavior, including the Joda-style
  /// format strings used to parse DATE and TIMESTAMP columns.
  dwio::common::JsonSerDeOptions serDeOptions;

  /// Formatter compiled once from serDeOptions.dateFormat, reused across
  /// rows to parse DATE columns. simdjson hands us the JSON string; this
  /// turns it into days since the epoch.
  std::shared_ptr<functions::DateTimeFormatter> dateFormatter;

  /// Formatter compiled once from serDeOptions.timestampFormat, reused
  /// across rows to parse TIMESTAMP columns.
  std::shared_ptr<functions::DateTimeFormatter> timestampFormatter;

  /// Decompressed byte stream for the file. Owned here so JsonRowReader
  /// can read from it without taking ownership.
  std::unique_ptr<dwio::common::BufferedInput> input;

  /// Lowercased top-level field name to column index in the schema.
  /// Built once from the schema; reused across rows. The iterate-once
  /// dispatch pattern requires
  /// a fast name lookup because simdjson On-Demand is forward-only and
  /// values cannot be stashed for later association with a column.
  std::unordered_map<std::string, size_t> fieldIndex;
};

/// Reader for the JSON file format (JSON Lines, matching Hive
/// org.apache.hive.hcatalog.data.JsonSerDe). Constructs JsonRowReader
/// instances that parse records out of the underlying stream.
///
/// Known v1 divergence: VARCHAR-formatted JSON numbers may differ in exact
/// string form from Presto's Hive JSON connector for edge cases involving
/// trailing zeros and scientific notation (Presto canonicalizes via
/// BigDecimal; v1 emits the original lexeme). The semantic numeric value is
/// preserved; only the textual representation may diverge.
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

/// Row reader for the JSON file format. Reads one JSON object per line,
/// dispatching each field to the matching column via the schema field
/// index. The whole file is loaded into memory at construction; split
/// support is deferred (see json-reader-pr-roadmap.md PR-7).
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
  // Reads the next newline-terminated line from the file buffer into
  // lineBuffer_, padded with SIMDJSON_PADDING zero bytes for safe
  // simdjson parsing. Returns false when there are no more lines.
  bool readNextLine();

  // Parses lineBuffer_ as a single JSON object and writes its fields
  // into the corresponding columns of row at rowIndex. Fields not in
  // the schema are silently ignored; fields in the schema but absent
  // from the JSON object remain NULL.
  void writeRow(RowVector& row, vector_size_t rowIndex);

  // Per-file shared state (input stream, schema, options).
  const std::shared_ptr<FileContents> contents_;

  // Caller-supplied row reader options (range, selector, scan spec).
  dwio::common::RowReaderOptions options_;

  // Entire file contents loaded at construction. Split support
  // is deferred to a later PR.
  std::string fileBuffer_;

  // Length of valid bytes in fileBuffer_. fileBuffer_ has additional
  // SIMDJSON_PADDING bytes of zeroes after fileLength_ so the last
  // line can be parsed in place.
  size_t fileLength_{0};

  // Current read offset into fileBuffer_. Records start at this position.
  size_t pos_{0};

  // Reusable padded buffer holding the current line. Sized to fit the
  // longest line seen so far plus SIMDJSON_PADDING.
  std::string lineBuffer_;

  // Length of valid line content in lineBuffer_ (excluding padding).
  size_t lineLength_{0};
};

} // namespace facebook::velox::json

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

#include <cctype>
#include <cstring>

#include <folly/Conv.h>

#include "velox/dwio/common/exception/Exceptions.h"
#include "velox/functions/prestosql/json/SIMDJsonWrapper.h"

namespace facebook::velox::json {
namespace {

// Unwraps a simdjson_result, throwing VELOX_USER_FAIL on error.
template <typename T>
T unwrap(simdjson::simdjson_result<T>&& result) {
  if (result.error() != simdjson::SUCCESS) {
    VELOX_USER_FAIL(
        "JSON parse error: {}", simdjson::error_message(result.error()));
  }
  return std::move(result).value_unsafe();
}

// Lowercases an ASCII string. JSON field names that fall outside ASCII go
// through unchanged; full Unicode folding lands with nested ROW support
// (see json-reader-pr-roadmap.md PR-6c).
std::string asciiLower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

// Casts a JSON-parsed double to int64, wrapping through uint64 for values
// in [2^63, 2^64). Values outside [-2^63, 2^64) are out of representable
// range and diverge from Presto/Jackson for inputs that exceed 2^64 in
// magnitude. The v1 contract is "no
// throw", not "bit-for-bit parity".
int64_t doubleToInt64Truncating(double d) {
  constexpr double kInt64MinAsDouble = -9.2233720368547758e18; // -2^63
  constexpr double kInt64MaxPlusOne = 9.2233720368547758e18; // 2^63
  constexpr double kUint64MaxPlusOne = 1.8446744073709552e19; // 2^64
  if (d >= kInt64MinAsDouble && d < kInt64MaxPlusOne) {
    return static_cast<int64_t>(d);
  }
  if (d >= kInt64MaxPlusOne && d < kUint64MaxPlusOne) {
    return static_cast<int64_t>(static_cast<uint64_t>(d));
  }
  // Out-of-range input. Velox v1 diverges from Presto/Jackson here.
  // The test contract is "does not throw".
  return 0;
}

// Returns the int64 coercion of a JSON scalar value per the empirical table.
// Throws VELOX_USER_FAIL on container-shape
// mismatch (object or array). null becomes SQL NULL via the caller's
// is_null() check before this function is called.
int64_t coerceToInt64(simdjson::ondemand::value& value) {
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::number: {
      auto num = unwrap(value.get_number());
      switch (num.get_number_type()) {
        case simdjson::ondemand::number_type::signed_integer:
          return num.get_int64();
        case simdjson::ondemand::number_type::unsigned_integer:
          // [2^63, 2^64) wraps via two's complement to match Presto/Jackson.
          return static_cast<int64_t>(num.get_uint64());
        case simdjson::ondemand::number_type::floating_point_number:
          // -3.7 -> -3 (truncate toward zero, not floor).
          return doubleToInt64Truncating(num.get_double());
        case simdjson::ondemand::number_type::big_integer:
          // simdjson reports big_integer for integer literals beyond
          // uint64. value.get_number() returns NUMBER_OUT_OF_RANGE in
          // that case, so the unwrap above already threw. Floating
          // forms like 1e20 (the test's overflow case) parse as
          // floating_point_number above and diverge from
          // Presto/Jackson.
          VELOX_UNREACHABLE();
      }
      VELOX_UNREACHABLE();
    }
    case simdjson::ondemand::json_type::boolean:
      return unwrap(value.get_bool()) ? 1 : 0;
    case simdjson::ondemand::json_type::string: {
      auto s = unwrap(value.get_string());
      // Full-fail to 0 — NOT partial-parse.
      // "12abc" -> 0, not 12.
      auto parsed = folly::tryTo<int64_t>(s);
      return parsed.hasValue() ? parsed.value() : 0;
    }
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array:
      VELOX_USER_FAIL(
          "Container shape mismatch: integer column received a JSON object or array.");
    case simdjson::ondemand::json_type::null:
      // Caller checks is_null() before invoking; reaching here is a bug.
      VELOX_UNREACHABLE();
    case simdjson::ondemand::json_type::unknown:
      VELOX_USER_FAIL("Unrecognized JSON value type.");
  }
  VELOX_UNREACHABLE();
}

double coerceToDouble(simdjson::ondemand::value& value) {
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::number:
      return unwrap(value.get_double());
    case simdjson::ondemand::json_type::boolean:
      return unwrap(value.get_bool()) ? 1.0 : 0.0;
    case simdjson::ondemand::json_type::string: {
      auto s = unwrap(value.get_string());
      auto parsed = folly::tryTo<double>(s);
      return parsed.hasValue() ? parsed.value() : 0.0;
    }
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array:
      VELOX_USER_FAIL(
          "Container shape mismatch: floating-point column received a JSON object or array.");
    case simdjson::ondemand::json_type::null:
      VELOX_UNREACHABLE();
    case simdjson::ondemand::json_type::unknown:
      VELOX_USER_FAIL("Unrecognized JSON value type.");
  }
  VELOX_UNREACHABLE();
}

// Writes one JSON value into a FlatVector cell. The caller has verified
// that the JSON value is not `null` (it set the cell to NULL beforehand).
void writeValue(
    simdjson::ondemand::value& value,
    const TypePtr& type,
    BaseVector& column,
    vector_size_t rowIndex) {
  switch (type->kind()) {
    case TypeKind::BIGINT: {
      column.asUnchecked<FlatVector<int64_t>>()->set(
          rowIndex, coerceToInt64(value));
      return;
    }
    case TypeKind::INTEGER: {
      // Narrowing wrap from int64 follows the empirical table.
      column.asUnchecked<FlatVector<int32_t>>()->set(
          rowIndex, static_cast<int32_t>(coerceToInt64(value)));
      return;
    }
    case TypeKind::SMALLINT: {
      column.asUnchecked<FlatVector<int16_t>>()->set(
          rowIndex, static_cast<int16_t>(coerceToInt64(value)));
      return;
    }
    case TypeKind::TINYINT: {
      column.asUnchecked<FlatVector<int8_t>>()->set(
          rowIndex, static_cast<int8_t>(coerceToInt64(value)));
      return;
    }
    case TypeKind::DOUBLE: {
      column.asUnchecked<FlatVector<double>>()->set(
          rowIndex, coerceToDouble(value));
      return;
    }
    case TypeKind::REAL: {
      column.asUnchecked<FlatVector<float>>()->set(
          rowIndex, static_cast<float>(coerceToDouble(value)));
      return;
    }
    default:
      VELOX_NYI(
          "JSON reader does not yet support column type: {}",
          type->toString());
  }
}

} // namespace

FileContents::FileContents(
    memory::MemoryPool& pool,
    std::shared_ptr<const RowType> schema,
    dwio::common::JsonSerDeOptions serDeOptions)
    : pool{pool},
      schema{std::move(schema)},
      serDeOptions{serDeOptions},
      input{nullptr} {
  fieldIndex.reserve(this->schema->size());
  for (size_t i = 0; i < this->schema->size(); ++i) {
    // Last-write-wins on case-collision: a schema with both `x` and `X`
    // produces a single index entry pointing to the second field. JSON
    // field matching is case-insensitive.
    fieldIndex[asciiLower(this->schema->nameOf(i))] = i;
  }
}

JsonReader::JsonReader(
    const dwio::common::ReaderOptions& options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : options_{options} {
  auto schema = options_.fileSchema();
  VELOX_USER_CHECK_NOT_NULL(schema, "File schema for JSON must be set.");
  VELOX_USER_CHECK(schema->isRow(), "File schema for JSON must be a ROW type.");

  contents_ = std::make_shared<FileContents>(
      options_.memoryPool(), std::move(schema), dwio::common::JsonSerDeOptions{});
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
    : contents_{std::move(contents)}, options_{options} {
  const auto& readFile = contents_->input->getReadFile();
  fileLength_ = readFile->size();
  // Pad the buffer so the last line can be parsed by simdjson without
  // copying. Lines other than the last are still copied into lineBuffer_
  // because simdjson requires padding bytes after the parsed region.
  fileBuffer_.assign(fileLength_ + simdjson::SIMDJSON_PADDING, '\0');
  if (fileLength_ > 0) {
    readFile->pread(0, fileLength_, fileBuffer_.data());
  }
}

bool JsonRowReader::readNextLine() {
  if (pos_ >= fileLength_) {
    return false;
  }
  const char* start = fileBuffer_.data() + pos_;
  size_t remaining = fileLength_ - pos_;
  const char* nl = static_cast<const char*>(std::memchr(start, '\n', remaining));
  size_t length = (nl == nullptr) ? remaining : static_cast<size_t>(nl - start);

  // Reserve enough room for the line plus simdjson's required trailing
  // padding. Reuse the buffer across rows; std::string keeps capacity.
  if (lineBuffer_.size() < length + simdjson::SIMDJSON_PADDING) {
    lineBuffer_.assign(length + simdjson::SIMDJSON_PADDING, '\0');
  } else {
    // Zero the padding region in case the previous line was longer.
    std::memset(lineBuffer_.data() + length, 0, simdjson::SIMDJSON_PADDING);
  }
  std::memcpy(lineBuffer_.data(), start, length);
  lineLength_ = length;
  pos_ += length + (nl == nullptr ? 0 : 1);
  return true;
}

void JsonRowReader::writeRow(RowVector& row, vector_size_t rowIndex) {
  // Initialize every column at this row to NULL. Fields absent from the
  // JSON object stay NULL; present fields overwrite below.
  for (size_t i = 0; i < row.childrenSize(); ++i) {
    auto* child = row.childAt(i).get();
    if (child != nullptr) {
      child->setNull(rowIndex, true);
    }
  }

  simdjson::padded_string_view padded(
      lineBuffer_.data(),
      lineLength_,
      lineLength_ + simdjson::SIMDJSON_PADDING);
  thread_local simdjson::ondemand::parser parser;
  auto docResult = parser.iterate(padded);
  if (docResult.error() != simdjson::SUCCESS) {
    VELOX_USER_FAIL(
        "JSON parse error: {}",
        simdjson::error_message(docResult.error()));
  }
  simdjson::ondemand::document doc = std::move(docResult).value_unsafe();

  auto objResult = doc.get_object();
  if (objResult.error() != simdjson::SUCCESS) {
    VELOX_USER_FAIL(
        "JSON record is not an object: {}",
        simdjson::error_message(objResult.error()));
  }
  simdjson::ondemand::object obj = objResult.value_unsafe();

  // Iterate-once dispatch. simdjson On-Demand is forward-only — we
  // cannot stash ondemand::value handles for later association with a
  // column. Instead, look each field up in the schema's lowercase-name
  // -> column-index map and dispatch directly into the matching column.
  // Last-write-wins on case-duplicate keys falls out for free.
  for (auto field : obj) {
    auto keyResult = field.unescaped_key();
    if (keyResult.error() != simdjson::SUCCESS) {
      VELOX_USER_FAIL(
          "JSON parse error: {}",
          simdjson::error_message(keyResult.error()));
    }
    std::string_view key = keyResult.value_unsafe();
    auto it = contents_->fieldIndex.find(asciiLower(key));
    if (it == contents_->fieldIndex.end()) {
      // Extra field — silently ignored.
      continue;
    }

    auto valueResult = field.value();
    if (valueResult.error() != simdjson::SUCCESS) {
      VELOX_USER_FAIL(
          "JSON parse error: {}",
          simdjson::error_message(valueResult.error()));
    }
    simdjson::ondemand::value value = valueResult.value_unsafe();
    auto* child = row.childAt(it->second).get();
    if (child == nullptr) {
      continue;
    }
    if (value.is_null()) {
      // Already NULL from the per-row initialization above.
      continue;
    }
    writeValue(value, contents_->schema->childAt(it->second), *child, rowIndex);
  }
}

uint64_t JsonRowReader::next(
    uint64_t size,
    VectorPtr& result,
    const dwio::common::Mutation* /*mutation*/) {
  if (size == 0 || pos_ >= fileLength_) {
    return 0;
  }

  auto rowVector = BaseVector::create<RowVector>(
      contents_->schema, static_cast<vector_size_t>(size), &contents_->pool);

  vector_size_t rowsRead = 0;
  while (rowsRead < static_cast<vector_size_t>(size) && readNextLine()) {
    writeRow(*rowVector, rowsRead);
    ++rowsRead;
  }

  rowVector->resize(rowsRead);
  result = rowVector;
  return static_cast<uint64_t>(rowsRead);
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

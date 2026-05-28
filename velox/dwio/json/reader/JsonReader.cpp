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

#include "velox/common/encode/Base64.h"
#include "velox/dwio/common/exception/Exceptions.h"
#include "velox/functions/prestosql/json/SIMDJsonWrapper.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::json {
namespace {

// Unwraps a simdjson_result, throwing VELOX_USER_FAIL on error.
template <typename T>
T unwrap(simdjson::simdjson_result<T> result) {
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

// Trims trailing ASCII whitespace from a raw simdjson token. raw_json_token()
// returns the value's bytes up to (but not including) the next structural
// character, which can leave trailing spaces or newlines for scalars.
std::string_view trimTrailingWhitespace(std::string_view token) {
  size_t size = token.size();
  while (size > 0) {
    char c = token[size - 1];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      break;
    }
    --size;
  }
  return token.substr(0, size);
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

// Returns the boolean coercion of a JSON scalar.
// Numbers coerce by nonzero (any nonzero, including negatives, is true).
// Strings match the strict literal "true" (lowercase, exactly four bytes);
// everything else — "True", "TRUE", "yes", "", "false" — is false. Throws on
// container-shape mismatch. null is handled by the caller's is_null() check.
bool coerceToBool(simdjson::ondemand::value& value) {
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::boolean:
      return unwrap(value.get_bool());
    case simdjson::ondemand::json_type::number: {
      auto num = unwrap(value.get_number());
      switch (num.get_number_type()) {
        case simdjson::ondemand::number_type::signed_integer:
          return num.get_int64() != 0;
        case simdjson::ondemand::number_type::unsigned_integer:
          return num.get_uint64() != 0;
        case simdjson::ondemand::number_type::floating_point_number:
          return num.get_double() != 0.0;
        case simdjson::ondemand::number_type::big_integer:
          // get_number() already returned NUMBER_OUT_OF_RANGE and threw.
          VELOX_UNREACHABLE();
      }
      VELOX_UNREACHABLE();
    }
    case simdjson::ondemand::json_type::string: {
      auto s = unwrap(value.get_string());
      // Strict literal match — case-sensitive — per the probe: only the
      // exact lowercase "true" is true. "True"/"TRUE" are false.
      return s.size() == 4 && std::memcmp(s.data(), "true", 4) == 0;
    }
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array:
      VELOX_USER_FAIL(
          "Container shape mismatch: boolean column received a JSON object or array.");
    case simdjson::ondemand::json_type::null:
      VELOX_UNREACHABLE();
    case simdjson::ondemand::json_type::unknown:
      VELOX_USER_FAIL("Unrecognized JSON value type.");
  }
  VELOX_UNREACHABLE();
}

// JSON-escapes str into out, emitting compact escapes for the mandatory
// characters (quote, backslash, control bytes). Forward slash is left bare
// because Jackson's compact serialization does not escape it — this is what
// makes "http:\/\/x" re-serialize as http://x. The surrounding quotes are
// the caller's responsibility.
void appendEscapedJsonString(std::string_view str, std::string& out) {
  for (char c : str) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        const auto byte = static_cast<unsigned int>(static_cast<unsigned char>(c));
        if (byte < 0x20) {
          constexpr char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(byte >> 4U) & 0xFU];
          out += kHex[byte & 0xFU];
        } else {
          out += c;
        }
    }
  }
}

// Forward declaration: serializeJsonValue and serializeJsonObject/Array
// recurse into one another.
void serializeJsonValue(simdjson::ondemand::value& value, std::string& out);

// Minifies a JSON object into out: whitespace stripped, escapes decoded then
// re-encoded, input key order preserved (NOT canonicalized — distinct from
// the JSON-typed-column rule). simdjson On-Demand
// is forward-only, so each field is visited exactly once in document order.
void serializeJsonObject(simdjson::ondemand::object& obj, std::string& out) {
  out += '{';
  bool first = true;
  for (auto field : obj) {
    if (!first) {
      out += ',';
    }
    first = false;
    out += '"';
    appendEscapedJsonString(unwrap(field.unescaped_key()), out);
    out += "\":";
    auto value = unwrap(field.value());
    serializeJsonValue(value, out);
  }
  out += '}';
}

// Minifies a JSON array into out, preserving element order.
void serializeJsonArray(simdjson::ondemand::array& arr, std::string& out) {
  out += '[';
  bool first = true;
  for (auto element : arr) {
    if (!first) {
      out += ',';
    }
    first = false;
    auto value = unwrap(element);
    serializeJsonValue(value, out);
  }
  out += ']';
}

void serializeJsonValue(simdjson::ondemand::value& value, std::string& out) {
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::object: {
      auto obj = unwrap(value.get_object());
      serializeJsonObject(obj, out);
      return;
    }
    case simdjson::ondemand::json_type::array: {
      auto arr = unwrap(value.get_array());
      serializeJsonArray(arr, out);
      return;
    }
    case simdjson::ondemand::json_type::string:
      out += '"';
      appendEscapedJsonString(unwrap(value.get_string()), out);
      out += '"';
      return;
    case simdjson::ondemand::json_type::number:
    case simdjson::ondemand::json_type::boolean:
    case simdjson::ondemand::json_type::null:
      // Scalars carry no whitespace within the token; emit the lexeme as-is
      // (trailing whitespace before the next structural char is trimmed).
      out += trimTrailingWhitespace(value.raw_json_token());
      return;
    case simdjson::ondemand::json_type::unknown:
      VELOX_USER_FAIL("Unrecognized JSON value type.");
  }
  VELOX_UNREACHABLE();
}

// Returns the VARCHAR coercion of a JSON value.
// Strings pass through with escapes decoded by the parser; booleans become
// the lowercase literals; objects and arrays are re-serialized minified with
// key order preserved; numbers use a best-effort lexeme. null is handled by
// the caller's is_null() check.
//
// VARCHAR-from-number is a known v1 divergence: Presto/Jackson formats via
// BigDecimal(input).stripTrailingZeros().toString(), which C++ has no
// standard equivalent for. v1 emits the original lexeme, so the semantic
// value is preserved but the exact textual form (trailing zeros, scientific
// notation case/threshold) may differ. This is the documented
// VARCHAR-from-number v1 divergence.
std::string coerceToString(simdjson::ondemand::value& value) {
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::string:
      return std::string{unwrap(value.get_string())};
    case simdjson::ondemand::json_type::boolean:
      return unwrap(value.get_bool()) ? "true" : "false";
    case simdjson::ondemand::json_type::number:
      return std::string{trimTrailingWhitespace(value.raw_json_token())};
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array: {
      std::string out;
      serializeJsonValue(value, out);
      return out;
    }
    case simdjson::ondemand::json_type::null:
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

// Parses a JSON value into a decimal of the given precision and scale from
// the original lexeme, NOT through double. Routing a high-precision number
// through double silently rounds away digits a double cannot hold (a double
// carries ~15-16 significant decimal digits); the raw lexeme preserves them.
// Numbers use the raw token; strings use the decoded contents. Throws on a
// container-shape mismatch or a lexeme the decimal parser rejects.
template <typename T>
T coerceToDecimal(
    simdjson::ondemand::value& value,
    uint8_t precision,
    uint8_t scale) {
  std::string_view lexeme;
  // Backs lexeme when the source is a JSON string; must outlive the parse.
  std::string decoded;
  auto type = unwrap(value.type());
  switch (type) {
    case simdjson::ondemand::json_type::number:
      lexeme = trimTrailingWhitespace(value.raw_json_token());
      break;
    case simdjson::ondemand::json_type::string:
      decoded = std::string{unwrap(value.get_string())};
      lexeme = decoded;
      break;
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array:
      VELOX_USER_FAIL(
          "Container shape mismatch: decimal column received a JSON object or array.");
    case simdjson::ondemand::json_type::boolean:
      VELOX_USER_FAIL("Cannot coerce a JSON boolean to a decimal column.");
    case simdjson::ondemand::json_type::null:
      VELOX_UNREACHABLE();
    case simdjson::ondemand::json_type::unknown:
      VELOX_USER_FAIL("Unrecognized JSON value type.");
  }

  T out{0};
  const auto status = DecimalUtil::castFromString<T>(
      StringView(lexeme.data(), static_cast<int32_t>(lexeme.size())),
      precision,
      scale,
      out);
  if (!status.ok()) {
    VELOX_USER_FAIL(
        "Cannot parse decimal from JSON lexeme: {} ({})",
        lexeme,
        status.message());
  }
  return out;
}

// Base64-decodes a JSON string into raw bytes for a VARBINARY column. Presto
// carries binary data base64-encoded in JSON text, so the reader decodes on
// the way in. Throws on a non-string value or invalid base64.
std::string coerceToVarbinary(simdjson::ondemand::value& value) {
  auto type = unwrap(value.type());
  if (type != simdjson::ondemand::json_type::string) {
    VELOX_USER_FAIL("VARBINARY column requires a base64-encoded JSON string.");
  }
  auto encoded = unwrap(value.get_string());
  // calculateDecodedSize adjusts inputSize for padding; the adjusted value
  // must be the one passed to decode.
  size_t inputSize = encoded.size();
  auto decodedSize =
      encoding::Base64::calculateDecodedSize(encoded.data(), inputSize);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL(
        "Invalid base64 in VARBINARY column: {}",
        decodedSize.error().message());
  }
  std::string out;
  out.resize(decodedSize.value());
  const auto status = encoding::Base64::decode(
      encoded.data(), inputSize, out.data(), out.size());
  if (!status.ok()) {
    VELOX_USER_FAIL(
        "Invalid base64 in VARBINARY column: {}", status.message());
  }
  return out;
}

// Writes one JSON value into a FlatVector cell. The caller has verified
// that the JSON value is not `null` (it set the cell to NULL beforehand).
void writeValue(
    simdjson::ondemand::value& value,
    const TypePtr& type,
    BaseVector& column,
    vector_size_t rowIndex) {
  // DECIMAL is read from the raw lexeme, not coerced through a numeric kind.
  // It must be intercepted before the kind switch because a short decimal's
  // TypeKind is BIGINT and a long decimal's is HUGEINT — the switch would
  // otherwise misroute them. JSON-typed columns are not handled here: the
  // Hive connector does not accept JSON column declarations today, so
  // JSON-shaped data flows through VARCHAR. If
  // a future Presto adds JSON columns to the Hive path, this dispatch reopens.
  if (type->isDecimal()) {
    const auto [precision, scale] = getDecimalPrecisionScale(*type);
    if (type->isShortDecimal()) {
      column.asUnchecked<FlatVector<int64_t>>()->set(
          rowIndex, coerceToDecimal<int64_t>(value, precision, scale));
    } else {
      column.asUnchecked<FlatVector<int128_t>>()->set(
          rowIndex, coerceToDecimal<int128_t>(value, precision, scale));
    }
    return;
  }

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
    case TypeKind::BOOLEAN: {
      column.asUnchecked<FlatVector<bool>>()->set(
          rowIndex, coerceToBool(value));
      return;
    }
    case TypeKind::VARCHAR: {
      auto str = coerceToString(value);
      // FlatVector::set() copies the StringView's data into the vector's
      // owned string buffer, so the temporary str can safely go out of scope.
      column.asUnchecked<FlatVector<StringView>>()->set(
          rowIndex, StringView(str.data(), str.size()));
      return;
    }
    case TypeKind::VARBINARY: {
      auto bytes = coerceToVarbinary(value);
      // FlatVector::set() copies the bytes into the vector's owned string
      // buffer, so the temporary can safely go out of scope.
      column.asUnchecked<FlatVector<StringView>>()->set(
          rowIndex, StringView(bytes.data(), bytes.size()));
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

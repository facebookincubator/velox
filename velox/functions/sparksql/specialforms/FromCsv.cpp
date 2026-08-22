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

#include "velox/functions/sparksql/specialforms/FromCsv.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fast_float/fast_float.h>

#include "velox/dwio/text/reader/TextFieldParser.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Maximum CSV input line size (10 MB) to prevent DoS from unbounded allocation.
constexpr size_t kMaxCsvLineSize = 10 * 1024 * 1024;

// Trims leading and trailing ASCII CSV whitespace (space, tab, CR, LF).
// Used for the per-type REAL/DOUBLE trim path (Java's parseFloat/parseDouble
// accept surrounding whitespace).
// Integer, boolean, decimal, VARCHAR, and VARBINARY fields are NOT trimmed.
std::string_view trimWhitespace(std::string_view input) {
  size_t start{0};
  while (start < input.size() &&
         (input[start] == ' ' || input[start] == '\t' || input[start] == '\r' ||
          input[start] == '\n')) {
    ++start;
  }
  size_t end{input.size()};
  while (end > start &&
         (input[end - 1] == ' ' || input[end - 1] == '\t' ||
          input[end - 1] == '\r' || input[end - 1] == '\n')) {
    --end;
  }
  return input.substr(start, end - start);
}

// Reusable per-row buffers for `splitCsvLine`. Grouping keeps the parser
// signature small and makes it clear which state is meant to be recycled
// across rows to avoid per-row allocation.
//
// - `fields`: parsed field values.
// - `fieldWasQuoted`: parallel to `fields` (same size), 1 if the field was
//   successfully parsed as a properly-closed quoted CSV token, 0 for
//   unquoted fields or literal-fallback fields from malformed quotes. This
//   distinction is required for correct `nullValue` handling: Spark's
//   `from_csv` `nullValue` (default "") matches only unquoted-empty tokens,
//   so quoted-empty `""` must produce an empty string rather than NULL.
//   `vector<char>` (not `vector<bool>`) is used to avoid the proxy-reference
//   specialization and keep pointer/index semantics identical to `fields`.
// - `quotedContent`: scratch buffer used while building the value of a
//   quoted field. Reused across rows and across fields within a row.
struct SplitCsvBuffers {
  std::vector<std::string> fields;
  std::vector<char> fieldWasQuoted;
  std::string quotedContent;
};

// Splits a single CSV line into fields, handling quoted fields.
// Follows Spark's CSV parsing rules (Univocity defaults):
// - Fields may be enclosed in double quotes.
// - Inside a quoted field, the escape character (backslash '\' by Spark
//   default) followed by a quote emits a literal quote. Doubled quotes ("")
//   also emit a literal quote (RFC 4180 compatibility, matching Univocity's
//   default quoteEscape == quote). Pass '\0' as `escape` to disable.
// - Unquoted fields are taken as-is.
// - unescapedQuoteHandling=STOP_AT_DELIMITER: after the closing quote,
//   only ASCII whitespace may appear before the next delimiter. Any other
//   trailing characters cause the entire field — from the opening quote
//   up to the next delimiter — to be taken as literal text.
// maxFields: stop parsing once this many fields are produced (0 = unlimited).
// buffers: reusable per-row state. `fields` and `fieldWasQuoted` are cleared
//   at function entry; `quotedContent` is cleared lazily on entry to each
//   quoted-field branch (so callers do not need to reset it between rows).
void splitCsvLine(
    std::string_view line,
    char delimiter,
    char escape,
    size_t maxFields,
    SplitCsvBuffers& buffers) {
  auto& fields = buffers.fields;
  auto& fieldWasQuoted = buffers.fieldWasQuoted;
  auto& quotedContent = buffers.quotedContent;
  fields.clear();
  fieldWasQuoted.clear();
  VELOX_DCHECK_LE(
      line.size(),
      kMaxCsvLineSize,
      "splitCsvLine called with oversized input — caller should guard.");
  if (maxFields > 0) {
    fields.reserve(maxFields);
    fieldWasQuoted.reserve(maxFields);
  }
  size_t i{0};
  bool trailingDelimiter{false};
  while (i < line.size()) {
    // Stop parsing once we have enough fields.
    if (maxFields > 0 && fields.size() >= maxFields) {
      break;
    }
    trailingDelimiter = false;
    if (line[i] == '"') {
      // Quoted field: build content, handling escaped quotes.
      // Spark's default unescapedQuoteHandling=STOP_AT_DELIMITER: if the
      // quote is not properly closed before EOL, treat it as literal text
      // and re-parse from the opening quote as an unquoted field.
      quotedContent.clear();
      size_t j = i + 1;
      bool closedProperly{false};
      while (j < line.size()) {
        if (escape != '\0' && line[j] == escape && j + 1 < line.size() &&
            line[j + 1] == '"') {
          // Escape char followed by quote — emit a literal quote.
          quotedContent.push_back('"');
          j += 2;
        } else if (line[j] == '"') {
          if (j + 1 < line.size() && line[j + 1] == '"') {
            // Doubled quote — emit a single quote (RFC 4180).
            quotedContent.push_back('"');
            j += 2;
          } else {
            // End of quoted field.
            closedProperly = true;
            break;
          }
        } else {
          quotedContent.push_back(line[j]);
          ++j;
        }
      }
      if (!closedProperly) {
        // Unclosed quote: Spark's UnivocityParser strips the opening quote
        // and returns the rest as the field value. Skip past the opening
        // quote character (at position i) and take the remainder up to the
        // next delimiter.
        size_t start = i + 1; // skip opening quote
        i = start;
        while (i < line.size() && line[i] != delimiter) {
          ++i;
        }
        fields.emplace_back(line.substr(start, i - start));
        fieldWasQuoted.push_back(0);
        if (i < line.size()) {
          ++i; // skip delimiter
          trailingDelimiter = true;
        }
      } else {
        // Advance past closing quote.
        ++j;
        // Spark/Univocity default unescapedQuoteHandling=STOP_AT_DELIMITER:
        // skip ASCII whitespace (chars <= ' ') between the closing quote and
        // the next delimiter. If the next non-whitespace character is the
        // delimiter or end of line, accept the parsed quoted content.
        // Otherwise, treat the entire field — from the opening quote up to
        // the next delimiter — as literal text.
        size_t k = j;
        while (k < line.size() && static_cast<unsigned char>(line[k]) <= ' ' &&
               line[k] != delimiter) {
          ++k;
        }
        if (k == line.size() || line[k] == delimiter) {
          fields.push_back(std::move(quotedContent));
          fieldWasQuoted.push_back(1);
          if (k < line.size()) {
            ++k; // Skip delimiter.
            trailingDelimiter = true;
          }
          i = k;
        } else {
          // Garbage after close — fall back to literal from opening quote.
          size_t start = i;
          while (i < line.size() && line[i] != delimiter) {
            ++i;
          }
          fields.emplace_back(line.substr(start, i - start));
          fieldWasQuoted.push_back(0);
          if (i < line.size()) {
            ++i; // Skip delimiter.
            trailingDelimiter = true;
          }
        }
      }
    } else {
      // Unquoted field.
      size_t start = i;
      while (i < line.size() && line[i] != delimiter) {
        ++i;
      }
      fields.emplace_back(line.substr(start, i - start));
      fieldWasQuoted.push_back(0);
      if (i < line.size()) {
        ++i; // skip delimiter
        trailingDelimiter = true;
      }
    }
  }
  // A trailing delimiter means there is one more empty field after it.
  // An empty input line also produces one empty field.
  if ((trailingDelimiter || line.empty()) &&
      (maxFields == 0 || fields.size() < maxFields)) {
    fields.emplace_back();
    fieldWasQuoted.push_back(0);
  }
}

// Strips a leading '+' sign from a numeric string view (Spark accepts +123).
// Returns empty view for malformed inputs like "+-123" or "++123".
std::string_view stripLeadingPlus(std::string_view input) {
  if (!input.empty() && input[0] == '+') {
    auto rest = input.substr(1);
    // Reject "+-N", "++N" — Java parseInt("+-123") throws.
    if (!rest.empty() && (rest[0] == '-' || rest[0] == '+')) {
      return {};
    }
    return rest;
  }
  return input;
}

// Checks if a numeric string starts with hex prefix (0x/0X), optionally after
// a sign character. Java's Integer.parseInt and Double.parseDouble reject hex.
bool hasHexPrefix(std::string_view input) {
  if (input.size() >= 2 && input[0] == '0' &&
      (input[1] == 'x' || input[1] == 'X')) {
    return true;
  }
  if (input.size() >= 3 && (input[0] == '+' || input[0] == '-') &&
      input[1] == '0' && (input[2] == 'x' || input[2] == 'X')) {
    return true;
  }
  return false;
}

// Parses an integer field matching Spark/Java semantics by delegating to the
// shared text-reader helper with strict trailing-character rejection. Adds
// from_csv-specific preprocessing: optional leading '+' sign (Java accepts
// "+123" but the shared helper does not, since TextReader rejects it).
//
// Hex literals (e.g. "0x1F", "-0X10") are rejected automatically: the shared
// parser stops at the non-digit 'x', and allowTrailingDecimal=false rejects
// the unparsed remainder.
template <typename T>
bool parseInt(std::string_view input, T& out) {
  if (input.empty()) {
    return false;
  }
  auto numStr = stripLeadingPlus(input);
  if (numStr.empty()) {
    return false;
  }
  auto parsed = ::facebook::velox::text::TextFieldParser::parseNarrowInteger<T>(
      numStr, /*allowTrailingDecimal=*/false);
  if (!parsed.has_value()) {
    return false;
  }
  out = *parsed;
  return true;
}

// Parses a floating-point string matching Spark/Java semantics.
// Accepts: decimal notation, "NaN" (exact), "Infinity"/"-Infinity"/"+Infinity",
//          "Inf"/"-Inf" (Spark CSV defaults: positiveInf="Inf",
//          negativeInf="-Inf").
// Rejects: hex floats (0x...), case-insensitive nan/inf variants.
// Overflow returns ±Infinity (matching Java's parseDouble("1e400") = Infinity).
//
// TODO: Extract a shared float parser into TextFieldParser.h once the
// TextReader path (sscanf %f with lenient ERANGE handling and case-insensitive
// nan/inf via boost::iequals) is reconciled with Spark's case-sensitive,
// Java-style overflow→±Inf / underflow→±0 semantics.

// Classification of what `fast_float::from_chars` overflow means for a given
// input, per Java's Float.parseFloat / Double.parseDouble semantics:
//   - kPositiveInfinity: magnitude too large, positive sign (e.g. "1e400").
//   - kNegativeInfinity: magnitude too large, negative sign (e.g. "-1e400").
//   - kPositiveZero: magnitude too small (underflow), positive sign.
//   - kNegativeZero: magnitude too small (underflow), negative sign.
enum class FloatOverflow {
  kPositiveInfinity,
  kNegativeInfinity,
  kPositiveZero,
  kNegativeZero,
};

// Determines which of the four Java overflow outcomes `numStr` maps to when
// `fast_float::from_chars` reports `result_out_of_range`. Called only after
// preprocessing has stripped leading '+' and type-suffix, so `numStr` is a
// canonical decimal (possibly with exponent) beginning with '-' or a digit.
FloatOverflow classifyFloatOverflow(std::string_view numStr) {
  const bool negative = !numStr.empty() && numStr[0] == '-';

  // Explicit "e-" / "E-" exponent → underflow to zero.
  for (size_t k{0}; k < numStr.size(); ++k) {
    const char c = numStr[k];
    if (c == 'e' || c == 'E') {
      const bool negativeExponent =
          (k + 1 < numStr.size() && numStr[k + 1] == '-');
      if (negativeExponent) {
        return negative ? FloatOverflow::kNegativeZero
                        : FloatOverflow::kPositiveZero;
      }
      return negative ? FloatOverflow::kNegativeInfinity
                      : FloatOverflow::kPositiveInfinity;
    }
  }

  // No exponent notation: distinguish "0.000...0001" (underflow, |x| < 1)
  // from "999...999" (overflow, |x| >= 1) by looking at leading zeros and
  // the position of the decimal point.
  std::string_view absStr = numStr;
  if (negative) {
    absStr.remove_prefix(1);
  }
  size_t i{0};
  while (i < absStr.size() && absStr[i] == '0') {
    ++i;
  }
  const bool magnitudeBelowOne = !absStr.empty() &&
      (absStr[0] == '.' || (i > 0 && i < absStr.size() && absStr[i] == '.') ||
       i == absStr.size());
  if (magnitudeBelowOne) {
    return negative ? FloatOverflow::kNegativeZero
                    : FloatOverflow::kPositiveZero;
  }
  return negative ? FloatOverflow::kNegativeInfinity
                  : FloatOverflow::kPositiveInfinity;
}

template <typename T>
T floatOverflowValue(FloatOverflow overflow) {
  switch (overflow) {
    case FloatOverflow::kPositiveInfinity:
      return std::numeric_limits<T>::infinity();
    case FloatOverflow::kNegativeInfinity:
      return -std::numeric_limits<T>::infinity();
    case FloatOverflow::kPositiveZero:
      return T(0);
    case FloatOverflow::kNegativeZero:
      return T(-0.0);
  }
  VELOX_UNREACHABLE();
}

template <typename T>
bool parseFloat(std::string_view input, T& out) {
  if (input.empty()) {
    return false;
  }
  // Match Spark: exact "NaN" only (Java's parseDouble("NaN") returns NaN).
  if (input == "NaN") {
    out = std::numeric_limits<T>::quiet_NaN();
    return true;
  }
  // Match Spark CSV defaults: positiveInf="Inf", negativeInf="-Inf".
  // Also accept "Infinity"/"+Infinity"/"-Infinity" per Java's parseDouble.
  if (input == "Infinity" || input == "+Infinity" || input == "Inf") {
    out = std::numeric_limits<T>::infinity();
    return true;
  }
  if (input == "-Infinity" || input == "-Inf") {
    out = -std::numeric_limits<T>::infinity();
    return true;
  }
  // Reject hex float notation (0x/0X) which Java's parseDouble does not accept.
  if (hasHexPrefix(input)) {
    return false;
  }
  // Strip leading '+' before parsing (Spark/Java accepts "+1.23" but
  // fast_float rejects it). Guard against malformed "+-" or "++".
  auto numStr = stripLeadingPlus(input);
  if (numStr.empty()) {
    return false;
  }
  // Java's Float.parseFloat / Double.parseDouble accept a single trailing
  // type-suffix character (f/F/d/D), e.g. "1.0f" or "2D". Strip exactly one
  // such suffix before delegating to fast_float; a second suffix or other
  // trailing garbage will be caught as a parse error below.
  if (numStr.size() >= 2) {
    const char last = numStr.back();
    if (last == 'f' || last == 'F' || last == 'd' || last == 'D') {
      const char prev = numStr[numStr.size() - 2];
      // Only strip if the preceding character is a digit or '.', so we don't
      // truncate "Inf"/"NaN" tokens (already handled above) or accept inputs
      // like "abcf".
      if ((prev >= '0' && prev <= '9') || prev == '.') {
        numStr = numStr.substr(0, numStr.size() - 1);
      }
    }
  }
  // Use fast_float: locale-independent, allocation-free, cross-platform.
  T value;
  auto [ptr, ec] = fast_float::from_chars(
      numStr.data(), numStr.data() + numStr.size(), value);
  if (ec == std::errc::result_out_of_range &&
      ptr == numStr.data() + numStr.size() && !numStr.empty()) {
    // Numeric overflow or underflow — Java returns ±Infinity for overflow,
    // ±0 for underflow. Delegate the classification to a dedicated helper
    // so the hot fast_float path stays legible.
    out = floatOverflowValue<T>(classifyFloatOverflow(numStr));
    return true;
  }
  if (ec != std::errc{} || ptr != numStr.data() + numStr.size()) {
    return false;
  }
  // Reject case-insensitive nan/inf variants that from_chars may accept
  // (e.g., "nan", "inf", "INFINITY"). Only the exact forms handled above
  // are valid per Spark/Java semantics.
  if (std::isnan(value) || std::isinf(value)) {
    return false;
  }
  out = value;
  return true;
}

bool isSupportedLeafType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      return true;
    case TypeKind::HUGEINT:
      // Only long decimal (precision > 18) is supported; plain HUGEINT is not.
      return type->isLongDecimal();
    default:
      return type->isShortDecimal() || type->isDate();
  }
}

// Parses a CSV string into a ROW (struct) type. Fields are matched
// positionally to the schema columns.
//
// Note: We do not reuse velox/dwio/text/reader/TextReader because it is
// designed for file-level multi-line streaming I/O with headers and schema
// inference. from_csv operates on single-string inputs (one row = one CSV line)
// where the schema is known at plan time. A custom lightweight parser avoids
// that overhead.
//
// Supported field types:
//   BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, VARCHAR,
//   VARBINARY, DECIMAL (short and long), DATE, TIMESTAMP.
//
// Key Behavior (matches Spark's from_csv with default options):
// - NULL input returns NULL struct.
// - Empty or whitespace-only input returns a non-null struct with null fields.
// - Fewer CSV fields than schema columns: remaining columns are NULL.
// - More CSV fields than schema columns: extra fields are ignored.
// - Fields that cannot be parsed to the target type become NULL.
// - Whitespace is NOT trimmed from fields (Spark's from_csv defaults:
//   ignoreLeadingWhiteSpace=false, ignoreTrailingWhiteSpace=false).
// - Quoted fields are supported with backslash escape (Spark default).
// - TIMESTAMP fields use session timezone for naive timestamps.
class FromCsvFunction : public exec::VectorFunction {
 public:
  // Spark's `from_csv` SQL function (1-arg form) currently exposes no options
  // to the caller, so all Univocity-parser settings are pinned to Spark's
  // documented defaults: delimiter=',', quote='"', escape='\\' (backslash),
  // nullValue="", ignoreLeadingWhiteSpace=false,
  // ignoreTrailingWhiteSpace=false. Only `sessionTimeZone` varies (resolved
  // from QueryConfig at plan time and used for naive TIMESTAMP fields).
  FromCsvFunction(
      const TypePtr& outputType,
      const tz::TimeZone* sessionTimeZone)
      : outputType_(outputType), sessionTimeZone_(sessionTimeZone) {
    VELOX_CHECK_EQ(outputType->kind(), TypeKind::ROW);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto& input = args[0];
    context.ensureWritable(rows, outputType_, result);
    auto* flatResult = result->as<RowVector>();
    const auto& rowType = outputType_->asRow();
    const auto numFields = rowType.size();

    // Decode input to flat.
    exec::LocalDecodedVector decodedInput(context, *input, rows);

    auto* rawResultNulls = flatResult->mutableRawNulls();

    // Hoist per-column metadata outside row loop for performance.
    struct ColumnInfo {
      BaseVector* child;
      TypeKind kind;
      TypePtr type;
    };
    std::vector<ColumnInfo> columns(numFields);
    for (column_index_t col = 0; col < numFields; ++col) {
      columns[col] = {
          flatResult->childAt(col).get(),
          rowType.childAt(col)->kind(),
          rowType.childAt(col)};
    }

    // Reusable per-row buffers to avoid per-row allocation. See
    // `SplitCsvBuffers` for what each member holds and why they are grouped.
    SplitCsvBuffers buffers;

    rows.applyToSelected([&](auto row) {
      if (decodedInput->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto csvStr = decodedInput->valueAt<StringView>(row);
      auto csvView = std::string_view(csvStr.data(), csvStr.size());

      bits::clearNull(rawResultNulls, row);

      // Guard against extremely large inputs — return non-null row with
      // all-null fields (DoS mitigation, consistent with Spark permissive
      // mode).
      if (csvView.size() > kMaxCsvLineSize) {
        for (column_index_t col = 0; col < numFields; ++col) {
          columns[col].child->setNull(row, true);
        }
        return;
      }

      // Short-circuit for zero-column schema: no fields to parse.
      if (numFields == 0) {
        return;
      }

      splitCsvLine(csvView, kDelimiter, kEscape, numFields, buffers);

      for (column_index_t col = 0; col < numFields; ++col) {
        auto* childVector = columns[col].child;
        auto typeKind = columns[col].kind;
        const auto& colType = columns[col].type;

        if (static_cast<size_t>(col) < buffers.fields.size()) {
          const auto& fieldStr = buffers.fields[col];
          // With the 1-arg `from_csv` form Spark pins
          // ignoreLeadingWhiteSpace=false and ignoreTrailingWhiteSpace=false,
          // so no field-level trim runs here. Per-type trim (REAL/DOUBLE)
          // still happens inside setCsvFieldToChild. When the 3-arg
          // `from_csv(str, schema, map)` overload lands, reintroduce the
          // configurable trim branch that reads from per-instance members.
          std::string_view fieldView(fieldStr);
          // Check against nullValue sentinel (default: empty string → null).
          // Skip this check for quoted fields: Spark's `from_csv` `nullValue`
          // matches only unquoted-empty tokens; a quoted-empty `""` yields a
          // literal empty string via the `emptyValue` option (default "").
          if (!buffers.fieldWasQuoted[col] && fieldView == kNullValue) {
            childVector->setNull(row, true);
          } else {
            childVector->setNull(row, false);
            setCsvFieldToChild(childVector, row, fieldView, typeKind, colType);
          }
        } else {
          // Missing field — set null.
          childVector->setNull(row, true);
        }
      }
    });
  }

 private:
  // Sets a parsed CSV field value into the child vector at the given row.
  //
  // Whitespace handling matches Spark's from_csv defaults
  // (ignoreLeadingWhiteSpace=false, ignoreTrailingWhiteSpace=false):
  // - BOOLEAN: No trimming. Scala's toBoolean rejects " true ".
  // - Integer types: No trimming. Java's parseInt rejects " 123 ".
  // - REAL/DOUBLE: Trimmed. Java's parseDouble/parseFloat accept whitespace.
  // - DECIMAL: No trimming. Java's BigDecimal(String) rejects surrounding
  //   whitespace.
  // - DATE: No trimming. Parsed as yyyy-MM-dd (Spark's default dateFormat).
  // - TIMESTAMP: No trimming. Parsed as yyyy-MM-dd'T'HH:mm:ss (best-effort).
  // - VARCHAR/VARBINARY: No trimming. Whitespace is preserved as-is.
  void setCsvFieldToChild(
      BaseVector* child,
      vector_size_t row,
      std::string_view field,
      TypeKind typeKind,
      const TypePtr& type) const {
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        auto* flat = child->asFlatVector<bool>();
        // Spark's Scala toBoolean is case-insensitive but does NOT trim and
        // does NOT accept "0"/"1" — delegate to the shared text-reader helper
        // with the strict (allowOneZero=false) flag.
        auto parsed = ::facebook::velox::text::TextFieldParser::parseBoolean(
            field, /*allowOneZero=*/false);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::TINYINT: {
        auto* flat = child->asFlatVector<int8_t>();
        // No whitespace trimming — Java's parseInt rejects whitespace.
        int8_t value;
        if (parseInt<int8_t>(field, value)) {
          flat->set(row, value);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::SMALLINT: {
        auto* flat = child->asFlatVector<int16_t>();
        int16_t value;
        if (parseInt<int16_t>(field, value)) {
          flat->set(row, value);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::INTEGER: {
        // Could be either INTEGER or DATE (days since epoch).
        if (type->isDate()) {
          // Parse as yyyy-MM-dd (Spark's default dateFormat for from_csv).
          auto result = util::fromDateString(
              StringView(field.data(), field.size()),
              util::ParseMode::kSparkCast);
          if (result.hasValue()) {
            child->asFlatVector<int32_t>()->set(row, result.value());
          } else {
            child->setNull(row, true);
          }
        } else {
          auto* flat = child->asFlatVector<int32_t>();
          int32_t value;
          if (parseInt<int32_t>(field, value)) {
            flat->set(row, value);
          } else {
            child->setNull(row, true);
          }
        }
        break;
      }
      case TypeKind::BIGINT: {
        // Could be either BIGINT or SHORT DECIMAL (precision <= 18).
        if (type->isShortDecimal()) {
          auto [precision, scale] = getDecimalPrecisionScale(*type);
          StringView sv(field.data(), field.size());
          int64_t decimalValue{0};
          auto status =
              DecimalUtil::castFromString(sv, precision, scale, decimalValue);
          if (status.ok()) {
            child->asFlatVector<int64_t>()->set(row, decimalValue);
          } else {
            child->setNull(row, true);
          }
        } else {
          auto* flat = child->asFlatVector<int64_t>();
          int64_t value;
          if (parseInt<int64_t>(field, value)) {
            flat->set(row, value);
          } else {
            child->setNull(row, true);
          }
        }
        break;
      }
      case TypeKind::REAL: {
        auto* flat = child->asFlatVector<float>();
        // Trim whitespace: Java's Float.parseFloat accepts surrounding
        // whitespace. This is idempotent if the caller already trimmed via
        // ignoreLeadingWhiteSpace/ignoreTrailingWhiteSpace config.
        auto trimmed = trimWhitespace(field);
        float value;
        if (parseFloat<float>(trimmed, value)) {
          flat->set(row, value);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::DOUBLE: {
        auto* flat = child->asFlatVector<double>();
        // Trim whitespace: Java's Double.parseDouble accepts surrounding
        // whitespace.
        auto trimmed = trimWhitespace(field);
        double value;
        if (parseFloat<double>(trimmed, value)) {
          flat->set(row, value);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::VARCHAR: {
        auto* flat = child->asFlatVector<StringView>();
        flat->set(row, StringView(field.data(), field.size()));
        break;
      }
      case TypeKind::VARBINARY: {
        // Spark's from_csv treats BINARY fields as raw UTF-8 bytes.
        auto* flat = child->asFlatVector<StringView>();
        flat->set(row, StringView(field.data(), field.size()));
        break;
      }
      case TypeKind::HUGEINT: {
        if (type->isLongDecimal()) {
          // LONG DECIMAL (precision > 18).
          auto [precision, scale] = getDecimalPrecisionScale(*type);
          StringView sv(field.data(), field.size());
          int128_t decimalValue{0};
          auto status =
              DecimalUtil::castFromString(sv, precision, scale, decimalValue);
          if (status.ok()) {
            child->asFlatVector<int128_t>()->set(row, decimalValue);
          } else {
            child->setNull(row, true);
          }
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::TIMESTAMP: {
        // Parse timestamp string and apply session timezone for naive
        // timestamps (no timezone offset in the string). Spark's from_csv
        // uses session timezone via TimestampFormatter.
        auto result = util::fromTimestampWithTimezoneString(
            StringView(field.data(), field.size()),
            util::TimestampParseMode::kSparkCast);
        if (result.hasValue()) {
          auto ts = util::fromParsedTimestampWithTimeZone(
              result.value(), sessionTimeZone_);
          child->asFlatVector<Timestamp>()->set(row, ts);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      default:
        child->setNull(row, true);
        break;
    }
  }

  // ROW type defining the expected output schema.
  const TypePtr outputType_;

  // Timezone for parsing naive TIMESTAMP fields (no offset in the string).
  // Resolved from QueryConfig::sessionTimezone at plan time (see
  // constructSpecialForm) and captured for the lifetime of this Expr. Velox
  // creates a fresh Expr per query, so per-query TZ overrides take effect.
  const tz::TimeZone* sessionTimeZone_;

  // Pinned Univocity CSV options — Spark 1-arg `from_csv` exposes no way to
  // override these. If/when the 3-arg `from_csv(str, schema, map)` overload
  // is added, these become per-instance fields wired from the options map.
  //
  // Additional options currently pinned to Spark defaults but not visible here
  // because the code path they gate is inlined at its point of use (or absent
  // for the 1-arg form):
  //   - ignoreLeadingWhiteSpace = false, ignoreTrailingWhiteSpace = false
  //     (no per-field trim; per-type trim for REAL/DOUBLE only).
  //   - quote = '"' (hard-coded in splitCsvLine).
  //   - mode = PERMISSIVE (parse failures produce NULLs, never throw).
  static constexpr char kDelimiter{','};
  static constexpr char kEscape{'\\'};
  static constexpr std::string_view kNullValue{""};
};

} // namespace

TypePtr FromCsvCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL("from_csv function does not support type resolution.");
}

exec::ExprPtr FromCsvCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK_EQ(args.size(), 1, "from_csv expects one argument.");
  VELOX_USER_CHECK_EQ(
      args[0]->type()->kind(),
      TypeKind::VARCHAR,
      "The first argument of from_csv should be of varchar type.");
  VELOX_USER_CHECK_EQ(
      type->kind(), TypeKind::ROW, "from_csv output type must be ROW.");

  const auto& rowType = type->asRow();
  for (column_index_t i = 0; i < rowType.size(); ++i) {
    const auto& childType = rowType.childAt(i);
    VELOX_USER_CHECK(
        isSupportedLeafType(childType),
        "Unsupported field type for from_csv: column '{}' has type {}. "
        "Nested types (ARRAY/MAP/ROW) are not supported.",
        rowType.nameOf(i),
        childType->toString());
  }

  // Resolve session timezone for timestamp parsing. Spark's from_csv
  // interprets naive timestamps (no timezone in string) using the session
  // timezone.
  const tz::TimeZone* sessionTimeZone = nullptr;
  const auto sessionTzName = config.sessionTimezone();
  if (!sessionTzName.empty()) {
    sessionTimeZone = tz::locateZone(sessionTzName);
  }

  auto func = std::make_shared<FromCsvFunction>(type, sessionTimeZone);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      func,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      kFromCsv,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql

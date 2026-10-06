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

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fast_float/fast_float.h>
#include <folly/Conv.h>

#include "velox/common/text/TextFieldParser.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/TimestampUtils.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Maximum CSV input line size (10 MB) to prevent DoS from unbounded allocation.
constexpr size_t kMaxCsvLineSize = 10 * 1024 * 1024;

// Trims leading and trailing code units from U+0000 through U+0020, matching
// the Java String.trim semantics used by Float.parseFloat and
// Double.parseDouble.
// Integer, boolean, decimal, VARCHAR, and VARBINARY fields are NOT trimmed.
std::string_view trimJavaWhitespace(std::string_view input) {
  size_t start{0};
  while (start < input.size() &&
         static_cast<unsigned char>(input[start]) <= 0x20) {
    ++start;
  }
  size_t end{input.size()};
  while (end > start && static_cast<unsigned char>(input[end - 1]) <= 0x20) {
    --end;
  }
  return input.substr(start, end - start);
}

// Reusable per-row buffers for `splitCsvLine`. Grouping keeps the parser
// signature small and makes it clear which state is meant to be recycled
// across rows to avoid per-row allocation.
//
// - `fields`: parsed field values as views. Unquoted fields point directly
//   into the input line. Every field that starts with a quote (closed,
//   unclosed, or malformed fallback) points into `quotedArena`.
// - `quotedArena`: owns processed bytes for quoted fields. On the first quoted
//   field, it reserves the line length. Each field adds no more bytes than its
//   raw input span; the malformed fallback is exactly tight because it
//   re-emits the opening quote that was not copied earlier. This prevents
//   mid-row reallocation without allocating for rows containing only unquoted
//   fields and keeps all views stable until the next `splitCsvLine` call.
struct SplitCsvBuffers {
  std::vector<std::string_view> fields;
  std::string quotedArena;
};

// Splits a single CSV line into fields, handling quoted fields.
// Follows Spark's CSV parsing rules (Univocity defaults):
// - Fields may be enclosed in double quotes.
// - Inside a quoted field, Spark's default backslash escape decodes \" and
//   \\. In a run of N quotes, N-1 quotes are emitted and the last quote is
//   considered as a possible closing quote. Pass '\0' as `escape` to disable.
// - Unquoted fields are taken as-is.
// - unescapedQuoteHandling=STOP_AT_DELIMITER: after the closing quote,
//   only ASCII whitespace may appear before the next delimiter. Any other
//   trailing characters produce the opening quote, already-decoded quoted
//   content, and the raw remainder up to the next delimiter.
// maxFields: stop parsing once this many fields are produced (0 = unlimited).
// buffers: reusable per-row state. `fields` and `quotedArena` are cleared at
//   function entry and reserved lazily when the first quoted field is found.
void splitCsvLine(
    std::string_view line,
    char delimiter,
    char escape,
    size_t maxFields,
    SplitCsvBuffers& buffers) {
  auto& fields = buffers.fields;
  auto& quotedArena = buffers.quotedArena;
  fields.clear();
  quotedArena.clear();
  VELOX_DCHECK_LE(
      line.size(),
      kMaxCsvLineSize,
      "splitCsvLine called with oversized input — caller should guard.");
  if (maxFields > 0) {
    fields.reserve(maxFields);
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
      // Each quoted field contributes at most its raw span, so one line-sized
      // reservation keeps every arena-backed view stable for the row.
      if (quotedArena.capacity() < line.size()) {
        quotedArena.reserve(line.size());
      }
      // Quoted field using Spark's fixed Univocity defaults: quote='"',
      // escape='\\', escapeEscape='\\', and STOP_AT_DELIMITER.
      const size_t quotedStart = quotedArena.size();
      size_t j = i + 1;
      bool closedProperly{false};
      bool literalFallback{false};
      while (j < line.size()) {
        if (escape != '\0' && line[j] == escape) {
          if (j + 1 < line.size() &&
              (line[j + 1] == escape || line[j + 1] == '"')) {
            quotedArena.push_back(line[j + 1]);
            j += 2;
          } else if (j + 1 == line.size()) {
            ++j;
          } else {
            quotedArena.push_back(line[j]);
            ++j;
          }
        } else if (line[j] != '"') {
          quotedArena.push_back(line[j]);
          ++j;
        } else {
          // With Spark's default escape='\\', a doubled quote emits one quote,
          // while the second quote remains the candidate closing quote.
          while (j + 1 < line.size() && line[j + 1] == '"') {
            quotedArena.push_back('"');
            ++j;
          }

          const size_t afterQuote = j + 1;
          size_t k = afterQuote;
          while (k < line.size() &&
                 static_cast<unsigned char>(line[k]) <= ' ' &&
                 line[k] != delimiter) {
            ++k;
          }
          if (k == line.size() || line[k] == delimiter) {
            closedProperly = true;
            j = k;
            break;
          }

          // An unescaped quote followed by non-whitespace is retained
          // literally, together with the opening quote and all content through
          // the next delimiter.
          size_t end = k;
          while (end < line.size() && line[end] != delimiter) {
            ++end;
          }
          quotedArena.push_back('"');
          quotedArena.append(line.data() + afterQuote, end - afterQuote);
          quotedArena.push_back('"');
          std::rotate(
              quotedArena.begin() + quotedStart,
              quotedArena.end() - 1,
              quotedArena.end());
          fields.push_back(std::string_view(quotedArena).substr(quotedStart));
          i = end;
          if (i < line.size()) {
            ++i;
            trailingDelimiter = true;
          }
          literalFallback = true;
          break;
        }
      }
      if (literalFallback) {
        continue;
      }
      if (!closedProperly) {
        // At EOF, Univocity returns the processed field content. A bare
        // opening quote is retained as a literal quote.
        if (i + 1 == line.size()) {
          quotedArena.push_back('"');
        }
        fields.push_back(std::string_view(quotedArena).substr(quotedStart));
        i = line.size();
      } else {
        fields.push_back(std::string_view(quotedArena).substr(quotedStart));
        if (j < line.size()) {
          ++j; // Skip delimiter.
          trailingDelimiter = true;
        }
        i = j;
      }
    } else {
      // Unquoted field.
      size_t start = i;
      while (i < line.size() && line[i] != delimiter) {
        ++i;
      }
      fields.push_back(line.substr(start, i - start));
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
    fields.push_back(std::string_view{});
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

// Checks if a numeric string starts with a hex prefix (0x/0X), optionally
// after a sign character.
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
  auto numericString = stripLeadingPlus(input);
  if (numericString.empty()) {
    return false;
  }
  auto parsed = ::facebook::velox::text::TextFieldParser::parseNarrowInteger<T>(
      numericString, /*allowTrailingDecimal=*/false);
  if (!parsed.has_value()) {
    return false;
  }
  out = *parsed;
  return true;
}

// Parses the supported Spark/Java floating-point formats.
// Accepts: decimal notation, "NaN"/"+NaN"/"-NaN",
//          "Infinity"/"-Infinity"/"+Infinity", "Inf"/"-Inf" (Spark CSV
//          defaults: positiveInf="Inf", negativeInf="-Inf").
// Rejects: hex floats (0x...), case-insensitive nan/inf variants.
// Overflow returns ±Infinity (matching Java's parseDouble("1e400") = Infinity).
//
// TODO: Extract a shared float parser into TextFieldParser.h once the
// TextReader path (sscanf %f with lenient ERANGE handling and case-insensitive
// nan/inf via boost::iequals) is reconciled with Spark's case-sensitive,
// Java-style overflow→±Inf / underflow→±0 semantics.

template <typename T>
bool parseFloat(std::string_view input, T& out) {
  if (input.empty()) {
    return false;
  }

  // Match Spark CSV's short infinity sentinels against the untrimmed token.
  if (input == "Inf") {
    out = std::numeric_limits<T>::infinity();
    return true;
  }
  if (input == "-Inf") {
    out = -std::numeric_limits<T>::infinity();
    return true;
  }

  const auto trimmed = trimJavaWhitespace(input);
  if (trimmed.empty()) {
    return false;
  }
  // Java's floating-point parser accepts an optional sign on NaN.
  if (trimmed == "NaN" || trimmed == "+NaN" || trimmed == "-NaN") {
    out = std::numeric_limits<T>::quiet_NaN();
    return true;
  }
  if (trimmed == "Infinity" || trimmed == "+Infinity") {
    out = std::numeric_limits<T>::infinity();
    return true;
  }
  if (trimmed == "-Infinity") {
    out = -std::numeric_limits<T>::infinity();
    return true;
  }
  // Java's parser does not accept the short Spark CSV sentinels.
  if (trimmed == "Inf" || trimmed == "-Inf") {
    return false;
  }
  // The decimal fast_float path does not support hexadecimal literals.
  if (hasHexPrefix(trimmed)) {
    return false;
  }
  // Strip leading '+' before parsing (Spark/Java accepts "+1.23" but
  // fast_float rejects it). Guard against malformed "+-" or "++".
  auto numericString = stripLeadingPlus(trimmed);
  if (numericString.empty()) {
    return false;
  }
  // Java's Float.parseFloat / Double.parseDouble accept a single trailing
  // type-suffix character (f/F/d/D), e.g. "1.0f" or "2D". Strip exactly one
  // such suffix before delegating to fast_float; a second suffix or other
  // trailing garbage will be caught as a parse error below.
  if (numericString.size() >= 2) {
    const char last = numericString.back();
    if (last == 'f' || last == 'F' || last == 'd' || last == 'D') {
      const char previous = numericString[numericString.size() - 2];
      // Only strip if the preceding character is a digit or '.', so we don't
      // truncate "Inf"/"NaN" tokens (already handled above) or accept inputs
      // like "abcf".
      if ((previous >= '0' && previous <= '9') || previous == '.') {
        numericString = numericString.substr(0, numericString.size() - 1);
      }
    }
  }
  // Use fast_float: locale-independent, allocation-free, cross-platform.
  T value;
  auto [parseEnd, parseError] = fast_float::from_chars(
      numericString.data(), numericString.data() + numericString.size(), value);
  if (parseError == std::errc::result_out_of_range &&
      parseEnd == numericString.data() + numericString.size() &&
      !numericString.empty()) {
    // Overflow or underflow. fast_float already saturates `value` to
    // ±infinity (overflow) or ±0 (underflow) with the correct sign before
    // reporting result_out_of_range, matching Java's Float.parseFloat /
    // Double.parseDouble semantics, so the saturated value is returned as-is.
    out = value;
    return true;
  }
  if (parseError != std::errc{} ||
      parseEnd != numericString.data() + numericString.size()) {
    return false;
  }
  // Reject case-insensitive nan/inf variants that from_chars may accept.
  // Only the exact forms handled above are valid per Spark/Java semantics.
  if (std::isnan(value) || std::isinf(value)) {
    return false;
  }
  out = value;
  return true;
}

bool normalizeScientificDecimal(
    std::string_view input,
    uint8_t precision,
    uint8_t targetScale,
    std::string& normalized,
    bool& isZero) {
  const auto exponentPos = input.find_first_of("eE");
  if (exponentPos == std::string_view::npos) {
    return true;
  }

  auto exponentText = input.substr(exponentPos + 1);
  bool negativeExponent{false};
  if (!exponentText.empty() &&
      (exponentText.front() == '+' || exponentText.front() == '-')) {
    negativeExponent = exponentText.front() == '-';
    exponentText.remove_prefix(1);
  }
  if (exponentText.empty()) {
    return false;
  }
  uint64_t magnitude{0};
  const auto [end, error] = std::from_chars(
      exponentText.data(),
      exponentText.data() + exponentText.size(),
      magnitude);
  if (error != std::errc{} ||
      end != exponentText.data() + exponentText.size() ||
      magnitude > 9'999'999'999ULL) {
    return false;
  }

  auto mantissa = input.substr(0, exponentPos);
  bool negative{false};
  if (!mantissa.empty() &&
      (mantissa.front() == '+' || mantissa.front() == '-')) {
    negative = mantissa.front() == '-';
    mantissa.remove_prefix(1);
  }
  std::string digits;
  digits.reserve(mantissa.size());
  bool sawDecimalPoint{false};
  size_t fractionDigits{0};
  for (const char c : mantissa) {
    if (c == '.') {
      if (sawDecimalPoint) {
        return false;
      }
      sawDecimalPoint = true;
    } else if (c >= '0' && c <= '9') {
      digits.push_back(c);
      if (sawDecimalPoint) {
        ++fractionDigits;
      }
    } else {
      return false;
    }
  }
  if (digits.empty()) {
    return false;
  }

  const int64_t exponent = negativeExponent ? -static_cast<int64_t>(magnitude)
                                            : static_cast<int64_t>(magnitude);
  const int64_t decimalScale = static_cast<int64_t>(fractionDigits) - exponent;
  if (decimalScale < std::numeric_limits<int32_t>::min() ||
      decimalScale > std::numeric_limits<int32_t>::max()) {
    return false;
  }

  const auto firstNonZero = digits.find_first_not_of('0');
  if (firstNonZero == std::string::npos) {
    isZero = true;
    return true;
  }
  digits.erase(0, firstNonZero);

  if (decimalScale > (int64_t{1} << 29)) {
    return false;
  }

  normalized.clear();
  if (negative) {
    normalized.push_back('-');
  }
  if (decimalScale <= 0) {
    const auto trailingZeros = static_cast<uint64_t>(-decimalScale);
    if (digits.size() + trailingZeros > precision) {
      return false;
    }
    normalized.append(digits);
    normalized.append(trailingZeros, '0');
  } else if (static_cast<uint64_t>(decimalScale) >= digits.size()) {
    const auto leadingZeros =
        static_cast<uint64_t>(decimalScale) - digits.size();
    if (leadingZeros > targetScale) {
      isZero = true;
      return true;
    }
    normalized.append("0.");
    normalized.append(leadingZeros, '0');
    normalized.append(digits);
  } else {
    const auto integerDigits = digits.size() - decimalScale;
    if (integerDigits > precision) {
      return false;
    }
    normalized.append(digits.data(), integerDigits);
    normalized.push_back('.');
    normalized.append(digits.data() + integerDigits, decimalScale);
  }
  return true;
}

template <typename T>
bool parseDecimal(
    std::string_view field,
    uint8_t precision,
    uint8_t scale,
    T& value) {
  std::string normalized;
  if (field.find(',') != std::string_view::npos) {
    normalized.reserve(field.size());
    for (const char c : field) {
      if (c != ',') {
        normalized.push_back(c);
      }
    }
    field = normalized;
  }
  std::string scientific;
  bool isZero{false};
  if (!normalizeScientificDecimal(
          field, precision, scale, scientific, isZero)) {
    return false;
  }
  if (isZero) {
    value = 0;
    return true;
  }
  if (!scientific.empty()) {
    field = scientific;
  }

  try {
    return DecimalUtil::castFromString(
               StringView(field.data(), field.size()), precision, scale, value)
        .ok();
  } catch (const folly::ConversionError&) {
    return false;
  } catch (const VeloxException&) {
    return false;
  }
}

Timestamp toSparkTimestamp(
    util::ParsedTimestampWithTimeZone parsed,
    const tz::TimeZone* sessionTimeZone) {
  if (parsed.timeZone != nullptr) {
    toGMTWithGapCorrection(parsed.timestamp, *parsed.timeZone);
    return parsed.timestamp;
  }
  if (!parsed.offsetMillis.has_value() && sessionTimeZone != nullptr) {
    toGMTWithGapCorrection(parsed.timestamp, *sessionTimeZone);
    return parsed.timestamp;
  }
  return util::fromParsedTimestampWithTimeZone(parsed, nullptr);
}

std::string_view trimSparkDateTimeWhitespace(std::string_view input) {
  const auto isTrimmedByte = [](char c) {
    const auto byte = static_cast<unsigned char>(c);
    return byte <= 0x20 || byte == 0x7f;
  };
  while (!input.empty() && isTrimmedByte(input.front())) {
    input.remove_prefix(1);
  }
  while (!input.empty() && isTrimmedByte(input.back())) {
    input.remove_suffix(1);
  }
  return input;
}

bool isSparkTimestampRange(const Timestamp& timestamp) {
  const int128_t micros =
      int128_t(timestamp.getSeconds()) * util::kMicrosPerSec +
      timestamp.getNanos() / util::kNanosPerMicro;
  return micros >= std::numeric_limits<int64_t>::min() &&
      micros <= std::numeric_limits<int64_t>::max();
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
// Supported field types:
//   BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, VARCHAR,
//   VARBINARY, DECIMAL (short and long), DATE, TIMESTAMP.
//
// Key behavior (matches the supported Spark from_csv default options):
// - NULL input returns NULL struct.
// - Empty input returns a non-null struct with null fields.
// - Whitespace-only fields are parsed according to the target type.
// - Fewer CSV fields than schema columns: remaining columns are NULL.
// - More CSV fields than schema columns: extra fields are ignored.
// - Fields that cannot be parsed to the target type become NULL.
// - Whitespace is NOT trimmed from fields (Spark's from_csv defaults:
//   ignoreLeadingWhiteSpace=false, ignoreTrailingWhiteSpace=false).
// - Quoted fields are supported with backslash escape (Spark default).
// - TIMESTAMP fields use session timezone for naive timestamps.
class FromCsvFunction : public exec::VectorFunction {
 public:
  // Spark's supported two-argument SQL overload resolves the schema into the
  // result type, leaving one expression child for the CSV value. It exposes no
  // options map, so all Univocity-parser settings are pinned to Spark's
  // documented defaults: delimiter=',', quote='"', escape='\\' (backslash),
  // nullValue="", ignoreLeadingWhiteSpace=false, and
  // ignoreTrailingWhiteSpace=false. Only `sessionTimeZone` varies.
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
    for (column_index_t columnIndex = 0; columnIndex < numFields;
         ++columnIndex) {
      columns[columnIndex] = {
          flatResult->childAt(columnIndex).get(),
          rowType.childAt(columnIndex)->kind(),
          rowType.childAt(columnIndex)};
    }

    // Reusable per-row buffers to avoid per-row allocation. See
    // `SplitCsvBuffers` for what each member holds and why they are grouped.
    SplitCsvBuffers buffers;

    rows.applyToSelected([&](auto row) {
      if (decodedInput->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto csvString = decodedInput->valueAt<StringView>(row);
      const auto csvStringView =
          std::string_view(csvString.data(), csvString.size());

      bits::clearNull(rawResultNulls, row);

      // Guard against extremely large inputs with a permissive-shaped result:
      // a non-null row whose fields are null.
      if (csvStringView.size() > kMaxCsvLineSize) {
        for (column_index_t columnIndex = 0; columnIndex < numFields;
             ++columnIndex) {
          columns[columnIndex].child->setNull(row, true);
        }
        return;
      }

      // Short-circuit for zero-column schema: no fields to parse.
      if (numFields == 0) {
        return;
      }

      splitCsvLine(csvStringView, kDelimiter, kEscape, numFields, buffers);

      for (column_index_t columnIndex = 0; columnIndex < numFields;
           ++columnIndex) {
        auto* childVector = columns[columnIndex].child;
        const auto typeKind = columns[columnIndex].kind;
        const auto& columnType = columns[columnIndex].type;

        if (static_cast<size_t>(columnIndex) < buffers.fields.size()) {
          const auto field = buffers.fields[columnIndex];
          // The supported overload pins parser-level whitespace trimming off.
          // Type-specific conversion rules are applied below.
          // Check against nullValue sentinel (default: empty string → null).
          if (field == kNullValue) {
            childVector->setNull(row, true);
          } else {
            childVector->setNull(row, false);
            setCsvFieldToChild(childVector, row, field, typeKind, columnType);
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
  // - DATE/TIMESTAMP: Spark trimAll bytes are removed before parsing.
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
          try {
            const auto cleaned = trimSparkDateTimeWhitespace(field);
            auto result = util::fromDateString(
                StringView(cleaned.data(), cleaned.size()),
                util::ParseMode::kSparkCast);
            if (result.hasValue()) {
              child->asFlatVector<int32_t>()->set(row, result.value());
            } else {
              child->setNull(row, true);
            }
          } catch (const VeloxException&) {
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
          int64_t decimalValue{0};
          if (parseDecimal(field, precision, scale, decimalValue)) {
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
        float value;
        if (parseFloat<float>(field, value)) {
          flat->set(row, value);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::DOUBLE: {
        auto* flat = child->asFlatVector<double>();
        double value;
        if (parseFloat<double>(field, value)) {
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
          int128_t decimalValue{0};
          if (parseDecimal(field, precision, scale, decimalValue)) {
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
        try {
          const auto cleaned = trimSparkDateTimeWhitespace(field);
          auto result = util::fromTimestampWithTimezoneString(
              StringView(cleaned.data(), cleaned.size()),
              util::TimestampParseMode::kSparkCast);
          if (result.hasValue()) {
            const auto timestamp =
                toSparkTimestamp(result.value(), sessionTimeZone_)
                    .toPrecision(TimestampPrecision::kMicroseconds);
            if (isSparkTimestampRange(timestamp)) {
              child->asFlatVector<Timestamp>()->set(row, timestamp);
            } else {
              child->setNull(row, true);
            }
          } else {
            child->setNull(row, true);
          }
        } catch (const VeloxException&) {
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

  // Pinned Univocity CSV options for the supported two-argument SQL overload.
  // Additional defaults are enforced at their points of use:
  //   - ignoreLeadingWhiteSpace = false, ignoreTrailingWhiteSpace = false
  //     (type-specific conversion still trims REAL/DOUBLE and DATE/TIMESTAMP).
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
  VELOX_USER_CHECK_EQ(
      args.size(),
      1,
      "from_csv expects one value argument after schema resolution.");
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
  const auto sessionTimezoneName = config.sessionTimezone();
  if (!sessionTimezoneName.empty()) {
    sessionTimeZone = tz::locateZone(sessionTimezoneName);
  }

  auto function = std::make_shared<FromCsvFunction>(type, sessionTimeZone);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      function,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      kFromCsv,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql

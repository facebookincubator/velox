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

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorFunction.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {
namespace {

// Maximum CSV input line size (10 MB) to prevent DoS from unbounded allocation.
constexpr size_t kMaxCsvLineSize = 10 * 1'024 * 1'024;

// Trims leading and trailing ASCII whitespace from a string_view.
// Used for numeric/boolean fields before parsing (Spark's Double.parseDouble
// accepts surrounding whitespace; for integer types we trim to be lenient in
// PERMISSIVE mode). VARCHAR fields are NOT trimmed.
std::string_view trimWhitespace(std::string_view s) {
  size_t start = 0;
  while (start < s.size() &&
         (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' ||
          s[start] == '\n')) {
    ++start;
  }
  size_t end = s.size();
  while (end > start &&
         (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r' ||
          s[end - 1] == '\n')) {
    --end;
  }
  return s.substr(start, end - start);
}

// Splits a single CSV line into fields, handling quoted fields.
// Follows RFC 4180 quoting rules:
// - Fields may be enclosed in double quotes.
// - A double quote inside a quoted field is escaped by another double quote
//   (unescaped to a single double quote in the output).
// - Unquoted fields are taken as-is (no escape character — matches Spark's
//   from_csv default where escape is disabled).
// - Characters between a closing quote and the next delimiter are skipped
//   (permissive handling of malformed CSV, matching Spark's default mode).
// maxFields: stop parsing once this many fields are produced (0 = unlimited).
// Populates the provided `fields` vector (cleared first) to allow reuse.
void splitCsvLine(
    std::string_view line,
    char delimiter,
    size_t maxFields,
    std::vector<std::string>& fields) {
  fields.clear();
  if (maxFields > 0) {
    fields.reserve(maxFields);
  }
  // Guard against extremely large inputs (DoS mitigation).
  if (line.size() > kMaxCsvLineSize) {
    return;
  }
  size_t i = 0;
  bool trailingDelimiter = false;
  while (i < line.size()) {
    // Stop parsing once we have enough fields.
    if (maxFields > 0 && fields.size() >= maxFields) {
      break;
    }
    trailingDelimiter = false;
    if (line[i] == '"') {
      // Quoted field: build content, unescaping doubled quotes.
      // Spark's default unescapedQuoteHandling=STOP_AT_DELIMITER: if the
      // quote is not properly closed before EOL, treat it as literal text
      // and re-parse from the opening quote as an unquoted field.
      std::string content;
      // Reserve an estimate for quoted field content to reduce reallocations.
      size_t remaining = line.size() - i - 1;
      content.reserve(std::min(remaining, size_t{128}));
      size_t j = i + 1;
      bool closedProperly = false;
      while (j < line.size()) {
        if (line[j] == '"') {
          if (j + 1 < line.size() && line[j + 1] == '"') {
            // Escaped double quote — emit a single quote.
            content.push_back('"');
            j += 2;
          } else {
            // End of quoted field.
            closedProperly = true;
            break;
          }
        } else {
          content.push_back(line[j]);
          ++j;
        }
      }
      if (!closedProperly) {
        // Unclosed quote: treat opening quote as literal, re-parse as
        // unquoted field (matches Spark STOP_AT_DELIMITER behavior).
        size_t start = i;
        i = start;
        while (i < line.size() && line[i] != delimiter) {
          ++i;
        }
        fields.emplace_back(line.substr(start, i - start));
        if (i < line.size()) {
          ++i; // skip delimiter
          trailingDelimiter = true;
        }
      } else {
        fields.push_back(std::move(content));
        // Advance past closing quote.
        ++j;
        // Skip any characters between closing quote and next delimiter
        // (permissive mode — absorb trailing garbage).
        while (j < line.size() && line[j] != delimiter) {
          ++j;
        }
        // Advance past delimiter.
        if (j < line.size() && line[j] == delimiter) {
          ++j;
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
      fields.emplace_back(line.substr(start, i - start));
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
  }
}

// Strips a leading '+' sign from a numeric string view (Spark accepts +123).
// Returns empty view for malformed inputs like "+-123" or "++123".
std::string_view stripLeadingPlus(std::string_view s) {
  if (!s.empty() && s[0] == '+') {
    auto rest = s.substr(1);
    // Reject "+-N", "++N" — Java parseInt("+-123") throws.
    if (!rest.empty() && (rest[0] == '-' || rest[0] == '+')) {
      return {};
    }
    return rest;
  }
  return s;
}

// Checks if a numeric string starts with hex prefix (0x/0X), optionally after
// a sign character. Java's Integer.parseInt and Double.parseDouble reject hex.
bool hasHexPrefix(std::string_view s) {
  if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    return true;
  }
  if (s.size() >= 3 && (s[0] == '+' || s[0] == '-') && s[1] == '0' &&
      (s[2] == 'x' || s[2] == 'X')) {
    return true;
  }
  return false;
}

// Checks if a string contains a decimal point. Java's Integer.parseInt rejects
// strings with decimal points (e.g., "123.0" → NumberFormatException).
bool hasDecimalPoint(std::string_view s) {
  return s.find('.') != std::string_view::npos;
}

// Parses an integer field matching Spark/Java semantics.
// Rejects: hex prefix (0x), decimal points, leading/trailing non-whitespace
// garbage. Accepts: optional leading '+', optional sign '-'.
template <typename T>
bool parseInt(std::string_view s, T& out) {
  if (s.empty()) {
    return false;
  }
  // Reject hex literals (Java parseInt/parseLong rejects "0x1F").
  if (hasHexPrefix(s)) {
    return false;
  }
  // Reject decimal notation (Java parseInt("123.0") throws).
  if (hasDecimalPoint(s)) {
    return false;
  }
  auto numStr = stripLeadingPlus(s);
  if (numStr.empty()) {
    return false;
  }
  T val;
  auto [ptr, ec] =
      std::from_chars(numStr.data(), numStr.data() + numStr.size(), val);
  if (ec == std::errc{} && ptr == numStr.data() + numStr.size()) {
    out = val;
    return true;
  }
  return false;
}

// Parses a floating-point string matching Spark/Java semantics.
// Accepts: decimal notation, "NaN" (exact), "Infinity"/"-Infinity"/"+Infinity",
//          "Inf"/"-Inf" (Spark CSV defaults: positiveInf="Inf",
//          negativeInf="-Inf").
// Rejects: hex floats (0x...), case-insensitive nan/inf variants.
// Overflow returns ±Infinity (matching Java's parseDouble("1e400") = Infinity).
template <typename T>
bool parseFloat(std::string_view s, T& out) {
  if (s.empty()) {
    return false;
  }
  // Match Spark: exact "NaN" only (Java's parseDouble("NaN") returns NaN).
  if (s == "NaN") {
    out = std::numeric_limits<T>::quiet_NaN();
    return true;
  }
  // Match Spark CSV defaults: positiveInf="Inf", negativeInf="-Inf".
  // Also accept "Infinity"/"+Infinity"/"-Infinity" per Java's parseDouble.
  if (s == "Infinity" || s == "+Infinity" || s == "Inf") {
    out = std::numeric_limits<T>::infinity();
    return true;
  }
  if (s == "-Infinity" || s == "-Inf") {
    out = -std::numeric_limits<T>::infinity();
    return true;
  }
  // Reject hex float notation (0x/0X) which Java's parseDouble does not accept.
  if (hasHexPrefix(s)) {
    return false;
  }
  // Strip leading '+' before from_chars (Spark/Java accepts "+1.23" but
  // std::from_chars rejects it). Guard against malformed "+-" or "++".
  auto numStr = stripLeadingPlus(s);
  // Use std::from_chars: locale-independent and allocation-free.
  T val;
  auto [ptr, ec] = std::from_chars(
      numStr.data(),
      numStr.data() + numStr.size(),
      val,
      std::chars_format::general);
  if (ec == std::errc::result_out_of_range &&
      ptr == numStr.data() + numStr.size() && !numStr.empty()) {
    // Numeric overflow/underflow. Java returns ±Infinity for overflow, 0 for
    // underflow. Determine direction from the exponent sign.
    bool hasNegativeExponent = false;
    for (size_t k = 0; k < numStr.size(); ++k) {
      if (numStr[k] == 'e' || numStr[k] == 'E') {
        hasNegativeExponent = (k + 1 < numStr.size() && numStr[k + 1] == '-');
        break;
      }
    }
    if (hasNegativeExponent) {
      out = T(0);
    } else {
      out = (numStr[0] == '-') ? -std::numeric_limits<T>::infinity()
                               : std::numeric_limits<T>::infinity();
    }
    return true;
  }
  if (ec != std::errc{} || ptr != numStr.data() + numStr.size()) {
    return false;
  }
  // Reject case-insensitive nan/inf variants that from_chars may accept
  // (e.g., "nan", "inf", "INFINITY"). Only the exact forms handled above
  // are valid per Spark/Java semantics.
  if (std::isnan(val) || std::isinf(val)) {
    return false;
  }
  out = val;
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
      return true;
    default:
      return false;
  }
}

// Parses a CSV string into a ROW (struct) type. Fields are matched
// positionally to the schema columns.
//
// Key Behavior (matches Spark's from_csv with default options):
// - NULL input returns NULL.
// - Fewer CSV fields than schema columns: remaining columns are NULL.
// - More CSV fields than schema columns: extra fields are ignored.
// - Fields that cannot be parsed to the target type become NULL.
// - Whitespace is NOT trimmed from fields (Spark's from_csv defaults:
//   ignoreLeadingWhiteSpace=false, ignoreTrailingWhiteSpace=false).
// - Quoted fields (RFC 4180) are supported.
// - No escape character (Spark's from_csv default: escape='\u0000').
class FromCsvFunction : public exec::VectorFunction {
 public:
  explicit FromCsvFunction(const TypePtr& outputType, char delimiter = ',')
      : outputType_(outputType), delimiter_(delimiter) {
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
    };
    std::vector<ColumnInfo> columns(numFields);
    for (column_index_t col = 0; col < numFields; ++col) {
      columns[col] = {
          flatResult->childAt(col).get(), rowType.childAt(col)->kind()};
    }

    // Reuse fields vector across rows to avoid per-row allocation.
    std::vector<std::string> fields;

    rows.applyToSelected([&](auto row) {
      if (decodedInput->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto csvStr = decodedInput->valueAt<StringView>(row);
      // Spark's from_csv returns NULL struct for empty input (UnivocityParser
      // returns null for empty/whitespace-only strings).
      if (csvStr.size() == 0) {
        flatResult->setNull(row, true);
        return;
      }

      bits::clearNull(rawResultNulls, row);

      splitCsvLine(
          std::string_view(csvStr.data(), csvStr.size()),
          delimiter_,
          numFields,
          fields);

      for (column_index_t col = 0; col < numFields; ++col) {
        auto* childVector = columns[col].child;
        auto typeKind = columns[col].kind;

        if (static_cast<size_t>(col) < fields.size()) {
          const auto& fieldStr = fields[col];
          // Spark default nullValue="": empty field → null for all types.
          // No whitespace trimming — matches Spark's from_csv defaults
          // (ignoreLeadingWhiteSpace=false, ignoreTrailingWhiteSpace=false).
          if (fieldStr.empty()) {
            childVector->setNull(row, true);
          } else {
            childVector->setNull(row, false);
            std::string_view fieldView(fieldStr);
            setCsvFieldToChild(childVector, row, fieldView, typeKind);
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
  // - VARCHAR: No trimming. Whitespace is preserved as-is.
  void setCsvFieldToChild(
      BaseVector* child,
      vector_size_t row,
      std::string_view field,
      TypeKind typeKind) const {
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        auto* flat = child->asFlatVector<bool>();
        // Spark's Scala toBoolean is case-insensitive but does NOT trim.
        if (field.size() == 4 && (field[0] == 't' || field[0] == 'T') &&
            (field[1] == 'r' || field[1] == 'R') &&
            (field[2] == 'u' || field[2] == 'U') &&
            (field[3] == 'e' || field[3] == 'E')) {
          flat->set(row, true);
        } else if (
            field.size() == 5 && (field[0] == 'f' || field[0] == 'F') &&
            (field[1] == 'a' || field[1] == 'A') &&
            (field[2] == 'l' || field[2] == 'L') &&
            (field[3] == 's' || field[3] == 'S') &&
            (field[4] == 'e' || field[4] == 'E')) {
          flat->set(row, false);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::TINYINT: {
        auto* flat = child->asFlatVector<int8_t>();
        // No whitespace trimming — Java's parseInt rejects whitespace.
        int8_t val;
        if (parseInt<int8_t>(field, val)) {
          flat->set(row, val);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::SMALLINT: {
        auto* flat = child->asFlatVector<int16_t>();
        int16_t val;
        if (parseInt<int16_t>(field, val)) {
          flat->set(row, val);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::INTEGER: {
        auto* flat = child->asFlatVector<int32_t>();
        int32_t val;
        if (parseInt<int32_t>(field, val)) {
          flat->set(row, val);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::BIGINT: {
        auto* flat = child->asFlatVector<int64_t>();
        int64_t val;
        if (parseInt<int64_t>(field, val)) {
          flat->set(row, val);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::REAL: {
        auto* flat = child->asFlatVector<float>();
        // Trim whitespace: Java's Float.parseFloat accepts surrounding
        // whitespace.
        auto trimmed = trimWhitespace(field);
        float val;
        if (parseFloat<float>(trimmed, val)) {
          flat->set(row, val);
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
        double val;
        if (parseFloat<double>(trimmed, val)) {
          flat->set(row, val);
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
      default:
        child->setNull(row, true);
        break;
    }
  }

  const TypePtr outputType_;
  const char delimiter_;
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
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_EQ(args.size(), 1, "from_csv expects one argument.");
  VELOX_USER_CHECK_EQ(
      args[0]->type()->kind(),
      TypeKind::VARCHAR,
      "The first argument of from_csv should be of varchar type.");
  VELOX_USER_CHECK_EQ(
      type->kind(), TypeKind::ROW, "from_csv output type must be ROW.");

  const auto& rowType = type->asRow();
  for (column_index_t i = 0; i < rowType.size(); ++i) {
    VELOX_USER_CHECK(
        isSupportedLeafType(rowType.childAt(i)),
        "Unsupported field type {} for from_csv.",
        rowType.childAt(i)->toString());
  }

  auto func = std::make_shared<FromCsvFunction>(type);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      func,
      exec::VectorFunctionMetadata{},
      kFromCsv,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql

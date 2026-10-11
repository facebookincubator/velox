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

#include <optional>
#include <string>
#include <vector>

#include <folly/Conv.h>

#include "velox/common/text/DelimitedTextParser.h"
#include "velox/common/text/TextFieldParser.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {
namespace {

using ::facebook::velox::text::DelimitedTextBuffers;
using ::facebook::velox::text::DelimitedTextOptions;
using ::facebook::velox::text::DelimitedTextParser;
using ::facebook::velox::text::TextFieldParser;

template <typename T>
std::optional<T>
parseDecimalField(std::string_view field, uint8_t precision, uint8_t scale) {
  try {
    const auto parsed =
        TextFieldParser::parseDecimal<T>(field, precision, scale);
    return parsed.hasValue() ? std::optional<T>{parsed.value()} : std::nullopt;
  } catch (const folly::ConversionError&) {
    return std::nullopt;
  } catch (const VeloxException&) {
    return std::nullopt;
  }
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
// Key behavior:
// - NULL input returns NULL struct.
// - Empty input returns a non-null struct with null fields.
// - Fewer CSV fields than schema columns: remaining columns are NULL.
// - More CSV fields than schema columns: extra fields are ignored.
// - Fields that cannot be parsed to the target type become NULL.
// - Quoted fields are supported with backslash escape (Spark default).
// - Primitive conversion uses the same canonical parsers as TextReader.
class FromCsvFunction : public exec::VectorFunction {
 public:
  explicit FromCsvFunction(const TypePtr& outputType)
      : outputType_(outputType) {
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

    DelimitedTextBuffers buffers;
    std::string varbinaryBuffer;

    rows.applyToSelected([&](auto row) {
      if (decodedInput->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto csvString = decodedInput->valueAt<StringView>(row);
      const auto csvStringView =
          std::string_view(csvString.data(), csvString.size());

      bits::clearNull(rawResultNulls, row);

      // Short-circuit for zero-column schema: no fields to parse.
      if (numFields == 0) {
        return;
      }

      DelimitedTextParser::splitLine(
          csvStringView,
          DelimitedTextOptions{
              .delimiter = kDelimiter, .escape = kEscape, .quote = kQuote},
          numFields,
          buffers);

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
          if (DelimitedTextParser::isNullField(field, kNullValue)) {
            childVector->setNull(row, true);
          } else {
            childVector->setNull(row, false);
            setCsvFieldToChild(
                childVector, row, field, typeKind, columnType, varbinaryBuffer);
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
  void setCsvFieldToChild(
      BaseVector* child,
      vector_size_t row,
      std::string_view field,
      TypeKind typeKind,
      const TypePtr& type,
      std::string& varbinaryBuffer) const {
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        auto* flat = child->asFlatVector<bool>();
        const auto parsed = TextFieldParser::parseBoolean(field);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::TINYINT: {
        auto* flat = child->asFlatVector<int8_t>();
        const auto parsed = TextFieldParser::parseNarrowInteger<int8_t>(field);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::SMALLINT: {
        auto* flat = child->asFlatVector<int16_t>();
        const auto parsed = TextFieldParser::parseNarrowInteger<int16_t>(field);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::INTEGER: {
        // Could be either INTEGER or DATE (days since epoch).
        if (type->isDate()) {
          const auto parsed = TextFieldParser::parseDate(field);
          if (parsed.hasValue()) {
            child->asFlatVector<int32_t>()->set(row, parsed.value());
          } else {
            child->setNull(row, true);
          }
        } else {
          auto* flat = child->asFlatVector<int32_t>();
          const auto parsed =
              TextFieldParser::parseNarrowInteger<int32_t>(field);
          if (parsed.has_value()) {
            flat->set(row, *parsed);
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
          const auto parsed =
              parseDecimalField<int64_t>(field, precision, scale);
          if (parsed.has_value()) {
            child->asFlatVector<int64_t>()->set(row, *parsed);
          } else {
            child->setNull(row, true);
          }
        } else {
          auto* flat = child->asFlatVector<int64_t>();
          const auto parsed =
              TextFieldParser::parseNarrowInteger<int64_t>(field);
          if (parsed.has_value()) {
            flat->set(row, *parsed);
          } else {
            child->setNull(row, true);
          }
        }
        break;
      }
      case TypeKind::REAL: {
        auto* flat = child->asFlatVector<float>();
        const auto parsed = TextFieldParser::parseFloatingPoint<float>(field);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::DOUBLE: {
        auto* flat = child->asFlatVector<double>();
        const auto parsed = TextFieldParser::parseFloatingPoint<double>(field);
        if (parsed.has_value()) {
          flat->set(row, *parsed);
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
        TextFieldParser::parseVarbinary(field, varbinaryBuffer);
        auto* flat = child->asFlatVector<StringView>();
        flat->set(
            row,
            StringView(
                varbinaryBuffer.data(),
                static_cast<int32_t>(varbinaryBuffer.size())));
        break;
      }
      case TypeKind::HUGEINT: {
        if (type->isLongDecimal()) {
          // LONG DECIMAL (precision > 18).
          auto [precision, scale] = getDecimalPrecisionScale(*type);
          const auto parsed =
              parseDecimalField<int128_t>(field, precision, scale);
          if (parsed.has_value()) {
            child->asFlatVector<int128_t>()->set(row, *parsed);
          } else {
            child->setNull(row, true);
          }
        } else {
          child->setNull(row, true);
        }
        break;
      }
      case TypeKind::TIMESTAMP: {
        const auto parsed = TextFieldParser::parseTimestamp(field);
        if (parsed.hasValue()) {
          child->asFlatVector<Timestamp>()->set(row, parsed.value());
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

  // Pinned CSV options for the supported two-argument SQL overload.
  static constexpr char kDelimiter{','};
  static constexpr char kEscape{'\\'};
  static constexpr char kQuote{'"'};
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
    const core::QueryConfig& /*config*/) {
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

  auto function = std::make_shared<FromCsvFunction>(type);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      function,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      kFromCsv,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql

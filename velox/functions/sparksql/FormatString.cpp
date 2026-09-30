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
#include "velox/functions/sparksql/FormatString.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/lib/string/StringCore.h"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <optional>
#include <string_view>

namespace facebook::velox::functions::sparksql {

namespace {

// Mirrors Spark's UTF8String.toString(), which decodes bytes through
// java.lang.String and replaces malformed UTF-8 sequences with the Unicode
// replacement character (U+FFFD). Pure-ASCII input is copied directly so the
// more expensive replacement path runs only for non-ASCII input.
std::string normalizeUtf8(StringView value) {
  if (stringCore::isAscii(value.data(), value.size())) {
    return std::string(value.data(), value.size());
  }
  std::string result;
  // StringView's length constructor enforces a non-negative int32_t size, so
  // the cast never overflows or produces a negative length.
  replaceInvalidUTF8Characters(
      result, value.data(), static_cast<int32_t>(value.size()));
  return result;
}

// Converts a non-VARCHAR value to the representation used by Java's
// Object.toString().
std::string valueToString(
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& type) {
  VELOX_DCHECK(!decoded.isNullAt(row));
  VELOX_USER_CHECK(
      !type->isDecimal(), "format_string does not support decimal arguments");
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return decoded.valueAt<bool>(row) ? "true" : "false";
    case TypeKind::TINYINT:
      return std::to_string(decoded.valueAt<int8_t>(row));
    case TypeKind::SMALLINT:
      return std::to_string(decoded.valueAt<int16_t>(row));
    case TypeKind::INTEGER:
      return std::to_string(decoded.valueAt<int32_t>(row));
    case TypeKind::BIGINT:
      return std::to_string(decoded.valueAt<int64_t>(row));
    default:
      VELOX_USER_FAIL(
          "Unsupported type for format_string: {}", type->toString());
  }
}

// Converts a value to int64_t for %d. REAL and DOUBLE are rejected to match
// Java Formatter's integral-conversion behavior.
int64_t valueToLong(
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& type) {
  VELOX_DCHECK(!decoded.isNullAt(row));
  VELOX_USER_CHECK(
      !type->isDecimal(), "format_string does not support decimal arguments");
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return decoded.valueAt<int8_t>(row);
    case TypeKind::SMALLINT:
      return decoded.valueAt<int16_t>(row);
    case TypeKind::INTEGER:
      return decoded.valueAt<int32_t>(row);
    case TypeKind::BIGINT:
      return decoded.valueAt<int64_t>(row);
    default:
      VELOX_USER_FAIL(
          "format_string: %d requires an integral type, got {}",
          type->toString());
  }
}

// Converts a value to the unsigned representation used by Java's integral
// formatting, preserving the source wrapper's bit width.
uint64_t valueToUnsigned(
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& type) {
  VELOX_DCHECK(!decoded.isNullAt(row));
  VELOX_USER_CHECK(
      !type->isDecimal(), "format_string does not support decimal arguments");
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return static_cast<uint8_t>(decoded.valueAt<int8_t>(row));
    case TypeKind::SMALLINT:
      return static_cast<uint16_t>(decoded.valueAt<int16_t>(row));
    case TypeKind::INTEGER:
      return static_cast<uint32_t>(decoded.valueAt<int32_t>(row));
    case TypeKind::BIGINT:
      return static_cast<uint64_t>(decoded.valueAt<int64_t>(row));
    default:
      VELOX_USER_FAIL(
          "format_string: %x/%o requires an integral type, got {}",
          type->toString());
  }
}

// Limits formatted output per specifier to prevent excessive allocation from
// large width specifiers such as "%999999999d".
static constexpr int kMaxFormattedSize = 1 << 20; // 1 MiB.

// Appends a printf-formatted result using a stack buffer for common outputs.
template <typename... Values>
void snprintfAppend(std::string& output, const char* format, Values... values) {
  std::array<char, 128> buffer;
  int formattedSize =
      std::snprintf(buffer.data(), buffer.size(), format, values...);
  VELOX_CHECK_GE(formattedSize, 0, "snprintf encoding error in format_string");
  VELOX_CHECK_LE(
      formattedSize,
      kMaxFormattedSize,
      "format_string produced an unexpectedly large output ({} bytes)",
      formattedSize);
  const auto outputSize = static_cast<size_t>(formattedSize);
  if (outputSize < buffer.size()) {
    output.append(buffer.data(), outputSize);
    return;
  }

  size_t oldSize = output.size();
  output.resize(oldSize + outputSize + 1);
  std::snprintf(output.data() + oldSize, outputSize + 1, format, values...);
  output.resize(oldSize + outputSize); // Trim the trailing NUL.
}

// Builds the string specifier Java uses for null values: width and
// left-alignment are preserved, while numeric flags are ignored.
std::string makeNullSpecifier(std::string_view specifier) {
  std::string result = "%";
  size_t i = 1;
  bool leftAlign = false;
  while (i < specifier.size() &&
         (specifier[i] == '-' || specifier[i] == '+' || specifier[i] == '0' ||
          specifier[i] == ' ')) {
    leftAlign |= specifier[i] == '-';
    ++i;
  }
  if (leftAlign) {
    result += '-';
  }
  while (i < specifier.size() && specifier[i] >= '0' && specifier[i] <= '9') {
    result += specifier[i++];
  }
  result += 's';
  return result;
}

// Keeps parsed pattern data reusable across rows with a constant pattern.
struct FormatPart {
  // Contains literal bytes when argumentIndex is empty; otherwise, contains
  // the original validated conversion specifier.
  std::string formatText;

  // Uses the evaluated-argument index; nullopt distinguishes literal parts.
  std::optional<size_t> argumentIndex;

  // Expands the conversion suffix to the platform's 64-bit printf suffix.
  std::string printfSpecifier;

  // Retains only width and left alignment for Java-compatible null formatting.
  std::string nullSpecifier;
};

// Formats one argument using the supported Java Formatter subset.
void formatOneValue(
    const FormatPart& part,
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& argumentType,
    std::string& output) {
  const auto& specifier = part.formatText;
  char conversion = specifier.back();

  // Java's Formatter formats null as "null" for all conversion types,
  // applying string width from the specifier. Uppercase conversions
  // uppercase the null representation.
  if (decoded.isNullAt(row)) {
    const char* nullValue = conversion == 'X' ? "NULL" : "null";
    if (part.nullSpecifier == "%s") {
      output += nullValue;
    } else {
      snprintfAppend(output, part.nullSpecifier.c_str(), nullValue);
    }
    return;
  }

  switch (conversion) {
    case 's': {
      VELOX_DCHECK_EQ(specifier, "%s");
      if (argumentType->isVarchar()) {
        auto value = decoded.valueAt<StringView>(row);
        if (stringCore::isAscii(value.data(), value.size())) {
          output.append(value.data(), value.size());
        } else {
          output += normalizeUtf8(value);
        }
      } else {
        output += valueToString(decoded, row, argumentType);
      }
      break;
    }
    case 'd': {
      int64_t value = valueToLong(decoded, row, argumentType);
      snprintfAppend(output, part.printfSpecifier.c_str(), value);
      break;
    }
    case 'o': {
      uint64_t value = valueToUnsigned(decoded, row, argumentType);
      snprintfAppend(output, part.printfSpecifier.c_str(), value);
      break;
    }
    case 'x': {
      uint64_t value = valueToUnsigned(decoded, row, argumentType);
      snprintfAppend(output, part.printfSpecifier.c_str(), value);
      break;
    }
    case 'X': {
      uint64_t value = valueToUnsigned(decoded, row, argumentType);
      snprintfAppend(output, part.printfSpecifier.c_str(), value);
      break;
    }
    default:
      VELOX_UNREACHABLE(
          "Unexpected format_string conversion: '{}'", conversion);
  }
}

// Parses and validates a normalized pattern into reusable formatting parts.
std::vector<FormatPart> parsePattern(
    std::string_view pattern,
    size_t numArguments) {
  std::vector<FormatPart> parts;
  std::string literal;
  size_t argumentIndex = 1;
  size_t i = 0;

  auto flushLiteral = [&]() {
    if (!literal.empty()) {
      parts.push_back({std::move(literal), std::nullopt, "", ""});
      literal.clear();
    }
  };

  while (i < pattern.size()) {
    if (pattern[i] != '%') {
      literal += pattern[i++];
      continue;
    }

    if (i + 1 < pattern.size() && pattern[i + 1] == '%') {
      literal += '%';
      i += 2;
      continue;
    }

    flushLiteral();
    std::string specifier{"%"};
    ++i;

    std::string flags;
    while (i < pattern.size() &&
           (pattern[i] == '-' || pattern[i] == '+' || pattern[i] == '0' ||
            pattern[i] == ' ')) {
      VELOX_USER_CHECK(
          flags.find(pattern[i]) == std::string::npos,
          "Duplicate flag in format_string: '{}'",
          pattern[i]);
      flags += pattern[i];
      specifier += pattern[i++];
    }

    const auto widthStart = i;
    size_t width = 0;
    while (i < pattern.size() && pattern[i] >= '0' && pattern[i] <= '9') {
      const auto digit = static_cast<size_t>(pattern[i] - '0');
      VELOX_USER_CHECK(
          width <= (static_cast<size_t>(kMaxFormattedSize) - digit) / 10,
          "format_string width must not exceed {}",
          kMaxFormattedSize);
      width = width * 10 + digit;
      specifier += pattern[i++];
    }
    const bool hasWidth = i > widthStart;

    bool hasPrecision = false;
    if (i < pattern.size() && pattern[i] == '.') {
      hasPrecision = true;
      specifier += pattern[i++];
      const auto precisionStart = i;
      size_t precision = 0;
      while (i < pattern.size() && pattern[i] >= '0' && pattern[i] <= '9') {
        const auto digit = static_cast<size_t>(pattern[i] - '0');
        VELOX_USER_CHECK(
            precision <= (static_cast<size_t>(kMaxFormattedSize) - digit) / 10,
            "format_string precision must not exceed {}",
            kMaxFormattedSize);
        precision = precision * 10 + digit;
        specifier += pattern[i++];
      }
      VELOX_USER_CHECK(
          i > precisionStart,
          "format_string precision requires at least one digit");
    }

    VELOX_USER_CHECK(
        i < pattern.size(), "Incomplete format specifier in format_string");
    const char conversion = pattern[i++];
    specifier += conversion;

    const bool hasLeftAlign = flags.find('-') != std::string::npos;
    const bool hasPlus = flags.find('+') != std::string::npos;
    const bool hasSpace = flags.find(' ') != std::string::npos;
    const bool hasZero = flags.find('0') != std::string::npos;
    VELOX_USER_CHECK(
        (!hasLeftAlign && !hasZero) || hasWidth,
        "format_string flags '-' and '0' require a width");
    VELOX_USER_CHECK(
        !(hasLeftAlign && hasZero),
        "format_string does not support combining '-' and '0'");
    VELOX_USER_CHECK(
        !(hasPlus && hasSpace),
        "format_string does not support combining '+' and space");

    switch (conversion) {
      case 's':
        VELOX_USER_CHECK(
            flags.empty() && !hasWidth && !hasPrecision,
            "format_string supports only bare %s");
        break;
      case 'd':
        VELOX_USER_CHECK(
            !hasPrecision, "format_string does not support integer precision");
        break;
      case 'o':
      case 'x':
      case 'X':
        VELOX_USER_CHECK(
            !hasPrecision && !hasPlus && !hasSpace,
            "Unsupported flags or precision for integral format_string: {}",
            specifier);
        break;
      default:
        VELOX_USER_FAIL(
            "Unsupported format conversion character: '{}'", conversion);
    }

    VELOX_USER_CHECK(
        argumentIndex < numArguments,
        "Not enough arguments for format_string: specifier {} ({}) has no "
        "argument; {} provided",
        argumentIndex,
        specifier,
        numArguments - 1);
    FormatPart part{std::move(specifier), argumentIndex++, "", ""};
    part.nullSpecifier = makeNullSpecifier(part.formatText);
    switch (conversion) {
      case 'd':
        part.printfSpecifier =
            part.formatText.substr(0, part.formatText.size() - 1) + PRId64;
        break;
      case 'o':
        part.printfSpecifier =
            part.formatText.substr(0, part.formatText.size() - 1) + PRIo64;
        break;
      case 'x':
        part.printfSpecifier =
            part.formatText.substr(0, part.formatText.size() - 1) + PRIx64;
        break;
      case 'X':
        part.printfSpecifier =
            part.formatText.substr(0, part.formatText.size() - 1) + PRIX64;
        break;
      case 's':
        break;
      default:
        VELOX_UNREACHABLE();
    }
    parts.push_back(std::move(part));
  }

  flushLiteral();
  return parts;
}

// Separates row formatting from the conditional child evaluation below.
class FormatStringFunction : public exec::VectorFunction {
 public:
  // Assumes null format-pattern rows were removed before argument evaluation.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& arguments,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    auto* flatResult = result->asFlatVector<StringView>();

    // Decode all arguments.
    std::vector<exec::LocalDecodedVector> decodedArguments;
    decodedArguments.reserve(arguments.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      decodedArguments.emplace_back(context, *arguments[i], rows);
    }

    // Normalize and parse constant patterns once for the whole batch.
    std::optional<std::vector<FormatPart>> constantParts;
    if (decodedArguments[0]->isConstantMapping() && rows.hasSelections()) {
      try {
        constantParts = parsePattern(
            normalizeUtf8(
                decodedArguments[0]->valueAt<StringView>(rows.begin())),
            arguments.size());
      } catch (const std::exception&) {
        context.setErrors(rows, std::current_exception());
        return;
      }
    }

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      std::vector<FormatPart> perRowParts;
      const std::vector<FormatPart>* parts;
      if (constantParts.has_value()) {
        parts = &constantParts.value();
      } else {
        perRowParts = parsePattern(
            normalizeUtf8(decodedArguments[0]->valueAt<StringView>(row)),
            arguments.size());
        parts = &perRowParts;
      }

      std::string output;
      size_t minimumOutputSize = 0;
      for (const auto& part : *parts) {
        minimumOutputSize += part.formatText.size();
      }
      output.reserve(minimumOutputSize);
      for (const auto& part : *parts) {
        if (!part.argumentIndex.has_value()) {
          output += part.formatText;
          continue;
        }
        const auto argumentIndex = part.argumentIndex.value();
        formatOneValue(
            part,
            *decodedArguments[argumentIndex],
            row,
            arguments[argumentIndex]->type(),
            output);
      }

      flatResult->set(row, StringView(output));
    });
  }
};

// Evaluates the format pattern before evaluating the remaining arguments.
class FormatStringExpr final : public exec::SpecialForm {
 public:
  FormatStringExpr(
      TypePtr type,
      std::vector<exec::ExprPtr>&& inputs,
      bool trackCpuUsage)
      : SpecialForm(
            exec::SpecialFormKind::kCustom,
            std::move(type),
            std::move(inputs),
            FormatStringCallToSpecialForm::kFormatString,
            false,
            trackCpuUsage) {}

  // Evaluates only rows with a non-null, error-free format pattern.
  void evalSpecialForm(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      VectorPtr& result) override {
    BaseVector::ensureWritable(rows, type(), context.pool(), result);
    result->addNulls(rows);

    exec::ScopedFinalSelectionSetter scopedFinalSelectionSetter(context, &rows);
    exec::LocalSelectivityVector activeRows(context, rows);

    std::vector<VectorPtr> arguments;
    arguments.reserve(inputs_.size());

    VectorPtr pattern;
    inputs_[0]->eval(rows, context, pattern);
    arguments.push_back(pattern);

    if (context.errors()) {
      context.deselectErrors(*activeRows);
    }
    if (!activeRows->hasSelections()) {
      return;
    }

    exec::LocalDecodedVector decodedPattern(context, *pattern, *activeRows);
    if (decodedPattern->mayHaveNulls()) {
      activeRows->deselectNulls(
          decodedPattern->nulls(activeRows.get()),
          activeRows->begin(),
          activeRows->end());
    }
    if (!activeRows->hasSelections()) {
      return;
    }

    for (size_t i = 1; i < inputs_.size(); ++i) {
      VectorPtr argument;
      inputs_[i]->eval(*activeRows, context, argument);
      arguments.push_back(argument);

      if (context.errors()) {
        context.deselectErrors(*activeRows);
      }
      if (!activeRows->hasSelections()) {
        return;
      }
    }

    function_.apply(*activeRows, arguments, type(), context, result);
  }

  bool isConditional() const override {
    return true;
  }

  bool evaluatesArgumentsOnNonIncreasingSelection() const override {
    return true;
  }

 private:
  void computePropagatesNulls() override {
    propagatesNulls_ = false;
  }

  // Performs row-level formatting after conditional argument evaluation.
  FormatStringFunction function_;
};

} // namespace

TypePtr FormatStringCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argumentTypes*/) {
  return VARCHAR();
}

exec::ExprPtr FormatStringCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& arguments,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  auto numArguments = arguments.size();
  VELOX_USER_CHECK(
      numArguments >= 1,
      "format_string requires at least one argument: {}",
      numArguments);
  VELOX_USER_CHECK(
      arguments[0]->type()->isVarchar(),
      "The first argument of format_string must be a varchar: {}",
      arguments[0]->type()->toString());

  return std::make_shared<FormatStringExpr>(
      type, std::move(arguments), trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql

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

#include "velox/functions/prestosql/types/JsonType.h"

#include <expression/PeeledEncoding.h>
#include <expression/StringWriter.h>
#include <expression/VectorFunction.h>
#include <functions/lib/DateTimeFormatter.h>
#include <functions/lib/RowsTranslationUtil.h>
#include <functions/lib/TimeUtils.h>
#include <functions/prestosql/json/JsonStringUtil.h>

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
std::enable_if_t<std::is_integral_v<T>, size_t> append(
    T value,
    char* const buffer) {
  const auto oute = buffer + folly::to_ascii_size_max_decimal<uint64_t> + 1;
  auto uvalue = value < 0 ? ~static_cast<uint64_t>(value) + 1
                          : static_cast<uint64_t>(value);
  size_t p = 0;
  char* writtenPosition = buffer;
  if (value < 0) {
    *writtenPosition++ = '-';
    p += 1;
  };
  p += folly::to_ascii_decimal(writtenPosition, oute, uvalue);
  return p;
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, size_t> append(
    T value,
    char* const buffer) {
  std::string result;
  if (FOLLY_UNLIKELY(std::isinf(value) || std::isnan(value))) {
    result = fmt::format(
        "\"{}\"", util::Converter<TypeKind::VARCHAR>::tryCast(value).value());
  } else {
    result = util::Converter<TypeKind::VARCHAR>::tryCast(value).value();
  }
  std::memcpy(buffer, result.c_str(), result.size());
  return result.size();
}

template <TypeKind kind, typename T>
size_t convertToString(
    T value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  VELOX_FAIL("{} is not supported in to_json.", type->toString());
}

template <>
size_t convertToString<TypeKind::BOOLEAN>(
    bool value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  static const char TRUE[] = "true";
  static const char FALSE[] = "false";
  char* pos = buffer;
  const char* res = value ? TRUE : FALSE;
  const size_t size = value ? 4 : 5;
  std::memcpy(pos, res, size);
  return size;
}

template <>
size_t convertToString<TypeKind::TINYINT>(
    int8_t value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  return append(value, buffer);
}

template <>
size_t convertToString<TypeKind::SMALLINT>(
    int16_t value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  return append(value, buffer);
}

template <>
size_t convertToString<TypeKind::INTEGER>(
    int32_t value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  if (type->isDate()) {
    std::string stringValue = DATE()->toString(value);
    return snprintf(
        buffer, stringValue.size() + 3, "\"%s\"", stringValue.c_str());
  } else {
    return append(value, buffer);
  }
}

template <>
size_t convertToString<TypeKind::BIGINT>(
    int64_t value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  if (type->isDecimal()) {
    auto [precision, scale] = getDecimalPrecisionScale(*type);
    auto size = DecimalUtil::maxStringViewSize(precision, scale);
    return DecimalUtil::castToString(value, scale, size, buffer);
  } else {
    return append(value, buffer);
  }
}

template <>
size_t convertToString<TypeKind::HUGEINT>(
    int128_t value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  const auto oute = buffer + folly::detail::digitsEnough<uint128_t>() + 1;
  size_t p;
  if (value < 0) {
    *buffer = '-';
    p = 1 + folly::detail::unsafeTelescope128(buffer + 1, oute, -value);
  } else {
    p = folly::detail::unsafeTelescope128(buffer, oute, value);
  }
  return p;
}

template <>
size_t convertToString<TypeKind::REAL>(
    float value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  return append(value, buffer);
}

template <>
size_t convertToString<TypeKind::DOUBLE>(
    double value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  return append(value, buffer);
}

template <>
size_t convertToString<TypeKind::VARCHAR>(
    StringView value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  size_t size = normalizedSizeForJsonCast(value.data(), value.size());
  *buffer = '"';
  normalizeForJsonCast(value.data(), size, buffer + 1);
  *(buffer + size + 1) = '"';
  return size + 2;
}

template <>
size_t convertToString<TypeKind::TIMESTAMP>(
    Timestamp value,
    char* const buffer,
    exec::EvalCtx& context,
    const TypePtr& type) {
  // Spark converts Timestamp in ISO8601 format by default.
  static const auto formatter =
      functions::buildJodaDateTimeFormatter("yyyy-MM-dd'T'HH:mm:ss.SSSZZ")
          .value();
  const auto* timeZone =
      getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
  const auto maxResultSize = formatter->maxResultSize(timeZone);
  *buffer = '"';
  const auto resultSize =
      formatter->format(value, timeZone, maxResultSize, buffer + 1, false, "Z");
  *(buffer + resultSize + 1) = '"';
  return resultSize + 2;
}

template <typename T>
size_t estimateRowSize(const TypePtr& type) {
  if constexpr (std::is_same_v<T, bool>) {
    return 5;
  } else if constexpr (std::is_integral_v<T>) {
    return folly::detail::digitsEnough<T>() + 1;
  } else if constexpr (std::is_same_v<T, Timestamp>) {
    // yyyy-MM-dd'T'HH:mm:ss.SSSZZ
    return 40;
  } else if (type->isDate()) {
    // yyyy-MM-dd.
    return 12;
  } else {
    // For variable-length types, the initial size is set to 10.
    return 10;
  }
}

// Convert primitive-type input vectors to Json string.
template <
    TypeKind kind,
    typename std::enable_if_t<TypeTraits<kind>::isPrimitiveType, int> = 0>
void toJson(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  using T = typename TypeTraits<kind>::NativeType;

  // input is guaranteed to be in flat or constant encodings when passed in.
  auto inputVector = input.as<SimpleVector<T>>();

  size_t rowSize = estimateRowSize<T>(inputVector->type());
  Buffer* buffer =
      flatResult.getBufferWithSpace(rows.countSelected() * rowSize);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputVector->isNullAt(row)) {
      flatResult.set(row, "null");
    } else {
      auto size = VELOX_DYNAMIC_TYPE_DISPATCH(
          convertToString,
          kind,
          inputVector->valueAt(row),
          rawBuffer,
          context,
          inputVector->type());

      flatResult.setNoCopy(row, StringView(rawBuffer, size));
      rawBuffer += size;
    }
  });
  // Update the exact buffer size.
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
}

// Forward declaration.
void toJsonFromRow(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

void toJsonFromArray(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

void toJsonFromMap(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

// Convert complex-type input vectors to Json string.
template <
    TypeKind kind,
    typename std::enable_if_t<!TypeTraits<kind>::isPrimitiveType, int> = 0>
void toJson(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  if constexpr (kind == TypeKind::ROW) {
    toJsonFromRow(input, context, rows, flatResult);
  } else if constexpr (kind == TypeKind::ARRAY) {
    toJsonFromArray(input, context, rows, flatResult);
  } else if constexpr (kind == TypeKind::MAP) {
    toJsonFromMap(input, context, rows, flatResult);
  } else {
    VELOX_FAIL("{} is not supported in to_json.", input.type()->toString());
  }
}

// Helper struct representing the Json vector of input.
struct AsJson {
  AsJson(
      exec::EvalCtx& context,
      const VectorPtr& input,
      const SelectivityVector& rows,
      const BufferPtr& elementToTopLevelRows)
      : decoded_(context) {
    VELOX_CHECK(rows.hasSelections());

    exec::EvalErrorsPtr oldErrors;
    context.swapErrors(oldErrors);
    if (isJsonType(input->type())) {
      json_ = input;
    } else {
      if (!exec::PeeledEncoding::isPeelable(input->encoding())) {
        serialize(context, input, rows, json_);
      } else {
        exec::withContextSaver([&](exec::ContextSaver& saver) {
          exec::LocalSelectivityVector newRowsHodler(*context.execCtx());

          exec::LocalDecodedVector localDecoded(context);
          std::vector<VectorPtr> peeledVectors;
          auto peeledEncoding = exec::PeeledEncoding::peel(
              {input}, rows, localDecoded, true, peeledVectors);
          VELOX_CHECK_EQ(peeledVectors.size(), 1);
          auto newRows =
              peeledEncoding->translateToInnerRows(rows, newRowsHodler);
          // Save context and set the peel
          context.saveAndReset(saver, rows);
          context.setPeeledEncoding(peeledEncoding);

          serialize(context, peeledVectors[0], *newRows, json_);
          json_ = context.getPeeledEncoding()->wrap(
              json_->type(), context.pool(), json_, rows);
        });
      }
    }
    decoded_.get()->decode(*json_, rows);
    jsonStrings_ = decoded_->base()->as<SimpleVector<StringView>>();

    combineErrors(context, rows, elementToTopLevelRows, oldErrors);
  }

  StringView at(vector_size_t i) const {
    return jsonStrings_->valueAt(decoded_->index(i));
  }

  // Returns the length of the json string of the value at i, when this
  // value will be inlined as an element in the json string of an array, map, or
  // row.
  vector_size_t lengthAt(vector_size_t i) const {
    if (decoded_->isNullAt(i)) {
      // Null values are inlined as "null".
      return 4;
    } else {
      return this->at(i).size();
    }
  }

  // Appends the json string of the value at i to a string writer.
  void append(vector_size_t i, exec::StringWriter& proxy) const {
    if (decoded_->isNullAt(i)) {
      proxy.append("null");
    } else {
      proxy.append(this->at(i));
    }
  }

 private:
  void serialize(
      exec::EvalCtx& context,
      const VectorPtr& input,
      const SelectivityVector& baseRows,
      VectorPtr& result) {
    context.ensureWritable(baseRows, JSON(), result);
    auto flatJsonStrings = result->as<FlatVector<StringView>>();

    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        toJson, input->typeKind(), *input, context, baseRows, *flatJsonStrings);
  }

  // Combine exceptions in oldErrors into context.errors_ with a transformation
  // of rows mapping provided by elementToTopLevelRows. If there are exceptions
  // at the same row in both context.errors_ and oldErrors, the one in oldErrors
  // remains. elementToTopLevelRows can be a nullptr, meaning that the rows in
  // context.errors_ correspond to rows in oldErrors exactly.
  void combineErrors(
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const BufferPtr& elementToTopLevelRows,
      exec::EvalErrorsPtr& oldErrors) {
    if (context.errors()) {
      if (elementToTopLevelRows) {
        context.addElementErrorsToTopLevel(
            rows, elementToTopLevelRows, oldErrors);
      } else {
        context.addErrors(rows, *context.errorsPtr(), oldErrors);
      }
    }
    context.swapErrors(oldErrors);
  }

  exec::LocalDecodedVector decoded_;
  VectorPtr json_;
  const SimpleVector<StringView>* jsonStrings_;
};

void toJsonFromRow(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guaranteed to be in flat encoding when passed in.
  VELOX_CHECK_EQ(input.encoding(), VectorEncoding::Simple::ROW);
  auto inputRow = input.as<RowVector>();
  auto childrenSize = inputRow->childrenSize();

  auto& rowType = inputRow->type()->asRow();
  VELOX_CHECK_EQ(rowType.size(), childrenSize, "Mismatch in row type size");

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all children Json strings and the
  // delimiters to be added.
  size_t childrenStringSize = 0;
  std::vector<AsJson> childrenAsJson;
  for (int i = 0; i < childrenSize; ++i) {
    childrenAsJson.emplace_back(context, inputRow->childAt(i), rows, nullptr);

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      if (inputRow->isNullAt(row)) {
        // "null" will be inlined in the StringView.
        return;
      }
      childrenStringSize += childrenAsJson[i].lengthAt(row);
    });
  }

  // Extra length for commas and brackets.
  childrenStringSize +=
      rows.countSelected() * (childrenSize > 0 ? childrenSize + 1 : 2);
  flatResult.getBufferWithSpace(childrenStringSize);

  // Constructs Json string of each row from Json strings of its children.
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputRow->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto proxy = exec::StringWriter(&flatResult, row);

    proxy.append("{"_sv);
    for (int i = 0; i < childrenSize; ++i) {
      if (i > 0) {
        proxy.append(","_sv);
      }

      proxy.append("\""_sv);
      proxy.append(rowType.nameOf(i));
      proxy.append("\":"_sv);

      childrenAsJson[i].append(row, proxy);
    }
    proxy.append("}"_sv);

    proxy.finalize();
  });
}

void toJsonFromArray(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guranteed to be in flat encoding when passed in.
  auto inputArray = input.as<ArrayVector>();

  auto elements = inputArray->elements();
  auto elementsRows =
      functions::toElementRows(elements->size(), rows, inputArray);
  if (!elementsRows.hasSelections()) {
    // All arrays are null or empty.
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      if (inputArray->isNullAt(row)) {
        flatResult.set(row, "null");
      } else {
        VELOX_CHECK_EQ(
            inputArray->sizeAt(row),
            0,
            "All arrays are expected to be null or empty");
        flatResult.set(row, "[]");
      }
    });
    return;
  }

  auto elementToTopLevelRows = functions::getElementToTopLevelRows(
      elements->size(), rows, inputArray, context.pool());
  AsJson elementsAsJson{context, elements, elementsRows, elementToTopLevelRows};

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all elements Json strings and the
  // delimiters to be added.
  size_t elementsStringSize = 0;
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputArray->isNullAt(row)) {
      // "null" will be inlined in the StringView.
      return;
    }

    auto offset = inputArray->offsetAt(row);
    auto size = inputArray->sizeAt(row);
    for (auto i = offset, end = offset + size; i < end; ++i) {
      elementsStringSize += elementsAsJson.lengthAt(i);
    }

    // Extra length for commas and brackets.
    elementsStringSize += size > 0 ? size + 1 : 2;
  });

  flatResult.getBufferWithSpace(elementsStringSize);

  // Constructs the Json string of each array from Json strings of its elements.
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputArray->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto offset = inputArray->offsetAt(row);
    auto size = inputArray->sizeAt(row);

    auto proxy = exec::StringWriter(&flatResult, row);

    proxy.append("["_sv);
    for (int i = offset, end = offset + size; i < end; ++i) {
      if (i > offset) {
        proxy.append(","_sv);
      }
      elementsAsJson.append(i, proxy);
    }
    proxy.append("]"_sv);

    proxy.finalize();
  });
}

void toJsonFromMap(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guaranteed to be in flat encoding when passed in.
  auto inputMap = input.as<MapVector>();
  auto& mapType = inputMap->type()->asMap();

  auto mapKeys = inputMap->mapKeys();
  auto mapValues = inputMap->mapValues();
  auto elementsRows = functions::toElementRows(mapKeys->size(), rows, inputMap);
  if (!elementsRows.hasSelections()) {
    // All maps are null or empty.
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      if (inputMap->isNullAt(row)) {
        flatResult.set(row, "null");
      } else {
        VELOX_CHECK_EQ(
            inputMap->sizeAt(row),
            0,
            "All maps are expected to be null or empty");
        flatResult.set(row, "{}");
      }
    });
    return;
  }

  auto elementToTopLevelRows = functions::getElementToTopLevelRows(
      mapKeys->size(), rows, inputMap, context.pool());

  AsJson keysAsJson{context, mapKeys, elementsRows, elementToTopLevelRows};
  AsJson valuesAsJson{context, mapValues, elementsRows, elementToTopLevelRows};

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all elements Json strings and the
  // delimiters to be added.
  size_t elementsStringSize = 0;
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputMap->isNullAt(row)) {
      // "null" will be inlined in the StringView.
      return;
    }

    auto offset = inputMap->offsetAt(row);
    auto size = inputMap->sizeAt(row);
    for (auto i = offset, end = offset + size; i < end; ++i) {
      // The construction of keysAsJson ensured there is no null in keysAsJson
      elementsStringSize += keysAsJson.at(i).size() + valuesAsJson.lengthAt(i);
    }

    // Extra length for commas, semicolons, and curly braces.
    elementsStringSize += size > 0 ? size * 2 + 1 : 2;
  });

  flatResult.getBufferWithSpace(elementsStringSize);

  // Constructs the Json string of each map from Json strings of its keys and
  // values.
  std::vector<std::pair<StringView, vector_size_t>> sortedKeys;
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputMap->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto offset = inputMap->offsetAt(row);
    auto size = inputMap->sizeAt(row);

    // Sort entries by keys in each map.
    sortedKeys.clear();
    for (int i = offset, end = offset + size; i < end; ++i) {
      sortedKeys.push_back(std::make_pair(keysAsJson.at(i), i));
    }
    std::sort(sortedKeys.begin(), sortedKeys.end());

    auto proxy = exec::StringWriter(&flatResult, row);

    proxy.append("{"_sv);
    for (auto it = sortedKeys.begin(); it != sortedKeys.end(); ++it) {
      if (it != sortedKeys.begin()) {
        proxy.append(","_sv);
      }
      std::string keyFormat =
          mapType.childAt(0)->isVarchar() ? "{}:" : "\"{}\":";
      proxy.append(fmt::format(keyFormat, it->first));
      valuesAsJson.append(it->second, proxy);
    }
    proxy.append("}"_sv);

    proxy.finalize();
  });
}

class ToJsonFunction final : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    VELOX_USER_CHECK_EQ(args.size(), 1, "to_json takes one argument.");
    auto kind = args[0]->typeKind();
    VELOX_USER_CHECK(
        kind == TypeKind::ROW || kind == TypeKind::ARRAY ||
            kind == TypeKind::MAP,
        "to_json only support ROW/ARRAY/MAP inputs.");
    context.ensureWritable(rows, outputType, result);
    result->clearNulls(rows);
    auto* rawResults = result->as<FlatVector<StringView>>();

    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        toJson, kind, *args[0], context, rows, *rawResults);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T(ROW/ARRAY/MAP) -> varchar
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("varchar")
                .argumentType("T")
                .build()};
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_to_json,
    ToJsonFunction::signatures(),
    std::make_unique<ToJsonFunction>());

} // namespace facebook::velox::functions::sparksql

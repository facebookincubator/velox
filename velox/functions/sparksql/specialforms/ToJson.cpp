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

#include "velox/functions/sparksql/specialforms/ToJson.h"
#include "velox/functions/prestosql/types/JsonType.h"

#include <expression/PeeledEncoding.h>
#include <expression/StringWriter.h>
#include <expression/VectorFunction.h>
#include <folly/Likely.h>
#include <functions/lib/RowsTranslationUtil.h>
#include <functions/prestosql/json/JsonStringUtil.h>
#include <type/Conversions.h>
#include <type/DecimalUtil.h>
#include <type/StringView.h>
#include <type/Timestamp.h>
#include <type/Type.h>
#include <utility>
#include <vector/BaseVector.h>
#include <vector/ComplexVector.h>
#include <vector/SelectivityVector.h>
#include <vector/SimpleVector.h>
#include <vector/TypeAliases.h>
#include <vector/VectorEncoding.h>

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
void generateJsonTyped(
    const SimpleVector<T>& input,
    int row,
    std::string& result,
    const TypePtr& type) {
  auto value = input.valueAt(row);

  if constexpr (std::is_same_v<T, StringView>) {
    size_t resultSize = escapedStringSize(value.data(), value.size());
    result.resize(resultSize + 2);
    result.data()[0] = '"';
    escapeString(value.data(), value.size(), result.data() + 1);
    result.data()[resultSize + 1] = '"';
  } else if constexpr (std::is_same_v<T, UnknownValue>) {
    VELOX_FAIL(
        "Convert UNKNOWN to JSON: Vextors of UNKNOWN type should not contain non-null rows");
  } else {
    if constexpr (std::is_same_v<T, bool>) {
      result.append(value ? "true" : "false");
    } else if constexpr (
        std::is_same_v<T, double> || std::is_same_v<T, float>) {
      if (FOLLY_UNLIKELY(std::isinf(value) || std::isnan(value))) {
        result.append(fmt::format(
            "\"{}\"",
            util::Converter<TypeKind::VARCHAR>::tryCast(value).value()));
      } else {
        result.append(
            util::Converter<TypeKind::VARCHAR>::tryCast(value).value());
      }
    } else if constexpr (std::is_same_v<T, Timestamp>) {
      std::string stringValue = std::to_string(value);
      result.reserve(stringValue.size() + 2);
      result.append("\"");
      result.append(stringValue);
      result.append("\"");
    } else if (type->isDate()) {
      std::string stringValue = DATE()->toString(value);
      result.reserve(stringValue.size() + 2);
      result.append("\"");
      result.append(stringValue);
      result.append("\"");
    } else if (type->isDecimal()) {
      result.append(DecimalUtil::toString(value, type));
    } else {
      folly::toAppend<std::string, T>(value, &result);
    }
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

  std::string result;
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputVector->isNullAt(row)) {
      flatResult.set(row, "null");
    } else {
      result.clear();
      generateJsonTyped<T>(*inputVector, row, result, inputVector->type());

      flatResult.set(row, StringView(result));
    }
  });
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
        exec::withContextSaver([&](exec::ContextSaver& saver){
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
  void append(vector_size_t i, exec::StringWriter<>& proxy) const {
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
        toJson,
        input->typeKind(),
        *input,
        context,
        baseRows,
        *flatJsonStrings);
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

  // Extra length for commas and brackets
  childrenStringSize +=
      rows.countSelected() * (childrenSize > 0 ? childrenSize + 1 : 2);
  flatResult.getBufferWithSpace(childrenStringSize);

  // Constructs Json string of each row from Json strings of its children.
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    if (inputRow->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto proxy = exec::StringWriter<>(&flatResult, row);

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
  AsJson elementsAsJson{
      context, elements, elementsRows, elementToTopLevelRows};

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

    auto proxy = exec::StringWriter<>(&flatResult, row);

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

  AsJson keysAsJson{
      context, mapKeys, elementsRows, elementToTopLevelRows};
  AsJson valuesAsJson{
      context, mapValues, elementsRows, elementToTopLevelRows};

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

    auto proxy = exec::StringWriter<>(&flatResult, row);

    proxy.append("{"_sv);
    for (auto it = sortedKeys.begin(); it != sortedKeys.end(); ++it) {
      if (it != sortedKeys.begin()) {
        proxy.append(","_sv);
      }
      std::string keyFormat = mapType.childAt(0)->isVarchar() ? "{}:" : "\"{}\":";
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
    context.ensureWritable(rows, outputType, result);
    result->clearNulls(rows);
    auto* rawResults = result->as<FlatVector<StringView>>();

    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        toJson, args[0]->typeKind(), *args[0], context, rows, *rawResults);
  }
};

} // namespace

TypePtr ToJsonCallToSpecialForm::resolveType(const std::vector<TypePtr>& /* argTypes */) {
  VELOX_FAIL("to_json function does not support type resolution.");
}

exec::ExprPtr ToJsonCallToSpecialForm::constructSpecialForm(
    const TypePtr &type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /* config */) {
  VELOX_USER_CHECK(type->isVarchar(), "The result type of to_json should be VARCHAR");
  VELOX_USER_CHECK_EQ(args.size(), 1, "to_json expects one argument.");
  VELOX_USER_CHECK(
    args[0]->type()->isRow() || args[0]->type()->isArray() || args[0]->type()->isMap(),
    "The argument type of to_json should be row, array or map.");

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::make_shared<ToJsonFunction>(),
      exec::VectorFunctionMetadata{},
      kToJson,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql

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

#include <string>

#include "folly/CPortability.h"
#include "folly/Conv.h"
#include "folly/json.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/VectorUdfTypeSystem.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {

template <typename T>
void generateJsonTyped(
    const SimpleVector<T>& input,
    int row,
    std::string& result) {
  auto value = input.valueAt(row);

  if constexpr (std::is_same_v<T, StringView>) {
    folly::json::escapeString(value, result, folly::json::serialization_opts{});
  } else if constexpr (std::is_same_v<T, bool>) {
    result.append(value ? "true" : "false");
  } else if constexpr (
      std::is_same_v<T, Date> || std::is_same_v<T, Timestamp>) {
    result.append(std::to_string(value));
  } else {
    folly::toAppend<std::string, T>(value, &result);
  }
}

// Casts primitive-type input vectors to Json type.
template <
    TypeKind kind,
    typename std::enable_if<TypeTraits<kind>::isPrimitiveType, int>::type = 0>
void castToJson(
    const BaseVector& input,
    exec::EvalCtx* /*context*/,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  using T = typename TypeTraits<kind>::NativeType;

  // input is guaranteed to be in flat or constant encodings when passed in.
  auto inputVector = input.as<SimpleVector<T>>();

  std::string result;
  rows.applyToSelected([&](auto row) {
    if (inputVector->isNullAt(row)) {
      flatResult.set(row, "null");
    } else {
      result.clear();
      generateJsonTyped(*inputVector, row, result);

      flatResult.set(row, StringView{result});
    }
  });
}

// Forward declaration.
void castToJsonFromArray(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

void castToJsonFromMap(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

void castToJsonFromRow(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult);

// Casts complex-type input vectors to Json type.
template <
    TypeKind kind,
    typename std::enable_if<!TypeTraits<kind>::isPrimitiveType, int>::type = 0>
void castToJson(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  if constexpr (kind == TypeKind::ARRAY) {
    castToJsonFromArray(input, context, rows, flatResult);
  } else if constexpr (kind == TypeKind::MAP) {
    castToJsonFromMap(input, context, rows, flatResult);
  } else if constexpr (kind == TypeKind::ROW) {
    castToJsonFromRow(input, context, rows, flatResult);
  } else {
    VELOX_FAIL(
        "Casting from {} to JSON is not supported.", input.type()->toString());
  }
}

// Helper struct representing the Json vector of input. If throwIfNull is
// true, throws an error when input is null at any of the selected rows.
struct AsJson {
  AsJson(
      exec::EvalCtx* context,
      const VectorPtr& input,
      const SelectivityVector& rows,
      bool throwIfNull = false)
      : decoded_(context, *input, rows) {
    // Translates the selected rows of input into the corresponding rows of the
    // base of the decoded input.
    exec::LocalSelectivityVector baseRows(
        context->execCtx(), decoded_->base()->size());
    baseRows->clearAll();
    rows.applyToSelected(
        [&](auto row) { baseRows->setValid(decoded_->index(row), true); });
    baseRows->updateBounds();

    if (throwIfNull) {
      baseRows->applyToSelected([&](auto row) {
        if (decoded_->base()->isNullAt(row)) {
          VELOX_FAIL("Unexpected null value in input vector.");
        }
      });
    }

    BaseVector::ensureWritable(*baseRows, JSON(), context->pool(), &json_);
    jsonStrings_ = json_->as<FlatVector<StringView>>();

    VELOX_DYNAMIC_TYPE_DISPATCH(
        castToJson,
        input->typeKind(),
        *decoded_->base(),
        context,
        *baseRows,
        *jsonStrings_);
  }

  StringView at(vector_size_t i) {
    return jsonStrings_->valueAt(decoded_->index(i));
  }

  exec::LocalDecodedVector decoded_;
  VectorPtr json_;
  FlatVector<StringView>* jsonStrings_;
};

void castToJsonFromArray(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guaranteed to be in flat encoding when passed in.
  auto inputArray = input.as<ArrayVector>();

  auto elements = inputArray->elements();
  auto elementsRows =
      functions::toElementRows(elements->size(), rows, inputArray);
  AsJson elementsAsJson{context, elements, elementsRows};

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all elements Json strings and the
  // delimiters to be added.
  size_t elementsStringSize = 0;
  rows.applyToSelected([&](auto row) {
    if (inputArray->isNullAt(row)) {
      // "null" will be inlined in the StringView.
      return;
    }

    auto offset = inputArray->offsetAt(row);
    auto size = inputArray->sizeAt(row);
    for (auto i = offset, end = offset + size; i < end; ++i) {
      elementsStringSize += elementsAsJson.at(i).size();
    }

    // Extra length for commas and brackets.
    elementsStringSize += size > 0 ? size + 1 : 2;
  });

  flatResult.getBufferWithSpace(elementsStringSize);

  // Constructs the Json string of each array from Json strings of its elements.
  rows.applyToSelected([&](auto row) {
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
      proxy.append(elementsAsJson.at(i));
    }
    proxy.append("]"_sv);

    proxy.finalize();
  });
}

void castToJsonFromMap(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guaranteed to be in flat encoding when passed in.
  auto inputMap = input.as<MapVector>();

  auto mapKeys = inputMap->mapKeys();
  auto mapValues = inputMap->mapValues();
  auto elementsRows = functions::toElementRows(mapKeys->size(), rows, inputMap);

  VELOX_CHECK_EQ(
      mapKeys->typeKind(),
      TypeKind::VARCHAR,
      "The type of map keys must be VARCHAR.");

  AsJson keysAsJson{context, mapKeys, elementsRows, true};
  AsJson valuesAsJson{context, mapValues, elementsRows};

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all elements Json strings and the
  // delimiters to be added.
  size_t elementsStringSize = 0;
  rows.applyToSelected([&](auto row) {
    if (inputMap->isNullAt(row)) {
      // "null" will be inlined in the StringView.
      return;
    }

    auto offset = inputMap->offsetAt(row);
    auto size = inputMap->sizeAt(row);
    for (auto i = offset, end = offset + size; i < end; ++i) {
      elementsStringSize += keysAsJson.at(i).size() + valuesAsJson.at(i).size();
    }

    // Extra length for commas, semicolons, and curly braces.
    elementsStringSize += size > 0 ? size * 2 + 1 : 2;
  });

  flatResult.getBufferWithSpace(elementsStringSize);

  // Constructs the Json string of each map from Json strings of its keys and
  // values.
  rows.applyToSelected([&](auto row) {
    if (inputMap->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto offset = inputMap->offsetAt(row);
    auto size = inputMap->sizeAt(row);

    auto proxy = exec::StringWriter<>(&flatResult, row);

    proxy.append("{"_sv);
    for (int i = offset, end = offset + size; i < end; ++i) {
      if (i > offset) {
        proxy.append(","_sv);
      }
      proxy.append(keysAsJson.at(i));
      proxy.append(":"_sv);
      proxy.append(valuesAsJson.at(i));
    }
    proxy.append("}"_sv);

    proxy.finalize();
  });
}

void castToJsonFromRow(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    FlatVector<StringView>& flatResult) {
  // input is guaranteed to be in flat encoding when passed in.
  auto inputRow = input.as<RowVector>();
  auto childrenSize = inputRow->childrenSize();

  // Estimates an upperbound of the total length of all Json strings for the
  // input according to the length of all children Json strings and the
  // delimiters to be added.
  size_t childrenStringSize = 0;
  std::vector<AsJson> childrenAsJson;
  for (int i = 0; i < childrenSize; ++i) {
    childrenAsJson.emplace_back(context, inputRow->childAt(i), rows);

    rows.applyToSelected([&](auto row) {
      if (inputRow->isNullAt(row)) {
        // "null" will be inlined in the StringView.
        return;
      }
      childrenStringSize += childrenAsJson[i].at(row).size();
    });
  }

  // Extra length for commas and brackets.
  childrenStringSize +=
      rows.countSelected() * (childrenSize > 0 ? childrenSize + 1 : 2);
  flatResult.getBufferWithSpace(childrenStringSize);

  // Constructs Json string of each row from Json strings of its children.
  rows.applyToSelected([&](auto row) {
    if (inputRow->isNullAt(row)) {
      flatResult.set(row, "null");
      return;
    }

    auto proxy = exec::StringWriter<>(&flatResult, row);

    proxy.append("["_sv);
    for (int i = 0; i < childrenSize; ++i) {
      if (i > 0) {
        proxy.append(","_sv);
      }
      proxy.append(childrenAsJson[i].at(row));
    }
    proxy.append("]"_sv);

    proxy.finalize();
  });
}

} // namespace

bool JsonCastOperator::isSupportedType(const TypePtr& other) const {
  switch (other->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DOUBLE:
    case TypeKind::REAL:
    case TypeKind::VARCHAR:
    case TypeKind::DATE:
    case TypeKind::TIMESTAMP:
      return true;
    case TypeKind::ARRAY:
      return isSupportedType(other->childAt(0));
    case TypeKind::ROW:
      for (auto& child : other->as<TypeKind::ROW>().children()) {
        if (!isSupportedType(child)) {
          return false;
        }
      }
      return true;
    case TypeKind::MAP:
      return (
          other->childAt(0)->kind() == TypeKind::VARCHAR &&
          isSupportedType(other->childAt(1)));
    default:
      return false;
  }
}

/// Converts an input vector of a supported type to Json type. The
/// implementation follows the structure below.
/// JsonOperator::castTo: type dispatch for castToJson
/// +- castToJson (simple types)
///    +- generateJsonTyped: appends actual data to string
/// +- castToJson (complex types, via SFINAE)
///    +- castToJsonFrom{Row, Map, Array}:
///         Generates data for child vectors in temporary vectors. Copies this
///         data and adds delimiters and separators.
///       +- castToJson (recursive)
void JsonCastOperator::castTo(
    const BaseVector& input,
    exec::EvalCtx* context,
    const SelectivityVector& rows,
    BaseVector& result) const {
  // result is guaranteed to be a flat writable vector.
  auto* flatResult = result.as<FlatVector<StringView>>();

  // Casting from VARBINARY is not supported and should have been rejected by
  // isSupportedType() in the caller.
  VELOX_DYNAMIC_TYPE_DISPATCH(
      castToJson, input.typeKind(), input, context, rows, *flatResult);
}

} // namespace facebook::velox

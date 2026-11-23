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
#include "velox/functions/sparksql/MapFromArrays.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/functions/sparksql/LastWin.h"
#include "velox/functions/sparksql/Map.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Can only take the fast path if keys and values have an equal number of arrays
// and the offsets and sizes of these arrays match 1:1. The map must be well
// formed for all elements, also ones not in 'rows' in apply(). This is because
// canonicalize() will touch all elements in any case.
bool canTakeFastPath(
    const ArrayVector& keys,
    const ArrayVector& values,
    const SelectivityVector& rows) {
  VELOX_CHECK_GE(keys.size(), rows.end());
  VELOX_CHECK_GE(values.size(), rows.end());
  if (keys.size() != values.size()) {
    return false;
  }
  for (vector_size_t row = 0; row < keys.size(); ++row) {
    if (keys.isNullAt(row)) {
      continue;
    }

    if (values.isNullAt(row)) {
      return false;
    }

    if (keys.offsetAt(row) != values.offsetAt(row) ||
        keys.sizeAt(row) != values.sizeAt(row)) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<exec::VectorFunction> makeMapFromArrays(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<MapFromArraysFunction>(
      config.throwExceptionOnDuplicateMapKeys());
}
} // namespace

void MapFromArraysFunction::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  VELOX_CHECK_EQ(args.size(), 2);

  exec::DecodedArgs decodedArgs(rows, args, context);
  auto* decodedKeys = decodedArgs.at(0);
  auto* decodedValues = decodedArgs.at(1);

  auto* keysArray = decodedKeys->base()->as<ArrayVector>();
  auto* valuesArray = decodedValues->base()->as<ArrayVector>();
  const auto& keysElements = keysArray->elements();
  const auto keyElementRows = toElementRows(
      keysElements->size(),
      rows,
      keysArray,
      decodedKeys->nulls(&rows),
      decodedKeys->indices());
  exec::LocalDecodedVector decodedKeyElements(
      context, *keysElements, keyElementRows);

  // Under EXCEPTION no row can shrink, so the result can reference the input
  // arrays instead of building entry indices.
  if (throwExceptionOnDuplicateMapKeys_ && decodedKeys->isIdentityMapping() &&
      decodedValues->isIdentityMapping() &&
      canTakeFastPath(*keysArray, *valuesArray, rows)) {
    exec::LocalSelectivityVector remainingRows(context, rows);
    checkNullAndDuplicateKeys(
        *remainingRows, context, *keysArray, *decodedKeyElements);
    context.deselectErrors(*remainingRows);

    auto mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        keysArray->nulls(),
        rows.end(),
        keysArray->offsets(),
        keysArray->sizes(),
        keysElements,
        valuesArray->elements());
    context.moveOrCopyResult(mapVector, *remainingRows, result);
    return;
  }

  exec::LocalSelectivityVector remainingRows(context, rows);
  auto mapVector = buildDeduplicatedMap(
      *remainingRows,
      context,
      rows.end(),
      outputType,
      *decodedKeys,
      *decodedValues,
      *decodedKeyElements,
      keysElements,
      valuesArray->elements(),
      throwExceptionOnDuplicateMapKeys_);
  context.deselectErrors(*remainingRows);
  context.moveOrCopyResult(mapVector, *remainingRows, result);
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
MapFromArraysFunction::signatures() {
  // array(K), array(V) -> map(K,V)
  return {
      exec::FunctionSignatureBuilder()
          .knownTypeVariable("K")
          .typeVariable("V")
          .returnType("map(K,V)")
          .argumentType("array(K)")
          .argumentType("array(V)")
          .build(),
  };
}

void registerMapFromArraysFunction(const std::string& name) {
  // Default null behavior applies: Spark's MapFromArrays is NullIntolerant, so
  // a null key or value array yields a null map.
  exec::registerStatefulVectorFunction(
      name, MapFromArraysFunction::signatures(), makeMapFromArrays);
}

} // namespace facebook::velox::functions::sparksql

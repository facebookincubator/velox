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
#include "velox/functions/sparksql/TransformKeys.h"

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/functions/sparksql/Map.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Implements Spark's transform_keys(map(K1,V), function(K1,V,K2)) ->
// map(K2,V). Pairs each key the lambda returns with the original value, in
// entry order. A repeated key raises DUPLICATED_MAP_KEY, or under LAST_WIN
// overwrites the value of its first occurrence in place, so the key keeps that
// first position.
//
// Default null behavior does not apply: the lambda may capture other columns,
// and a null capture does not make the result null. A null map still yields a
// null map.
class TransformKeysFunction : public exec::VectorFunction {
 public:
  explicit TransformKeysFunction(bool throwExceptionOnDuplicateMapKeys)
      : throwExceptionOnDuplicateMapKeys_{throwExceptionOnDuplicateMapKeys} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "transform_keys requires two arguments");

    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    auto flatMap = flattenMap(rows, args[0], *mapDecoder.get());

    const auto numKeys = flatMap->mapKeys()->size();
    const std::vector<VectorPtr> lambdaArgs = {
        flatMap->mapKeys(), flatMap->mapValues()};
    SelectivityVector validRowsInReusedResult =
        toElementRows<MapVector>(numKeys, rows, flatMap.get());

    VectorPtr transformedKeys;
    applyLambdaToElements<MapVector>(
        args[1],
        rows,
        numKeys,
        flatMap,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        transformedKeys);

    // A null map has no keys to check and stays null in the result.
    exec::LocalSelectivityVector remainingRows(context, rows);
    context.deselectErrors(*remainingRows);
    if (flatMap->mayHaveNulls()) {
      remainingRows->deselectNulls(
          flatMap->rawNulls(), rows.begin(), rows.end());
    }
    exec::LocalDecodedVector decodedKeys(
        context,
        *transformedKeys,
        toElementRows<MapVector>(numKeys, *remainingRows, flatMap.get()));

    // Under EXCEPTION no row can shrink, so the result keeps the offsets and
    // sizes of the input map.
    if (throwExceptionOnDuplicateMapKeys_) {
      checkNullAndDuplicateKeys(
          *remainingRows, context, *flatMap, *decodedKeys);
      context.deselectErrors(*remainingRows);

      auto mapVector = std::make_shared<MapVector>(
          context.pool(),
          outputType,
          addNullsForUnselectedRows(flatMap, *remainingRows),
          rows.end(),
          flatMap->offsets(),
          flatMap->sizes(),
          transformedKeys,
          flatMap->mapValues());
      context.moveOrCopyResult(mapVector, rows, result);
      return;
    }

    // The offsets and sizes of the map locate both the transformed keys and the
    // values.
    exec::LocalDecodedVector decodedFlatMap(context, *flatMap, *remainingRows);
    auto mapVector = buildDeduplicatedMap(
        *remainingRows,
        context,
        rows.end(),
        outputType,
        *decodedFlatMap,
        *decodedFlatMap,
        *decodedKeys,
        transformedKeys,
        flatMap->mapValues(),
        /*throwOnDuplicateKey=*/false);
    context.deselectErrors(*remainingRows);
    mapVector->setNulls(addNullsForUnselectedRows(flatMap, *remainingRows));
    context.moveOrCopyResult(mapVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map(K1,V), function(K1,V,K2) -> map(K2,V)
    // K2 is orderable, which rejects a key type that is or contains a map, as
    // Spark does.
    return {
        exec::FunctionSignatureBuilder()
            .typeVariable("K1")
            .orderableTypeVariable("K2")
            .typeVariable("V")
            .returnType("map(K2,V)")
            .argumentType("map(K1,V)")
            .argumentType("function(K1,V,K2)")
            .build(),
    };
  }

 private:
  const bool throwExceptionOnDuplicateMapKeys_;
};

std::shared_ptr<exec::VectorFunction> makeTransformKeys(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<TransformKeysFunction>(
      config.throwExceptionOnDuplicateMapKeys());
}
} // namespace

void registerTransformKeysFunction(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name,
      TransformKeysFunction::signatures(),
      makeTransformKeys,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build());
}

} // namespace facebook::velox::functions::sparksql

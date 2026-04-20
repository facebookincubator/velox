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
#include "velox/functions/lib/MapConcat.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/MapConcat.h"

namespace facebook::velox::functions {
namespace {

// See documentation at https://prestodb.io/docs/current/functions/map.html
class MapConcatFunction : public exec::VectorFunction {
 public:
  MapConcatFunction(bool emptyForNull, bool allowSingleArg)
      : emptyForNull_(emptyForNull), allowSingleArg_(allowSingleArg) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Ensure all input types and output type are of the same map type.
    const TypePtr& mapType = args[0]->type();
    VELOX_CHECK_EQ(mapType->kind(), TypeKind::MAP);
    for (const VectorPtr& arg : args) {
      VELOX_CHECK(mapType->kindEquals(arg->type()));
    }
    VELOX_CHECK(mapType->kindEquals(outputType));

    const auto numArgs = static_cast<int>(args.size());
    if (!allowSingleArg_) {
      VELOX_CHECK_GE(numArgs, 2);
    }

    exec::DecodedArgs decodedArgs(rows, args, context);

    // UNKNOWN key type means all maps must be empty.
    if (outputType->asMap().keyType()->kind() == TypeKind::UNKNOWN) {
      for (auto i = 0; i < numArgs; ++i) {
        auto* decoded = decodedArgs.at(i);
        auto* map = decoded->base()->as<MapVector>();
        rows.applyToSelected([&](vector_size_t row) {
          VELOX_CHECK_EQ(
              map->sizeAt(decoded->index(row)),
              0,
              "Map with UNKNOWN key type must be empty");
        });
      }
      auto emptyMap = std::make_shared<MapVector>(
          context.pool(),
          outputType,
          BufferPtr(nullptr),
          1,
          allocateOffsets(1, context.pool()),
          allocateSizes(1, context.pool()),
          BaseVector::create(outputType->asMap().keyType(), 0, context.pool()),
          BaseVector::create(
              outputType->asMap().valueType(), 0, context.pool()));
      auto constant = BaseVector::wrapInConstant(rows.end(), 0, emptyMap);
      context.moveOrCopyResult(constant, rows, result);
      return;
    }

    std::vector<DecodedVector*> inputs;
    inputs.reserve(numArgs);
    for (auto i = 0; i < numArgs; ++i) {
      inputs.push_back(decodedArgs.at(i));
    }

    MapConcatConfig config;
    config.emptyForNull = emptyForNull_;
    if (auto* ctx = context.execCtx()->queryCtx()) {
      config.throwOnDuplicateKeys =
          ctx->queryConfig().throwExceptionOnDuplicateMapKeys();
    }

    auto merged = mapConcat(context.pool(), outputType, inputs, rows, config);
    context.moveOrCopyResult(merged, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map(K,V), map(K,V), ... -> map(K,V)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("K")
                .typeVariable("V")
                .returnType("map(K,V)")
                .argumentType("map(K,V)")
                .variableArity("map(K,V)")
                .build()};
  }

 private:
  // When true, treat null map inputs as empty maps rather than propagating
  // null.
  const bool emptyForNull_;

  // When true, allow a single map argument (Spark semantics).
  const bool allowSingleArg_;
};
} // namespace

void registerMapConcatFunction(const std::string& name) {
  exec::registerVectorFunction(
      name,
      MapConcatFunction::signatures(),
      std::make_unique<MapConcatFunction>(
          /*emptyForNull=*/false, /*allowSingleArg=*/false));
}

void registerMapConcatAllowSingleArg(const std::string& name) {
  exec::registerVectorFunction(
      name,
      MapConcatFunction::signatures(),
      std::make_unique<MapConcatFunction>(
          /*emptyForNull=*/false, /*allowSingleArg=*/true));
}

void registerMapConcatEmptyNullsFunction(const std::string& name) {
  exec::registerVectorFunction(
      name,
      MapConcatFunction::signatures(),
      std::make_unique<MapConcatFunction>(
          /*emptyForNull=*/true, /*allowSingleArg=*/false),
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build());
}
} // namespace facebook::velox::functions

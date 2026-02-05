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

#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/TransformFunctionBase.h"

namespace facebook::velox::functions::sparksql {
namespace {

/// Spark's transform function supports both signatures:
/// 1. transform(array, x -> expr) - element only
/// 2. transform(array, (x, i) -> expr) - element + index (Spark-specific)
///
/// See Spark documentation:
/// https://spark.apache.org/docs/latest/api/sql/index.html#transform
class TransformFunction : public TransformFunctionBase {
 public:
  /// Returns both base signature and Spark-specific signature with index.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    auto sigs = TransformFunctionBase::signatures();
    // Add Spark-specific signature: array(T), function(T, integer, U) ->
    // array(U). Spark uses IntegerType (32-bit) for the index parameter.
    sigs.push_back(
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .typeVariable("U")
            .returnType("array(U)")
            .argumentType("array(T)")
            .argumentType("function(T, integer, U)")
            .build());
    return sigs;
  }

 private:
  // Adds index vector to lambda arguments if the lambda expects it.
  void addIndexVector(
      const std::vector<VectorPtr>& args,
      const ArrayVectorPtr& flatArray,
      vector_size_t numElements,
      exec::EvalCtx& context,
      std::vector<VectorPtr>& lambdaArgs) const override {
    // Check the lambda function type to see if it expects 2 input arguments.
    // function(T, U) has 2 children (input T, output U) -> 1 input arg.
    // function(T, integer, U) has 3 children (input T, index integer, output U)
    // -> 2 input args.
    if (args[1]->type()->size() == 3) {
      lambdaArgs.push_back(
          createIndexVector(flatArray, numElements, context.pool()));
    }
  }

  // Creates an index vector where each element contains its position within
  // its respective array. For example, if we have arrays [a, b] and [c, d, e],
  // the index vector will be [0, 1, 0, 1, 2].
  // Spark uses IntegerType (32-bit) for the index.
  static VectorPtr createIndexVector(
      const ArrayVectorPtr& flatArray,
      vector_size_t numElements,
      memory::MemoryPool* pool) {
    auto indexVector =
        BaseVector::create<FlatVector<int32_t>>(INTEGER(), numElements, pool);

    auto* rawOffsets = flatArray->rawOffsets();
    auto* rawSizes = flatArray->rawSizes();
    auto* rawNulls = flatArray->rawNulls();
    auto* rawIndices = indexVector->mutableRawValues();

    for (vector_size_t row = 0; row < flatArray->size(); ++row) {
      // Skip null arrays.
      if (rawNulls && bits::isBitNull(rawNulls, row)) {
        continue;
      }
      auto offset = rawOffsets[row];
      auto size = rawSizes[row];
      for (vector_size_t i = 0; i < size; ++i) {
        rawIndices[offset + i] = i;
      }
    }

    return indexVector;
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_spark_transform,
    TransformFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<TransformFunction>());

} // namespace facebook::velox::functions::sparksql

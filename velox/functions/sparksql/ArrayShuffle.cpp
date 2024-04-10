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
#include "velox/functions/lib/ArrayShuffle.h"

namespace facebook::velox::functions::sparksql {
namespace {

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("array(T)")
              .argumentType("array(T)")
              .constantArgumentType("bigint")
              .build()};
}

} // namespace

// Register function.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION_WITH_METADATA(
    udf_array_shuffle,
    signatures(),
    exec::VectorFunctionMetadataBuilder().deterministic(false).build(),
    [](const auto& /*name*/,
       const auto& inputs,
       const core::QueryConfig& config) {
      VELOX_USER_CHECK_EQ(inputs.size(), 2);
      VELOX_USER_CHECK_EQ(inputs[1].type->kind(), TypeKind::BIGINT);
      VELOX_USER_CHECK_NOT_NULL(inputs[1].constantValue);

      const auto seed =
          inputs[1]
              .constantValue->template as<ConstantVector<int64_t>>()
              ->valueAt(0);
      return std::make_shared<ArrayShuffleFunction>(
          seed + config.sparkPartitionId());
    });
} // namespace facebook::velox::functions::sparksql

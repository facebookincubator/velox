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
#include "velox/functions/lib/ArrayShuffleBase.h"

namespace facebook::velox::functions::sparksql {
namespace {
class ArrayShuffleFunction : public ArrayShuffleBaseFunction {
 public:
  explicit ArrayShuffleFunction(int64_t seed)
      : randGen_(std::make_shared<std::mt19937>(seed)) {}

  static std::shared_ptr<exec::VectorFunction> create(
      const std::string& /*name*/,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const core::QueryConfig& config) {
    VELOX_CHECK_GE(inputArgs.size(), 2);
    VELOX_CHECK_EQ(inputArgs[1].type->kind(), TypeKind::BIGINT);

    const auto seed = inputArgs[1]
                          .constantValue->template as<ConstantVector<int64_t>>()
                          ->valueAt(0);
    const int32_t partitionId = config.sparkPartitionId();

    return std::make_shared<ArrayShuffleFunction>(seed + partitionId);
  }

 protected:
  std::shared_ptr<std::mt19937> getRandGen() const override {
    return randGen_;
  }

 private:
  std::shared_ptr<std::mt19937> randGen_;
};

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
    ArrayShuffleFunction::create);
} // namespace facebook::velox::functions::sparksql

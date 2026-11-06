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

#include "velox/functions/sparksql/ArrayGetFunction.h"
#include "velox/functions/lib/SubscriptUtil.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {

namespace {

// get(array, index) and get_array_item(array, index) -> array[index]
//
// Both access the array at a 0-based index. Spark's get() never throws, and
// array[index] throws for a negative or out-of-bounds index when ANSI mode is
// enabled, so get_array_item picks the instance from the query config.
// - failOnError = false: returns NULL for a negative or out-of-bounds index.
// - failOnError = true: throws for a negative or out-of-bounds index.
template <bool failOnError>
class ArrayGetFunction : public SubscriptImpl<
                             /* allowNegativeIndices */ !failOnError,
                             /* nullOnNegativeIndices */ !failOnError,
                             /* allowOutOfBound */ !failOnError,
                             /* indexStartsAtOne */ false> {
 public:
  ArrayGetFunction()
      : SubscriptImpl<!failOnError, !failOnError, !failOnError, false>(false) {}

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;

    // array(T), tinyint|smallint|integer|bigint -> T
    for (const auto& indexType : {"tinyint", "smallint", "integer", "bigint"}) {
      signatures.push_back(
          exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("T")
              .argumentType("array(T)")
              .argumentType(indexType)
              .build());
    }

    return signatures;
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_get,
    ArrayGetFunction<false>::signatures(),
    std::make_unique<ArrayGetFunction<false>>());

void registerGetArrayItemFunction(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name,
      ArrayGetFunction<false>::signatures(),
      [](const std::string& /*name*/,
         const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
         const core::QueryConfig& config)
          -> std::shared_ptr<exec::VectorFunction> {
        static const auto kNullOnError =
            std::make_shared<ArrayGetFunction<false>>();
        static const auto kThrowOnError =
            std::make_shared<ArrayGetFunction<true>>();
        if (SparkQueryConfig{config}.ansiEnabled()) {
          return kThrowOnError;
        }
        return kNullOnError;
      });
}

} // namespace facebook::velox::functions::sparksql

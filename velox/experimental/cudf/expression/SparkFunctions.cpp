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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/expression/sparksql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::cudf_velox {

void registerSparkFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("integer")
           .argumentType("any")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});
}

} // namespace facebook::velox::cudf_velox

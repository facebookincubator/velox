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

#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::cudf_velox {

void registerPrestoFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "plus",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<prestosql::DatePlusIntervalFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<prestosql::DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .constantArgumentType("varchar")
           .argumentType("bigint")
           .argumentType("date")
           .build()},
      true,
      prestosql::DateAddFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_trunc",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateTruncFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .constantArgumentType("varchar")
           .argumentType("timestamp")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .constantArgumentType("varchar")
           .argumentType("date")
           .build()},
      true,
      DateTruncFunction::canEvaluate);
}

} // namespace facebook::velox::cudf_velox

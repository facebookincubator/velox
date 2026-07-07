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

#include "velox/experimental/cudf/expression/ArrayAccessFunctions.h"
#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

namespace facebook::velox::cudf_velox {

std::vector<exec::FunctionSignaturePtr> arrayAccessSignatures(
    std::initializer_list<const char*> indexTypes) {
  using exec::FunctionSignatureBuilder;

  std::vector<exec::FunctionSignaturePtr> signatures;
  signatures.reserve(indexTypes.size());
  for (const auto* indexType : indexTypes) {
    signatures.push_back(
        FunctionSignatureBuilder()
            .typeVariable("T")
            .returnType("T")
            .argumentType("array(T)")
            .argumentType(indexType)
            .build());
  }
  return signatures;
}

void registerArrayAccessFunction(
    const std::string& name,
    ArrayAccessPolicy policy,
    std::vector<exec::FunctionSignaturePtr> signatures) {
  registerCudfFunction(
      name,
      [policy](
          const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return makeArrayAccessFunction(expr, policy);
      },
      signatures);
}

} // namespace facebook::velox::cudf_velox

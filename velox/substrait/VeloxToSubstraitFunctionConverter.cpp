/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "VeloxToSubstraitFunctionConverter.h"
#include "velox/substrait/TypeUtils.h"

namespace facebook::velox::substrait {

const std::optional<std::shared_ptr<::substrait::Expression>>
VeloxToSubstraitScalarFunctionConverter::convert(
    const core::CallTypedExprPtr& callTypeExpr,
    google::protobuf::Arena& arena,
    const RowTypePtr& inputType) const {
  auto& functionName = callTypeExpr->name();

  if (functionSignatureMap_.find(functionName) == functionSignatureMap_.end()) {
    return std::nullopt;
  }

  if ("if" == functionName || "switch" == functionName) {
    return std::nullopt;
  }

  std::vector<std::shared_ptr<exec::TypeSignature>> typeSinatures;
  typeSinatures.reserve(callTypeExpr->inputs().size());
  for (const auto& arg : callTypeExpr->inputs()) {
    const auto typeSignature =
        std::make_shared<exec::TypeSignature>(typeToTypeSignature(arg->type()));
    typeSinatures.emplace_back(typeSignature);
  }

  auto* exprMessage =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
  auto substraitExpr = std::make_shared<::substrait::Expression>(*exprMessage);

  auto scalarExpr = substraitExpr->mutable_scalar_function();
  scalarExpr->set_function_reference(
      functionCollector_->getFunctionReference(callTypeExpr));

  for (auto& arg : callTypeExpr->inputs()) {
    scalarExpr->add_args()->MergeFrom(
        exprConvertorPtr_->toSubstraitExpr(arena, arg, inputType));
  }
  scalarExpr->mutable_output_type()->MergeFrom(
      typeConvertor_->toSubstraitType(arena, callTypeExpr->type()));

  return substraitExpr;
}

} // namespace facebook::velox::substrait
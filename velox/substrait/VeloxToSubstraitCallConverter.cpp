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

#include "VeloxToSubstraitCallConverter.h"

namespace facebook::velox::substrait {

const std::optional<::substrait::Expression*>
VeloxToSubstraitScalarFunctionConverter::convert(
    const core::CallTypedExprPtr& callTypeExpr,
    google::protobuf::Arena& arena,
    SubstraitExprConverter& topLevelConverter) const {
  auto& scalarFunctionOption =
      functionLookup_->lookupFunction(arena, callTypeExpr);

  if (!scalarFunctionOption.has_value()) {
    return std::nullopt;
  }
  auto* substraitExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
  auto scalarExpr = substraitExpr->mutable_scalar_function();
  scalarExpr->set_function_reference(
      functionCollector_->getFunctionReference(scalarFunctionOption.value()));

  for (auto& arg : callTypeExpr->inputs()) {
    scalarExpr->add_args()->MergeFrom(topLevelConverter(arg));
  }
  scalarExpr->mutable_output_type()->MergeFrom(
      typeConvertor_->toSubstraitType(arena, callTypeExpr->type()));

  return std::make_optional(substraitExpr);
}

} // namespace facebook::velox::substrait

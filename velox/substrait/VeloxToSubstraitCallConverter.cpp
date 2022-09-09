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

#include "velox/substrait/VeloxToSubstraitCallConverter.h"

namespace facebook::velox::substrait {

const std::optional<::substrait::Expression*>
VeloxToSubstraitIfThenConverter::convert(
    const core::CallTypedExprPtr& callTypeExpr,
    google::protobuf::Arena& arena,
    SubstraitExprConverter& topLevelConverter) const {
  if (callTypeExpr->name() == "if" && callTypeExpr->name() == "switch") {
    if (callTypeExpr->inputs().size() % 2 != 1) {
      VELOX_NYI(
          "Number of arguments are always going to be odd for if/then expression");
    }

    auto* substraitExpr =
        google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
    auto ifThenExpr = substraitExpr->mutable_if_then();

    auto last = callTypeExpr->inputs().size() - 1;
    for (int i = 0; i < last; i += 2) {
      auto ifClauseExpr = ifThenExpr->add_ifs();
      ifClauseExpr->mutable_if_()->MergeFrom(
          topLevelConverter(callTypeExpr->inputs().at(i)));

      ifClauseExpr->mutable_then()->MergeFrom(
          topLevelConverter(callTypeExpr->inputs().at(i + 1)));
    }
    ifThenExpr->mutable_else_()->MergeFrom(
        topLevelConverter(callTypeExpr->inputs().at(last)));
    return std::make_optional(substraitExpr);
  }
  return std::nullopt;
}

const std::optional<::substrait::Expression*>
VeloxToSubstraitScalarFunctionConverter::convert(
    const core::CallTypedExprPtr& callTypeExpr,
    google::protobuf::Arena& arena,
    SubstraitExprConverter& topLevelConverter) const {
  auto* substraitExpr =
      google::protobuf::Arena::CreateMessage<::substrait::Expression>(&arena);
  auto scalarExpr = substraitExpr->mutable_scalar_function();

  std::vector<TypePtr> arguments;
  arguments.reserve(callTypeExpr->inputs().size());
  for (const auto& input : callTypeExpr->inputs()) {
    arguments.emplace_back(input->type());
  }

  const auto& referenceId = extensionCollector_->getFunctionReference(
      callTypeExpr->name(), arguments);

  VELOX_CHECK(
      referenceId.has_value(),
      "Fail to get function reference for call typed expression, {}",
      callTypeExpr->toString());
  scalarExpr->set_function_reference(referenceId.value());

  for (auto& arg : callTypeExpr->inputs()) {
    const auto& message = topLevelConverter(arg);
    scalarExpr->add_args()->MergeFrom(message);
  }
  scalarExpr->mutable_output_type()->MergeFrom(
      typeConvertor_->toSubstraitType(arena, callTypeExpr->type()));

  return std::make_optional(substraitExpr);
}

} // namespace facebook::velox::substrait

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

#include "velox/expression/CastExpr.h"

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"
#include "velox/expression/PrestoCastKernel.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec {

TypePtr CastCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /* argTypes */) {
  VELOX_FAIL("CAST expressions do not support type resolution.");
}

ExprPtr CastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());
  const auto inputKind = compiledChildren[0]->type()->kind();
  if (type->kind() == TypeKind::VARBINARY &&
      (inputKind == TypeKind::TINYINT || inputKind == TypeKind::SMALLINT ||
       inputKind == TypeKind::INTEGER || inputKind == TypeKind::BIGINT)) {
    VELOX_UNSUPPORTED(
        "Cannot cast {} to VARBINARY.",
        compiledChildren[0]->type()->toString());
  }
  return std::make_shared<CastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      false,
      std::make_shared<PrestoCastKernel>(config));
}

TypePtr TryCastCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /* argTypes */) {
  VELOX_FAIL("TRY CAST expressions do not support type resolution.");
}

ExprPtr TryCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "TRY CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());
  return std::make_shared<CastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      true,
      std::make_shared<PrestoCastKernel>(config));
}
} // namespace facebook::velox::exec

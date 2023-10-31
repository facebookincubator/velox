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

#pragma once

#include "velox/expression/FunctionCallToSpecialForm.h"

namespace facebook::velox::functions::sparksql {
class DecimalRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  TypePtr resolveType(
      const std::vector<std::shared_ptr<const core::ITypedExpr>>& inputs)
      override;

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kRoundDecimal = "decimal_round";
};
} // namespace facebook::velox::functions::sparksql

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/expression/CastExpr.h"
#include "velox/functions/sparksql/specialforms/SparkCastHooks.h"

namespace facebook::velox::functions::sparksql {

class SparkCastExpr : public exec::CastExpr {
 public:
  SparkCastExpr(
      TypePtr type,
      exec::ExprPtr&& input,
      bool trackCpuUsage,
      bool isTryCast,
      std::shared_ptr<exec::CastHooks> hooks)
      : exec::CastExpr(
            std::move(type),
            std::move(input),
            trackCpuUsage,
            isTryCast,
            std::move(hooks)) {}
};

class SparkCastCallToSpecialForm : public exec::CastCallToSpecialForm {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

 private:
  /// Determines if ANSI mode is supported for casting from fromType to toType.
  /// TODO: Remove this function once all cast operations support ANSI mode.
  /// @param fromType The source type of the cast
  /// @param toType The target type of the cast
  /// @return true if ANSI mode is supported for this cast, false otherwise
  static bool isAnsiSupported(const TypePtr& fromType, const TypePtr& toType);
};

class SparkTryCastCallToSpecialForm : public exec::TryCastCallToSpecialForm {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;
};
} // namespace facebook::velox::functions::sparksql

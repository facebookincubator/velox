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

#include "velox/functions/sparksql/specialforms/SparkCastExpr.h"

#include "velox/expression/SpecialFormRegistry.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {
namespace {

bool isIntegralType(const TypePtr& type) {
  return type == TINYINT() || type == SMALLINT() || type == INTEGER() ||
      type == BIGINT();
}

exec::ExprPtr makeSparkCastExpr(
    const TypePtr& type,
    exec::ExprPtr&& input,
    bool trackCpuUsage,
    bool isTryCast,
    bool allowOverflow,
    const core::QueryConfig& config) {
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(input),
      trackCpuUsage,
      isTryCast,
      std::make_shared<SparkCastHooks>(config, allowOverflow));
}

// Per-expression ANSI cast special form. Spark cast support has four special
// forms:
// - `cast` uses ANSI behavior only when SparkQueryConfig::ansiEnabled() is
//   true and the cast pair is supported; otherwise it uses legacy behavior.
// - `try_cast` returns NULL on cast failures and disables overflow truncation.
// - This form forces ANSI behavior for supported cast pairs regardless of the
//   session ANSI setting. Unsupported cast pairs fall back to legacy behavior.
class SparkAnsiCastCallToSpecialForm : public exec::CastCallToSpecialForm {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override {
    VELOX_CHECK_EQ(
        compiledChildren.size(),
        1,
        "ANSI CAST statements expect exactly 1 argument, received {}.",
        compiledChildren.size());

    const auto& fromType = compiledChildren[0]->type();
    const bool isTryCast =
        !SparkCastCallToSpecialForm::isAnsiSupported(fromType, type);
    return makeSparkCastExpr(
        type,
        std::move(compiledChildren[0]),
        trackCpuUsage,
        isTryCast,
        isTryCast,
        config);
  }
};

// SparkLegacyCastCallToSpecialForm forces legacy behavior regardless of the
// session ANSI setting.
// Legacy behavior means cast failures return NULL and overflow truncation is
// allowed where Spark permits it.
class SparkLegacyCastCallToSpecialForm : public exec::CastCallToSpecialForm {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override {
    VELOX_CHECK_EQ(
        compiledChildren.size(),
        1,
        "LEGACY CAST statements expect exactly 1 argument, received {}.",
        compiledChildren.size());

    return makeSparkCastExpr(
        type,
        std::move(compiledChildren[0]),
        trackCpuUsage,
        true,
        true,
        config);
  }
};

} // namespace

bool SparkCastCallToSpecialForm::isAnsiSupported(
    const TypePtr& fromType,
    const TypePtr& toType) {
  if (fromType->isVarchar()) {
    if (toType->isBoolean() || toType->isDate() || toType->isDecimal()) {
      return true;
    }
    if (isIntegralType(toType)) {
      // For string-to-integral casts, ANSI mode rejects invalid formats (e.g.
      // decimal points) instead of returning NULL.
      return true;
    }
  }
  if (fromType->isTimestamp() && isIntegralType(toType)) {
    return true;
  }

  return false;
}

exec::ExprPtr SparkCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());

  const auto& fromType = compiledChildren[0]->type();

  // In Spark SQL (with ANSI mode off), both CAST and TRY_CAST behave like
  // Velox's try_cast, so we set 'isTryCast' to true when ANSI is disabled or
  // the specific cast operation doesn't support ANSI mode.
  const bool isTryCast = !SparkQueryConfig{config}.ansiEnabled() ||
      !isAnsiSupported(fromType, type);

  // For ANSI-supported casts, CAST mirrors TRY_CAST when ANSI is disabled.
  // The distinction is controlled by the 'allowOverflow' flag in
  // SparkCastHooks.
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      isTryCast,
      std::make_shared<SparkCastHooks>(config, isTryCast));
}

exec::ExprPtr SparkTryCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "TRY_CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());

  // TRY_CAST always uses allowOverflow=false to return NULL on cast failures.
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      true,
      std::make_shared<SparkCastHooks>(config, false));
}

void registerSparkCastModeSpecialForms() {
  exec::registerFunctionCallToSpecialForm(
      "spark_ansi_cast", std::make_unique<SparkAnsiCastCallToSpecialForm>());
  exec::registerFunctionCallToSpecialForm(
      "spark_legacy_cast",
      std::make_unique<SparkLegacyCastCallToSpecialForm>());
}

} // namespace facebook::velox::functions::sparksql

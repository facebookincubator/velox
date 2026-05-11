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

#include "velox/expression/CastExpr.h"
#include "velox/expression/SpecialForm.h"

namespace facebook::velox::exec {

/// NULLIF(a, b): returns NULL if a equals b, otherwise returns a.
///
/// The comparison casts both inputs to 'commonType' internally. The return type
/// is a's original type. Evaluates each input exactly once.
class NullIfExpr : public SpecialForm {
 public:
  /// @param inputs Exactly two inputs: the value and the comparand.
  /// @param castExpr Optional. Used to cast inputs to a common type for
  /// comparison. Must be provided when input types differ. When null, inputs
  /// must have the same type and are compared directly.
  NullIfExpr(
      std::vector<ExprPtr>&& inputs,
      std::shared_ptr<CastExpr> castExpr,
      bool trackCpuUsage);

  /// Creates a NullIfExpr, building a CastExpr with proper hooks when input
  /// types differ from commonType.
  static ExprPtr create(
      std::vector<ExprPtr>&& inputs,
      const TypePtr& commonType,
      bool trackCpuUsage,
      const core::QueryConfig& config);

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

 private:
  // NULLIF does not propagate nulls: NULLIF(NULL, x) returns NULL (from the
  // first argument), but NULLIF(x, NULL) returns x, not NULL.
  void computePropagatesNulls() override {
    propagatesNulls_ = false;
  }

  // Casts evaluated vectors to the common type for comparison. Null when both
  // input types already match (no casting needed).
  std::shared_ptr<CastExpr> castExpr_;

  // Whether each input needs to be cast to the common type for comparison.
  const bool needsCastA_;
  const bool needsCastB_;
};

} // namespace facebook::velox::exec

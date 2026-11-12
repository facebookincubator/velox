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

#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/expression/JitExpression.h"

#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox::test_utils {

/// Drops the AST and JIT evaluators below GPU SFI for its lifetime, so integral
/// arithmetic reaches GPU SFI; AST and JIT do not check for overflow.
class PreferGpuSfi {
 public:
  PreferGpuSfi()
      : registry_(
            (ensureBuiltinExpressionEvaluatorsRegistered(),
             getCudfExpressionEvaluatorRegistry())),
        ast_(registry_.at(kAstEvaluatorName)),
        jit_(registry_.at(kJitEvaluatorName)) {
    registry_.at(kAstEvaluatorName).priority = 0;
    registry_.at(kJitEvaluatorName).priority = 0;
  }

  ~PreferGpuSfi() {
    registry_.at(kAstEvaluatorName) = ast_;
    registry_.at(kJitEvaluatorName) = jit_;
  }

 private:
  std::unordered_map<std::string, CudfExpressionEvaluatorEntry>& registry_;
  const CudfExpressionEvaluatorEntry ast_;
  const CudfExpressionEvaluatorEntry jit_;
};

} // namespace facebook::velox::cudf_velox::test_utils

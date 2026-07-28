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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/CudfExpressionCompiler.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"

namespace facebook::velox::cudf_velox {

std::shared_ptr<CudfExpression> compile(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool) {
  const auto* best = findBestEvaluator(expr);
  VELOX_CHECK_NOT_NULL(
      best, "No cuDF expression evaluator can handle: {}", expr->toString());
  return best->create(expr, inputRowSchema, pool);
}

} // namespace facebook::velox::cudf_velox

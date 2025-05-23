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

#include <velox/common/memory/Memory.h>
#include <velox/expression/Expr.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/ComplexVector.h>
#include "velox4j/memory/MemoryManager.h"

namespace velox4j {
class Evaluator {
 public:
  Evaluator(MemoryManager* memoryManager, const std::string& exprJson);

  facebook::velox::VectorPtr eval(
      const facebook::velox::SelectivityVector& rows,
      const facebook::velox::RowVector& input);

 private:
  const std::string exprJson_;
  std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx_;
  std::unique_ptr<facebook::velox::core::ExpressionEvaluator> ee_;
  std::unique_ptr<facebook::velox::exec::ExprSet> exprSet_;
};
} // namespace velox4j

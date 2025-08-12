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
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::expression {

/// Flattens inputs to conjunct expressions 'AND' and 'OR'. Simplifies conjunct
/// expressions 'AND' and 'OR' to constant boolean expressions 'false' and
/// 'true' respectively if any of the inputs can be constant folded to 'false'
/// and 'true'. The rewrite also prunes inputs to conjunct expressions 'AND'
/// and 'OR' that can be constant folded to 'true' and 'false' respectively.
core::TypedExprPtr rewriteConjunctExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool);

} // namespace facebook::velox::expression

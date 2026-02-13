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

#include "velox/core/Expressions.h"

namespace facebook::velox::cudf_velox {

/// Operator/plan-node level predicate: can a group of expressions
/// (e.g. filter + projections) be evaluated by the CUDF operator.
bool canBeEvaluatedByCudf(const std::vector<core::TypedExprPtr>& exprs);

} // namespace facebook::velox::cudf_velox

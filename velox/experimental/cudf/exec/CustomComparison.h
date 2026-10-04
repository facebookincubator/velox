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
#include "velox/type/Type.h"

#include <vector>

namespace facebook::velox::cudf_velox {

/// Returns true if 'type' or any type nested in it provides a custom
/// comparison. cuDF receives only the physical representation of such values,
/// so hashing, equality and ordering on the GPU diverge from Velox semantics.
/// TIMESTAMP WITH TIME ZONE, for example, packs a zone key into the low bits
/// of the BIGINT cuDF sees, while Velox compares the UTC millis alone.
bool containsCustomComparison(const TypePtr& type);

/// Returns true if any of 'keys' has a type that contains a custom comparison.
/// Operators that hash, compare, sort or partition by such keys must stay on
/// the CPU.
bool keysUseCustomComparison(
    const std::vector<core::FieldAccessTypedExprPtr>& keys);

/// Returns true if 'expr' or any of its sub-expressions has a type that
/// contains a custom comparison. Covers join conditions, whose compared
/// columns are not exposed as keys.
bool exprUsesCustomComparison(const core::TypedExprPtr& expr);

} // namespace facebook::velox::cudf_velox

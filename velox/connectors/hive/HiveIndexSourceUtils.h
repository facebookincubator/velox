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

#include <optional>
#include <utility>
#include "velox/core/PlanNode.h"
#include "velox/type/Filter.h"
#include "velox/type/Variant.h"

namespace facebook::velox::connector::hive {

/// Extracts a constant value from a point lookup filter (single value
/// equality). Returns nullopt if the filter is not a point lookup or cannot be
/// converted.
std::optional<variant> extractPointLookupValue(const common::Filter* filter);

/// Extracts range bounds from a range filter. Returns a pair of (lower, upper)
/// inclusive variants suitable for creating a BetweenIndexLookupCondition.
/// Returns nullopt if the filter is not a range filter or cannot be converted.
std::optional<std::pair<variant, variant>> extractRangeBounds(
    const common::Filter* filter);

/// Creates an EqualIndexLookupCondition with a constant value.
core::IndexLookupConditionPtr createEqualConditionWithConstant(
    const std::string& columnName,
    const TypePtr& type,
    const variant& value);

/// Creates a BetweenIndexLookupCondition with constant bounds.
core::IndexLookupConditionPtr createBetweenConditionWithConstants(
    const std::string& columnName,
    const TypePtr& type,
    const variant& lowerValue,
    const variant& upperValue);

} // namespace facebook::velox::connector::hive

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

#include "velox/functions/lib/aggregates/MinMaxAggregateBase.h"

namespace facebook::velox::functions::aggregate::detail {

// Negative INF is the smallest value of floating point type.
template <>
const float MaxAggregate<float>::kInitialValue_ =
    -1 * MinMaxTrait<float>::infinity();

template <>
const double MaxAggregate<double>::kInitialValue_ =
    -1 * MinMaxTrait<double>::infinity();

// In velox, NaN is considered larger than infinity for floating point types.
template <>
const float MinAggregate<float>::kInitialValue_ =
    MinMaxTrait<float>::quiet_NaN();

template <>
const double MinAggregate<double>::kInitialValue_ =
    MinMaxTrait<double>::quiet_NaN();

} // namespace facebook::velox::functions::aggregate::detail

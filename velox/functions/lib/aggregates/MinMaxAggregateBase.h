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

#include "velox/exec/Aggregate.h"

namespace facebook::velox::functions::aggregate {

/// Min & Max functions in Presto and Spark have different semantics:
/// 1. Nested nulls are allowed in Spark but not Presto.
/// 2. The map type is not orderable in Spark.
/// 3. The timestamp type represents a time instant in microsecond precision in
/// Spark, but millis precision in Presto.
/// We add parameters 'nestedNullAllowed', 'mapTypeSupported',
/// 'useMillisPrecision' to register min and max functions with different
/// behaviors.
exec::AggregateFunctionFactory getMinFunctionFactory(
    const std::string& name,
    bool nestedNullAllowed,
    bool mapTypeSupported,
    bool useMillisPrecision);

exec::AggregateFunctionFactory getMaxFunctionFactory(
    const std::string& name,
    bool nestedNullAllowed,
    bool mapTypeSupported,
    bool useMillisPrecision);
} // namespace facebook::velox::functions::aggregate

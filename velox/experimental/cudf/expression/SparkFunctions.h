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
#include "velox/core/QueryConfig.h"

#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Returns true when an expression contains a Spark date_format call.
bool containsSparkDateFormat(
    const std::vector<core::TypedExprPtr>& expressions);

/// Returns true when an expression contains Spark date_format whose configured
/// CPU semantics cannot be reproduced by cuDF.
bool requiresCpuSparkDateFormat(
    const std::vector<core::TypedExprPtr>& expressions,
    const core::QueryConfig& queryConfig);

/// Register Spark-specific CUDF functions.
void registerSparkFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox

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

#include <folly/FixedString.h>

/// Error codes for Spark error conditions, as defined in Spark's
/// common/utils/src/main/resources/error/error-conditions.json. Each code is
/// "SPARK:<condition>". The "SPARK:" prefix tells them apart from the error
/// codes of other engines.
namespace facebook::velox::functions::sparksql::spark_error_code {

/// Spark error condition DIVIDE_BY_ZERO.
inline constexpr auto kDivideByZero =
    folly::makeFixedString("SPARK:DIVIDE_BY_ZERO");

} // namespace facebook::velox::functions::sparksql::spark_error_code

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

#include <folly/Likely.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::functions::sparksql {

/// Standard "throw under ANSI, return NULL otherwise" branch for Spark vector
/// functions whose row-processing helpers return std::optional<T>.
///
/// When `ansiEnabled` is true, throws a user error formatted from the trailing
/// fmt + args (forwarded to VELOX_USER_FAIL). When false, returns
/// `std::nullopt` from the *enclosing function*. The macro exits control flow
/// either way; the `RETURN_NULL_OR_FAIL` name is chosen so this is obvious at
/// the call site.
///
/// Usage:
///   if (hour < 0 || hour >= 24) {
///     VELOX_SPARK_RETURN_NULL_OR_FAIL(
///         ansiEnabled, "Invalid value for hour, must be in [0, 24): {}",
///         hour);
///   }
///
/// Caller contract:
///   - The enclosing function must have a return type convertible from
///     `std::nullopt` (e.g. std::optional<T>).
///   - Error messages should follow Velox convention: static description
///     first, runtime values at the end of the format string.
#define VELOX_SPARK_RETURN_NULL_OR_FAIL(ansiEnabled, /*fmt, args...*/...) \
  do {                                                                    \
    if (FOLLY_UNLIKELY(ansiEnabled)) {                                    \
      VELOX_USER_FAIL(__VA_ARGS__);                                       \
    }                                                                     \
    return std::nullopt;                                                  \
  } while (0)

} // namespace facebook::velox::functions::sparksql

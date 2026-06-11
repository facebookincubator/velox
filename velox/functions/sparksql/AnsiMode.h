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

#include <fmt/format.h>
#include <folly/Likely.h>

#include <optional>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::functions::sparksql {

/// Returns std::nullopt when ANSI mode is disabled; throws a user error
/// formatted from 'fmt' and 'args' when enabled. Use at validation sites in
/// Spark functions that return NULL on invalid input unless ANSI mode
/// requires an error:
///
///   if (hour < 0 || hour >= 24) {
///     return nullOrUserFail(
///         ansiEnabled, "Invalid value for hour, must be in [0, 24): {}",
///         hour);
///   }
///
/// Error messages should follow Velox convention: static description first,
/// runtime values at the end of the format string.
template <typename... Args>
std::nullopt_t nullOrUserFail(
    bool ansiEnabled,
    fmt::format_string<Args...> fmt,
    Args&&... args) {
  if (FOLLY_UNLIKELY(ansiEnabled)) {
    VELOX_USER_FAIL("{}", fmt::format(fmt, std::forward<Args>(args)...));
  }
  return std::nullopt;
}

} // namespace facebook::velox::functions::sparksql

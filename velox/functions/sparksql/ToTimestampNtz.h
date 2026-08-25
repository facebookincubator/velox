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

#include "velox/functions/sparksql/AnsiMode.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::functions::sparksql {

/// to_timestamp_ntz(timestamp_str). Ignores timezone suffixes, not
/// session-adjusted.
template <typename T>
struct ToTimestampNtzFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TimestampUtc>& result,
      const arg_type<Varchar>& input) {
    auto parsed = util::fromTimestampWithTimezoneString(
        input, util::TimestampParseMode::kSparkCast);
    if (parsed.hasError()) {
      nullOrUserFail(ansiEnabled_, "{}", parsed.error().message());
      return false;
    }
    result = parsed.value().timestamp;
    return true;
  }

 private:
  bool ansiEnabled_{false};
};

} // namespace facebook::velox::functions::sparksql

/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/common/vectorindex/CodecOptions.h"

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

// static
CodecOptions CodecOptions::parse(
    const std::vector<std::pair<std::string_view, std::string_view>>& options) {
  CodecOptions result;
  for (const auto& [key, value] : options) {
    VELOX_USER_CHECK(
        key == kMaxFlatClusters, "Unknown vector index codec option: {}", key);
    const auto parsed = folly::tryTo<int64_t>(value);
    VELOX_USER_CHECK(
        parsed.hasValue(),
        "Vector index codec option must be an integer: {}={}",
        key,
        value);
    VELOX_USER_CHECK_GT(
        parsed.value(),
        0,
        "Vector index codec option must be positive: {}",
        key);
    result.maxFlatClusters = parsed.value();
  }
  return result;
}

} // namespace facebook::velox::common::vectorindex

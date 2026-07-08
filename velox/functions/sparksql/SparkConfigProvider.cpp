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
#include "velox/functions/sparksql/SparkConfigProvider.h"

#include <folly/Conv.h>
#include "velox/common/base/Exceptions.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {

std::vector<config::ConfigProperty> SparkConfigProvider::properties() const {
  return SparkQueryConfig::registeredProperties();
}

std::string SparkConfigProvider::normalize(
    std::string_view name,
    std::string_view value) const {
  if (name == SparkQueryConfig::kPartitionId) {
    const auto parsed = folly::tryTo<int32_t>(value);
    VELOX_USER_CHECK(
        parsed.hasValue(), "Invalid Spark partition id: {}", value);
    VELOX_USER_CHECK_GE(
        parsed.value(), 0, "Spark partition id must be non-negative.");
  }
  return std::string{value};
}

} // namespace facebook::velox::functions::sparksql

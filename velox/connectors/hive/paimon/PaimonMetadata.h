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

#include <folly/dynamic.h>
#include <limits>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::paimon {

/// Wire integers are signed JSON int64 values. Validate before narrowing or
/// converting to an unsigned size; unknown optional values are JSON null.
inline int64_t paimonInt(
    const folly::dynamic& obj,
    const char* field,
    int64_t min = 0,
    int64_t max = std::numeric_limits<int64_t>::max()) {
  const auto* value = obj.get_ptr(field);
  VELOX_USER_CHECK(
      value && value->isInt(), "Paimon '{}' must be an int64", field);
  const auto number = value->asInt();
  VELOX_USER_CHECK(
      number >= min && number <= max,
      "Paimon '{}' out of range [{}, {}]: {}",
      field,
      min,
      max,
      number);
  return number;
}

} // namespace facebook::velox::connector::hive::paimon

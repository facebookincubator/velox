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

#include <string_view>
#include <utility>

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::dwrf {

struct DwrfRuntimeStats {
  /// Count of string dictionary values that were flattened during reading.
  inline static constexpr std::string_view kFlattenStringDictionaryValues =
      "flattenStringDictionaryValues";

  /// Describes the flatten-string-dictionary runtime metric.
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kFlattenStringDictionaryValuesMetric = {
          kFlattenStringDictionaryValues,
          RuntimeCounter::Unit::kNone};
};

} // namespace facebook::velox::dwrf

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <unordered_map>
#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::exec {

struct DriverStats {
  static constexpr std::string_view kTotalPauseTime =
      "totalDriverPauseWallNanos";
  static constexpr std::string_view kTotalOffThreadTime =
      "totalDriverOffThreadWallNanos";

  /// Number of silent Velox throws during operator execution.
  static constexpr std::string_view kNumSilentThrow = "numSilentThrow";
  /// Time an operator spent queued before execution.
  static constexpr std::string_view kQueuedWallNanos = "queuedWallNanos";
  /// Number of dynamic filters accepted by an operator.
  static constexpr std::string_view kDynamicFiltersAccepted =
      "dynamicFiltersAccepted";
  /// Number of dynamic filters produced by an operator.
  static constexpr std::string_view kDynamicFiltersProduced =
      "dynamicFiltersProduced";

  std::unordered_map<std::string, RuntimeMetric> runtimeStats;
};

} // namespace facebook::velox::exec

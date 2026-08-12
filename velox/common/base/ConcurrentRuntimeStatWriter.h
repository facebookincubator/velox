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

#include <folly/Synchronized.h>
#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox {

/// Accumulates runtime metrics by name behind a lock, and exposes a snapshot.
/// Unlike writers that forward each sample to an owning operator, this one owns
/// the values, so any number of threads may record into it concurrently.
class ConcurrentRuntimeStatWriter final : public BaseRuntimeStatWriter {
 public:
  /// Merges 'value' under 'name'. All samples under a name must share the same
  /// unit; a mismatched unit throws.
  void addRuntimeStat(std::string_view name, const RuntimeCounter& value)
      override;

  /// Replaces any existing metric under 'name'.
  void setRuntimeStat(std::string_view name, const RuntimeMetric& metric)
      override;

  /// Returns a snapshot of the accumulated metrics.
  std::unordered_map<std::string, RuntimeMetric> runtimeStats() const;

  /// Drops all accumulated metrics.
  void clear();

 private:
  // All samples under a name share one unit; addRuntimeStat enforces it.
  folly::Synchronized<std::unordered_map<std::string, RuntimeMetric>>
      runtimeStats_;
};

} // namespace facebook::velox

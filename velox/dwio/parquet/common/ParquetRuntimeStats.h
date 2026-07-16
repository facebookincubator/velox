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

namespace facebook::velox::parquet {

struct ParquetRuntimeStats {
  /// Time spent loading Parquet pages in nanoseconds.
  inline static constexpr std::string_view kPageLoadTimeNs = "pageLoadTimeNanos";

  /// Describes the page-load-time runtime metric.
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kPageLoadTimeNsMetric = {kPageLoadTimeNs, RuntimeCounter::Unit::kNanos};

  /// Estimated memory used by the deserialized Parquet footer in bytes.
  inline static constexpr std::string_view kFooterEstimatedBytes =
      "footerEstimatedBytes";

  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kFooterEstimatedBytesMetric = {
          kFooterEstimatedBytes,
          RuntimeCounter::Unit::kBytes};
};

} // namespace facebook::velox::parquet

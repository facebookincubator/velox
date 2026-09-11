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

/// Names and descriptors for Parquet-specific runtime metrics.
struct ParquetRuntimeStats {
  /// Time spent loading Parquet pages in nanoseconds.
  inline static constexpr std::string_view kPageLoadTimeNs =
      "pageLoadTimeNanos";
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kPageLoadTimeNsMetric = {kPageLoadTimeNs, RuntimeCounter::Unit::kNanos};

  /// Number of pages skipped without reading or decompressing page data.
  inline static constexpr std::string_view kSkippedPages = "skippedPages";
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kSkippedPagesMetric = {kSkippedPages, RuntimeCounter::Unit::kNone};

  /// Number of pages whose data was read and processed.
  inline static constexpr std::string_view kProcessedPages = "processedPages";
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kProcessedPagesMetric = {kProcessedPages, RuntimeCounter::Unit::kNone};

  /// Estimated memory used by the deserialized Parquet footer in bytes.
  inline static constexpr std::string_view kFooterEstimatedBytes =
      "footerEstimatedBytes";
  inline static constexpr std::pair<std::string_view, RuntimeCounter::Unit>
      kFooterEstimatedBytesMetric = {
          kFooterEstimatedBytes,
          RuntimeCounter::Unit::kBytes};
};

} // namespace facebook::velox::parquet

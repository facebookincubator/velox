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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::exec::trace {

/// Defines the shared constants used by query trace implementation.
struct TraceTraits {
  static constexpr std::string_view kPlanNodeKey = "planNode";
  static constexpr std::string_view kQueryConfigKey = "queryConfig";
  static constexpr std::string_view kConnectorPropertiesKey =
      "connectorProperties";

  static constexpr std::string_view kTaskMetaFileName = "task_trace_meta.json";
};

struct OperatorTraceTraits {
  static constexpr std::string_view kSummaryFileName = "op_trace_summary.json";
  static constexpr std::string_view kInputFileName = "op_input_trace.data";
  static constexpr std::string_view kSplitFileName = "op_split_trace.split";

  /// Keys for operator trace summary file.
  static constexpr std::string_view kOpTypeKey = "opType";
  static constexpr std::string_view kPeakMemoryKey = "peakMemory";
  static constexpr std::string_view kInputRowsKey = "inputRows";
  static constexpr std::string_view kInputBytesKey = "inputBytes";
  static constexpr std::string_view kRawInputRowsKey = "rawInputRows";
  static constexpr std::string_view kRawInputBytesKey = "rawInputBytes";
  static constexpr std::string_view kNumSplitsKey = "numSplits";
};

/// Contains the summary of an operator trace.
struct OperatorTraceSummary {
  std::string opType;
  /// The number of splits processed by a table scan operator, nullopt for the
  /// other operator types.
  std::optional<uint32_t> numSplits{std::nullopt};

  uint64_t inputRows{0};
  uint64_t inputBytes{0};
  uint64_t rawInputRows{0};
  uint64_t rawInputBytes{0};
  uint64_t peakMemory{0};

  std::string toString() const;
};

} // namespace facebook::velox::exec::trace

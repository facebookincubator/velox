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
#include <vector>

#include "velox/core/PlanNode.h"

namespace facebook::velox::exec {

/// Transport-neutral statistics for one destination of a task's output buffer.
struct DestinationBufferStats {
  /// True once all data for this destination has been delivered.
  bool finished{false};

  /// Bytes / rows / pages currently buffered for this destination.
  int64_t bytesBuffered{0};
  int64_t rowsBuffered{0};
  int64_t pagesBuffered{0};

  /// Bytes / rows / pages already sent to this destination.
  int64_t bytesSent{0};
  int64_t rowsSent{0};
  int64_t pagesSent{0};
};

/// Transport-neutral statistics for a task's output buffer. Returned by
/// OutputBufferManager::stats(); each concrete manager maps its own accounting
/// onto it. Counts are in bytes, rows, and pages, where a "page" is one unit of
/// output data (a serialized page for the in-memory transport, a packed-columns
/// block for a GPU/UCX transport), so the type does not commit to any
/// transport's data model.
struct OutputBufferStats {
  core::PartitionedOutputNode::Kind kind;

  /// State of the output buffer.
  bool noMoreBuffers{false};
  bool noMoreData{false};
  bool finished{false};

  /// Bytes / pages currently buffered across all destinations.
  int64_t bufferedBytes{0};
  int64_t bufferedPages{0};

  /// Total bytes / rows / pages sent across all destinations.
  int64_t totalBytesSent{0};
  int64_t totalRowsSent{0};
  int64_t totalPagesSent{0};

  /// Average time a unit of data stays buffered, in milliseconds.
  int64_t averageBufferTimeMs{0};

  /// Number of destinations that together hold 80% of the data. A skew signal;
  /// 0 if the transport does not track it.
  int32_t numTopBuffers{0};

  /// Per-destination stats.
  std::vector<DestinationBufferStats> buffersStats;
};

} // namespace facebook::velox::exec

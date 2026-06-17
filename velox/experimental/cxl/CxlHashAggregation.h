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

#include "velox/exec/HashAggregation.h"

namespace facebook::velox::cxl {

/// HashAggregation that builds its group table in a CXL-backed CxlHashTable.
/// Under memory pressure it relocates the table from DRAM to the CXL pool
/// instead of spilling, and falls back to the inherited spill path when the CXL
/// pool is exhausted.
class CxlHashAggregation : public exec::HashAggregation {
 public:
  CxlHashAggregation(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode);

  void initialize() override;

  // Emits the CXL-relocated groups first, then delegates to HashAggregation for
  // the DRAM-resident groups. GroupingSet's own output drains only the DRAM row
  // container, so the relocated groups must be drained here.
  RowVectorPtr getOutput() override;

  bool canReclaim() const override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

 private:
  // Output iteration over the CXL row container, drained before the DRAM rows.
  exec::RowContainerIterator cxlIterator_;
  bool cxlDrained_{false};
};

/// Prototype diagnostics for tests (process-wide). Not a stable API.
int64_t numCxlHashAggregationsInitialized();
int64_t numCxlHashAggregationsWithCxlPool();
int64_t numCxlPartitionsMigrated();

/// Resets the diagnostics above.
void resetCxlHashAggregationCounters();

} // namespace facebook::velox::cxl

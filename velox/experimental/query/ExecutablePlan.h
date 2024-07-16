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

#include "velox/exec/Task.h"

namespace facebook::velox::exec {

/// Describes an exchange source for an ExchangeNode a non-leaf stage.
struct InputStage {
  core::PlanNodeId consumer;
  std::string producerTaskPrefix;
};

/// Describes a fragment of a distributed plan. This allows a run
/// time to distribute fragments across workers and to set up
/// exchanges. A complete plan is a vector of these with the last
/// being the fragment that gathers results from the complete
/// plan. Different runtimes, e.g. local, streaming or
/// materialized shuffle can use this to describe exchange
/// parallel execution. Decisions on number of workers, location
/// of workers and mode of exchange are up to the runtime.
struct ExecutableFragment {
  std::string taskPrefix;
  int32_t width;
  velox::core::PlanFragment fragment;
  std::vector<InputStage> inputStages;
  std::vector<std::shared_ptr<const core::TableScanNode>> scans;
  int32_t numBroadcastDestinations{0};
};

/// Describes options for generating an executable plan.
struct ExecutablePlanOptions {
  // Maximum Number of independent Tasks for one stage of execution. If 1, there
  // are no exchanges.
  int32_t numWorkers;

  // Number of threads in a fragment in a worker. If 1, there are no local
  // exchanges.
  int32_t numDrivers;

  // If true, prevents planning of local aggregation.
  bool noLocalAggregation{false};
};

} // namespace facebook::velox::exec

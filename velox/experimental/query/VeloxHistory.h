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

#include "velox/exec/TaskStats.h"
#include "velox/experimental/query/ExecutablePlan.h"
#include "velox/experimental/query/Plan.h"

namespace facebook::verax {

class VeloxHistory : public History {
 public:
  virtual std::optional<Cost> findCost(RelationOp& op) override {
    return std::nullopt;
  }

  void recordCost(const RelationOp& op, Cost cost) override {}

  bool setLeafSelectivity(BaseTable& table) override;

  /// Stores observed costs and cardinalities from a query execution. If 'op' is
  /// non-null, non-leaf costs from non-leaf levels are recorded. Otherwise only
  /// leaf scan selectivities  are recorded.
  void recordVeloxExecution(
      const RelationOp* op,
      const std::vector<velox::exec::ExecutableFragment>& plan,
      const std::vector<velox::exec::TaskStats>& stats);
};

} // namespace facebook::verax

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

#include "velox/experimental/query/VeloxHistory.h"
#include "velox/connectors/hive/TableHandle.h"
namespace facebook::verax {

using namespace facebook::velox::exec;

bool VeloxHistory::setLeafSelectivity(BaseTable& table) {
  auto optimization = queryCtx()->optimization();
  auto& handle = *dynamic_cast<velox::connector::hive::HiveTableHandle*>(
      optimization->leafHandle(table.id()).get());
  if (handle.subfieldFilters().empty() && !handle.remainingFilter()) {
    table.filterSelectivity = 1;
    return true;
  }
  auto string = handle.toString();
  auto it = leafSelectivities_.find(string);
  if (it != leafSelectivities_.end()) {
    table.filterSelectivity = it->second;
    return true;
  }
  table.filterSelectivity = 0.1;
  return false;
}

void VeloxHistory::recordVeloxExecution(
    const RelationOp* op,
    const std::vector<ExecutableFragment>& plan,
    const std::vector<velox::exec::TaskStats>& stats) {
  std::unordered_map<std::string, const velox::exec::OperatorStats*> map;
  for (auto& task : stats) {
    for (auto& pipeline : task.pipelineStats) {
      for (auto& op : pipeline.operatorStats) {
        map[op.planNodeId] = &op;
      }
    }
  }
  for (auto& fragment : plan) {
    for (auto& scan : fragment.scans) {
      auto scanStats = map[scan->id()];
      std::string handle = scan->tableHandle()->toString();
      recordLeafSelectivity(
          handle,
          scanStats->outputPositions /
              std::max<float>(1, scanStats->rawInputPositions));
    }
  }
}

} // namespace facebook::verax

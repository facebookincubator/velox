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

#include <string>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::exec::test {

/// Contains a deserialized Velox query plan and optional data files keyed on
/// source plan node ID. Structurally identical to TpchPlan; works with
/// AssertQueryBuilder (plan + optional splits).
struct VeloxPlan {
  core::PlanNodePtr plan;
  std::unordered_map<core::PlanNodeId, std::vector<std::string>> dataFiles;
  dwio::common::FileFormat dataFileFormat{dwio::common::FileFormat::PARQUET};
};

/// Backward-compatible alias.
using TpcdsPlan = VeloxPlan;

/// Generic loader for Velox plan JSON files (e.g. dumped from a Presto
/// worker's plan-dump-dir). Not specific to any benchmark suite -- works for
/// TPC-DS, TPC-H, or any arbitrary single-node query plan.
///
/// Plan directory layout (for loadPlanByQueryId):
/// - Canonical names Q1.json, Q2.json, ... Q99.json.
///
/// When stripPartitionedOutput is true (default), the loader strips the
/// PartitionedOutput root node that Presto always adds for distributed
/// shuffle. This makes the plan runnable locally via AssertQueryBuilder /
/// TaskCursor.
class VeloxPlanLoader {
 public:
  /// @param planDirectory  Directory containing plan JSON files.
  /// @param pool           Memory pool for plan deserialization (creates one
  ///                       if nullptr).
  /// @param stripPartitionedOutput  If true, strip PartitionedOutput root.
  VeloxPlanLoader(
      const std::string& planDirectory,
      memory::MemoryPool* pool = nullptr,
      bool stripPartitionedOutput = true);

  /// Resolve a plan directory from either an environment variable or a
  /// supplied default. If envVarName is non-empty and the env var is set, its
  /// value is used; otherwise defaultDir is returned.
  static std::string resolvePlanDirectory(
      const std::string& defaultDir,
      const std::string& envVarName = "");

  /// Load a plan from a specific file path.
  VeloxPlan loadPlan(const std::string& path) const;

  /// Load plan by query ID (1..99). Looks for planDir/Q{id}.json.
  VeloxPlan loadPlanByQueryId(int queryId) const;

  /// Collects all TableScan plan nodes in the plan tree.
  static std::vector<core::TableScanNodePtr> collectTableScanNodes(
      const core::PlanNodePtr& plan);

 private:
  std::string planDir_;
  memory::MemoryPool* pool_;
  std::shared_ptr<memory::MemoryPool> ownedPool_;
  bool stripPartitionedOutput_;

  std::string pathForQuery(int queryId) const;

  /// If stripPartitionedOutput_ is true and the plan root is a
  /// PartitionedOutputNode, replace it with its single child.
  void maybeStripPartitionedOutput(core::PlanNodePtr& plan) const;
};

/// Backward-compatible alias.
using TpcdsPlanFromJson = VeloxPlanLoader;

} // namespace facebook::velox::exec::test

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

#include <folly/Optional.h>
#include <string>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

/// Contains the query plan and optional data files keyed on source plan node
/// ID. Mirrors TpchPlan for use with AssertQueryBuilder (plan + optional
/// splits).
struct TpcdsPlan {
  core::PlanNodePtr plan;
  std::unordered_map<core::PlanNodeId, std::vector<std::string>> dataFiles;
  dwio::common::FileFormat dataFileFormat{dwio::common::FileFormat::PARQUET};
};

/// Loads TPC-DS query plans from pre-dumped Velox plan JSON files (e.g. from
/// Presto worker plan-dump-dir). No hand-written getQ1Plan()...getQ99Plan();
/// plans are loaded by query ID from the filesystem.
///
/// Plan directory layout:
/// - Option A: Canonical names q1.json, q2.json, ... q99.json.
/// - Option B: If q{id}.json is missing, a manifest.json file can map query ID
///   to filename (e.g. task-ID named files).
/// - Option C (Q1): For query 1, a Q1/ subdirectory can hold all plan
/// fragments;
///   all *.json in planDir/Q1/ are read (getQueryPlanFragments(1) returns all).
///
/// Plan directory can be set via constructor or env TPCDS_PLAN_DIR.
class TpcdsPlanFromJson {
 public:
  /// @param planDirectory  Directory containing plan JSON files (q1.json, ...)
  ///                       and optionally manifest.json.
  /// @param pool           Memory pool for plan deserialization.
  TpcdsPlanFromJson(
      const std::string& planDirectory,
      memory::MemoryPool* pool = nullptr);

  /// Uses TPCDS_PLAN_DIR env var if set; otherwise planDirectory passed to
  /// constructor.
  static std::string resolvePlanDirectory(const std::string& planDirectory);

  /// Load plan for TPC-DS query 1..99. Throws on missing file or parse error.
  /// For Q1, when planDir/Q1/ exists, returns the first plan from that
  /// subdirectory (see getQueryPlanFragments(1) for all fragments).
  TpcdsPlan getQueryPlan(int queryId) const;

  /// Collects all TableScan plan nodes in the plan tree. Use when building
  /// splits for AssertQueryBuilder (e.g. from a data path layout).
  static std::vector<core::TableScanNodePtr> collectTableScanNodes(
      const core::PlanNodePtr& plan);

 private:
  std::string planDir_;
  memory::MemoryPool* pool_;
  std::shared_ptr<memory::MemoryPool> ownedPool_;
  std::string pathForQuery(int queryId) const;
  TpcdsPlan loadPlanFromPath(const std::string& path) const;
};

} // namespace facebook::velox::exec::test

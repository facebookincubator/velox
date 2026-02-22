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

#include "velox/benchmarks/QueryBenchmarkBase.h"
#include "velox/exec/tests/utils/TpcdsQueryBuilder.h"

DECLARE_string(plan_path);

class TpcdsBenchmark : public facebook::velox::QueryBenchmarkBase {
 public:
  void initialize() override;

  void shutdown() override;

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats) override;

  void runQuery(int32_t queryId);

  /// Override to stamp splits with the plan's connector ID (e.g. "hive")
  /// instead of the default "test-hive" used by the base class.
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

 protected:
  /// Override to create a different query builder (e.g. CudfTpcdsQueryBuilder).
  virtual void initQueryBuilder();

  std::unique_ptr<facebook::velox::exec::test::TpcdsQueryBuilder> queryBuilder_;
  std::unordered_map<std::string, std::string> queryConfigs_;
  std::string planDir_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
};

extern std::unique_ptr<TpcdsBenchmark> tpcdsBenchmark;

void tpcdsBenchmarkMain();

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
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

class TpchBenchmark : public facebook::velox::QueryBenchmarkBase {
 public:
  void initialize() override;

  void shutdown() override;

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats) override;

  void runQuery(int32_t queryId) {
    const auto planContext = queryBuilder_->getQueryPlan(queryId);
    run(planContext, queryConfigs_);
  }

 protected:
  std::unordered_map<std::string, std::string> queryConfigs_;

 private:
  void initQueryBuilder();

  std::shared_ptr<facebook::velox::exec::test::TpchQueryBuilder> queryBuilder_;
};

extern std::unique_ptr<TpchBenchmark> benchmark;

void tpchBenchmarkMain();

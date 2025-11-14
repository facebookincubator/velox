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

#include "velox/benchmarks/tpch/TpchBenchmark.h"

#include <memory>
#include <string>
#include <vector>

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

  void shutdown() override;
};

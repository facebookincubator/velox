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

#include "velox/connectors/hive/benchmarks/IcebergInsertBenchmark.h"

namespace facebook::velox::connector::hive::benchmark {

class IcebergTransformBenchmark : public Benchmark {
 public:
  IcebergTransformBenchmark() {}

  BenchmarkStats runBenchmark(
      const TypePtr& dataType,
      connector::hive::iceberg::TransformType transformType,
      std::optional<int32_t> parameter,
      uint32_t numRows);
};

void run(
    unsigned int iters,
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows,
    folly::UserCounters& counters);

} // namespace facebook::velox::connector::hive::benchmark

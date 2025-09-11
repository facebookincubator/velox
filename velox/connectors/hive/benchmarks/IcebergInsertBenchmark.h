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

#include "velox/connectors/hive/benchmarks/Benchmark.h"

namespace facebook::velox::connector::hive::benchmark {

class IcebergInsertBenchmark : public Benchmark {
 public:
  IcebergInsertBenchmark() : Benchmark() {}

  BenchmarkStats runBenchmark(
      const TypePtr& dataType,
      connector::hive::iceberg::TransformType transformType,
      std::optional<int32_t> parameter,
      uint32_t numRows);

 private:
  void addIcebergColumnHandles(
      const RowTypePtr& rowType_,
      const std::vector<PartitionField>& partitionFields,
      std::vector<
          std::shared_ptr<const connector::hive::iceberg::IcebergColumnHandle>>&
          columnHandles);

  std::shared_ptr<connector::hive::iceberg::IcebergDataSink>
  createIcebergDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields);

  void writeWithIcebergDataSink(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<PartitionField>& partitionFields);
};

void run(
    unsigned int iters,
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows,
    folly::UserCounters& counters);

} // namespace facebook::velox::connector::hive::benchmark

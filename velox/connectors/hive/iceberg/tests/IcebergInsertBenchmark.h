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

#include <folly/Benchmark.h>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::iceberg::insert::test {

struct PartitionField {
  int32_t id;
  connector::hive::iceberg::TransformType type;
  std::optional<int32_t> parameter;
};

struct BenchmarkStats {
  std::chrono::microseconds duration;
  int64_t memoryUsedMB;
  int64_t peakMemoryMB;
};

class IcebergInsertBenchmark {
 public:
  IcebergInsertBenchmark();
  ~IcebergInsertBenchmark();

  BenchmarkStats runBenchmark(
      const TypePtr& dataType,
      connector::hive::iceberg::TransformType transformType,
      std::optional<int32_t> parameter,
      uint32_t numRows);

 private:
  void setUp();
  void tearDown();

  std::vector<RowVectorPtr> createTestData(
      int32_t numBatches,
      vector_size_t rowsPerBatch);

  std::shared_ptr<connector::hive::iceberg::IcebergDataSink>
  createIcebergDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields);

  void writeWithIcebergDataSink(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<PartitionField>& partitionFields);

  std::shared_ptr<exec::test::TempDirectoryPath> testDir_;
  RowTypePtr rowType_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;
  std::shared_ptr<connector::hive::HiveConfig> connectorConfig_;
  std::unique_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<VectorFuzzer> fuzzer_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

void run(
    unsigned int iters,
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows,
    folly::UserCounters& counters);

} // namespace facebook::velox::iceberg::insert::test

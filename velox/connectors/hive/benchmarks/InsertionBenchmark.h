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
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::connector::hive::insert::test {

struct BenchmarkStats {
  std::chrono::microseconds duration;
  int64_t memoryUsedMB;
  int64_t peakMemoryMB;
  int64_t filesWritten;
  int64_t rowsWritten;
};

struct ComparisonStats {
  BenchmarkStats hiveStats;
  BenchmarkStats icebergStats;
  double hiveToIcebergRatio;
};

struct PartitionField {
  int32_t id;
  connector::hive::iceberg::TransformType type;
  std::optional<int32_t> parameter;
};

class HiveInsertBenchmark {
 public:
  RowTypePtr rowType_;
  HiveInsertBenchmark();
  ~HiveInsertBenchmark();

  ComparisonStats runComparisonBenchmark(
      const TypePtr& dataType,
      uint32_t numRows);

  std::vector<RowVectorPtr> createTestDataWithSeed(
      const TypePtr& dataType,
      int32_t numBatches,
      vector_size_t rowsPerBatch,
      bool seed);

  BenchmarkStats writeWithHiveDataSink(
      const std::vector<RowVectorPtr>& batches);

  BenchmarkStats writeWithIcebergDataSink(
      const std::vector<RowVectorPtr>& batches,
      const PartitionField& field);

 private:
  void setUp();
  void tearDown();

  std::shared_ptr<HiveDataSink> createHiveDataSink(
      const std::string& outputDirectoryPath);

  std::shared_ptr<hive::iceberg::IcebergDataSink> createIcebergDataSink(
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields);

  std::shared_ptr<exec::test::TempDirectoryPath> testDir_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;
  std::shared_ptr<HiveConfig> connectorConfig_;
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

void runHive(unsigned int iters, const TypePtr& dataType, uint32_t numRows);

void runIceberg(
    unsigned int iters,
    const TypePtr& dataType,
    iceberg::TransformType transformType,
    std::optional<int32_t> param,
    uint32_t numRows,
    bool limitedRange = true);

void runHiveOnly(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters);

void runIcebergOnly(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters);

void runComparison(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters);

} // namespace facebook::velox::connector::hive::insert::test

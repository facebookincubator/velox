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

#include "velox/connectors/hive/benchmarks/IcebergTransformBenchmark.h"

#include "velox/connectors/hive/iceberg/IcebergPartitionIdGenerator.h"
#include "velox/connectors/hive/iceberg/TransformFactory.h"

namespace facebook::velox::connector::hive::benchmark {

constexpr int32_t kMaxPartitions = 1024;

BenchmarkStats IcebergTransformBenchmark::runBenchmark(
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows) {
  rowType_ = ROW(
      {{"partition_col", dataType},
       {"id", BIGINT()},
       {"image", VARBINARY()},
       {"yes", BOOLEAN()},
       {"level", SMALLINT()},
       {"days", INTEGER()},
       {"date", DATE()},
       {"data", VARCHAR()},
       {"timestamp", TIMESTAMP()},
       {"price", DOUBLE()},
       {"ask", REAL()}});

  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;
  auto initialMemory = rootPool_->usedBytes();

  auto batches =
      createTestData(dataType, numBatches, rowsPerBatch, kMaxPartitions);
  std::vector<PartitionField> partitionFields;
  partitionFields.push_back({0, transformType, parameter});

  std::vector<connector::hive::iceberg::IcebergPartitionSpec::Field> specFields;
  specFields.reserve(partitionFields.size());
  for (const auto& f : partitionFields) {
    specFields.push_back(
        {rowType_->nameOf(f.id), rowType_->childAt(f.id), f.type, f.parameter});
  }
  auto transforms = connector::hive::iceberg::parsePartitionTransformSpecs(
      specFields, opPool_.get());
  std::vector<column_index_t> partitionChannels;
  partitionChannels.reserve(partitionFields.size());
  for (const auto& f : partitionFields) {
    partitionChannels.push_back(f.id);
  }
  connector::hive::iceberg::IcebergPartitionIdGenerator generator(
      partitionChannels, kMaxPartitions, opPool_.get(), transforms, true);

  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& batch : batches) {
    raw_vector<uint64_t> ids;
    generator.run(batch, ids);
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto finalMemory = rootPool_->usedBytes();
  auto peakMemory = rootPool_->peakBytes();

  return BenchmarkStats{
      duration,
      (finalMemory - initialMemory) / (1024 * 1024),
      peakMemory / (1024 * 1024)};
}

void run(
    unsigned int iters,
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows,
    folly::UserCounters& counters) {
  IcebergTransformBenchmark benchmark;
  auto stats =
      benchmark.runBenchmark(dataType, transformType, parameter, numRows);
  counters["Elapsed (Milli)"] = stats.duration.count();
  counters["MemoryMB"] = stats.memoryUsedMB;
  counters["PeakMB"] = stats.peakMemoryMB;
}

} // namespace facebook::velox::connector::hive::benchmark

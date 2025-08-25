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
#include "velox/connectors/hive/iceberg/tests/IcebergInsertBenchmark.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/type/Type.h"

namespace facebook::velox::iceberg::insert::test {

IcebergInsertBenchmark::IcebergInsertBenchmark() {
  setUp();
}

IcebergInsertBenchmark::~IcebergInsertBenchmark() {
  tearDown();
}

void IcebergInsertBenchmark::setUp() {
  filesystems::registerLocalFileSystem();
  parquet::registerParquetWriterFactory();
  dwio::common::registerFileSinks();
  Type::registerSerDe();

  testDir_ = exec::test::TempDirectoryPath::create();
  rootPool_ =
      memory::memoryManager()->addRootPool("IcebergInsertBenchmark", 1L << 30);
  opPool_ = rootPool_->addLeafChild("operator");
  connectorPool_ = rootPool_->addAggregateChild("connector");
  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);
  connectorConfig_ = std::make_shared<connector::hive::HiveConfig>(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
      opPool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      nullptr,
      common::PrefixSortConfig(),
      nullptr,
      nullptr,
      "queryBenchmark",
      "taskBenchmark",
      "planNodeBenchmark",
      0,
      "");

  VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = 100;
  fuzzerOptions.nullRatio = 0.0;
  fuzzerOptions.stringVariableLength = true;
  fuzzer_ = std::make_unique<VectorFuzzer>(fuzzerOptions, opPool_.get());
  vectorMaker_ = std::make_unique<velox::test::VectorMaker>(opPool_.get());
}

void IcebergInsertBenchmark::tearDown() {
  vectorMaker_.reset();
  fuzzer_.reset();
  connectorQueryCtx_.reset();
  connectorPool_.reset();
  opPool_.reset();
  rootPool_.reset();
}

std::vector<RowVectorPtr> IcebergInsertBenchmark::createTestData(
    int32_t numBatches,
    vector_size_t rowsPerBatch) {
  std::vector<RowVectorPtr> batches;
  batches.reserve(numBatches);
  for (auto i = 0; i < numBatches; ++i) {
    auto batch = fuzzer_->fuzzRow(rowType_, rowsPerBatch);
    batches.push_back(batch);
  }

  return batches;
}

void addColumnHandles(
    const RowTypePtr& rowType,
    const std::vector<PartitionField>& partitionFields,
    std::vector<
        std::shared_ptr<const connector::hive::iceberg::IcebergColumnHandle>>&
        columnHandles) {
  std::unordered_set<int32_t> partitionColumnIds;
  for (const auto& field : partitionFields) {
    partitionColumnIds.insert(field.id);
  }
  connector::hive::HiveColumnHandle::ColumnParseParameters
      columnParseParameters;

  std::function<connector::hive::iceberg::IcebergNestedField(
      const TypePtr&, int32_t&)>
      collectNestedField = [&](const TypePtr& type, int32_t& columnOrdinal)
      -> connector::hive::iceberg::IcebergNestedField {
    int32_t currentId = columnOrdinal++;
    std::vector<connector::hive::iceberg::IcebergNestedField> children;
    if (type->isRow()) {
      auto rowType = asRowType(type);
      for (auto i = 0; i < rowType->size(); ++i) {
        children.push_back(
            collectNestedField(rowType->childAt(i), columnOrdinal));
      }
    } else if (type->isArray()) {
      auto arrayType = std::dynamic_pointer_cast<const ArrayType>(type);
      for (auto i = 0; i < arrayType->size(); ++i) {
        children.push_back(
            collectNestedField(arrayType->childAt(i), columnOrdinal));
      }
    } else if (type->isMap()) {
      auto mapType = std::dynamic_pointer_cast<const MapType>(type);
      for (auto i = 0; i < mapType->size(); ++i) {
        children.push_back(
            collectNestedField(mapType->childAt(i), columnOrdinal));
      }
    }

    return connector::hive::iceberg::IcebergNestedField{currentId, children};
  };

  int32_t startIndex = 1;
  for (auto i = 0; i < rowType->size(); ++i) {
    auto columnName = rowType->nameOf(i);
    auto type = rowType->childAt(i);
    auto field = collectNestedField(type, startIndex);
    columnHandles.push_back(
        std::make_shared<connector::hive::iceberg::IcebergColumnHandle>(
            columnName,
            partitionColumnIds.count(i) > 0
                ? connector::hive::HiveColumnHandle::ColumnType::kPartitionKey
                : connector::hive::HiveColumnHandle::ColumnType::kRegular,
            type,
            type,
            field,
            std::vector<common::Subfield>{},
            columnParseParameters));
  }
}

std::shared_ptr<connector::hive::iceberg::IcebergDataSink>
IcebergInsertBenchmark::createIcebergDataSink(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields) {
  std::vector<
      std::shared_ptr<const connector::hive::iceberg::IcebergColumnHandle>>
      columnHandles;
  addColumnHandles(rowType, partitionFields, columnHandles);

  auto locationHandle = std::make_shared<connector::hive::LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      connector::hive::LocationHandle::TableType::kNew);

  auto partitionSpec =
      std::make_shared<connector::hive::iceberg::IcebergPartitionSpec>(
          1,
          std::vector<connector::hive::iceberg::IcebergPartitionSpec::Field>{});

  auto tableHandle =
      std::make_shared<connector::hive::iceberg::IcebergInsertTableHandle>(
          columnHandles,
          locationHandle,
          partitionSpec,
          opPool_.get(),
          dwio::common::FileFormat::PARQUET,
          std::vector<connector::hive::iceberg::IcebergSortingColumn>{},
          common::CompressionKind::CompressionKind_ZSTD);

  return std::make_shared<connector::hive::iceberg::IcebergDataSink>(
      rowType,
      tableHandle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      connectorConfig_);
}

void IcebergInsertBenchmark::writeWithIcebergDataSink(
    const std::vector<RowVectorPtr>& batches,
    const std::vector<PartitionField>& partitionFields) {
  auto dataSink =
      createIcebergDataSink(rowType_, testDir_->getPath(), partitionFields);

  for (const auto& batch : batches) {
    dataSink->appendData(batch);
  }

  dataSink->finish();
  dataSink->close();
}

BenchmarkStats IcebergInsertBenchmark::runBenchmark(
    const TypePtr& dataType,
    connector::hive::iceberg::TransformType transformType,
    std::optional<int32_t> parameter,
    uint32_t numRows) {
  rowType_ = ROW(
      {{"id", BIGINT()},
       {"image", VARBINARY()},
       {"yes", BOOLEAN()},
       {"level", SMALLINT()},
       {"days", INTEGER()},
       {"date", DATE()},
       {"partition_col", dataType},
       {"data", VARCHAR()},
       {"timestamp", TIMESTAMP()},
       {"price", DOUBLE()},
       {"ask", REAL()}});

  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;
  auto initialMemory = rootPool_->usedBytes();

  auto batches = createTestData(numBatches, rowsPerBatch);
  std::vector<PartitionField> partitionFields;
  partitionFields.push_back({1, transformType, parameter});

  auto start = std::chrono::high_resolution_clock::now();
  writeWithIcebergDataSink(batches, partitionFields);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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
  IcebergInsertBenchmark benchmark;
  auto stats =
      benchmark.runBenchmark(dataType, transformType, parameter, numRows);
  counters["Elapsed"] = stats.duration.count() / 1000;
  counters["MemoryMB"] = stats.memoryUsedMB;
  counters["PeakMB"] = stats.peakMemoryMB;
}

} // namespace facebook::velox::iceberg::insert::test

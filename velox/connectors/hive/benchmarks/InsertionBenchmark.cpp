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
#include "velox/connectors/hive/benchmarks/InsertionBenchmark.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::insert::test {

constexpr int32_t kMaxValues = 128;

HiveInsertBenchmark::HiveInsertBenchmark() {
  setUp();
}

HiveInsertBenchmark::~HiveInsertBenchmark() {
  tearDown();
}

void HiveInsertBenchmark::setUp() {
  filesystems::registerLocalFileSystem();
  parquet::registerParquetWriterFactory();
  dwrf::registerDwrfWriterFactory();
  dwio::common::registerFileSinks();
  Type::registerSerDe();

  testDir_ = exec::test::TempDirectoryPath::create();
  rootPool_ = memory::memoryManager()->addRootPool("InsertBenchmark", 1L << 30);
  opPool_ = rootPool_->addLeafChild("operator");
  connectorPool_ = rootPool_->addAggregateChild("connector");
  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);
  connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  connectorQueryCtx_ = std::make_unique<ConnectorQueryCtx>(
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

  rowType_ = ROW(
      {{"id", BIGINT()},
       {"image", VARBINARY()},
       {"yes", BOOLEAN()},
       {"level", SMALLINT()},
       {"days", INTEGER()},
       {"date", DATE()},
       {"partition_col", VARCHAR()},
       {"data", VARCHAR()},
       {"timestamp", TIMESTAMP()},
       {"price", DOUBLE()},
       {"ask", REAL()}});

  vectorMaker_ = std::make_unique<velox::test::VectorMaker>(opPool_.get());
}

void HiveInsertBenchmark::tearDown() {
  vectorMaker_.reset();
  connectorQueryCtx_.reset();
  connectorPool_.reset();
  opPool_.reset();
  rootPool_.reset();
}

class LimitedUniqueGenerator : public AbstractInputGenerator {
 public:
  LimitedUniqueGenerator(const TypePtr& targetType, int maxUniqueValues)
      : AbstractInputGenerator(0, targetType, nullptr, 0.0),
        maxUniqueValues_(maxUniqueValues) {}

  variant generate() override {
    auto uniqueValue =
        std::uniform_int_distribution<int32_t>(1, maxUniqueValues_)(rng_);
    switch (type_->kind()) {
      case TypeKind::INTEGER:
        return variant(static_cast<int32_t>(uniqueValue));
      case TypeKind::BIGINT:
        return variant(static_cast<int64_t>(uniqueValue));
      case TypeKind::SMALLINT:
        return variant(static_cast<int16_t>(uniqueValue));
      case TypeKind::VARCHAR:
        return variant(fmt::format("string_0000_1111_2222_{}", uniqueValue));
      case TypeKind::VARBINARY: {
        std::string v(fmt::format("binary_0000_1111_2222_{}", uniqueValue));
        return variant::binary(v);
      }
      case TypeKind::BOOLEAN:
        return variant((uniqueValue % 2) == 0);
      case TypeKind::HUGEINT:
        return variant(static_cast<int128_t>(uniqueValue));
      default:
        return variant(static_cast<int32_t>(uniqueValue));
    }
  }

 private:
  int maxUniqueValues_;
};

std::vector<RowVectorPtr> HiveInsertBenchmark::createTestDataWithSeed(
    const TypePtr& dataType,
    int32_t numBatches,
    vector_size_t rowsPerBatch,
    bool limitedRange) {
  VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = rowsPerBatch;
  fuzzerOptions.nullRatio = 0.0;
  fuzzerOptions.stringVariableLength = true;
  auto seededFuzzer =
      std::make_unique<VectorFuzzer>(fuzzerOptions, opPool_.get());
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

  std::vector<RowVectorPtr> batches;
  batches.reserve(numBatches);

  std::shared_ptr<LimitedUniqueGenerator> generator = nullptr;
  if (limitedRange) {
    generator = make_shared<LimitedUniqueGenerator>(dataType, kMaxValues);
  }
  std::vector<AbstractInputGeneratorPtr> generators = {
      generator,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr};
  for (auto i = 0; i < numBatches; ++i) {
    auto batch =
        seededFuzzer->fuzzRow(rowType_, rowsPerBatch, false, generators);
    batches.push_back(batch);
  }
  return batches;
}

std::shared_ptr<HiveDataSink> HiveInsertBenchmark::createHiveDataSink(
    const std::string& outputDirectoryPath) {
  std::vector<std::shared_ptr<const HiveColumnHandle>> columnHandles;
  for (auto i = 0; i < rowType_->size(); ++i) {
    auto columnName = rowType_->nameOf(i);
    auto type = rowType_->childAt(i);

    columnHandles.push_back(std::make_shared<HiveColumnHandle>(
        columnName,
        i == 0 ? HiveColumnHandle::ColumnType::kPartitionKey
               : HiveColumnHandle::ColumnType::kRegular,
        type,
        type));
  }

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      LocationHandle::TableType::kNew);

  std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      128, // bucket count.
      std::vector<std::string>{"id"}, // bucketed by first column.
      std::vector<TypePtr>{rowType_->childAt(1)}, // type of first column.
      std::vector<std::shared_ptr<const HiveSortingColumn>>{}); // no sorting.

  auto tableHandle = std::make_shared<HiveInsertTableHandle>(
      columnHandles,
      locationHandle,
      dwio::common::FileFormat::PARQUET,
      nullptr,
      std::nullopt, // no compression.
      std::unordered_map<std::string, std::string>{},
      nullptr, // writerOptions.
      false, // ensureFiles.
      std::make_shared<const HiveInsertFileNameGenerator>());

  return std::make_shared<HiveDataSink>(
      rowType_,
      tableHandle,
      connectorQueryCtx_.get(),
      CommitStrategy::kNoCommit,
      connectorConfig_);
}

void addIcebergColumnHandles(
    const RowTypePtr& rowType_,
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
      auto rowType_ = asRowType(type);
      for (auto i = 0; i < rowType_->size(); ++i) {
        children.push_back(
            collectNestedField(rowType_->childAt(i), columnOrdinal));
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
  for (auto i = 0; i < rowType_->size(); ++i) {
    auto columnName = rowType_->nameOf(i);
    auto type = rowType_->childAt(i);
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
HiveInsertBenchmark::createIcebergDataSink(
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields) {
  std::vector<
      std::shared_ptr<const connector::hive::iceberg::IcebergColumnHandle>>
      columnHandles;
  addIcebergColumnHandles(rowType_, partitionFields, columnHandles);

  auto locationHandle = std::make_shared<connector::hive::LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      connector::hive::LocationHandle::TableType::kNew);

  // Create partition spec fields based on the partition field
  std::vector<connector::hive::iceberg::IcebergPartitionSpec::Field> specFields;
  for (const auto& field : partitionFields) {
    specFields.push_back(
        {rowType_->nameOf(0),
         rowType_->childAt(0),
         field.type,
         field.parameter});
  }

  auto partitionSpec =
      std::make_shared<connector::hive::iceberg::IcebergPartitionSpec>(
          1, specFields);

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
      rowType_,
      tableHandle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      connectorConfig_);
}

BenchmarkStats HiveInsertBenchmark::writeWithHiveDataSink(
    const std::vector<RowVectorPtr>& batches) {
  auto initialMemory = rootPool_->usedBytes();
  auto dataSink = createHiveDataSink(testDir_->getPath() + "/hive");
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& batch : batches) {
    dataSink->appendData(batch);
  }
  dataSink->finish();
  dataSink->close();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto finalMemory = rootPool_->usedBytes();
  auto peakMemory = rootPool_->peakBytes();

  int64_t totalRows = 0;
  for (const auto& batch : batches) {
    totalRows += batch->size();
  }

  return BenchmarkStats{
      duration,
      (finalMemory - initialMemory) / (1024 * 1024),
      peakMemory / (1024 * 1024),
      1,
      totalRows};
}

BenchmarkStats HiveInsertBenchmark::writeWithIcebergDataSink(
    const std::vector<RowVectorPtr>& batches,
    const PartitionField& field) {
  auto initialMemory = rootPool_->usedBytes();
  std::vector<PartitionField> partitionFields;
  partitionFields.push_back(field);
  auto dataSink =
      createIcebergDataSink(testDir_->getPath() + "/data", partitionFields);
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& batch : batches) {
    dataSink->appendData(batch);
  }
  dataSink->finish();
  dataSink->close();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto finalMemory = rootPool_->usedBytes();
  auto peakMemory = rootPool_->peakBytes();

  int64_t totalRows = 0;
  for (const auto& batch : batches) {
    totalRows += batch->size();
  }

  return BenchmarkStats{
      duration,
      (finalMemory - initialMemory) / (1024 * 1024),
      peakMemory / (1024 * 1024),
      1,
      totalRows};
}

ComparisonStats HiveInsertBenchmark::runComparisonBenchmark(
    const TypePtr& dataType,
    uint32_t numRows) {
  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;

  auto batches =
      createTestDataWithSeed(dataType, numBatches, rowsPerBatch, true);

  auto hiveStats = writeWithHiveDataSink(batches);

  PartitionField field{0, iceberg::TransformType::kBucket, 16};
  auto icebergStats = writeWithIcebergDataSink(batches, field);

  double ratio = static_cast<double>(hiveStats.duration.count()) /
      static_cast<double>(icebergStats.duration.count());

  return {hiveStats, icebergStats, ratio};
}

void runHiveCounter(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters) {
  HiveInsertBenchmark benchmark;
  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;

  auto batches = benchmark.createTestDataWithSeed(
      dataType, numBatches, rowsPerBatch, true);
  auto stats = benchmark.writeWithHiveDataSink(batches);

  counters["Elapsed"] = stats.duration.count() / 1000;
  counters["MemoryMB"] = stats.memoryUsedMB;
  counters["PeakMB"] = stats.peakMemoryMB;
  counters["RowsWritten"] = stats.rowsWritten;
}

void runIcebergCounter(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters) {
  HiveInsertBenchmark benchmark;
  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;

  auto batches = benchmark.createTestDataWithSeed(
      dataType, numBatches, rowsPerBatch, true);
  PartitionField field{6, iceberg::TransformType::kBucket, 16};
  auto stats = benchmark.writeWithIcebergDataSink(batches, field);

  counters["Elapsed"] = stats.duration.count() / 1000;
  counters["MemoryMB"] = stats.memoryUsedMB;
  counters["PeakMB"] = stats.peakMemoryMB;
  counters["RowsWritten"] = stats.rowsWritten;
}

void runHive(unsigned int iters, const TypePtr& dataType, uint32_t numRows) {
  HiveInsertBenchmark benchmark;
  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;

  auto batches = benchmark.createTestDataWithSeed(
      dataType, numBatches, rowsPerBatch, true);
  benchmark.writeWithHiveDataSink(batches);
}

void runIceberg(
    unsigned int iters,
    const TypePtr& dataType,
    iceberg::TransformType transformType,
    std::optional<int32_t> param,
    uint32_t numRows,
    bool limitedRange) {
  HiveInsertBenchmark benchmark;
  const uint32_t batchSize = 10'000;
  const uint32_t numBatches = (numRows + batchSize - 1) / batchSize;
  const uint32_t rowsPerBatch = numRows / numBatches;

  auto batches = benchmark.createTestDataWithSeed(
      dataType, numBatches, rowsPerBatch, limitedRange);

  PartitionField partitionField{0, transformType, param};
  benchmark.writeWithIcebergDataSink(batches, partitionField);
}

void runComparison(
    unsigned int iters,
    const TypePtr& dataType,
    uint32_t numRows,
    folly::UserCounters& counters) {
  HiveInsertBenchmark benchmark;
  auto stats = benchmark.runComparisonBenchmark(dataType, numRows);

  counters["HiveElapsed"] = stats.hiveStats.duration.count() / 1000;
  counters["IcebergElapsed"] = stats.icebergStats.duration.count() / 1000;
  counters["HiveMemoryMB"] = stats.hiveStats.memoryUsedMB;
  counters["IcebergMemoryMB"] = stats.icebergStats.memoryUsedMB;
  counters["HivePeakMB"] = stats.hiveStats.peakMemoryMB;
  counters["IcebergPeakMB"] = stats.icebergStats.peakMemoryMB;
  counters["HiveToIcebergRatio"] =
      stats.hiveToIcebergRatio * 100; // As percentage
}

} // namespace facebook::velox::connector::hive::insert::test

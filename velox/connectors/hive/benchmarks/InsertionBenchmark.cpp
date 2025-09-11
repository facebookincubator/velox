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
  // Set session properties. Configure max partitions per writers to avoid
  // throttling
  std::unordered_map<std::string, std::string> sessionProps = {
      {HiveConfig::kMaxPartitionsPerWritersSession, "1024"},
  };
  connectorSessionProperties_ =
      std::make_shared<config::ConfigBase>(std::move(sessionProps), true);
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

  auto tableHandle = std::make_shared<HiveInsertTableHandle>(
      columnHandles,
      locationHandle,
      dwio::common::FileFormat::PARQUET,
      std::make_shared<HiveBucketProperty>(
          HiveBucketProperty::Kind::kHiveCompatible,
          4, // bucket count.
          std::vector<std::string>{"id"}, // bucketed by first column.
          std::vector<TypePtr>{rowType_->childAt(1)}, // type of first column.
          std::vector<std::shared_ptr<const HiveSortingColumn>>{}), // no
                                                                    // sorting.,
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

} // namespace facebook::velox::connector::hive::insert::test

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
#include "velox/connectors/hive/paimon/PaimonConnector.h"

#include <folly/Conv.h>
#include <folly/json.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/TokenProvider.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/reader/ParquetReader.h"
#endif

namespace facebook::velox::connector::hive::paimon {
namespace {

static const std::string kPaimonConnectorId = "test-paimon";

class TestTokenProvider final : public filesystems::TokenProvider {
 public:
  bool equals(const filesystems::TokenProvider& other) const override {
    return this == &other;
  }
  size_t hash() const override {
    return 17;
  }
  std::shared_ptr<filesystems::AccessToken> getToken(
      const filesystems::AccessTokenKey&) const override {
    return nullptr;
  }
};

/// Opening a real file requires the expected per-file access context. This
/// catches dropped options on both the first and subsequent physical readers.
class AccessFileSystem final : public filesystems::FileSystem {
 public:
  AccessFileSystem() : FileSystem(nullptr) {}
  std::function<void(std::string_view, const filesystems::FileOptions&)> check;
  std::string name() const override {
    return "Paimon access test";
  }
  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view path,
      const filesystems::FileOptions& options) override {
    auto realPath = path.substr(std::string_view("paimon-access:").size());
    check(realPath, options);
    return filesystems::getFileSystem(realPath, nullptr)
        ->openFileForRead(realPath, options);
  }
  std::unique_ptr<WriteFile> openFileForWrite(
      std::string_view,
      const filesystems::FileOptions&) override {
    VELOX_UNSUPPORTED();
  }
  void remove(std::string_view) override {
    VELOX_UNSUPPORTED();
  }
  void rename(std::string_view, std::string_view, bool) override {
    VELOX_UNSUPPORTED();
  }
  bool exists(std::string_view) override {
    VELOX_UNSUPPORTED();
  }
  std::vector<std::string> list(std::string_view) override {
    VELOX_UNSUPPORTED();
  }
  void mkdir(std::string_view, const filesystems::DirectoryOptions&) override {
    VELOX_UNSUPPORTED();
  }
  void rmdir(std::string_view) override {
    VELOX_UNSUPPORTED();
  }
};

class PaimonConnectorTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    Type::registerSerDe();
    common::Filter::registerSerDe();
    core::ITypedExpr::registerSerDe();
    core::PlanNode::registerSerDe();
    PaimonConnector::registerSerDe();
#ifdef VELOX_ENABLE_PARQUET
    if (!dwio::common::hasReaderFactory(dwio::common::FileFormat::PARQUET)) {
      parquet::registerParquetReaderFactory();
    }
#endif
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    auto connector =
        PaimonConnectorFactory().newConnector(kPaimonConnectorId, config);
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
  }

  void TearDown() override {
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    HiveConnectorTestBase::TearDown();
  }

  /// Creates a table handle for the Paimon connector.
  static std::shared_ptr<PaimonTableHandle> makePaimonTableHandle(
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      std::vector<int32_t> partitionIds = {}) {
    std::vector<int32_t> ids(dataColumns->size());
    std::iota(ids.begin(), ids.end(), 0);
    const auto primary = tableType == PaimonTableType::kPrimaryKey;
    return std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        tableType,
        0,
        std::vector<PaimonSchema>{{0, dataColumns, std::move(ids)}},
        std::move(partitionIds),
        primary ? std::vector<int32_t>{0} : std::vector<int32_t>{},
        std::move(subfieldFilters),
        remainingFilter,
        primary ? std::unordered_map<
                      std::string,
                      std::string>{{"merge-engine", "deduplicate"}}
                : std::unordered_map<std::string, std::string>{});
  }

  /// Creates column assignments for all columns as regular columns.
  static connector::ColumnHandleMap makePaimonColumnHandles(
      const RowTypePtr& rowType) {
    connector::ColumnHandleMap assignments;
    assignments.reserve(rowType->size());
    for (uint32_t i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      assignments[name] =
          std::make_shared<PaimonColumnHandle>(name, i, rowType->childAt(i));
    }
    return assignments;
  }

  /// Builds a table scan plan using the Paimon connector.
  core::PlanNodePtr makePaimonScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      PaimonTableType tableType = PaimonTableType::kAppendOnly) {
    auto tableHandle = makePaimonTableHandle(
        dataColumns ? dataColumns : outputType,
        std::move(subfieldFilters),
        remainingFilter,
        tableType);
    auto assignments =
        makePaimonColumnHandles(dataColumns ? dataColumns : outputType);
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(tableHandle)
        .assignments(assignments)
        .endTableScan()
        .planNode();
  }

  /// Creates a PaimonConnectorSplit from file paths.
  std::shared_ptr<PaimonConnectorSplit> makePaimonSplit(
      const std::vector<std::string>& filePaths,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      bool rawConvertible = true) {
    std::vector<PaimonDataFile> files;
    for (const auto& filePath : filePaths) {
      PaimonDataFile file;
      file.path = filePath;
      file.size = std::filesystem::file_size(filePath);
      file.rowCount = writtenRows_.at(filePath);
      file.schemaId = 0;
      file.fileFormat = format;
      file.deleteRowCount = 0;
      files.push_back(std::move(file));
    }
    return std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        1,
        tableType,
        format,
        files,
        partitionKeys,
        std::nullopt,
        rawConvertible);
  }

  void writeToFile(
      const std::string& path,
      const std::vector<RowVectorPtr>& vectors) {
    HiveConnectorTestBase::writeToFile(path, vectors);
    auto& count = writtenRows_[path];
    count = 0;
    for (const auto& rows : vectors) {
      count += rows->size();
    }
  }

  std::unordered_map<std::string, uint64_t> writtenRows_;

  struct DirectScan {
    std::shared_ptr<core::QueryCtx> query = core::QueryCtx::create();
    config::ConfigBase session{{}};
    std::unique_ptr<ConnectorQueryCtx> ctx;
    std::unique_ptr<DataSource> source;
  };

  std::unique_ptr<DirectScan> directScan(const core::PlanNodePtr& plan) {
    auto scan = std::make_unique<DirectScan>();
    scan->ctx = std::make_unique<ConnectorQueryCtx>(
        pool(),
        pool(),
        &scan->session,
        nullptr,
        common::PrefixSortConfig(),
        std::make_unique<exec::SimpleExpressionEvaluator>(
            scan->query.get(), pool()),
        nullptr,
        "query",
        "task",
        "scan",
        0,
        "");
    const auto* node = plan->as<core::TableScanNode>();
    scan->source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                       ->createDataSource(
                           node->outputType(),
                           node->tableHandle(),
                           node->assignments(),
                           scan->ctx.get());
    return scan;
  }

#ifdef VELOX_ENABLE_PARQUET
  static RowTypePtr fixtureType() {
    return ROW(
        {"id", "Payload", "amount", "active", "big", "p"},
        {INTEGER(), VARCHAR(), DOUBLE(), BOOLEAN(), BIGINT(), VARCHAR()});
  }

  static folly::dynamic fixture() {
    std::ifstream input(
        std::filesystem::path(PAIMON_TEST_DATA_DIR) / "manifest.json");
    VELOX_CHECK(input.good(), "Paimon fixture manifest is missing");
    return folly::parseJson(
        std::string(std::istreambuf_iterator<char>(input), {}));
  }

  PaimonDataFile fixtureFile(const folly::dynamic& object) {
    auto file = PaimonDataFile::create(object);
    file.path =
        (std::filesystem::path(PAIMON_TEST_DATA_DIR) / file.path).string();
    return file;
  }

  std::shared_ptr<PaimonConnectorSplit> fixtureSplit(
      const folly::dynamic& object,
      bool cacheable = true) {
    std::vector<PaimonDataFile> files;
    for (const auto& file : object["files"]) {
      files.push_back(fixtureFile(file));
    }
    auto partition = object["partition"].isNull()
        ? variant(TypeKind::VARCHAR)
        : variant(object["partition"].asString());
    return std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        object["snapshotId"].asInt(),
        PaimonTableType::kAppendOnly,
        dwio::common::FileFormat::PARQUET,
        files,
        std::unordered_map<std::string, std::optional<std::string>>{},
        object["bucket"].asInt(),
        true,
        cacheable,
        PaimonPartition{{5, std::move(partition)}});
  }

  std::vector<std::shared_ptr<ConnectorSplit>> fixtureSplits(
      const std::string& name) {
    std::vector<std::shared_ptr<ConnectorSplit>> splits;
    const auto manifest = fixture();
    for (const auto& split : manifest[name]["splits"]) {
      splits.push_back(fixtureSplit(split));
    }
    return splits;
  }

  core::PlanNodePtr fixturePlan(
      const RowTypePtr& outputType,
      common::SubfieldFilters filters = {},
      core::TypedExprPtr remaining = nullptr) {
    auto table = makePaimonTableHandle(
        fixtureType(),
        std::move(filters),
        remaining,
        PaimonTableType::kAppendOnly,
        {5});
    auto assignments = makePaimonColumnHandles(fixtureType());
    assignments["p"] = std::make_shared<PaimonColumnHandle>(
        "p", 5, VARCHAR(), FileColumnHandle::ColumnType::kPartitionKey);
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(table)
        .assignments(assignments)
        .endTableScan()
        .planNode();
  }

  RowVectorPtr fixtureRows(const folly::dynamic& rows) {
    VELOX_CHECK_LE(rows.size(), std::numeric_limits<vector_size_t>::max());
    const auto rowCount = static_cast<vector_size_t>(rows.size());
    auto result = std::static_pointer_cast<RowVector>(
        BaseVector::create(fixtureType(), rowCount, pool()));
    for (vector_size_t i = 0; i < rowCount; ++i) {
      for (auto c = 0; c < fixtureType()->size(); ++c) {
        const auto& value = rows[i][c];
        auto& vector = result->childAt(c);
        if (value.isNull()) {
          vector->setNull(i, true);
          continue;
        }
        switch (vector->typeKind()) {
          case TypeKind::INTEGER:
            vector->asFlatVector<int32_t>()->set(
                i, folly::to<int32_t>(value.asInt()));
            break;
          case TypeKind::BIGINT:
            vector->asFlatVector<int64_t>()->set(i, value.asInt());
            break;
          case TypeKind::DOUBLE:
            vector->asFlatVector<double>()->set(i, value.asDouble());
            break;
          case TypeKind::BOOLEAN:
            vector->asFlatVector<bool>()->set(i, value.asBool());
            break;
          case TypeKind::VARCHAR:
            vector->asFlatVector<StringView>()->set(
                i, StringView(value.getString()));
            break;
          default:
            VELOX_FAIL("Unexpected fixture type");
        }
      }
    }
    return result;
  }
#endif
};

TEST_F(PaimonConnectorTest, connectorRegistration) {
  auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  ASSERT_NE(connector, nullptr);
  ASSERT_NE(connector->connectorConfig(), nullptr);
}

TEST_F(PaimonConnectorTest, connectorFactory) {
  PaimonConnectorFactory factory;
  EXPECT_EQ(
      std::string(PaimonConnectorFactory::kPaimonConnectorName), "paimon");

  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {HiveConfig::kEnableFileHandleCache, "true"},
          {HiveConfig::kNumCacheFileHandles, "500"}});

  auto connector = factory.newConnector("test-paimon-2", config);
  ASSERT_NE(connector, nullptr);

  HiveConfig hiveConfig(connector->connectorConfig());
  EXPECT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  EXPECT_EQ(hiveConfig.numCacheFileHandles(), 500);
}

// E2E test: read a single file from an append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlySingleFile) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors = makeVectors(rowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit({filePaths[0]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(vectors);
}

// E2E test: read multiple files from a single append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyMultipleFiles) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors1 = makeVectors(rowType, 1, 100);
  auto vectors2 = makeVectors(rowType, 1, 50);

  auto filePaths = makeFilePaths(2);
  writeToFile(filePaths[0]->getPath(), vectors1);
  writeToFile(filePaths[1]->getPath(), vectors2);

  auto split =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  // Expected result is all rows from both files.
  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1.begin(), vectors1.end());
  expected.insert(expected.end(), vectors2.begin(), vectors2.end());

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(expected);
}

// Verify that empty splits (no data files) are rejected by
// PaimonConnectorSplit.
TEST_F(PaimonConnectorTest, rejectsEmptySplit) {
  VELOX_ASSERT_THROW(
      makePaimonSplit({}), "PaimonConnectorSplit requires non-empty dataFiles");
}

// E2E test: read with partition keys from an append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyWithPartitionKeys) {
  // Data file columns (not including partition column).
  auto dataRowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(dataRowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  // The output includes both the data column and the partition column.
  auto outputType = ROW({"c0", "p0"}, {BIGINT(), VARCHAR()});
  auto tableHandle = makePaimonTableHandle(
      outputType, {}, nullptr, PaimonTableType::kAppendOnly, {1});

  connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<PaimonColumnHandle>("c0", 0, BIGINT());
  assignments["p0"] = std::make_shared<PaimonColumnHandle>(
      "p0", 1, VARCHAR(), FileColumnHandle::ColumnType::kPartitionKey);

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(kPaimonConnectorId)
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      {{"p0", std::optional<std::string>("2024-01-01")}});

  // Build expected output with the partition value filled in.
  auto expectedC0 = vectors[0]->childAt(0);
  auto numRows = vectors[0]->size();
  auto expectedP0 =
      BaseVector::createConstant(VARCHAR(), "2024-01-01", numRows, pool());

  auto expected = makeRowVector({"c0", "p0"}, {expectedC0, expectedP0});

  exec::test::AssertQueryBuilder(plan).split(split).assertResults({expected});
}

TEST_F(PaimonConnectorTest, typedPartitionTaskRoundTrip) {
  auto vectors = makeVectors(ROW({"c0"}, {BIGINT()}), 1, 10);
  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);
  const auto dataSplit = makePaimonSplit({filePaths[0]->getPath()});
  const auto outputType =
      ROW({"c0", "flag", "small", "large"},
          {BIGINT(), BOOLEAN(), INTEGER(), BIGINT()});
  auto table = makePaimonTableHandle(
      outputType, {}, nullptr, PaimonTableType::kAppendOnly, {1, 2, 3});
  auto assignments = makePaimonColumnHandles(outputType);
  for (const auto id : {1, 2, 3}) {
    const auto& name = outputType->nameOf(id);
    assignments[name] = std::make_shared<PaimonColumnHandle>(
        name,
        id,
        outputType->childAt(id),
        FileColumnHandle::ColumnType::kPartitionKey);
  }
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(kPaimonConnectorId)
                  .outputType(outputType)
                  .tableHandle(table)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();
  plan = ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());

  const std::vector<PaimonPartition> partitions{
      {{1, variant(false)},
       {2, variant(std::numeric_limits<int32_t>::min())},
       {3, variant(std::numeric_limits<int64_t>::max())}},
      {{1, variant(true)},
       {2, variant(std::numeric_limits<int32_t>::max())},
       {3, variant(std::numeric_limits<int64_t>::min())}},
      {{1, variant(TypeKind::BOOLEAN)},
       {2, variant(TypeKind::INTEGER)},
       {3, variant(TypeKind::BIGINT)}}};
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  std::vector<RowVectorPtr> expected;
  for (const auto& partition : partitions) {
    auto split = std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        1,
        PaimonTableType::kAppendOnly,
        dwio::common::FileFormat::DWRF,
        dataSplit->dataFiles(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        true,
        true,
        partition);
    splits.push_back(PaimonConnectorSplit::create(split->serialize()));
    std::vector<VectorPtr> children{vectors[0]->childAt(0)};
    for (const auto id : {1, 2, 3}) {
      children.push_back(
          BaseVector::createConstant(
              outputType->childAt(id),
              partition.at(id),
              vectors[0]->size(),
              pool()));
    }
    expected.push_back(makeRowVector(outputType->names(), children));
  }
  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(expected);
}

// Verify that non-rawConvertible splits trigger NYI (merge-on-read).
TEST_F(PaimonConnectorTest, rejectsNonRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);

  auto plan = makePaimonScanPlan(rowType);

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Paimon merge-on-read is not yet implemented");
}

// E2E test: primary-key table with rawConvertible=true reads successfully.
// When fully compacted (rawConvertible), primary-key tables can be read
// the same way as append-only — each file is independent, no merge needed.
TEST_F(PaimonConnectorTest, primaryKeyRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kPrimaryKey,
      dwio::common::FileFormat::DWRF);

  auto plan = makePaimonScanPlan(
      rowType, nullptr, {}, nullptr, PaimonTableType::kPrimaryKey);

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(vectors);
}

// Verify that primary-key table with rawConvertible=false triggers NYI.
TEST_F(PaimonConnectorTest, rejectsPrimaryKeyNonRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kPrimaryKey,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);

  auto plan = makePaimonScanPlan(
      rowType, nullptr, {}, nullptr, PaimonTableType::kPrimaryKey);

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Paimon merge-on-read is not yet implemented");
}

// E2E test: multiple splits each with multiple files.
TEST_F(PaimonConnectorTest, appendOnlyMultipleSplits) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});

  // Split 1 with 2 files.
  auto vectors1a = makeVectors(rowType, 1, 50);
  auto vectors1b = makeVectors(rowType, 1, 30);
  // Split 2 with 1 file.
  auto vectors2a = makeVectors(rowType, 1, 40);

  auto filePaths = makeFilePaths(3);
  writeToFile(filePaths[0]->getPath(), vectors1a);
  writeToFile(filePaths[1]->getPath(), vectors1b);
  writeToFile(filePaths[2]->getPath(), vectors2a);

  auto split1 =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto split2 = makePaimonSplit({filePaths[2]->getPath()});

  auto plan = makePaimonScanPlan(rowType);

  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1a.begin(), vectors1a.end());
  expected.insert(expected.end(), vectors1b.begin(), vectors1b.end());
  expected.insert(expected.end(), vectors2a.begin(), vectors2a.end());

  exec::test::AssertQueryBuilder(plan)
      .splits({split1, split2})
      .assertResults(expected);
}

TEST_F(PaimonConnectorTest, capabilityBoundaries) {
  auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  EXPECT_FALSE(connector->supportsSplitPreload());
  EXPECT_FALSE(connector->supportsIndexLookup());
  VELOX_ASSERT_THROW(
      connector->createDataSink(
          nullptr, nullptr, nullptr, CommitStrategy::kNoCommit),
      "Paimon task writer is not yet implemented");
  VELOX_ASSERT_THROW(
      connector->createIndexSource(nullptr, {}, nullptr, nullptr, {}, nullptr),
      "Paimon index lookup is not yet implemented");
}

TEST_F(PaimonConnectorTest, schemaAndColumnIdentity) {
  const auto type = ROW({"x", "p"}, {BIGINT(), VARCHAR()});
  auto table = makePaimonTableHandle(
      type, {}, nullptr, PaimonTableType::kAppendOnly, {1});
  const auto wire = table->serialize();
  auto copy = ISerializable::deserialize<PaimonTableHandle>(wire, pool());
  EXPECT_EQ(copy->serialize(), wire);
  auto bad = wire;
  bad["schemas"][0]["fieldIds"][1] = 0;
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(bad, pool()), "Duplicate Paimon fieldId");
  bad = wire;
  bad["targetSchemaId"] = 99;
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(bad, pool()), "Missing Paimon schemaId 99");
  bad = wire;
  bad["schemas"][0]["fieldIds"][1] = int64_t{1} << 32;
  VELOX_ASSERT_THROW(PaimonTableHandle::create(bad, pool()), "out of range");
  bad = wire;
  bad["tableType"] = "PRIMARY_KEY";
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(bad, pool()), "explicit primary keys disagree");
  bad = wire;
  bad.erase("tableType");
  EXPECT_ANY_THROW(PaimonTableHandle::create(bad, pool()));
  auto column = std::make_shared<PaimonColumnHandle>(
      "p", 1, VARCHAR(), FileColumnHandle::ColumnType::kPartitionKey);
  EXPECT_EQ(
      ISerializable::deserialize<PaimonColumnHandle>(column->serialize())
          ->serialize(),
      column->serialize());
}

TEST_F(PaimonConnectorTest, primaryKeyUnknownDeletionCountIsNotZero) {
  auto type = ROW({"c0"}, {BIGINT()});
  auto paths = makeFilePaths(1);
  writeToFile(paths[0]->getPath(), makeVectors(type, 1, 10));
  auto split =
      makePaimonSplit({paths[0]->getPath()}, PaimonTableType::kPrimaryKey);
  auto wire = split->serialize();
  wire["dataFiles"][0]["deleteRowCount"] = nullptr;
  auto plan = makePaimonScanPlan(
      type, nullptr, {}, nullptr, PaimonTableType::kPrimaryKey);
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan)
          .split(PaimonConnectorSplit::create(wire))
          .copyResults(pool()),
      "require known zero deleteRowCount");
}

TEST_F(PaimonConnectorTest, rejectsUndeclaredLambdaCaptures) {
  const auto type = ROW({"c0"}, {BIGINT()});
  auto rows = makeRowVector(
      {"c0", "_ROW_ID", "x"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({0, 0, 0}),
       makeFlatVector<int64_t>({0, 0, 0})});
  auto paths = makeFilePaths(1);
  writeToFile(paths[0]->getPath(), {rows});
  for (const auto& [filter, field] :
       std::vector<std::pair<std::string, std::string>>{
           {"any_match(array_constructor(c0), x -> x > \"_ROW_ID\")",
            "_ROW_ID"},
           {"any_match(array_constructor(c0), x -> "
            "any_match(array_constructor(x), y -> y > \"_ROW_ID\"))",
            "_ROW_ID"},
           {"any_match(array_constructor(c0), x -> x > 0) AND "
            "any_match(array_constructor(c0), y -> y > x)",
            "x"}}) {
    SCOPED_TRACE(filter);
    auto plan =
        makePaimonScanPlan(type, type, {}, parseExpr(filter, rows->rowType()));
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit({paths[0]->getPath()}))
            .copyResults(pool()),
        fmt::format("Unsupported Paimon field '{}'", field));
  }
}

TEST_F(PaimonConnectorTest, lambdaParametersAndValidCaptures) {
  auto rows = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({0, 5, 1, 6})});
  auto paths = makeFilePaths(1);
  writeToFile(paths[0]->getPath(), {rows});
  const auto outputType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 3})});
  for (
      const auto* filter :
      {"any_match(array_constructor(c0), x -> x > c1)",
       "any_match(array_constructor(cast(c0 as double)), "
       "c0 -> c0 > cast(c1 as double))",
       "any_match(array_constructor(c0), x -> "
       "any_match(array_constructor(c1), y -> x > y))",
       "any_match(array_constructor(cast(row_constructor(c0) as struct(v bigint))), "
       "x -> x.v > c1)",
       "any_match(array_constructor(c0), \"_ROW_ID\" -> \"_ROW_ID\" > c1)"}) {
    SCOPED_TRACE(filter);
    auto plan = makePaimonScanPlan(
        outputType, rows->rowType(), {}, parseExpr(filter, rows->rowType()));
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit({paths[0]->getPath()}))
        .assertResults({expected});
  }
}

TEST_F(PaimonConnectorTest, rejectsLambdaCaptureTypeMismatch) {
  const auto type = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto filter = parseExpr(
      "any_match(array_constructor(c0), x -> cast(x as varchar) = c1)",
      ROW({"c0", "c1"}, {BIGINT(), VARCHAR()}));
  auto plan = makePaimonScanPlan(ROW({"c0"}, {BIGINT()}), type, {}, filter);
  VELOX_ASSERT_THROW(
      directScan(plan), "Paimon field 'c1' type differs from target schema");
}

#ifdef VELOX_ENABLE_PARQUET
TEST_F(PaimonConnectorTest, realAppendAndCowWithSplitBuilder) {
  const auto manifest = fixture();
  for (const auto* name : {"append", "cow"}) {
    SCOPED_TRACE(name);
    std::vector<std::shared_ptr<ConnectorSplit>> splits;
    for (const auto& input : manifest[name]["splits"]) {
      auto builder = PaimonConnectorSplitBuilder(
          kPaimonConnectorId,
          input["snapshotId"].asInt(),
          PaimonTableType::kAppendOnly,
          dwio::common::FileFormat::PARQUET);
      builder
          .partitionKey(
              "p",
              input["partition"].isNull()
                  ? std::nullopt
                  : std::make_optional(input["partition"].asString()))
          .tableBucketNumber(folly::to<int32_t>(input["bucket"].asInt()));
      for (const auto& file : input["files"]) {
        builder.addFile(fixtureFile(file));
      }
      splits.push_back(
          PaimonConnectorSplit::create(builder.build()->serialize()));
    }
    exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
        .splits(splits)
        .assertResults({fixtureRows(manifest[name]["rows"])});
  }
}

TEST_F(PaimonConnectorTest, realAppendAndCowSnapshots) {
  const auto manifest = fixture();
  EXPECT_EQ(
      manifest["paimonCommit"], "e28c6582c6c864b82c22b6bb4768c092aeafdebd");
  EXPECT_EQ(manifest["append"]["rows"].size(), 8);
  EXPECT_EQ(manifest["cow"]["rows"].size(), 7);
  for (const auto* name : {"append", "cow"}) {
    SCOPED_TRACE(name);
    auto expected = fixtureRows(manifest[name]["rows"]);
    exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
        .splits(fixtureSplits(name))
        .maxDrivers(2)
        .config(core::QueryConfig::kPreferredOutputBatchRows, "2")
        .assertResults({expected});
  }
  auto splits = fixtureSplits("append");
  for (auto& split : splits) {
    auto wire = split->serialize();
    for (auto& file : wire["dataFiles"]) {
      file.erase("deleteRowCount");
    }
    split = PaimonConnectorSplit::create(wire);
  }
  exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
      .splits(splits)
      .assertResults({fixtureRows(manifest["append"]["rows"])});
}

TEST_F(PaimonConnectorTest, realProjectionFiltersAndSerde) {
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(2, 5, false));
  auto remaining = parseExpr(
      "length(\"Payload\") > 0 AND amount < cast(0 as double)", fixtureType());
  auto table = makePaimonTableHandle(
      fixtureType(),
      std::move(filters),
      remaining,
      PaimonTableType::kAppendOnly,
      {5});
  auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(ROW({"renamed"}, {VARCHAR()}))
          .tableHandle(table)
          .assignments(
              {{"renamed",
                std::make_shared<PaimonColumnHandle>("Payload", 1, VARCHAR())}})
          .endTableScan()
          .planNode();
  auto decoded =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  auto splits = fixtureSplits("append");
  for (auto& split : splits) {
    split = std::const_pointer_cast<PaimonConnectorSplit>(
        ISerializable::deserialize<PaimonConnectorSplit>(split->serialize()));
  }
  auto expected =
      makeRowVector({"renamed"}, {makeFlatVector<std::string>({"beta"})});
  for (const auto& input : {plan, decoded}) {
    exec::test::AssertQueryBuilder(input).splits(splits).assertResults(
        {expected});
  }
}

TEST_F(PaimonConnectorTest, realEmptyProjectionWithRemainingFilter) {
  auto plan =
      exec::test::PlanBuilder(
          fixturePlan(
              ROW({}, {}),
              {},
              parseExpr("id >= 2 AND length(\"Payload\") > 0", fixtureType())),
          std::make_shared<core::PlanNodeIdGenerator>(1),
          pool())
          .singleAggregation({}, {"count(1)"})
          .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(fixtureSplits("append"))
      .assertResults({makeRowVector({makeFlatVector<int64_t>({5})})});
}

TEST_F(PaimonConnectorTest, realNullAndEmptyPartitionValues) {
  for (bool isNull : {true, false}) {
    auto plan = fixturePlan(
        ROW({"p"}, {VARCHAR()}),
        {},
        parseExpr(isNull ? "p IS NULL" : "p = ''", fixtureType()));
    auto expected = makeRowVector(
        {"p"},
        {makeNullableFlatVector<std::string>(
            {isNull ? std::optional<std::string>{}
                    : std::optional<std::string>{""}})});
    exec::test::AssertQueryBuilder(plan)
        .splits(fixtureSplits("append"))
        .assertResults({expected});
  }
}

TEST_F(PaimonConnectorTest, realEmptyAndFilteredFilesKeepAdvancing) {
  auto manifest = fixture();
  folly::dynamic east;
  for (const auto& split : manifest["append"]["splits"]) {
    if (split["partition"] == "east") {
      east = split;
      break;
    }
  }
  ASSERT_EQ(east["files"].size(), 2);
  const auto files = east["files"];
  east["files"] = folly::dynamic::array(
      manifest["emptyFile"],
      files[0],
      manifest["emptyFile"],
      files[1],
      manifest["emptyFile"]);
  folly::dynamic expected = folly::dynamic::array;
  for (const auto& row : east["rows"]) {
    if (row[0].asInt() >= 4) {
      expected.push_back(row);
    }
  }
  for (bool cacheable : {true, false}) {
    auto plan =
        fixturePlan(fixtureType(), {}, parseExpr("id >= 4", fixtureType()));
    auto task = exec::test::AssertQueryBuilder(plan)
                    .split(fixtureSplit(east, cacheable))
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
                    .assertResults({fixtureRows(expected)});
    EXPECT_GE(
        task->taskStats()
            .pipelineStats[0]
            .operatorStats[0]
            .runtimeStats.at("skippedSplits")
            .sum,
        1);
  }
  exec::test::AssertQueryBuilder(
      fixturePlan(fixtureType(), {}, parseExpr("id > 100", fixtureType())))
      .split(fixtureSplit(east))
      .assertResults({fixtureRows(folly::dynamic::array)});
}

TEST_F(PaimonConnectorTest, realInputFailures) {
  const auto manifest = fixture();
  auto split = fixtureSplit(manifest["append"]["splits"][0]);
  const auto original = split->serialize();
  const auto fails = [&](const folly::dynamic& wire,
                         const std::string& message) {
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
            .split(PaimonConnectorSplit::create(wire))
            .copyResults(pool()),
        message);
  };
  auto wire = original;
  wire["dataFiles"][0]["schemaId"] = nullptr;
  fails(wire, "Missing Paimon schemaId");
  wire = original;
  wire["dataFiles"][0]["schemaId"] = 999;
  fails(wire, "Missing Paimon schemaId 999");
  wire = original;
  wire["dataFiles"][0]["fileFormat"] = "orc";
  fails(wire, "Unsupported Paimon file format");
  wire = original;
  wire["dataFiles"][0]["fileSize"] =
      wire["dataFiles"][0]["fileSize"].asInt() + 1;
  fails(wire, "fileSize disagrees with opened file");
  wire = original;
  wire["dataFiles"][0]["rowCount"] =
      wire["dataFiles"][0]["rowCount"].asInt() + 1;
  fails(wire, "rowCount disagrees with file footer");
  wire = original;
  wire["dataFiles"][0]["filePath"] =
      wire["dataFiles"][0]["filePath"].asString() + ".missing";
  fails(wire, "No such file or directory");
  wire = original;
  wire["dataFiles"][0]["deletionFile"] =
      PaimonDeletionFile("dv", 0, 16, 1).serialize();
  fails(wire, "deletion vector reading is not yet implemented");
  wire = original;
  wire["dataFiles"][0]["fileType"] = "CHANGELOG";
  fails(wire, "changelog file reading is not yet supported");
  wire = original;
  wire["rawConvertible"] = false;
  fails(wire, "merge-on-read is not yet implemented");
  wire = original;
  wire["readMode"] = "STREAMING";
  fails(wire, "only SNAPSHOT reads");
  wire = original;
  wire["wireVersion"] = 2;
  fails(wire, "out of range");
  wire = original;
  wire["partitionValues"] = folly::dynamic::array;
  fails(wire, "partition metadata is incomplete");
}
TEST_F(PaimonConnectorTest, realOptionsAndHiddenDependencies) {
  auto plan = fixturePlan(fixtureType());
  exec::test::AssertQueryBuilder(plan)
      .splits(fixtureSplits("append"))
      .connectorSessionProperty(
          kPaimonConnectorId,
          FileConfig::kFileColumnNamesReadAsLowerCaseSession,
          "true")
      .connectorSessionProperty(
          kPaimonConnectorId, FileConfig::kUseColumnNamesSession, "false")
      .assertResults({fixtureRows(fixture()["append"]["rows"])});
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan)
          .splits(fixtureSplits("append"))
          .connectorSessionProperty(
              kPaimonConnectorId,
              FileConfig::kIgnoreMissingFilesSession,
              "true")
          .copyResults(pool()),
      "cannot enable ignore_missing_files");
  for (const auto* key :
       {"data-evolution.enabled",
        "deletion-vectors.enabled",
        "force-lookup",
        "ignore-delete",
        "sequence.field",
        "skip.header.line.count",
        "unknown-option"}) {
    auto wire = plan->serialize();
    wire["tableHandle"]["options"][key] = "true";
    auto changed = ISerializable::deserialize<core::PlanNode>(wire, pool());
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(changed)
            .splits(fixtureSplits("append"))
            .copyResults(pool()),
        "Unsupported Paimon");
  }
  // Row tracking alone does not prevent ordinary projection; hidden reads are
  // rejected even if their field is used only by the remaining expression.
  auto wire = plan->serialize();
  wire["tableHandle"]["options"]["row-tracking.enabled"] = "true";
  exec::test::AssertQueryBuilder(
      ISerializable::deserialize<core::PlanNode>(wire, pool()))
      .splits(fixtureSplits("append"))
      .assertResults({fixtureRows(fixture()["append"]["rows"])});
  auto primary = wire;
  primary["tableHandle"]["tableType"] = "PRIMARY_KEY";
  primary["tableHandle"]["primaryKeyFieldIds"] = folly::dynamic::array(0, 5);
  primary["tableHandle"]["options"]["merge-engine"] = "deduplicate";
  VELOX_ASSERT_THROW(
      directScan(ISerializable::deserialize<core::PlanNode>(primary, pool())),
      "row tracking on a primary-key table");
  auto hidden = parseExpr("\"_ROW_ID\" > 0", ROW({"_ROW_ID"}, {BIGINT()}));
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(fixturePlan(fixtureType(), {}, hidden))
          .splits(fixtureSplits("append"))
          .copyResults(pool()),
      "Unsupported Paimon field '_ROW_ID'");
}

TEST_F(PaimonConnectorTest, realPerFileAccessContextAndReadPath) {
  static const auto kFileSystem = []() {
    auto filesystem = std::make_shared<AccessFileSystem>();
    filesystems::registerFileSystem(
        [](std::string_view path) { return path.find("paimon-access:") == 0; },
        [filesystem](auto, auto) { return filesystem; });
    return filesystem;
  }();
  auto token = std::make_shared<TestTokenProvider>();
  auto query = core::QueryCtx::Builder()
                   .executor(executor_.get())
                   .tokenProvider(token)
                   .build();
  auto manifest = fixture();
  auto splits = fixtureSplits("append");
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> expected;
  int64_t index = 0;
  for (auto& input : splits) {
    auto wire = input->serialize();
    wire["cacheable"] = false;
    for (auto& file : wire["dataFiles"]) {
      const auto path = file["filePath"].asString();
      expected.emplace(path, std::make_pair(file["fileSize"].asInt(), ++index));
      file["physicalFilePath"] = "paimon-access:" + path;
      file["filePath"] = "immutable-file-" + std::to_string(index);
      FileProperties properties;
      properties.fileSize = file["fileSize"].asInt();
      properties.readRangeHint = index;
      properties.extraFileInfo =
          std::make_shared<std::string>("file-" + std::to_string(index));
      properties.fileReadOps["access-context"] = std::to_string(index);
      file["properties"] = properties.serialize();
    }
    input = PaimonConnectorSplit::create(wire);
    EXPECT_FALSE(input->cacheable);
  }
  std::atomic<int> opens{0};
  kFileSystem->check = [&](std::string_view path,
                           const filesystems::FileOptions& options) {
    const auto& [size, id] = expected.at(std::string(path));
    VELOX_CHECK_EQ(options.fileSize.value(), size);
    VELOX_CHECK_EQ(options.readRangeHint.value(), id);
    VELOX_CHECK_EQ(*options.extraFileInfo, "file-" + std::to_string(id));
    VELOX_CHECK_EQ(
        options.fileReadOps.at("access-context"), std::to_string(id));
    VELOX_CHECK_EQ(options.fileReadOps.at("tableName"), "paimon_table");
    VELOX_CHECK(options.tokenProvider == token);
    ++opens;
  };
  auto guard = folly::makeGuard([&]() { kFileSystem->check = nullptr; });
  exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
      .queryCtx(query)
      .splits(splits)
      .assertResults({fixtureRows(manifest["append"]["rows"])});
  EXPECT_EQ(opens, expected.size());
}

TEST_F(PaimonConnectorTest, realDynamicFilterAcrossFiles) {
  auto manifest = fixture();
  folly::dynamic east;
  for (const auto& split : manifest["append"]["splits"]) {
    if (split["partition"] == "east") {
      east = split;
    }
  }
  auto scan = directScan(fixturePlan(ROW({"id"}, {INTEGER()})));
  auto& source = scan->source;
  source->addSplit(fixtureSplit(east));
  ContinueFuture future;
  auto first = source->next(1, future);
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(*first, nullptr);
  test::assertEqualVectors(
      makeRowVector({"id"}, {makeFlatVector<int32_t>({1})}), *first);
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(4, 5, false));
  std::vector<int32_t> ids;
  bool ended = false;
  for (int i = 0; i < 20; ++i) {
    auto rows = source->next(1, future);
    ASSERT_TRUE(rows.has_value());
    if (!*rows) {
      ended = true;
      break;
    }
    DecodedVector decoded(*(*rows)->childAt(0));
    for (auto row = 0; row < (*rows)->size(); ++row) {
      ids.push_back(decoded.valueAt<int32_t>(row));
    }
    const auto stats = source->getRuntimeStats();
    const auto again = source->getRuntimeStats();
    for (const auto& [name, value] : stats) {
      EXPECT_EQ(value.sum, again.at(name).sum) << name;
    }
  }
  EXPECT_TRUE(ended);
  EXPECT_EQ(ids, (std::vector<int32_t>{4, 5}));
  EXPECT_EQ(source->getRuntimeStats().at("processedSplits").sum, 2);
}

TEST_F(PaimonConnectorTest, realCallbacksAndEarlyFinish) {
  auto query = core::QueryCtx::create(executor_.get());
  std::unordered_map<std::string, uint64_t> rowsByFile;
  query->setScanBatchCallback([&](const core::ScanBatchEvent& event) {
    const auto* file = dynamic_cast<const FileScanBatchEvent*>(&event);
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(file->fileFormat, dwio::common::FileFormat::PARQUET);
    EXPECT_EQ(file->tableName, "paimon_table");
    if (event.numRows) {
      rowsByFile[std::string(file->filePath)] += event.numRows;
    }
  });
  auto splits = fixtureSplits("append");
  exec::test::AssertQueryBuilder(fixturePlan(fixtureType()))
      .queryCtx(query)
      .splits(splits)
      .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
      .assertResults({fixtureRows(fixture()["append"]["rows"])});
  for (const auto& split : splits) {
    for (const auto& file : split->as<PaimonConnectorSplit>()->dataFiles()) {
      EXPECT_EQ(rowsByFile.at(file.path), file.rowCount);
    }
  }

  auto scan = directScan(fixturePlan(ROW({"id"}, {INTEGER()})));
  scan->source->addSplit(splits.front());
  ContinueFuture future;
  auto row = scan->source->next(1, future);
  ASSERT_TRUE(row.has_value());
  ASSERT_NE(*row, nullptr);
  (*row)->childAt(0)->loadedVector();
  const auto before = scan->source->getRuntimeStats();
  const auto bytes = scan->source->getCompletedBytes();
  EXPECT_GT(bytes, 0);
  scan->source->cancel();
  scan->source->cancel();
  const auto after = scan->source->getRuntimeStats();
  for (const auto& [name, value] : before) {
    EXPECT_EQ(after.at(name).sum, value.sum) << name;
  }
  EXPECT_EQ(scan->source->getCompletedBytes(), bytes);

  auto limit = exec::test::PlanBuilder(
                   fixturePlan(fixtureType()),
                   std::make_shared<core::PlanNodeIdGenerator>(1),
                   pool())
                   .limit(0, 1, false)
                   .planNode();
  auto result =
      exec::test::AssertQueryBuilder(limit).splits(splits).copyResults(pool());
  EXPECT_EQ(result->size(), 1);
}

#endif

} // namespace
} // namespace facebook::velox::connector::hive::paimon

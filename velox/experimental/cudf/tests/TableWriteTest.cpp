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
#include "folly/dynamic.h"
#include "velox/common/base/Fs.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/tests/utils/ParquetConnectorTestBase.h"

#include <re2/re2.h>
#include <string>
#include "folly/experimental/EventCount.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::common;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::common::hll;

constexpr uint64_t kQueryMemoryCapacity = 512 * MB;

enum class TestMode {
  kUnpartitioned,
};

std::string testModeString(TestMode mode) {
  switch (mode) {
    case TestMode::kUnpartitioned:
      return "UNPARTITIONED";
  }
  VELOX_UNREACHABLE();
}

static std::shared_ptr<core::AggregationNode> generateAggregationNode(
    const std::string& name,
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    AggregationNode::Step step,
    const PlanNodePtr& source) {
  core::TypedExprPtr inputField =
      std::make_shared<const core::FieldAccessTypedExpr>(BIGINT(), name);
  auto callExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{inputField}, "min");
  std::vector<std::string> aggregateNames = {"min"};
  std::vector<core::AggregationNode::Aggregate> aggregates = {
      core::AggregationNode::Aggregate{
          callExpr, {{BIGINT()}}, nullptr, {}, {}}};
  return std::make_shared<core::AggregationNode>(
      core::PlanNodeId(),
      step,
      groupingKeys,
      std::vector<core::FieldAccessTypedExprPtr>{},
      aggregateNames,
      aggregates,
      false, // ignoreNullKeys
      source);
}

std::function<PlanNodePtr(std::string, PlanNodePtr)> addTableWriter(
    const RowTypePtr& inputColumns,
    const std::vector<std::string>& tableColumnNames,
    const std::shared_ptr<core::AggregationNode>& aggregationNode,
    const std::shared_ptr<core::InsertTableHandle>& insertHandle,
    facebook::velox::connector::CommitStrategy commitStrategy =
        facebook::velox::connector::CommitStrategy::kNoCommit) {
  return [=](core::PlanNodeId nodeId,
             core::PlanNodePtr source) -> core::PlanNodePtr {
    return std::make_shared<core::TableWriteNode>(
        nodeId,
        inputColumns,
        tableColumnNames,
        aggregationNode,
        insertHandle,
        false,
        TableWriteTraits::outputType(aggregationNode),
        commitStrategy,
        std::move(source));
  };
}

FOLLY_ALWAYS_INLINE std::ostream& operator<<(std::ostream& os, TestMode mode) {
  os << testModeString(mode);
  return os;
}

// NOTE: google parameterized test framework can't handle complex test
// parameters properly. So we encode the different test parameters into one
// integer value.
struct TestParam {
  uint64_t value;

  explicit TestParam(uint64_t _value) : value(_value) {}

  TestParam(
      FileFormat fileFormat,
      TestMode testMode,
      CommitStrategy commitStrategy,
      bool multiDrivers,
      CompressionKind compressionKind,
      bool scaleWriter) {
    value = (scaleWriter ? 1ULL << 40 : 0) |
        static_cast<uint64_t>(compressionKind) << 32 |
        static_cast<uint64_t>(!!multiDrivers) << 24 |
        static_cast<uint64_t>(fileFormat) << 16 |
        static_cast<uint64_t>(testMode) << 8 |
        static_cast<uint64_t>(commitStrategy);
  }

  CompressionKind compressionKind() const {
    return static_cast<facebook::velox::common::CompressionKind>(
        (value & ((1L << 40) - 1)) >> 32);
  }

  bool multiDrivers() const {
    return (value >> 24) != 0;
  }

  FileFormat fileFormat() const {
    return static_cast<FileFormat>((value & ((1L << 24) - 1)) >> 16);
  }

  TestMode testMode() const {
    return static_cast<TestMode>((value & ((1L << 16) - 1)) >> 8);
  }

  CommitStrategy commitStrategy() const {
    return static_cast<CommitStrategy>((value & ((1L << 8) - 1)));
  }

  bool scaleWriter() const {
    return (value >> 40) != 0;
  }

  std::string toString() const {
    return fmt::format(
        "FileFormat[{}] TestMode[{}] commitStrategy[{}] multiDrivers[{}] compression[{}] scaleWriter[{}]",
        dwio::common::toString((fileFormat())),
        testModeString(testMode()),
        commitStrategyToString(commitStrategy()),
        multiDrivers(),
        compressionKindToString(compressionKind()),
        scaleWriter());
  }
};

class TableWriteTest : public ParquetConnectorTestBase {
 protected:
  explicit TableWriteTest(uint64_t testValue)
      : testParam_(static_cast<TestParam>(testValue)),
        fileFormat_(dwio::common::FileFormat::PARQUET),
        testMode_(testParam_.testMode()),
        numTableWriterCount_(
            testParam_.multiDrivers() ? kNumTableWriterCount : 1),
        commitStrategy_(testParam_.commitStrategy()),
        compressionKind_(testParam_.compressionKind()),
        scaleWriter_(testParam_.scaleWriter()) {
    LOG(INFO) << testParam_.toString();

    auto rowType =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
            {BIGINT(), INTEGER(), SMALLINT(), REAL(), DOUBLE(), VARCHAR()});
    setDataTypes(rowType);
  }

  void SetUp() override {
    ParquetConnectorTestBase::SetUp();
  }

  std::shared_ptr<Task> assertQueryWithWriterConfigs(
      const core::PlanNodePtr& plan,
      std::vector<std::shared_ptr<TempFilePath>> filePaths,
      const std::string& duckDbSql,
      bool spillEnabled = false) {
    std::vector<Split> splits;
    for (const auto& filePath : filePaths) {
      splits.push_back(facebook::velox::exec::Split(
          makeParquetConnectorSplit(filePath->getPath())));
    }
    if (!spillEnabled) {
      return AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2 * kNumTableWriterCount)
          .config(
              QueryConfig::kTaskWriterCount,
              std::to_string(numTableWriterCount_))
          // Scale writer settings to trigger partition rebalancing.
          .config(QueryConfig::kScaleWriterRebalanceMaxMemoryUsageRatio, "1.0")
          .config(
              QueryConfig::kScaleWriterMinProcessedBytesRebalanceThreshold, "0")
          .config(
              QueryConfig::
                  kScaleWriterMinPartitionProcessedBytesRebalanceThreshold,
              "0")
          .splits(splits)
          .assertResults(duckDbSql);
    }
  }

  std::shared_ptr<Task> assertQueryWithWriterConfigs(
      const core::PlanNodePtr& plan,
      const std::string& duckDbSql,
      bool enableSpill = false) {
    if (!enableSpill) {
      TestScopedSpillInjection scopedSpillInjection(100);
      return AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2 * kNumTableWriterCount)
          .config(
              QueryConfig::kTaskWriterCount,
              std::to_string(numTableWriterCount_))
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(QueryConfig::kWriterSpillEnabled, "true")
          // Scale writer settings to trigger partition rebalancing.
          .config(QueryConfig::kScaleWriterRebalanceMaxMemoryUsageRatio, "1.0")
          .config(
              QueryConfig::kScaleWriterMinProcessedBytesRebalanceThreshold, "0")
          .config(
              QueryConfig::
                  kScaleWriterMinPartitionProcessedBytesRebalanceThreshold,
              "0")
          .assertResults(duckDbSql);
    }
  }

  RowVectorPtr runQueryWithWriterConfigs(
      const core::PlanNodePtr& plan,
      bool spillEnabled = false) {
    if (!spillEnabled) {
      return AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2 * kNumTableWriterCount)
          .config(
              QueryConfig::kTaskWriterCount,
              std::to_string(numTableWriterCount_))
          // Scale writer settings to trigger partition rebalancing.
          .config(QueryConfig::kScaleWriterRebalanceMaxMemoryUsageRatio, "1.0")
          .config(
              QueryConfig::kScaleWriterMinProcessedBytesRebalanceThreshold, "0")
          .config(
              QueryConfig::
                  kScaleWriterMinPartitionProcessedBytesRebalanceThreshold,
              "0")
          .copyResults(pool());
    }
  }

  void setCommitStrategy(CommitStrategy commitStrategy) {
    commitStrategy_ = commitStrategy;
  }

  void setDataTypes(
      const RowTypePtr& inputType,
      const RowTypePtr& tableSchema = nullptr) {
    rowType_ = inputType;
    if (tableSchema != nullptr) {
      setTableSchema(tableSchema);
    } else {
      setTableSchema(rowType_);
    }
  }

  void setTableSchema(const RowTypePtr& tableSchema) {
    tableSchema_ = tableSchema;
  }

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  makeParquetConnectorSplits(
      const std::shared_ptr<TempDirectoryPath>& directoryPath) {
    return makeParquetConnectorSplits(directoryPath->getPath());
  }

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  makeParquetConnectorSplits(const std::string& directoryPath) {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;

    for (auto& path : fs::recursive_directory_iterator(directoryPath)) {
      if (path.is_regular_file()) {
        splits.push_back(ParquetConnectorTestBase::makeParquetConnectorSplits(
            path.path().string(), 1)[0]);
      }
    }

    return splits;
  }

  // Lists and returns all the regular files from a given directory recursively.
  std::vector<std::string> listAllFiles(const std::string& directoryPath) {
    std::vector<std::string> files;
    for (auto& path : fs::recursive_directory_iterator(directoryPath)) {
      if (path.is_regular_file()) {
        files.push_back(path.path().filename());
      }
    }
    return files;
  }

  // Builds and returns the parquet splits from the list of files with one split
  // per each file.
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  makeParquetConnectorSplits(
      const std::vector<std::filesystem::path>& filePaths) {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    for (const auto& filePath : filePaths) {
      splits.push_back(ParquetConnectorTestBase::makeParquetConnectorSplits(
          filePath.string(), 1)[0]);
    }
    return splits;
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t numVectors,
      int32_t rowsPerVector) {
    return ParquetConnectorTestBase::makeVectors(
        rowType_, numVectors, rowsPerVector);
  }

  RowVectorPtr makeConstantVector(size_t size) {
    return makeRowVector(
        rowType_->names(),
        {makeConstant((int64_t)123'456, size),
         makeConstant((int32_t)321, size),
         makeConstant((int16_t)12'345, size),
         // makeConstant(variant(TypeKind::REAL), size),
         makeConstant((double)1'234.01, size),
         makeConstant(variant(TypeKind::VARCHAR), size)});
  }

  std::vector<RowVectorPtr> makeBatches(
      vector_size_t numBatches,
      std::function<RowVectorPtr(int32_t)> makeVector) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (int32_t i = 0; i < numBatches; ++i) {
      batches.push_back(makeVector(i));
    }
    return batches;
  }

  std::set<std::string> getLeafSubdirectories(
      const std::string& directoryPath) {
    std::set<std::string> subdirectories;
    for (auto& path : fs::recursive_directory_iterator(directoryPath)) {
      if (path.is_regular_file()) {
        subdirectories.emplace(path.path().parent_path().string());
      }
    }
    return subdirectories;
  }

  std::vector<std::string> getRecursiveFiles(const std::string& directoryPath) {
    std::vector<std::string> files;
    for (auto& path : fs::recursive_directory_iterator(directoryPath)) {
      if (path.is_regular_file()) {
        files.push_back(path.path().string());
      }
    }
    return files;
  }

  uint32_t countRecursiveFiles(const std::string& directoryPath) {
    return getRecursiveFiles(directoryPath).size();
  }

  // Helper method to return InsertTableHandle.
  std::shared_ptr<core::InsertTableHandle> createInsertTableHandle(
      const RowTypePtr& outputRowType,
      const cudf_velox::connector::parquet::LocationHandle::TableType&
          outputTableType,
      const std::string& outputDirectoryPath,
      const std::optional<CompressionKind> compressionKind = {}) {
    return std::make_shared<core::InsertTableHandle>(
        kParquetConnectorId,
        makeParquetInsertTableHandle(
            outputRowType->names(),
            outputRowType->children(),
            makeLocationHandle(outputDirectoryPath, outputTableType),
            compressionKind));
  }

  // Returns a table insert plan node.
  PlanNodePtr createInsertPlan(
      PlanBuilder& inputPlan,
      const RowTypePtr& outputRowType,
      const std::string& outputDirectoryPath,
      const std::optional<CompressionKind> compressionKind = {},
      int numTableWriters = 1,
      const cudf_velox::connector::parquet::LocationHandle::TableType&
          outputTableType =
              cudf_velox::connector::parquet::LocationHandle::TableType::kNew,
      const CommitStrategy& outputCommitStrategy = CommitStrategy::kNoCommit,
      bool aggregateResult = true,
      std::shared_ptr<core::AggregationNode> aggregationNode = nullptr) {
    return createInsertPlan(
        inputPlan,
        inputPlan.planNode()->outputType(),
        outputRowType,
        outputDirectoryPath,
        compressionKind,
        numTableWriters,
        outputTableType,
        outputCommitStrategy,
        aggregateResult,
        aggregationNode);
  }

  PlanNodePtr createInsertPlan(
      PlanBuilder& inputPlan,
      const RowTypePtr& inputRowType,
      const RowTypePtr& tableRowType,
      const std::string& outputDirectoryPath,
      const std::optional<CompressionKind> compressionKind = {},
      int numTableWriters = 1,
      const cudf_velox::connector::parquet::LocationHandle::TableType&
          outputTableType =
              cudf_velox::connector::parquet::LocationHandle::TableType::kNew,
      const CommitStrategy& outputCommitStrategy = CommitStrategy::kNoCommit,
      bool aggregateResult = true,
      std::shared_ptr<core::AggregationNode> aggregationNode = nullptr) {
    if (numTableWriters == 1) {
      return createInsertPlanWithSingleWriter(
          inputPlan,
          inputRowType,
          tableRowType,
          outputDirectoryPath,
          compressionKind,
          outputTableType,
          outputCommitStrategy,
          aggregateResult,
          aggregationNode);
    }
  }

  PlanNodePtr createInsertPlanWithSingleWriter(
      PlanBuilder& inputPlan,
      const RowTypePtr& inputRowType,
      const RowTypePtr& tableRowType,
      const std::string& outputDirectoryPath,
      const std::optional<CompressionKind> compressionKind,
      const cudf_velox::connector::parquet::LocationHandle::TableType&
          outputTableType,
      const CommitStrategy& outputCommitStrategy,
      bool aggregateResult,
      std::shared_ptr<core::AggregationNode> aggregationNode) {
    const bool addScaleWriterExchange = false;
    auto insertPlan = inputPlan;
    insertPlan
        .addNode(addTableWriter(
            inputRowType,
            tableRowType->names(),
            aggregationNode,
            createInsertTableHandle(
                tableRowType,
                outputTableType,
                outputDirectoryPath,
                compressionKind),
            outputCommitStrategy))
        .capturePlanNodeId(tableWriteNodeId_);
    if (aggregateResult) {
      insertPlan.project({TableWriteTraits::rowCountColumnName()})
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())});
    }
    return insertPlan.planNode();
  }

  // Return the corresponding column names in 'inputRowType' of
  // 'tableColumnNames' from 'tableRowType'.
  static std::vector<std::string> inputColumnNames(
      const std::vector<std::string>& tableColumnNames,
      const RowTypePtr& tableRowType,
      const RowTypePtr& inputRowType) {
    std::vector<std::string> inputNames;
    inputNames.reserve(tableColumnNames.size());
    for (const auto& tableColumnName : tableColumnNames) {
      const auto columnIdx = tableRowType->getChildIdx(tableColumnName);
      inputNames.push_back(inputRowType->nameOf(columnIdx));
    }
    return inputNames;
  }

  // Parameter partitionName is string formatted in the Parquet style
  // key1=value1/key2=value2/... Parameter partitionTypes are types of partition
  // keys in the same order as in partitionName.The return value is a SQL
  // predicate with values single quoted for string and date and not quoted for
  // other supported types, ex., key1='value1' AND key2=value2 AND ...
  std::string partitionNameToPredicate(
      const std::string& partitionName,
      const std::vector<TypePtr>& partitionTypes) {
    std::vector<std::string> conjuncts;

    std::vector<std::string> partitionKeyValues;
    folly::split('/', partitionName, partitionKeyValues);
    VELOX_CHECK_EQ(partitionKeyValues.size(), partitionTypes.size());

    for (auto i = 0; i < partitionKeyValues.size(); ++i) {
      if (partitionTypes[i]->isVarchar() || partitionTypes[i]->isVarbinary() ||
          partitionTypes[i]->isDate()) {
        conjuncts.push_back(
            partitionKeyValues[i]
                .replace(partitionKeyValues[i].find("="), 1, "='")
                .append("'"));
      } else {
        conjuncts.push_back(partitionKeyValues[i]);
      }
    }

    return folly::join(" AND ", conjuncts);
  }

  // Verifies if a unbucketed file name is encoded properly based on the
  // used commit strategy.
  void verifyUnbucketedFilePath(
      const std::filesystem::path& filePath,
      const std::string& targetDir) {
    ASSERT_EQ(filePath.parent_path().string(), targetDir);
    if (commitStrategy_ == CommitStrategy::kNoCommit) {
      ASSERT_TRUE(RE2::FullMatch(
          filePath.filename().string(),
          fmt::format(
              "test_cursor.+_[0-{}]_{}_.+",
              numTableWriterCount_ - 1,
              tableWriteNodeId_)))
          << filePath.filename().string();
    } else {
      ASSERT_TRUE(RE2::FullMatch(
          filePath.filename().string(),
          fmt::format(
              ".tmp.velox.test_cursor.+_[0-{}]_{}_.+",
              numTableWriterCount_ - 1,
              tableWriteNodeId_)))
          << filePath.filename().string();
    }
  }

  // Verifies the file layout and data produced by a table writer.
  void verifyTableWriterOutput(
      const std::string& targetDir,
      const RowTypePtr& bucketCheckFileType,
      bool verifyPartitionedData = true,
      bool verifyBucketedData = true) {
    SCOPED_TRACE(testParam_.toString());
    std::vector<std::filesystem::path> filePaths;
    std::vector<std::filesystem::path> dirPaths;
    for (auto& path : fs::recursive_directory_iterator(targetDir)) {
      if (path.is_regular_file()) {
        filePaths.push_back(path.path());
      } else {
        dirPaths.push_back(path.path());
      }
    }
    if (testMode_ == TestMode::kUnpartitioned) {
      ASSERT_EQ(dirPaths.size(), 0);
      ASSERT_LE(filePaths.size(), numTableWriterCount_);
      verifyUnbucketedFilePath(filePaths[0], targetDir);
      return;
    }
  }

  int getNumWriters() {
    return numTableWriterCount_;
  }

  static inline int kNumTableWriterCount = 1;

  const TestParam testParam_;
  const FileFormat fileFormat_ = FileFormat::PARQUET;
  const TestMode testMode_;
  const int numTableWriterCount_;

  RowTypePtr rowType_;
  RowTypePtr tableSchema_;
  CommitStrategy commitStrategy_;
  std::optional<CompressionKind> compressionKind_;
  bool scaleWriter_;
  std::vector<column_index_t> sortColumnIndices_;
  std::vector<CompareFlags> sortedFlags_;
  core::PlanNodeId tableWriteNodeId_;
};

class BasicTableWriteTest : public ParquetConnectorTestBase {};

TEST_F(BasicTableWriteTest, roundTrip) {
  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(
          size, [](auto row) { return row * 2; }, nullEvery(7)),
  });

  auto sourceFilePath = TempFilePath::create();
  writeToFile(sourceFilePath->getPath(), data);

  auto targetDirectoryPath = TempDirectoryPath::create();

  auto rowType = asRowType(data->type());
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(ParquetConnectorTestBase::makeTableHandle())
                  .endTableScan()
                  .tableWrite(targetDirectoryPath->getPath())
                  .planNode();

  auto results =
      AssertQueryBuilder(plan)
          .split(makeParquetConnectorSplit(sourceFilePath->getPath()))
          .copyResults(pool());
  ASSERT_EQ(2, results->size());

  // First column has number of rows written in the first row and nulls in other
  // rows.
  auto rowCount = results->childAt(TableWriteTraits::kRowCountChannel)
                      ->as<FlatVector<int64_t>>();
  ASSERT_FALSE(rowCount->isNullAt(0));
  ASSERT_EQ(size, rowCount->valueAt(0));
  ASSERT_TRUE(rowCount->isNullAt(1));

  // Second column contains details about written files.
  auto details = results->childAt(TableWriteTraits::kFragmentChannel)
                     ->as<FlatVector<StringView>>();
  ASSERT_TRUE(details->isNullAt(0));
  ASSERT_FALSE(details->isNullAt(1));
  folly::dynamic obj = folly::parseJson(details->valueAt(1));

  ASSERT_EQ(size, obj["rowCount"].asInt());
  auto fileWriteInfos = obj["fileWriteInfos"];
  ASSERT_EQ(1, fileWriteInfos.size());

  auto writeFileName = fileWriteInfos[0]["writeFileName"].asString();

  // Read from 'writeFileName' and verify the data matches the original.
  plan = PlanBuilder().tableScan(rowType).planNode();

  auto copy = AssertQueryBuilder(plan)
                  .split(makeParquetConnectorSplit(fmt::format(
                      "{}/{}", targetDirectoryPath->getPath(), writeFileName)))
                  .copyResults(pool());
  assertEqualResults({data}, {copy});
}

TEST_F(BasicTableWriteTest, targetFileName) {
  constexpr const char* kFileName = "test.parquet";
  auto data = makeRowVector({makeFlatVector<int64_t>(10, folly::identity)});
  auto directory = TempDirectoryPath::create();
  auto plan = PlanBuilder()
                  .values({data})
                  .tableWrite(
                      directory->getPath(),
                      dwio::common::FileFormat::PARQUET,
                      {},
                      nullptr,
                      kFileName)
                  .planNode();
  auto results = AssertQueryBuilder(plan).copyResults(pool());
  auto* details = results->childAt(TableWriteTraits::kFragmentChannel)
                      ->asUnchecked<SimpleVector<StringView>>();
  auto detail = folly::parseJson(details->valueAt(1));
  auto fileWriteInfos = detail["fileWriteInfos"];
  ASSERT_EQ(1, fileWriteInfos.size());
  ASSERT_EQ(fileWriteInfos[0]["writeFileName"].asString(), kFileName);
  plan = PlanBuilder().tableScan(asRowType(data->type())).planNode();
  AssertQueryBuilder(plan)
      .split(makeParquetConnectorSplit(
          fmt::format("{}/{}", directory->getPath(), kFileName)))
      .assertResults(data);
}

#if 0
class PartitionedTableWriterTest
    : public TableWriteTest,
      public testing::WithParamInterface<uint64_t> {
 public:
  PartitionedTableWriterTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (bool scaleWriter : {false, true}) {
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

class UnpartitionedTableWriterTest
    : public TableWriteTest,
      public testing::WithParamInterface<uint64_t> {
 public:
  UnpartitionedTableWriterTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (bool scaleWriter : {false, true}) {
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kUnpartitioned,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_NONE,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kUnpartitioned,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_NONE,
              scaleWriter}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

class BucketedTableOnlyWriteTest
    : public TableWriteTest,
      public testing::WithParamInterface<uint64_t> {
 public:
  BucketedTableOnlyWriteTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    const std::vector<TestMode> bucketModes = {
        TestMode::kBucketed, TestMode::kOnlyBucketed};
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (auto bucketMode : bucketModes) {
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              true,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              true,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              true,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              true,
              multiDrivers,
              CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

class BucketSortOnlyTableWriterTest
    : public TableWriteTest,
      public testing::WithParamInterface<uint64_t> {
 public:
  BucketSortOnlyTableWriterTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    const std::vector<TestMode> bucketModes = {
        TestMode::kBucketed, TestMode::kOnlyBucketed};
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (auto bucketMode : bucketModes) {
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              true,
              multiDrivers,
              facebook::velox::common::CompressionKind_ZSTD,
              /*scaleWriter=*/false}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              bucketMode,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              true,
              multiDrivers,
              facebook::velox::common::CompressionKind_NONE,
              /*scaleWriter=*/false}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

class PartitionedWithoutBucketTableWriterTest
    : public TableWriteTest,
      public testing::WithParamInterface<uint64_t> {
 public:
  PartitionedWithoutBucketTableWriterTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (bool scaleWriter : {false, true}) {
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              true,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

class AllTableWriterTest : public TableWriteTest,
                           public testing::WithParamInterface<uint64_t> {
 public:
  AllTableWriterTest() : TableWriteTest(GetParam()) {}

  static std::vector<uint64_t> getTestParams() {
    std::vector<uint64_t> testParams;
    const std::vector<bool> multiDriverOptions = {false, true};
    std::vector<FileFormat> fileFormats = {FileFormat::DWRF};
    if (hasWriterFactory(FileFormat::PARQUET)) {
      fileFormats.push_back(FileFormat::PARQUET);
    }
    for (bool multiDrivers : multiDriverOptions) {
      for (FileFormat fileFormat : fileFormats) {
        for (bool scaleWriter : {false, true}) {
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kUnpartitioned,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kUnpartitioned,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kPartitioned,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kOnlyBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kOnlyBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kParquetCompatible,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kOnlyBucketed,
              CommitStrategy::kNoCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
          testParams.push_back(TestParam{
              fileFormat,
              TestMode::kOnlyBucketed,
              CommitStrategy::kTaskCommit,
              ParquetBucketProperty::Kind::kPrestoNative,
              false,
              multiDrivers,
              CompressionKind_ZSTD,
              scaleWriter}
                                   .value);
        }
      }
    }
    return testParams;
  }
};

// Runs a pipeline with read + filter + project (with substr) + write.
TEST_P(AllTableWriterTest, scanFilterProjectWrite) {
  auto filePaths = makeFilePaths(5);
  auto vectors = makeVectors(filePaths.size(), 500);
  for (int i = 0; i < filePaths.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }

  createDuckDbTable(vectors);

  auto outputDirectory = TempDirectoryPath::create();

  auto planBuilder = PlanBuilder();
  auto project = planBuilder.tableScan(rowType_).filter("c2 <> 0").project(
      {"c0", "c1", "c3", "c5", "c2 + c3", "substr(c5, 1, 1)"});

  auto intputTypes = project.planNode()->outputType()->children();
  std::vector<std::string> tableColumnNames = {
      "c0", "c1", "c3", "c5", "c2_plus_c3", "substr_c5"};
  const auto outputType =
      ROW(std::move(tableColumnNames), std::move(intputTypes));

  auto plan = createInsertPlan(
      project,
      outputType,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  assertQueryWithWriterConfigs(
      plan, filePaths, "SELECT count(*) FROM tmp WHERE c2 <> 0");

  // To test the correctness of the generated output,
  // We create a new plan that only read that file and then
  // compare that against a duckDB query that runs the whole query.
  if (partitionedBy_.size() > 0) {
    auto newOutputType = getNonPartitionsColumns(partitionedBy_, outputType);
    assertQuery(
        PlanBuilder().tableScan(newOutputType).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c3, c5, c2 + c3, substr(c5, 1, 1) FROM tmp WHERE c2 <> 0");
    verifyTableWriterOutput(outputDirectory->getPath(), newOutputType, false);
  } else {
    assertQuery(
        PlanBuilder().tableScan(outputType).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c0, c1, c3, c5, c2 + c3, substr(c5, 1, 1) FROM tmp WHERE c2 <> 0");
    verifyTableWriterOutput(outputDirectory->getPath(), outputType, false);
  }
}

TEST_P(AllTableWriterTest, renameAndReorderColumns) {
  auto filePaths = makeFilePaths(5);
  auto vectors = makeVectors(filePaths.size(), 500);
  for (int i = 0; i < filePaths.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }

  createDuckDbTable(vectors);

  auto outputDirectory = TempDirectoryPath::create();

  if (testMode_ == TestMode::kPartitioned || testMode_ == TestMode::kBucketed) {
    const std::vector<std::string> partitionBy = {"x", "y"};
    setPartitionBy(partitionBy);
  }
  if (testMode_ == TestMode::kBucketed ||
      testMode_ == TestMode::kOnlyBucketed) {
    setBucketProperty(
        bucketProperty_->kind(),
        bucketProperty_->bucketCount(),
        {"z", "v"},
        {REAL(), VARCHAR()},
        {});
  }

  auto inputRowType =
      ROW({"c2", "c5", "c4", "c1", "c0", "c3"},
          {SMALLINT(), VARCHAR(), DOUBLE(), INTEGER(), BIGINT(), REAL()});

  setTableSchema(
      ROW({"u", "v", "w", "x", "y", "z"},
          {SMALLINT(), VARCHAR(), DOUBLE(), INTEGER(), BIGINT(), REAL()}));

  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType_),
      inputRowType,
      tableSchema_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  assertQueryWithWriterConfigs(plan, filePaths, "SELECT count(*) FROM tmp");

  if (partitionedBy_.size() > 0) {
    auto newOutputType = getNonPartitionsColumns(partitionedBy_, tableSchema_);
    ParquetConnectorTestBase::assertQuery(
        PlanBuilder().tableScan(newOutputType).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c2, c5, c4, c3 FROM tmp");

    verifyTableWriterOutput(outputDirectory->getPath(), newOutputType, false);
  } else {
    ParquetConnectorTestBase::assertQuery(
        PlanBuilder().tableScan(tableSchema_).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c2, c5, c4, c1, c0, c3 FROM tmp");

    verifyTableWriterOutput(outputDirectory->getPath(), tableSchema_, false);
  }
}

// Runs a pipeline with read + write.
TEST_P(AllTableWriterTest, directReadWrite) {
  auto filePaths = makeFilePaths(5);
  auto vectors = makeVectors(filePaths.size(), 200);
  for (int i = 0; i < filePaths.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }

  createDuckDbTable(vectors);

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType_),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  assertQuery(plan, filePaths, "SELECT count(*) FROM tmp");

  // To test the correctness of the generated output,
  // We create a new plan that only read that file and then
  // compare that against a duckDB query that runs the whole query.

  if (partitionedBy_.size() > 0) {
    auto newOutputType = getNonPartitionsColumns(partitionedBy_, tableSchema_);
    assertQuery(
        PlanBuilder().tableScan(newOutputType).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c2, c3, c4, c5 FROM tmp");
    rowType_ = newOutputType;
    verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
  } else {
    assertQuery(
        PlanBuilder().tableScan(rowType_).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT * FROM tmp");

    verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
  }
}

// Tests writing constant vectors.
TEST_P(AllTableWriterTest, constantVectors) {
  vector_size_t size = 1'000;

  // Make constant vectors of various types with null and non-null values.
  auto vector = makeConstantVector(size);

  createDuckDbTable({vector});

  auto outputDirectory = TempDirectoryPath::create();
  auto op = createInsertPlan(
      PlanBuilder().values({vector}),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  assertQuery(op, fmt::format("SELECT {}", size));

  if (partitionedBy_.size() > 0) {
    auto newOutputType = getNonPartitionsColumns(partitionedBy_, tableSchema_);
    assertQuery(
        PlanBuilder().tableScan(newOutputType).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT c2, c3, c4, c5 FROM tmp");
    rowType_ = newOutputType;
    verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
  } else {
    assertQuery(
        PlanBuilder().tableScan(rowType_).planNode(),
        makeParquetConnectorSplits(outputDirectory),
        "SELECT * FROM tmp");

    verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
  }
}

TEST_P(AllTableWriterTest, emptyInput) {
  auto outputDirectory = TempDirectoryPath::create();
  auto vector = makeConstantVector(0);
  auto op = createInsertPlan(
      PlanBuilder().values({vector}),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  assertQuery(op, "SELECT 0");
}

TEST_P(AllTableWriterTest, commitStrategies) {
  auto filePaths = makeFilePaths(5);
  auto vectors = makeVectors(filePaths.size(), 100);

  createDuckDbTable(vectors);

  // Test the kTaskCommit commit strategy writing to one dot-prefixed
  // temporary file.
  {
    SCOPED_TRACE(CommitStrategy::kTaskCommit);
    auto outputDirectory = TempDirectoryPath::create();
    auto plan = createInsertPlan(
        PlanBuilder().values(vectors),
        rowType_,
        outputDirectory->getPath(),
        compressionKind_,
        getNumWriters(),
        connector::parquet::LocationHandle::TableType::kNew,
        commitStrategy_);

    assertQuery(plan, "SELECT count(*) FROM tmp");

    if (partitionedBy_.size() > 0) {
      auto newOutputType =
          getNonPartitionsColumns(partitionedBy_, tableSchema_);
      assertQuery(
          PlanBuilder().tableScan(newOutputType).planNode(),
          makeParquetConnectorSplits(outputDirectory),
          "SELECT c2, c3, c4, c5 FROM tmp");
      auto originalRowType = rowType_;
      rowType_ = newOutputType;
      verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
      rowType_ = originalRowType;
    } else {
      assertQuery(
          PlanBuilder().tableScan(rowType_).planNode(),
          makeParquetConnectorSplits(outputDirectory),
          "SELECT * FROM tmp");
      verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
    }
  }
  // Test kNoCommit commit strategy writing to non-temporary files.
  {
    SCOPED_TRACE(CommitStrategy::kNoCommit);
    auto outputDirectory = TempDirectoryPath::create();
    setCommitStrategy(CommitStrategy::kNoCommit);
    auto plan = createInsertPlan(
        PlanBuilder().values(vectors),
        rowType_,
        outputDirectory->getPath(),
        compressionKind_,
        getNumWriters(),
        connector::parquet::LocationHandle::TableType::kNew,
        commitStrategy_);

    assertQuery(plan, "SELECT count(*) FROM tmp");

    if (partitionedBy_.size() > 0) {
      auto newOutputType =
          getNonPartitionsColumns(partitionedBy_, tableSchema_);
      assertQuery(
          PlanBuilder().tableScan(newOutputType).planNode(),
          makeParquetConnectorSplits(outputDirectory),
          "SELECT c2, c3, c4, c5 FROM tmp");
      rowType_ = newOutputType;
      verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
    } else {
      assertQuery(
          PlanBuilder().tableScan(rowType_).planNode(),
          makeParquetConnectorSplits(outputDirectory),
          "SELECT * FROM tmp");
      verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
    }
  }
}

TEST_P(PartitionedTableWriterTest, specialPartitionName) {
  const int32_t numPartitions = 50;
  const int32_t numBatches = 2;

  const auto rowType =
      ROW({"c0", "p0", "p1", "c1", "c3", "c5"},
          {INTEGER(), INTEGER(), VARCHAR(), BIGINT(), REAL(), VARCHAR()});
  const std::vector<std::string> partitionKeys = {"p0", "p1"};
  const std::vector<TypePtr> partitionTypes = {INTEGER(), VARCHAR()};

  const std::vector charsToEscape = {
      '"',
      '#',
      '%',
      '\'',
      '*',
      '/',
      ':',
      '=',
      '?',
      '\\',
      '\x7F',
      '{',
      '[',
      ']',
      '^'};
  ASSERT_GE(numPartitions, charsToEscape.size());
  std::vector<RowVectorPtr> vectors = makeBatches(numBatches, [&](auto) {
    return makeRowVector(
        rowType->names(),
        {
            makeFlatVector<int32_t>(
                numPartitions, [&](auto row) { return row + 100; }),
            makeFlatVector<int32_t>(
                numPartitions, [&](auto row) { return row; }),
            makeFlatVector<StringView>(
                numPartitions,
                [&](auto row) {
                  // special character
                  return StringView::makeInline(
                      fmt::format("str_{}{}", row, charsToEscape.at(row % 15)));
                }),
            makeFlatVector<int64_t>(
                numPartitions, [&](auto row) { return row + 1000; }),
            makeFlatVector<float>(
                numPartitions, [&](auto row) { return row + 33.23; }),
            makeFlatVector<StringView>(
                numPartitions,
                [&](auto row) {
                  return StringView::makeInline(
                      fmt::format("bucket_{}", row * 3));
                }),
        });
  });
  createDuckDbTable(vectors);

  auto inputFilePaths = makeFilePaths(numBatches);
  for (int i = 0; i < numBatches; i++) {
    writeToFile(inputFilePaths[i]->getPath(), vectors[i]);
  }

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  auto task = assertQuery(plan, inputFilePaths, "SELECT count(*) FROM tmp");

  std::set<std::string> actualPartitionDirectories =
      getLeafSubdirectories(outputDirectory->getPath());

  std::set<std::string> expectedPartitionDirectories;
  const std::vector<std::string> expectedCharsAfterEscape = {
      "%22",
      "%23",
      "%25",
      "%27",
      "%2A",
      "%2F",
      "%3A",
      "%3D",
      "%3F",
      "%5C",
      "%7F",
      "%7B",
      "%5B",
      "%5D",
      "%5E"};
  for (auto i = 0; i < numPartitions; ++i) {
    // url encoded
    auto partitionName = fmt::format(
        "p0={}/p1=str_{}{}", i, i, expectedCharsAfterEscape.at(i % 15));
    expectedPartitionDirectories.emplace(
        fs::path(outputDirectory->getPath()) / partitionName);
  }
  EXPECT_EQ(actualPartitionDirectories, expectedPartitionDirectories);
}

TEST_P(PartitionedTableWriterTest, multiplePartitions) {
  int32_t numPartitions = 50;
  int32_t numBatches = 2;

  auto rowType =
      ROW({"c0", "p0", "p1", "c1", "c3", "c5"},
          {INTEGER(), INTEGER(), VARCHAR(), BIGINT(), REAL(), VARCHAR()});
  std::vector<std::string> partitionKeys = {"p0", "p1"};
  std::vector<TypePtr> partitionTypes = {INTEGER(), VARCHAR()};

  std::vector<RowVectorPtr> vectors = makeBatches(numBatches, [&](auto) {
    return makeRowVector(
        rowType->names(),
        {
            makeFlatVector<int32_t>(
                numPartitions, [&](auto row) { return row + 100; }),
            makeFlatVector<int32_t>(
                numPartitions, [&](auto row) { return row; }),
            makeFlatVector<StringView>(
                numPartitions,
                [&](auto row) {
                  return StringView::makeInline(fmt::format("str_{}", row));
                }),
            makeFlatVector<int64_t>(
                numPartitions, [&](auto row) { return row + 1000; }),
            makeFlatVector<float>(
                numPartitions, [&](auto row) { return row + 33.23; }),
            makeFlatVector<StringView>(
                numPartitions,
                [&](auto row) {
                  return StringView::makeInline(
                      fmt::format("bucket_{}", row * 3));
                }),
        });
  });
  createDuckDbTable(vectors);

  auto inputFilePaths = makeFilePaths(numBatches);
  for (int i = 0; i < numBatches; i++) {
    writeToFile(inputFilePaths[i]->getPath(), vectors[i]);
  }

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  auto task = assertQuery(plan, inputFilePaths, "SELECT count(*) FROM tmp");

  // Verify that there is one partition directory for each partition.
  std::set<std::string> actualPartitionDirectories =
      getLeafSubdirectories(outputDirectory->getPath());

  std::set<std::string> expectedPartitionDirectories;
  std::set<std::string> partitionNames;
  for (auto i = 0; i < numPartitions; i++) {
    auto partitionName = fmt::format("p0={}/p1=str_{}", i, i);
    partitionNames.emplace(partitionName);
    expectedPartitionDirectories.emplace(
        fs::path(outputDirectory->getPath()) / partitionName);
  }
  EXPECT_EQ(actualPartitionDirectories, expectedPartitionDirectories);

  // Verify distribution of records in partition directories.
  auto iterPartitionDirectory = actualPartitionDirectories.begin();
  auto iterPartitionName = partitionNames.begin();
  auto newOutputType = getNonPartitionsColumns(partitionKeys, rowType);
  while (iterPartitionDirectory != actualPartitionDirectories.end()) {
    assertQuery(
        PlanBuilder().tableScan(newOutputType).planNode(),
        makeParquetConnectorSplits(*iterPartitionDirectory),
        fmt::format(
            "SELECT c0, c1, c3, c5 FROM tmp WHERE {}",
            partitionNameToPredicate(*iterPartitionName, partitionTypes)));
    // In case of unbucketed partitioned table, one single file is written to
    // each partition directory for Parquet connector.
    if (testMode_ == TestMode::kPartitioned) {
      ASSERT_EQ(countRecursiveFiles(*iterPartitionDirectory), 1);
    } else {
      ASSERT_GE(countRecursiveFiles(*iterPartitionDirectory), 1);
    }

    ++iterPartitionDirectory;
    ++iterPartitionName;
  }
}

TEST_P(PartitionedTableWriterTest, singlePartition) {
  const int32_t numBatches = 2;
  auto rowType =
      ROW({"c0", "p0", "c3", "c5"}, {VARCHAR(), BIGINT(), REAL(), VARCHAR()});
  std::vector<std::string> partitionKeys = {"p0"};

  // Partition vector is constant vector.
  std::vector<RowVectorPtr> vectors = makeBatches(numBatches, [&](auto) {
    return makeRowVector(
        rowType->names(),
        {makeFlatVector<StringView>(
             1'000,
             [&](auto row) {
               return StringView::makeInline(fmt::format("str_{}", row));
             }),
         makeConstant((int64_t)365, 1'000),
         makeFlatVector<float>(1'000, [&](auto row) { return row + 33.23; }),
         makeFlatVector<StringView>(1'000, [&](auto row) {
           return StringView::makeInline(fmt::format("bucket_{}", row * 3));
         })});
  });
  createDuckDbTable(vectors);

  auto inputFilePaths = makeFilePaths(numBatches);
  for (int i = 0; i < numBatches; i++) {
    writeToFile(inputFilePaths[i]->getPath(), vectors[i]);
  }

  auto outputDirectory = TempDirectoryPath::create();
  const int numWriters = getNumWriters();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      numWriters,
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  auto task = assertQueryWithWriterConfigs(
      plan, inputFilePaths, "SELECT count(*) FROM tmp");

  std::set<std::string> partitionDirectories =
      getLeafSubdirectories(outputDirectory->getPath());

  // Verify only a single partition directory is created.
  ASSERT_EQ(partitionDirectories.size(), 1);
  EXPECT_EQ(
      *partitionDirectories.begin(),
      fs::path(outputDirectory->getPath()) / "p0=365");

  // Verify all data is written to the single partition directory.
  auto newOutputType = getNonPartitionsColumns(partitionKeys, rowType);
  assertQuery(
      PlanBuilder().tableScan(newOutputType).planNode(),
      makeParquetConnectorSplits(outputDirectory),
      "SELECT c0, c3, c5 FROM tmp");

  // In case of unbucketed partitioned table, one single file is written to
  // each partition directory for Parquet connector.
  if (testMode_ == TestMode::kPartitioned) {
    ASSERT_LE(countRecursiveFiles(*partitionDirectories.begin()), numWriters);
  } else {
    ASSERT_GE(countRecursiveFiles(*partitionDirectories.begin()), numWriters);
  }
}

TEST_P(PartitionedWithoutBucketTableWriterTest, fromSinglePartitionToMultiple) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  setDataTypes(rowType);
  std::vector<std::string> partitionKeys = {"c0"};

  // Partition vector is constant vector.
  std::vector<RowVectorPtr> vectors;
  // The initial vector has the same partition key value;
  vectors.push_back(makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>(1'000, [&](auto /*unused*/) { return 1; }),
       makeFlatVector<int64_t>(1'000, [&](auto row) { return row + 1; })}));
  // The second vector has different partition key value.
  vectors.push_back(makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>(1'000, [&](auto row) { return row * 234 % 30; }),
       makeFlatVector<int64_t>(1'000, [&](auto row) { return row + 1; })}));
  createDuckDbTable(vectors);

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().values(vectors),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      numTableWriterCount_);

  assertQueryWithWriterConfigs(plan, "SELECT count(*) FROM tmp");

  auto newOutputType = getNonPartitionsColumns(partitionKeys, rowType);
  assertQuery(
      PlanBuilder().tableScan(newOutputType).planNode(),
      makeParquetConnectorSplits(outputDirectory),
      "SELECT c1 FROM tmp");
}

TEST_P(PartitionedTableWriterTest, maxPartitions) {
  SCOPED_TRACE(testParam_.toString());
  const int32_t maxPartitions = 100;
  const int32_t numPartitions =
      testMode_ == TestMode::kBucketed ? 1 : maxPartitions + 1;
  if (testMode_ == TestMode::kBucketed) {
    setBucketProperty(
        testParam_.bucketKind(),
        1000,
        bucketProperty_->bucketedBy(),
        bucketProperty_->bucketedTypes(),
        bucketProperty_->sortedBy());
  }

  auto rowType = ROW({"p0", "c3", "c5"}, {BIGINT(), REAL(), VARCHAR()});
  std::vector<std::string> partitionKeys = {"p0"};

  RowVectorPtr vector;
  if (testMode_ == TestMode::kPartitioned) {
    vector = makeRowVector(
        rowType->names(),
        {makeFlatVector<int64_t>(numPartitions, [&](auto row) { return row; }),
         makeFlatVector<float>(
             numPartitions, [&](auto row) { return row + 33.23; }),
         makeFlatVector<StringView>(numPartitions, [&](auto row) {
           return StringView::makeInline(fmt::format("bucket_{}", row * 3));
         })});
  } else {
    vector = makeRowVector(
        rowType->names(),
        {makeFlatVector<int64_t>(4'000, [&](auto /*unused*/) { return 0; }),
         makeFlatVector<float>(4'000, [&](auto row) { return row + 33.23; }),
         makeFlatVector<StringView>(4'000, [&](auto row) {
           return StringView::makeInline(fmt::format("bucket_{}", row * 3));
         })});
  };

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().values({vector}),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  if (testMode_ == TestMode::kPartitioned) {
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan)
            .connectorSessionProperty(
                kParquetConnectorId,
                ParquetConfig::kMaxPartitionsPerWritersSession,
                folly::to<std::string>(maxPartitions))
            .copyResults(pool()),
        fmt::format(
            "Exceeded limit of {} distinct partitions.", maxPartitions));
  } else {
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan)
            .connectorSessionProperty(
                kParquetConnectorId,
                ParquetConfig::kMaxPartitionsPerWritersSession,
                folly::to<std::string>(maxPartitions))
            .copyResults(pool()),
        "Exceeded open writer limit");
  }
}

// Test TableWriter does not create a file if input is empty.
TEST_P(AllTableWriterTest, writeNoFile) {
  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType_).filter("false"),
      rowType_,
      outputDirectory->getPath());

  auto execute = [&](const std::shared_ptr<const core::PlanNode>& plan,
                     std::shared_ptr<core::QueryCtx> queryCtx) {
    CursorParameters params;
    params.planNode = plan;
    params.queryCtx = queryCtx;
    readCursor(params, [&](Task* task) { task->noMoreSplits("0"); });
  };

  execute(plan, core::QueryCtx::create(executor_.get()));
  ASSERT_TRUE(fs::is_empty(outputDirectory->getPath()));
}

TEST_P(UnpartitionedTableWriterTest, differentCompression) {
  std::vector<CompressionKind> compressions{
      CompressionKind_NONE,
      CompressionKind_ZLIB,
      CompressionKind_SNAPPY,
      CompressionKind_LZO,
      CompressionKind_ZSTD,
      CompressionKind_LZ4,
      CompressionKind_GZIP,
      CompressionKind_MAX};

  for (auto compressionKind : compressions) {
    auto input = makeVectors(10, 10);
    auto outputDirectory = TempDirectoryPath::create();
    if (compressionKind == CompressionKind_MAX) {
      VELOX_ASSERT_THROW(
          createInsertPlan(
              PlanBuilder().values(input),
              rowType_,
              outputDirectory->getPath(),
              compressionKind,
              numTableWriterCount_,
              connector::parquet::LocationHandle::TableType::kNew),
          "Unsupported compression type: CompressionKind_MAX");
      return;
    }
    auto plan = createInsertPlan(
        PlanBuilder().values(input),
        rowType_,
        outputDirectory->getPath(),
        compressionKind,
        numTableWriterCount_,
        connector::parquet::LocationHandle::TableType::kNew);

    // currently we don't support any compression in PARQUET format
    if (fileFormat_ == FileFormat::PARQUET &&
        compressionKind != CompressionKind_NONE) {
      continue;
    }
    if (compressionKind == CompressionKind_NONE ||
        compressionKind == CompressionKind_ZLIB ||
        compressionKind == CompressionKind_ZSTD) {
      auto result = AssertQueryBuilder(plan)
                        .config(
                            QueryConfig::kTaskWriterCount,
                            std::to_string(numTableWriterCount_))
                        .copyResults(pool());
      assertEqualResults(
          {makeRowVector({makeConstant<int64_t>(100, 1)})}, {result});
    } else {
      VELOX_ASSERT_THROW(
          AssertQueryBuilder(plan)
              .config(
                  QueryConfig::kTaskWriterCount,
                  std::to_string(numTableWriterCount_))
              .copyResults(pool()),
          "Unsupported compression type:");
    }
  }
}

TEST_P(UnpartitionedTableWriterTest, runtimeStatsCheck) {
  // The runtime stats test only applies for dwrf file format.
  if (fileFormat_ != dwio::common::FileFormat::DWRF) {
    return;
  }
  struct {
    int numInputVectors;
    std::string maxStripeSize;
    int expectedNumStripes;

    std::string debugString() const {
      return fmt::format(
          "numInputVectors: {}, maxStripeSize: {}, expectedNumStripes: {}",
          numInputVectors,
          maxStripeSize,
          expectedNumStripes);
    }
  } testSettings[] = {
      {10, "1GB", 1},
      {1, "1GB", 1},
      {2, "1GB", 1},
      {10, "1B", 10},
      {2, "1B", 2},
      {1, "1B", 1}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto rowType = ROW({"c0", "c1"}, {VARCHAR(), BIGINT()});

    VectorFuzzer::Options options;
    options.nullRatio = 0.0;
    options.vectorSize = 1;
    options.stringLength = 1L << 20;
    VectorFuzzer fuzzer(options, pool());

    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < testData.numInputVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }

    createDuckDbTable(vectors);

    auto outputDirectory = TempDirectoryPath::create();
    auto plan = createInsertPlan(
        PlanBuilder().values(vectors),
        rowType,
        outputDirectory->getPath(),
        compressionKind_,
        1,
        connector::parquet::LocationHandle::TableType::kNew);
    const std::shared_ptr<Task> task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .config(QueryConfig::kTaskWriterCount, std::to_string(1))
            .connectorSessionProperty(
                kParquetConnectorId,
                ParquetConfig::kOrcWriterMaxStripeSizeSession,
                testData.maxStripeSize)
            .assertResults("SELECT count(*) FROM tmp");
    auto stats = task->taskStats().pipelineStats.front().operatorStats;
    if (testData.maxStripeSize == "1GB") {
      ASSERT_GT(
          stats[1].memoryStats.peakTotalMemoryReservation,
          testData.numInputVectors * options.stringLength);
    }
    ASSERT_EQ(
        stats[1].runtimeStats["stripeSize"].count, testData.expectedNumStripes);
    ASSERT_EQ(stats[1].runtimeStats["numWrittenFiles"].sum, 1);
    ASSERT_EQ(stats[1].runtimeStats["numWrittenFiles"].count, 1);
    ASSERT_GE(stats[1].runtimeStats["writeIOTime"].sum, 0);
    ASSERT_EQ(stats[1].runtimeStats["writeIOTime"].count, 1);
  }
}

TEST_P(UnpartitionedTableWriterTest, immutableSettings) {
  struct {
    connector::parquet::LocationHandle::TableType dataType;
    bool immutablePartitionsEnabled;
    bool expectedInsertSuccees;

    std::string debugString() const {
      return fmt::format(
          "dataType:{}, immutablePartitionsEnabled:{}, operationSuccess:{}",
          dataType,
          immutablePartitionsEnabled,
          expectedInsertSuccees);
    }
  } testSettings[] = {
      {connector::parquet::LocationHandle::TableType::kNew, true, true},
      {connector::parquet::LocationHandle::TableType::kNew, false, true},
      {connector::parquet::LocationHandle::TableType::kExisting, true, false},
      {connector::parquet::LocationHandle::TableType::kExisting, false, true}};

  for (auto testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    std::unordered_map<std::string, std::string> propFromFile{
        {"parquet.immutable-partitions",
         testData.immutablePartitionsEnabled ? "true" : "false"}};
    std::shared_ptr<const config::ConfigBase> config{
        std::make_shared<config::ConfigBase>(std::move(propFromFile))};
    resetParquetConnector(config);

    auto input = makeVectors(10, 10);
    auto outputDirectory = TempDirectoryPath::create();
    auto plan = createInsertPlan(
        PlanBuilder().values(input),
        rowType_,
        outputDirectory->getPath(),
        CompressionKind_NONE,
        numTableWriterCount_,
        testData.dataType);

    if (!testData.expectedInsertSuccees) {
      VELOX_ASSERT_THROW(
          AssertQueryBuilder(plan).copyResults(pool()),
          "Unpartitioned Parquet tables are immutable.");
    } else {
      auto result = AssertQueryBuilder(plan)
                        .config(
                            QueryConfig::kTaskWriterCount,
                            std::to_string(numTableWriterCount_))
                        .copyResults(pool());
      assertEqualResults(
          {makeRowVector({makeConstant<int64_t>(100, 1)})}, {result});
    }
  }
}

TEST_P(BucketedTableOnlyWriteTest, bucketCountLimit) {
  SCOPED_TRACE(testParam_.toString());
  auto input = makeVectors(1, 100);
  createDuckDbTable(input);
  struct {
    uint32_t bucketCount;
    bool expectedError;

    std::string debugString() const {
      return fmt::format(
          "bucketCount:{} expectedError:{}", bucketCount, expectedError);
    }
  } testSettings[] = {
      {1, false},
      {3, false},
      {ParquetDataSink::maxBucketCount() - 1, false},
      {ParquetDataSink::maxBucketCount(), true},
      {ParquetDataSink::maxBucketCount() + 1, true},
      {ParquetDataSink::maxBucketCount() * 2, true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto outputDirectory = TempDirectoryPath::create();
    setBucketProperty(
        bucketProperty_->kind(),
        testData.bucketCount,
        bucketProperty_->bucketedBy(),
        bucketProperty_->bucketedTypes(),
        bucketProperty_->sortedBy());
    auto plan = createInsertPlan(
        PlanBuilder().values({input}),
        rowType_,
        outputDirectory->getPath(),
        compressionKind_,
        getNumWriters(),
        connector::parquet::LocationHandle::TableType::kNew,
        commitStrategy_);
    if (testData.expectedError) {
      VELOX_ASSERT_THROW(
          AssertQueryBuilder(plan)
              .connectorSessionProperty(
                  kParquetConnectorId,
                  ParquetConfig::kMaxPartitionsPerWritersSession,
                  // Make sure we have a sufficient large writer limit.
                  folly::to<std::string>(testData.bucketCount * 2))
              .copyResults(pool()),
          "bucketCount exceeds the limit");
    } else {
      assertQueryWithWriterConfigs(plan, "SELECT count(*) FROM tmp");

      if (partitionedBy_.size() > 0) {
        auto newOutputType =
            getNonPartitionsColumns(partitionedBy_, tableSchema_);
        assertQuery(
            PlanBuilder().tableScan(newOutputType).planNode(),
            makeParquetConnectorSplits(outputDirectory),
            "SELECT c2, c3, c4, c5 FROM tmp");
        auto originalRowType = rowType_;
        rowType_ = newOutputType;
        verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
        rowType_ = originalRowType;
      } else {
        assertQuery(
            PlanBuilder().tableScan(rowType_).planNode(),
            makeParquetConnectorSplits(outputDirectory),
            "SELECT * FROM tmp");
        verifyTableWriterOutput(outputDirectory->getPath(), rowType_);
      }
    }
  }
}

TEST_P(BucketedTableOnlyWriteTest, mismatchedBucketTypes) {
  SCOPED_TRACE(testParam_.toString());
  auto input = makeVectors(1, 100);
  createDuckDbTable(input);
  auto outputDirectory = TempDirectoryPath::create();
  std::vector<TypePtr> badBucketedBy = bucketProperty_->bucketedTypes();
  const auto oldType = badBucketedBy[0];
  badBucketedBy[0] = VARCHAR();
  setBucketProperty(
      bucketProperty_->kind(),
      bucketProperty_->bucketCount(),
      bucketProperty_->bucketedBy(),
      badBucketedBy,
      bucketProperty_->sortedBy());
  auto plan = createInsertPlan(
      PlanBuilder().values({input}),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      fmt::format(
          "Input column {} type {} doesn't match bucket type {}",
          bucketProperty_->bucketedBy()[0],
          oldType->toString(),
          bucketProperty_->bucketedTypes()[0]));
}

TEST_P(AllTableWriterTest, tableWriteOutputCheck) {
  SCOPED_TRACE(testParam_.toString());
  if (!testParam_.multiDrivers() ||
      testParam_.testMode() != TestMode::kUnpartitioned) {
    return;
  }
  auto input = makeVectors(10, 100);
  createDuckDbTable(input);
  auto outputDirectory = TempDirectoryPath::create();
  auto plan = createInsertPlan(
      PlanBuilder().values({input}),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_,
      false);

  auto result = runQueryWithWriterConfigs(plan);
  auto writtenRowVector = result->childAt(TableWriteTraits::kRowCountChannel)
                              ->asFlatVector<int64_t>();
  auto fragmentVector = result->childAt(TableWriteTraits::kFragmentChannel)
                            ->asFlatVector<StringView>();
  auto commitContextVector = result->childAt(TableWriteTraits::kContextChannel)
                                 ->asFlatVector<StringView>();
  const int64_t expectedRows = 10 * 100;
  std::vector<std::string> writeFiles;
  int64_t numRows{0};
  for (int i = 0; i < result->size(); ++i) {
    if (testParam_.multiDrivers()) {
      ASSERT_FALSE(commitContextVector->isNullAt(i));
      if (!fragmentVector->isNullAt(i)) {
        ASSERT_TRUE(writtenRowVector->isNullAt(i));
      }
    } else {
      if (i == 0) {
        ASSERT_TRUE(fragmentVector->isNullAt(i));
      } else {
        ASSERT_TRUE(writtenRowVector->isNullAt(i));
        ASSERT_FALSE(fragmentVector->isNullAt(i));
      }
      ASSERT_FALSE(commitContextVector->isNullAt(i));
    }
    if (!fragmentVector->isNullAt(i)) {
      ASSERT_FALSE(fragmentVector->isNullAt(i));
      folly::dynamic obj = folly::parseJson(fragmentVector->valueAt(i));
      if (testMode_ == TestMode::kUnpartitioned) {
        ASSERT_EQ(obj["targetPath"], outputDirectory->getPath());
        ASSERT_EQ(obj["writePath"], outputDirectory->getPath());
      } else {
        std::string partitionDirRe;
        for (const auto& partitionBy : partitionedBy_) {
          partitionDirRe += fmt::format("/{}=.+", partitionBy);
        }
        ASSERT_TRUE(RE2::FullMatch(
            obj["targetPath"].asString(),
            fmt::format("{}{}", outputDirectory->getPath(), partitionDirRe)))
            << obj["targetPath"].asString();
        ASSERT_TRUE(RE2::FullMatch(
            obj["writePath"].asString(),
            fmt::format("{}{}", outputDirectory->getPath(), partitionDirRe)))
            << obj["writePath"].asString();
      }
      numRows += obj["rowCount"].asInt();
      ASSERT_EQ(obj["updateMode"].asString(), "NEW");

      ASSERT_TRUE(obj["fileWriteInfos"].isArray());
      ASSERT_EQ(obj["fileWriteInfos"].size(), 1);
      folly::dynamic writerInfoObj = obj["fileWriteInfos"][0];
      const std::string writeFileName =
          writerInfoObj["writeFileName"].asString();
      writeFiles.push_back(writeFileName);
      const std::string targetFileName =
          writerInfoObj["targetFileName"].asString();
      const std::string writeFileFullPath =
          obj["writePath"].asString() + "/" + writeFileName;
      std::filesystem::path path{writeFileFullPath};
      const auto actualFileSize = fs::file_size(path);
      ASSERT_EQ(obj["onDiskDataSizeInBytes"].asInt(), actualFileSize);
      ASSERT_GT(obj["inMemoryDataSizeInBytes"].asInt(), 0);
      ASSERT_EQ(writerInfoObj["fileSize"], actualFileSize);
      if (commitStrategy_ == CommitStrategy::kNoCommit) {
        ASSERT_EQ(writeFileName, targetFileName);
      } else {
        const std::string kParquetSuffix = ".parquet";
        if (folly::StringPiece(targetFileName).endsWith(kParquetSuffix)) {
          // Remove the .parquet suffix.
          auto trimmedFilename = targetFileName.substr(
              0, targetFileName.size() - kParquetSuffix.size());
          ASSERT_TRUE(writeFileName.find(trimmedFilename) != std::string::npos);
        } else {
          ASSERT_TRUE(writeFileName.find(targetFileName) != std::string::npos);
        }
      }
    }
    if (!commitContextVector->isNullAt(i)) {
      ASSERT_TRUE(RE2::FullMatch(
          commitContextVector->valueAt(i).getString(),
          fmt::format(".*{}.*", commitStrategyToString(commitStrategy_))))
          << commitContextVector->valueAt(i);
    }
  }
  ASSERT_EQ(numRows, expectedRows);
  if (testMode_ == TestMode::kUnpartitioned) {
    ASSERT_GT(writeFiles.size(), 0);
    ASSERT_LE(writeFiles.size(), numTableWriterCount_);
  }
  auto diskFiles = listAllFiles(outputDirectory->getPath());
  std::sort(diskFiles.begin(), diskFiles.end());
  std::sort(writeFiles.begin(), writeFiles.end());
  ASSERT_EQ(diskFiles, writeFiles)
      << "\nwrite files: " << folly::join(",", writeFiles)
      << "\ndisk files: " << folly::join(",", diskFiles);
  // Verify the utilities provided by table writer traits.
  ASSERT_EQ(TableWriteTraits::getRowCount(result), 10 * 100);
  auto obj = TableWriteTraits::getTableCommitContext(result);
  ASSERT_EQ(
      obj[TableWriteTraits::kCommitStrategyContextKey],
      commitStrategyToString(commitStrategy_));
  ASSERT_EQ(obj[TableWriteTraits::klastPageContextKey], true);
  ASSERT_EQ(obj[TableWriteTraits::kLifeSpanContextKey], "TaskWide");
}

TEST_P(AllTableWriterTest, columnStatsDataTypes) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           REAL(),
           DOUBLE(),
           VARCHAR(),
           BOOLEAN(),
           MAP(DATE(), BIGINT()),
           ARRAY(BIGINT())});
  setDataTypes(rowType);
  std::vector<RowVectorPtr> input;
  input.push_back(makeRowVector(
      rowType_->names(),
      {
          makeFlatVector<int64_t>(1'000, [&](auto row) { return 1; }),
          makeFlatVector<int32_t>(1'000, [&](auto row) { return 1; }),
          makeFlatVector<int16_t>(1'000, [&](auto row) { return row; }),
          makeFlatVector<float>(1'000, [&](auto row) { return row + 33.23; }),
          makeFlatVector<double>(1'000, [&](auto row) { return row + 33.23; }),
          makeFlatVector<StringView>(
              1'000,
              [&](auto row) {
                return StringView(std::to_string(row).c_str());
              }),
          makeFlatVector<bool>(1'000, [&](auto row) { return true; }),
          makeMapVector<int32_t, int64_t>(
              1'000,
              [](auto /*row*/) { return 5; },
              [](auto row) { return row; },
              [](auto row) { return row * 3; }),
          makeArrayVector<int64_t>(
              1'000,
              [](auto /*row*/) { return 5; },
              [](auto row) { return row * 3; }),
      }));
  createDuckDbTable(input);
  auto outputDirectory = TempDirectoryPath::create();

  std::vector<FieldAccessTypedExprPtr> groupingKeyFields;
  for (int i = 0; i < partitionedBy_.size(); ++i) {
    groupingKeyFields.emplace_back(std::make_shared<core::FieldAccessTypedExpr>(
        partitionTypes_.at(i), partitionedBy_.at(i)));
  }

  // aggregation node
  core::TypedExprPtr intInputField =
      std::make_shared<const core::FieldAccessTypedExpr>(SMALLINT(), "c2");
  auto minCallExpr = std::make_shared<const core::CallTypedExpr>(
      SMALLINT(), std::vector<core::TypedExprPtr>{intInputField}, "min");
  auto maxCallExpr = std::make_shared<const core::CallTypedExpr>(
      SMALLINT(), std::vector<core::TypedExprPtr>{intInputField}, "max");
  auto distinctCountCallExpr = std::make_shared<const core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{intInputField},
      "approx_distinct");

  core::TypedExprPtr strInputField =
      std::make_shared<const core::FieldAccessTypedExpr>(VARCHAR(), "c5");
  auto maxDataSizeCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{strInputField},
      "max_data_size_for_stats");
  auto sumDataSizeCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{strInputField},
      "sum_data_size_for_stats");

  core::TypedExprPtr boolInputField =
      std::make_shared<const core::FieldAccessTypedExpr>(BOOLEAN(), "c6");
  auto countCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{boolInputField}, "count");
  auto countIfCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{boolInputField}, "count_if");

  core::TypedExprPtr mapInputField =
      std::make_shared<const core::FieldAccessTypedExpr>(
          MAP(DATE(), BIGINT()), "c7");
  auto countMapCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{mapInputField}, "count");
  auto sumDataSizeMapCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{mapInputField},
      "sum_data_size_for_stats");

  core::TypedExprPtr arrayInputField =
      std::make_shared<const core::FieldAccessTypedExpr>(
          MAP(DATE(), BIGINT()), "c7");
  auto countArrayCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{mapInputField}, "count");
  auto sumDataSizeArrayCallExpr = std::make_shared<const core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{mapInputField},
      "sum_data_size_for_stats");

  const std::vector<std::string> aggregateNames = {
      "min",
      "max",
      "approx_distinct",
      "max_data_size_for_stats",
      "sum_data_size_for_stats",
      "count",
      "count_if",
      "count",
      "sum_data_size_for_stats",
      "count",
      "sum_data_size_for_stats",
  };

  auto makeAggregate = [](const auto& callExpr) {
    std::vector<TypePtr> rawInputTypes;
    for (const auto& input : callExpr->inputs()) {
      rawInputTypes.push_back(input->type());
    }
    return core::AggregationNode::Aggregate{
        callExpr,
        rawInputTypes,
        nullptr, // mask
        {}, // sortingKeys
        {} // sortingOrders
    };
  };

  std::vector<core::AggregationNode::Aggregate> aggregates = {
      makeAggregate(minCallExpr),
      makeAggregate(maxCallExpr),
      makeAggregate(distinctCountCallExpr),
      makeAggregate(maxDataSizeCallExpr),
      makeAggregate(sumDataSizeCallExpr),
      makeAggregate(countCallExpr),
      makeAggregate(countIfCallExpr),
      makeAggregate(countMapCallExpr),
      makeAggregate(sumDataSizeMapCallExpr),
      makeAggregate(countArrayCallExpr),
      makeAggregate(sumDataSizeArrayCallExpr),
  };
  const auto aggregationNode = std::make_shared<core::AggregationNode>(
      core::PlanNodeId(),
      core::AggregationNode::Step::kPartial,
      groupingKeyFields,
      std::vector<core::FieldAccessTypedExprPtr>{},
      aggregateNames,
      aggregates,
      false, // ignoreNullKeys
      PlanBuilder().values({input}).planNode());

  auto plan = PlanBuilder()
                  .values({input})
                  .addNode(addTableWriter(
                      rowType_,
                      rowType_->names(),
                      aggregationNode,
                      std::make_shared<core::InsertTableHandle>(
                          kParquetConnectorId,
                          makeParquetInsertTableHandle(
                              rowType_->names(),
                              rowType_->children(),
                              partitionedBy_,
                              nullptr,
                              makeLocationHandle(outputDirectory->getPath()))),
                      CommitStrategy::kNoCommit))
                  .planNode();

  // the result is in format of : row/fragments/context/[partition]/[stats]
  int nextColumnStatsIndex = 3 + partitionedBy_.size();
  const RowVectorPtr result = AssertQueryBuilder(plan).copyResults(pool());
  auto minStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int16_t>();
  ASSERT_EQ(minStatsVector->valueAt(0), 0);
  const auto maxStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int16_t>();
  ASSERT_EQ(maxStatsVector->valueAt(0), 999);
  const auto distinctCountStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<StringView>();
  HashStringAllocator allocator{pool_.get()};
  DenseHll denseHll{
      std::string(distinctCountStatsVector->valueAt(0)).c_str(), &allocator};
  ASSERT_EQ(denseHll.cardinality(), 1000);
  const auto maxDataSizeStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(maxDataSizeStatsVector->valueAt(0), 7);
  const auto sumDataSizeStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(sumDataSizeStatsVector->valueAt(0), 6890);
  const auto countStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(countStatsVector->valueAt(0), 1000);
  const auto countIfStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(countIfStatsVector->valueAt(0), 1000);
  const auto countMapStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(countMapStatsVector->valueAt(0), 1000);
  const auto sumDataSizeMapStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(sumDataSizeMapStatsVector->valueAt(0), 64000);
  const auto countArrayStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(countArrayStatsVector->valueAt(0), 1000);
  const auto sumDataSizeArrayStatsVector =
      result->childAt(nextColumnStatsIndex++)->asFlatVector<int64_t>();
  ASSERT_EQ(sumDataSizeArrayStatsVector->valueAt(0), 64000);
}

TEST_P(AllTableWriterTest, columnStats) {
  auto input = makeVectors(1, 100);
  createDuckDbTable(input);
  auto outputDirectory = TempDirectoryPath::create();

  // 1. standard columns
  std::vector<std::string> output = {
      "numWrittenRows", "fragment", "tableCommitContext"};
  std::vector<TypePtr> types = {BIGINT(), VARBINARY(), VARBINARY()};
  std::vector<core::FieldAccessTypedExprPtr> groupingKeys;
  // 2. partition columns
  for (int i = 0; i < partitionedBy_.size(); i++) {
    groupingKeys.emplace_back(
        std::make_shared<const core::FieldAccessTypedExpr>(
            partitionTypes_.at(i), partitionedBy_.at(i)));
    output.emplace_back(partitionedBy_.at(i));
    types.emplace_back(partitionTypes_.at(i));
  }
  // 3. stats columns
  output.emplace_back("min");
  types.emplace_back(BIGINT());
  const auto writerOutputType = ROW(std::move(output), std::move(types));

  // aggregation node
  auto aggregationNode = generateAggregationNode(
      "c0",
      groupingKeys,
      core::AggregationNode::Step::kPartial,
      PlanBuilder().values({input}).planNode());

  auto plan = PlanBuilder()
                  .values({input})
                  .addNode(addTableWriter(
                      rowType_,
                      rowType_->names(),
                      aggregationNode,
                      std::make_shared<core::InsertTableHandle>(
                          kParquetConnectorId,
                          makeParquetInsertTableHandle(
                              rowType_->names(),
                              rowType_->children(),
                              partitionedBy_,
                              bucketProperty_,
                              makeLocationHandle(outputDirectory->getPath()))),
                      commitStrategy_))
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto rowVector = result->childAt(0)->asFlatVector<int64_t>();
  auto fragmentVector = result->childAt(1)->asFlatVector<StringView>();
  auto columnStatsVector =
      result->childAt(3 + partitionedBy_.size())->asFlatVector<int64_t>();

  std::vector<std::string> writeFiles;

  // For partitioned, expected result is as follows:
  // Row     Fragment           Context       partition           c1_min_value
  // null    null                x            partition1          0
  // null    null                x            partition2          10
  // null    null                x            partition3          15
  // count   null                x            null                null
  // null    partition1_update   x            null                null
  // null    partition1_update   x            null                null
  // null    partition2_update   x            null                null
  // null    partition2_update   x            null                null
  // null    partition3_update   x            null                null
  //
  // Note that we can have multiple same partition_update, they're for
  // different files, but for stats, we would only have one record for each
  // partition
  //
  // For unpartitioned, expected result is:
  // Row     Fragment           Context       partition           c1_min_value
  // null    null                x                                0
  // count   null                x            null                null
  // null    update              x            null                null

  int countRow = 0;
  while (!columnStatsVector->isNullAt(countRow)) {
    countRow++;
  }
  for (int i = 0; i < result->size(); ++i) {
    if (i < countRow) {
      ASSERT_FALSE(columnStatsVector->isNullAt(i));
      ASSERT_TRUE(rowVector->isNullAt(i));
      ASSERT_TRUE(fragmentVector->isNullAt(i));
    } else if (i == countRow) {
      ASSERT_TRUE(columnStatsVector->isNullAt(i));
      ASSERT_FALSE(rowVector->isNullAt(i));
      ASSERT_TRUE(fragmentVector->isNullAt(i));
    } else {
      ASSERT_TRUE(columnStatsVector->isNullAt(i));
      ASSERT_TRUE(rowVector->isNullAt(i));
      ASSERT_FALSE(fragmentVector->isNullAt(i));
    }
  }
}

TEST_P(AllTableWriterTest, columnStatsWithTableWriteMerge) {
  auto input = makeVectors(1, 100);
  createDuckDbTable(input);
  auto outputDirectory = TempDirectoryPath::create();

  // 1. standard columns
  std::vector<std::string> output = {
      "numWrittenRows", "fragment", "tableCommitContext"};
  std::vector<TypePtr> types = {BIGINT(), VARBINARY(), VARBINARY()};
  std::vector<core::FieldAccessTypedExprPtr> groupingKeys;
  // 2. partition columns
  for (int i = 0; i < partitionedBy_.size(); i++) {
    groupingKeys.emplace_back(
        std::make_shared<const core::FieldAccessTypedExpr>(
            partitionTypes_.at(i), partitionedBy_.at(i)));
    output.emplace_back(partitionedBy_.at(i));
    types.emplace_back(partitionTypes_.at(i));
  }
  // 3. stats columns
  output.emplace_back("min");
  types.emplace_back(BIGINT());
  const auto writerOutputType = ROW(std::move(output), std::move(types));

  // aggregation node
  auto aggregationNode = generateAggregationNode(
      "c0",
      groupingKeys,
      core::AggregationNode::Step::kPartial,
      PlanBuilder().values({input}).planNode());

  auto tableWriterPlan = PlanBuilder().values({input}).addNode(addTableWriter(
      rowType_,
      rowType_->names(),
      aggregationNode,
      std::make_shared<core::InsertTableHandle>(
          kParquetConnectorId,
          makeParquetInsertTableHandle(
              rowType_->names(),
              rowType_->children(),
              partitionedBy_,
              bucketProperty_,
              makeLocationHandle(outputDirectory->getPath()))),
      commitStrategy_));

  auto mergeAggregationNode = generateAggregationNode(
      "min",
      groupingKeys,
      core::AggregationNode::Step::kIntermediate,
      std::move(tableWriterPlan.planNode()));

  auto finalPlan = tableWriterPlan.capturePlanNodeId(tableWriteNodeId_)
                       .localPartition(std::vector<std::string>{})
                       .tableWriteMerge(std::move(mergeAggregationNode))
                       .planNode();

  auto result = AssertQueryBuilder(finalPlan).copyResults(pool());
  auto rowVector = result->childAt(0)->asFlatVector<int64_t>();
  auto fragmentVector = result->childAt(1)->asFlatVector<StringView>();
  auto columnStatsVector =
      result->childAt(3 + partitionedBy_.size())->asFlatVector<int64_t>();

  std::vector<std::string> writeFiles;

  // For partitioned, expected result is as follows:
  // Row     Fragment           Context       partition           c1_min_value
  // null    null                x            partition1          0
  // null    null                x            partition2          10
  // null    null                x            partition3          15
  // count   null                x            null                null
  // null    partition1_update   x            null                null
  // null    partition1_update   x            null                null
  // null    partition2_update   x            null                null
  // null    partition2_update   x            null                null
  // null    partition3_update   x            null                null
  //
  // Note that we can have multiple same partition_update, they're for
  // different files, but for stats, we would only have one record for each
  // partition
  //
  // For unpartitioned, expected result is:
  // Row     Fragment           Context       partition           c1_min_value
  // null    null                x                                0
  // count   null                x            null                null
  // null    update              x            null                null

  int statsRow = 0;
  while (columnStatsVector->isNullAt(statsRow) && statsRow < result->size()) {
    ++statsRow;
  }
  for (int i = 1; i < result->size(); ++i) {
    if (i < statsRow) {
      ASSERT_TRUE(rowVector->isNullAt(i));
      ASSERT_FALSE(fragmentVector->isNullAt(i));
      ASSERT_TRUE(columnStatsVector->isNullAt(i));
    } else if (i < result->size() - 1) {
      ASSERT_TRUE(rowVector->isNullAt(i));
      ASSERT_TRUE(fragmentVector->isNullAt(i));
      ASSERT_FALSE(columnStatsVector->isNullAt(i));
    } else {
      ASSERT_FALSE(rowVector->isNullAt(i));
      ASSERT_TRUE(fragmentVector->isNullAt(i));
      ASSERT_TRUE(columnStatsVector->isNullAt(i));
    }
  }
}

TEST_P(AllTableWriterTest, tableWriterStats) {
  const int32_t numBatches = 2;
  auto rowType =
      ROW({"c0", "p0", "c3", "c5"}, {VARCHAR(), BIGINT(), REAL(), VARCHAR()});
  std::vector<std::string> partitionKeys = {"p0"};

  VectorFuzzer::Options options;
  options.vectorSize = 1000;
  VectorFuzzer fuzzer(options, pool());
  // Partition vector is constant vector.
  std::vector<RowVectorPtr> vectors = makeBatches(numBatches, [&](auto) {
    return makeRowVector(
        rowType->names(),
        {fuzzer.fuzzFlat(VARCHAR()),
         fuzzer.fuzzConstant(BIGINT()),
         fuzzer.fuzzFlat(REAL()),
         fuzzer.fuzzFlat(VARCHAR())});
  });
  createDuckDbTable(vectors);

  auto inputFilePaths = makeFilePaths(numBatches);
  for (int i = 0; i < numBatches; i++) {
    writeToFile(inputFilePaths[i]->getPath(), vectors[i]);
  }

  auto outputDirectory = TempDirectoryPath::create();
  const int numWriters = getNumWriters();
  auto plan = createInsertPlan(
      PlanBuilder().tableScan(rowType),
      rowType,
      outputDirectory->getPath(),
      compressionKind_,
      numWriters,
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  auto task = assertQueryWithWriterConfigs(
      plan, inputFilePaths, "SELECT count(*) FROM tmp");

  // Each batch would create a new partition, numWrittenFiles is same as
  // partition num when not bucketed. When bucketed, it's partitionNum *
  // bucketNum, bucket number is 4
  const int numWrittenFiles =
      bucketProperty_ == nullptr ? numBatches : numBatches * 4;
  // The size of bytes (ORC_MAGIC_LEN) written when the DWRF writer
  // initializes a file.
  const int32_t ORC_HEADER_LEN{3};
  const auto fixedWrittenBytes =
      numWrittenFiles * (fileFormat_ == FileFormat::DWRF ? ORC_HEADER_LEN : 0);

  auto planStats = exec::toPlanStats(task->taskStats());
  auto& stats = planStats.at(tableWriteNodeId_);
  ASSERT_GT(stats.physicalWrittenBytes, fixedWrittenBytes);
  ASSERT_GT(
      stats.operatorStats.at("TableWrite")->physicalWrittenBytes,
      fixedWrittenBytes);
  ASSERT_EQ(
      stats.operatorStats.at("TableWrite")
          ->customStats.at("numWrittenFiles")
          .sum,
      numWrittenFiles);
  ASSERT_GE(
      stats.operatorStats.at("TableWrite")->customStats.at("writeIOTime").sum,
      0);
}

DEBUG_ONLY_TEST_P(
    UnpartitionedTableWriterTest,
    fileWriterFlushErrorOnDriverClose) {
  VectorFuzzer::Options options;
  const int batchSize = 1000;
  options.vectorSize = batchSize;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 10;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzRow(rowType_));
  }
  std::atomic<int> writeInputs{0};
  std::atomic<bool> triggerWriterOOM{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::addInput",
      std::function<void(Operator*)>([&](Operator* op) {
        if (op->operatorType() != "TableWrite") {
          return;
        }
        if (++writeInputs != 3) {
          return;
        }
        op->testingOperatorCtx()->task()->requestAbort();
        triggerWriterOOM = true;
      }));
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MemoryPoolImpl::reserveThreadSafe",
      std::function<void(memory::MemoryPool*)>([&](memory::MemoryPool* pool) {
        const std::string dictPoolRe(".*dictionary");
        const std::string generalPoolRe(".*general");
        const std::string compressionPoolRe(".*compression");
        if (!RE2::FullMatch(pool->name(), dictPoolRe) &&
            !RE2::FullMatch(pool->name(), generalPoolRe) &&
            !RE2::FullMatch(pool->name(), compressionPoolRe)) {
          return;
        }
        if (!triggerWriterOOM) {
          return;
        }
        VELOX_MEM_POOL_CAP_EXCEEDED("Inject write OOM");
      }));

  auto outputDirectory = TempDirectoryPath::create();
  auto op = createInsertPlan(
      PlanBuilder().values(vectors),
      rowType_,
      outputDirectory->getPath(),
      compressionKind_,
      getNumWriters(),
      connector::parquet::LocationHandle::TableType::kNew,
      commitStrategy_);

  VELOX_ASSERT_THROW(
      assertQuery(op, fmt::format("SELECT {}", numRows)),
      "Aborted for external error");
}

DEBUG_ONLY_TEST_P(UnpartitionedTableWriterTest, dataSinkAbortError) {
  if (fileFormat_ != FileFormat::DWRF) {
    // NOTE: only test on dwrf writer format as we inject write error in dwrf
    // writer.
    return;
  }
  VectorFuzzer::Options options;
  const int batchSize = 100;
  options.vectorSize = batchSize;
  VectorFuzzer fuzzer(options, pool());
  auto vector = fuzzer.fuzzInputRow(rowType_);

  std::atomic<bool> triggerWriterErrorOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::write",
      std::function<void(dwrf::Writer*)>([&](dwrf::Writer* /*unused*/) {
        if (!triggerWriterErrorOnce.exchange(false)) {
          return;
        }
        VELOX_FAIL("inject writer error");
      }));

  std::atomic<bool> triggerAbortErrorOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::connector::parquet::ParquetDataSink::closeInternal",
      std::function<void(const ParquetDataSink*)>(
          [&](const ParquetDataSink* /*unused*/) {
            if (!triggerAbortErrorOnce.exchange(false)) {
              return;
            }
            VELOX_FAIL("inject abort error");
          }));

  auto outputDirectory = TempDirectoryPath::create();
  auto plan = PlanBuilder()
                  .values({vector})
                  .tableWrite(outputDirectory->getPath(), fileFormat_)
                  .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()), "inject writer error");
  ASSERT_FALSE(triggerWriterErrorOnce);
  ASSERT_FALSE(triggerAbortErrorOnce);
}
#endif

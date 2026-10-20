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

#include "velox/connectors/hive/FileDataSource.h"

#include <future>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/LocalFile.h"
#include "velox/connectors/hive/FileScanState.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive {
namespace {
using exec::test::HiveConnectorTestBase;
using exec::test::kHiveConnectorId;
using exec::test::TempFilePath;

class InspectableFileDataSource : public FileDataSource {
 public:
  using FileDataSource::FileDataSource;

  const auto& scanState() const {
    return fileScanState_;
  }

  const auto& configuration() const {
    return fileScanConfig_;
  }

  void addPhysicalColumn(const std::string& name, const TypePtr& type) {
    auto names = readerOutputType_->names();
    auto types = readerOutputType_->children();
    names.push_back(name);
    types.push_back(type);
    readerOutputType_ = ROW(std::move(names), std::move(types));
    resetScanSpec();
  }
};

class FileDataSourceTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    queryCtx_ = core::QueryCtx::create();
    context_ = makeContext();
#ifdef VELOX_ENABLE_NIMBLE
    nimble::registerSelectiveNimbleReaderFactory();
#endif
  }

  void TearDown() override {
#ifdef VELOX_ENABLE_NIMBLE
    nimble::unregisterSelectiveNimbleReaderFactory();
#endif
    context_.reset();
    queryCtx_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::unique_ptr<ConnectorQueryCtx> makeContext(
      memory::MemoryPool* pool = nullptr) {
    pool = pool ? pool : pool_.get();
    return ConnectorQueryCtx::Builder()
        .operatorPool(pool)
        .connectorPool(pool)
        .sessionProperties(config_.get())
        .expressionEvaluator(
            std::make_unique<exec::SimpleExpressionEvaluator>(
                queryCtx_.get(), pool))
        .queryId("FileDataSourceTest")
        .taskId("FileDataSourceTest")
        .planNodeId("scan")
        .build();
  }

  std::shared_ptr<TempFilePath> writeScanFile(
      const RowVectorPtr& input,
      dwio::common::FileFormat format) {
    auto file = TempFilePath::create();
    if (format == dwio::common::FileFormat::DWRF) {
      writeToFile(file->getPath(), input);
    }
#ifdef VELOX_ENABLE_NIMBLE
    else {
      nimble::Writer writer(
          input->type(),
          std::make_unique<LocalWriteFile>(file->getPath(), false, false),
          *pool_,
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
    }
#endif
    return file;
  }

  static std::vector<dwio::common::FileFormat> fileFormats() {
    std::vector<dwio::common::FileFormat> formats{
        dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
    formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
    return formats;
  }

  RowVectorPtr next(DataSource& source, uint64_t size = 10) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source.next(size, future);
    VELOX_CHECK(result.has_value());
    VELOX_CHECK(!future.valid());
    return result.value();
  }

  static void expectStatsEqual(
      const std::unordered_map<std::string, RuntimeMetric>& expected,
      const std::unordered_map<std::string, RuntimeMetric>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (const auto& [name, metric] : expected) {
      SCOPED_TRACE(name);
      const auto it = actual.find(name);
      ASSERT_NE(it, actual.end());
      EXPECT_EQ(metric.unit, it->second.unit);
      EXPECT_EQ(metric.sum, it->second.sum);
      EXPECT_EQ(metric.count, it->second.count);
      EXPECT_EQ(metric.min, it->second.min);
      EXPECT_EQ(metric.max, it->second.max);
    }
  }

  void expectValues(
      const RowVectorPtr& result,
      const std::vector<int64_t>& values) {
    ASSERT_TRUE(result);
    test::assertEqualVectors(
        makeRowVector({makeFlatVector<int64_t>(values)}), result);
  }

  const RowTypePtr type_ = ROW({{"c0", BIGINT()}});
  const ColumnHandleMap assignments_{{"c0", regularColumn("c0", BIGINT())}};
  const std::shared_ptr<config::ConfigBase> config_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{});
  const std::shared_ptr<HiveConfig> hiveConfig_ =
      std::make_shared<HiveConfig>(config_);
  FileHandleFactory fileFactory_{
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(0),
      std::make_unique<FileHandleGenerator>()};
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> context_;
};

TEST_F(FileDataSourceTest, fileReaderReadFilterReuseAndTakeover) {
  std::vector<dwio::common::FileFormat> formats{dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
  formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
  std::vector<std::string> strings;
  strings.reserve(100);
  for (int i = 0; i < 100; ++i) {
    strings.push_back(fmt::format("out-of-line string value {}", i));
  }
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto i) { return i; }),
       makeFlatVector<std::string>(strings)});
  const auto fileType = asRowType(input->type());
  const ColumnHandleMap fileAssignments{
      {"c0", regularColumn("c0", BIGINT())},
      {"c1", regularColumn("c1", VARCHAR())}};
  for (auto format : formats) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto file = TempFilePath::create();
    if (format == dwio::common::FileFormat::DWRF) {
      writeToFile(file->getPath(), input);
    }
#ifdef VELOX_ENABLE_NIMBLE
    else {
      nimble::Writer writer(
          input->type(),
          std::make_unique<LocalWriteFile>(file->getPath(), false, false),
          *pool_,
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
    }
#endif
    auto makeFileSource = [&](const ConnectorQueryCtx* context,
                              const std::string& remaining = "") {
      return std::make_unique<HiveDataSource>(
          fileType,
          makeTableHandle(
              {},
              remaining.empty() ? nullptr : parseExpr(remaining, fileType),
              "test",
              fileType),
          fileAssignments,
          &fileFactory_,
          nullptr,
          context,
          hiveConfig_);
    };
    auto split = makeHiveConnectorSplit(file->getPath());
    split->fileFormat = format;
    auto preloadContext = makeContext();
    auto preloaded = makeFileSource(preloadContext.get());
    preloaded->addSplit(split);
    auto source = makeFileSource(context_.get());
    source->setFromDataSource(std::move(preloaded));
    preloadContext.reset();
    int callbacks = 0;
    source->setScanBatchCallback([&](const core::ScanBatchEvent& event) {
      const auto& fileEvent = static_cast<const FileScanBatchEvent&>(event);
      EXPECT_EQ(fileEvent.filePath, file->getPath());
      EXPECT_EQ(fileEvent.fileFormat, format);
      EXPECT_EQ(fileEvent.tableName, "test");
      ++callbacks;
    });
    auto first = next(*source, 10);
    source->fireScanBatchCallback({});
    ASSERT_EQ(callbacks, 1);
    // Load before advancing, as required by DWIO's lazy batch contract.
    first->loadedVector();
    auto second = next(*source, 10);
    second->loadedVector();
    ASSERT_EQ(source->getCompletedRows(), 20);
    const auto stats = source->getRuntimeStats();
    expectStatsEqual(stats, source->getRuntimeStats());
    source.reset();
    test::assertEqualVectors(input->slice(0, 10), first);
    test::assertEqualVectors(input->slice(10, 10), second);

    // An empty filtered batch is not EOF. A later batch still returns rows.
    auto filtered = makeFileSource(context_.get(), "c0 % 100 >= 10");
    filtered->addSplit(split);
    ASSERT_EQ(next(*filtered, 10)->size(), 0);
    auto rows = next(*filtered, 10);
    test::assertEqualVectors(input->slice(10, 10), rows);
    int count = rows->size();
    while (auto batch = next(*filtered, 10)) {
      count += batch->size();
    }
    ASSERT_EQ(count, 90);
    ASSERT_EQ(filtered->getCompletedRows(), 100);

    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_unique<common::BigintRange>(200, 300, false));
    auto skipped = std::make_unique<HiveDataSource>(
        fileType,
        makeTableHandle(std::move(filters), nullptr, "test", fileType),
        fileAssignments,
        &fileFactory_,
        nullptr,
        context_.get(),
        hiveConfig_);
    skipped->addSplit(split);
    while (auto batch = next(*skipped, 10)) {
      ASSERT_EQ(batch->size(), 0);
    }
  }
}

TEST_F(FileDataSourceTest, fileScanConfigPreservesLogicalInputsAndPolicy) {
  const auto dataType = ROW({{"c0", BIGINT()}, {"c1", BIGINT()}});
  const auto outputType = ROW({{"alias", BIGINT()}});
  ColumnHandleMap assignments{{"alias", regularColumn("c0", BIGINT())}};
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("c0"),
      std::make_unique<common::BigintRange>(0, 9, false));
  auto expression = parseExpr("c0 >= 3 AND c1 % 2 = 1", dataType);
  auto table =
      makeTableHandle(std::move(filters), expression, "test", dataType);
  for (bool extract : {true, false}) {
    SCOPED_TRACE(extract);
    auto fileScanConfig = std::make_shared<const FileScanConfig>(
        outputType,
        table,
        assignments,
        context_.get(),
        hiveConfig_,
        FileScanConfig::Options{extract});
    EXPECT_EQ(fileScanConfig->originalRemainingFilter(), expression);
    EXPECT_EQ(
        fileScanConfig->assignments().at("alias"), assignments.at("alias"));
    EXPECT_TRUE(fileScanConfig->outputType()->equivalent(*outputType));
    EXPECT_TRUE(fileScanConfig->readerOutputType()->equivalent(*dataType));
    EXPECT_TRUE(fileScanConfig->originalFilters()
                    .at(common::Subfield("c0"))
                    ->testInt64(0));
    EXPECT_EQ(
        fileScanConfig->filters().at(common::Subfield("c0"))->testInt64(0),
        !extract);
    EXPECT_TRUE(fileScanConfig->remainingFilterColumns().contains("c1"));
    if (!extract) {
      EXPECT_EQ(fileScanConfig->remainingFilter(), expression);
    }
    // The caller receives a separate predicate map, not mutable configuration
    // storage.
    auto copied = fileScanConfig->filters();
    copied.clear();
    ASSERT_EQ(fileScanConfig->filters().size(), 1);

    auto file = writeScanFile(
        makeRowVector(
            {makeFlatVector<int64_t>({0, 3, 4, 7, 10}),
             makeFlatVector<int64_t>({1, 1, 2, 3, 1})}),
        dwio::common::FileFormat::DWRF);
    FileDataSource source(
        fileScanConfig, &fileFactory_, nullptr, context_.get(), hiveConfig_);
    source.addSplit(makeHiveConnectorSplit(file->getPath()));
    expectValues(next(source), {3, 7});
    ASSERT_EQ(next(source), nullptr);
  }
}

TEST_F(FileDataSourceTest, fileScanConfigCreatesIndependentMetadataAndFilters) {
  const auto type = ROW({{"c0", BIGINT()}, {"c1", BIGINT()}});
  ColumnHandleMap assignments{
      {"c0", regularColumn("c0", BIGINT())},
      {"c1", regularColumn("c1", BIGINT())}};
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("c0"),
      std::make_unique<common::BigintRange>(0, 100, false));
  auto table = makeTableHandle(
      std::move(filters), parseExpr("c0 > 10 OR c1 > 20", type), "test", type);
  auto fileScanConfig = std::make_shared<const FileScanConfig>(
      type, table, assignments, context_.get(), hiveConfig_);
  auto first = fileScanConfig->newFileScanState(context_.get());
  auto second = fileScanConfig->newFileScanState(context_.get());
  ASSERT_NE(first.scanSpec, second.scanSpec);
  ASSERT_NE(first.metadataFilter, second.metadataFilter);
  ASSERT_TRUE(first.metadataFilter);
  auto* firstField = first.scanSpec->childByName("c0");
  auto* secondField = second.scanSpec->childByName("c0");
  ASSERT_GT(firstField->numMetadataFilters(), 0);
  ASSERT_GT(secondField->numMetadataFilters(), 0);
  ASSERT_NE(
      firstField->metadataFilterNodeAt(0),
      secondField->metadataFilterNodeAt(0));
  ASSERT_NE(firstField->filter(), secondField->filter());
  firstField->setConstantValue<int64_t>(99, BIGINT(), pool_.get());
  firstField->setFilter(std::make_shared<common::BigintRange>(90, 100, false));
  ASSERT_FALSE(secondField->isConstant());
  ASSERT_TRUE(secondField->filter()->testInt64(1));
  auto third = fileScanConfig->newFileScanState(context_.get());
  ASSERT_FALSE(third.scanSpec->childByName("c0")->isConstant());
  ASSERT_TRUE(third.scanSpec->childByName("c0")->filter()->testInt64(1));

  InspectableFileDataSource source(
      fileScanConfig, &fileFactory_, nullptr, context_.get(), hiveConfig_);
  const auto before = source.scanState();
  source.addPhysicalColumn("extra", BIGINT());
  const auto& after = source.scanState();
  ASSERT_NE(before->scanSpec, after->scanSpec);
  ASSERT_NE(before->metadataFilter, after->metadataFilter);
  ASSERT_TRUE(after->scanSpec->childByName("extra"));
  ASSERT_GT(after->scanSpec->childByName("c0")->numMetadataFilters(), 0);
  ASSERT_NE(
      before->scanSpec->childByName("c0")->metadataFilterNodeAt(0),
      after->scanSpec->childByName("c0")->metadataFilterNodeAt(0));
  ASSERT_FALSE(fileScanConfig->readerOutputType()->containsChild("extra"));
}

TEST_F(FileDataSourceTest, extractionRemainingFilterMetadata) {
  constexpr vector_size_t kNumRows = 20'000;
  // The physical ROW is non-null, but its extracted child is always null.
  const auto nulls = makeConstant<int64_t>(std::nullopt, kNumRows);
  const auto values = makeConstant<int64_t>(7, kNumRows);
  const auto negative = makeConstant<int64_t>(-1, kNumRows);
  const auto physicalRow = makeRowVector({"x", "y"}, {nulls, values});
  const auto input =
      makeRowVector({"r", "c", "d"}, {physicalRow, negative, negative});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto file = writeScanFile(input, format);
    const std::vector<NamedExtraction> extractions{
        {"r", {ExtractionPathElement::structField("x")}, BIGINT()}};
    const auto expected =
        makeRowVector({"r", "c", "d"}, {nulls, negative, negative});
    const auto outputType = asRowType(expected->type());
    const ColumnHandleMap assignments{
        {"r",
         std::make_shared<HiveColumnHandle>(
             "r",
             HiveColumnHandle::ColumnType::kRegular,
             BIGINT(),
             physicalRow->type(),
             std::vector<common::Subfield>{},
             extractions)},
        {"c", regularColumn("c", BIGINT())},
        {"d", regularColumn("d", BIGINT())}};
    const std::vector<std::pair<std::string, vector_size_t>> cases{
        {"r IS NULL OR c > 0", kNumRows},
        {"NOT (r IS NOT NULL AND c <= 0)", kNumRows},
        {"r IS NOT NULL OR c > 0", 0},
        {"(r IS NULL OR c > 0) AND c < 0", kNumRows},
        {"c > 0 OR d > 0", 0}};
    for (const auto& [filter, expectedRows] : cases) {
      SCOPED_TRACE(filter);
      for (bool extractFilter : {false, true}) {
        SCOPED_TRACE(extractFilter);
        const auto fileScanConfig = std::make_shared<const FileScanConfig>(
            outputType,
            makeTableHandle(
                {},
                parseExpr(filter, outputType),
                "test",
                asRowType(input->type())),
            assignments,
            context_.get(),
            hiveConfig_,
            FileScanConfig::Options{extractFilter});
        FileDataSource source(
            fileScanConfig,
            &fileFactory_,
            nullptr,
            context_.get(),
            hiveConfig_);
        source.addSplit(HiveConnectorSplitBuilder(file->getPath())
                            .connectorId(kHiveConnectorId)
                            .fileFormat(format)
                            .build());
        vector_size_t numRows = 0;
        while (auto batch = next(source, 1'024)) {
          numRows += batch->size();
          test::assertEqualVectors(expected->slice(0, batch->size()), batch);
        }
        EXPECT_EQ(numRows, expectedRows);
        // Preserve the baseline: extracted projections do not use metadata
        // pruning, including predicates on ordinary columns.
        if (format == dwio::common::FileFormat::DWRF &&
            filter == "c > 0 OR d > 0") {
          const auto stats = source.getRuntimeStats();
          auto it = stats.find("skippedStrides");
          EXPECT_EQ(it == stats.end() ? 0 : it->second.sum, 0);
        }
      }
    }
  }
}

TEST_F(FileDataSourceTest, bucketConversionPreservesMetadataPruning) {
  constexpr vector_size_t kNumRows = 20'000;
  auto negative = makeConstant<int64_t>(-1, kNumRows);
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {negative, negative, makeFlatVector<int64_t>(kNumRows, [](auto row) {
         return row;
       })});
  auto schema = asRowType(input->type());
  auto file = writeScanFile(input, dwio::common::FileFormat::DWRF);
  HiveDataSource source(
      type_,
      makeTableHandle(
          {}, parseExpr("c0 > 0 OR c1 > 0", schema), "test", schema),
      assignments_,
      &fileFactory_,
      nullptr,
      context_.get(),
      hiveConfig_);
  auto split = makeHiveConnectorSplit(file->getPath());
  split->tableBucketNumber = 0;
  split->bucketConversion =
      HiveBucketConversion{4, 1, {regularColumn("c2", BIGINT())}};
  source.addSplit(split);
  while (auto batch = next(source, 1'024)) {
    EXPECT_EQ(batch->size(), 0);
  }
  // Adding c2 for bucket conversion must not newly enable metadata pruning
  // of the remaining expression. The original path scans the physical rows.
  EXPECT_EQ(source.getCompletedRows(), kNumRows);
  auto stats = source.getRuntimeStats();
  auto skipped = stats.find("skippedStrides");
  EXPECT_EQ(skipped == stats.end() ? 0 : skipped->second.sum, 0);
}

TEST_F(FileDataSourceTest, metadataWithMultipleExtractions) {
  const auto rowType = ROW({{"x", BIGINT()}, {"y", BIGINT()}});
  const auto type = ROW({{"r", rowType}, {"c", BIGINT()}});
  // Swapping fields preserves the ROW type, but changes predicate semantics.
  // Multiple extractions use a transform with ExtractionType::kNone.
  const ColumnHandleMap assignments{
      {"r",
       std::make_shared<HiveColumnHandle>(
           "r",
           HiveColumnHandle::ColumnType::kRegular,
           rowType,
           rowType,
           std::vector<common::Subfield>{},
           std::vector<NamedExtraction>{
               {"x", {ExtractionPathElement::structField("y")}, BIGINT()},
               {"y", {ExtractionPathElement::structField("x")}, BIGINT()}})},
      {"c", regularColumn("c", BIGINT())}};
  const FileScanConfig fileScanConfig(
      type,
      makeTableHandle(
          {}, parseExpr("r.y IS NULL OR c > 0", type), "test", type),
      assignments,
      context_.get(),
      hiveConfig_);
  const auto state = fileScanConfig.newFileScanState(context_.get());
  ASSERT_FALSE(state.metadataFilter);
  const auto* r = state.scanSpec->childByName("r");
  ASSERT_TRUE(r->hasTransform());
  ASSERT_EQ(r->extractionType(), common::ScanSpec::ExtractionType::kNone);
  ASSERT_EQ(r->childByName("y")->numMetadataFilters(), 0);
  ASSERT_EQ(state.scanSpec->childByName("c")->numMetadataFilters(), 0);
}

TEST_F(FileDataSourceTest, sharedFileScanConfigConcurrentPhysicalReaders) {
  const auto type = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}, {"p", BIGINT()}});
  ColumnHandleMap assignments{
      {"c0", regularColumn("c0", BIGINT())},
      {"c1", regularColumn("c1", VARCHAR())},
      {"p", partitionKey("p", BIGINT())}};
  auto fileScanConfig = std::make_shared<const FileScanConfig>(
      type,
      makeTableHandle({}, parseExpr("c0 % 2 = 1", type), "test", type),
      assignments,
      context_.get(),
      hiveConfig_);
  auto firstPool = rootPool_->addLeafChild("first-reader");
  auto secondPool = rootPool_->addLeafChild("second-reader");
  auto firstContext = makeContext(firstPool.get());
  auto secondContext = makeContext(secondPool.get());
  const auto c0 = makeFlatVector<int64_t>(100, [](auto i) { return i; });
  const auto strings = makeFlatVector<std::string>(
      100, [](auto i) { return fmt::format("retained string value {}", i); });
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto firstFile = writeScanFile(makeRowVector({c0}), format);
    auto secondFile = writeScanFile(makeRowVector({c0, strings}), format);
    auto read =
        [&](const auto& file, const ConnectorQueryCtx* context, int partition) {
          FileDataSource source(
              fileScanConfig, &fileFactory_, nullptr, context, hiveConfig_);
          auto split = HiveConnectorSplitBuilder(file->getPath())
                           .connectorId(kHiveConnectorId)
                           .fileFormat(format)
                           .partitionKey("p", std::to_string(partition))
                           .build();
          source.addSplit(split);
          std::vector<RowVectorPtr> batches;
          while (auto batch = next(source, 7)) {
            batch->loadedVector();
            batches.push_back(std::move(batch));
          }
          return batches;
        };
    auto first = std::async(std::launch::async, [&] {
      return read(firstFile, firstContext.get(), 1);
    });
    auto second = std::async(std::launch::async, [&] {
      return read(secondFile, secondContext.get(), 2);
    });
    auto firstBatches = first.get();
    auto secondBatches = second.get();
    for (int reader = 0; reader < 2; ++reader) {
      int count = 0;
      for (const auto& batch : reader == 0 ? firstBatches : secondBatches) {
        DecodedVector decodedNumbers(*batch->childAt(0));
        DecodedVector decodedStrings(*batch->childAt(1));
        DecodedVector decodedPartition(*batch->childAt(2));
        for (int row = 0; row < batch->size(); ++row) {
          const auto value = 2 * count + 1;
          EXPECT_EQ(decodedNumbers.valueAt<int64_t>(row), value);
          if (reader == 0) {
            EXPECT_TRUE(batch->childAt(1)->isNullAt(row));
          } else {
            EXPECT_EQ(
                decodedStrings.valueAt<StringView>(row).str(),
                fmt::format("retained string value {}", value));
          }
          EXPECT_EQ(decodedPartition.valueAt<int64_t>(row), reader + 1);
          ++count;
        }
      }
      EXPECT_EQ(count, 50);
    }
    ASSERT_FALSE(fileScanConfig->newFileScanState(context_.get())
                     .scanSpec->childByName("c1")
                     ->isConstant());
  }
}

TEST_F(FileDataSourceTest, newFileStatePreservesDynamicFiltersAndOwnership) {
  const auto type = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}, {"p", BIGINT()}});
  ColumnHandleMap assignments{
      {"c0", regularColumn("c0", BIGINT())},
      {"c1", regularColumn("c1", VARCHAR())},
      {"p", partitionKey("p", BIGINT())}};
  auto fileScanConfig = std::make_shared<const FileScanConfig>(
      type,
      makeTableHandle({}, nullptr, "test", type),
      assignments,
      context_.get(),
      hiveConfig_);
  const auto numbers = makeFlatVector<int64_t>({1, 2, 3});
  const auto strings = makeFlatVector<std::string>(
      {"out of line first value",
       "out of line second value",
       "out of line third value"});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto missing = writeScanFile(makeRowVector({numbers}), format);
    auto present = writeScanFile(makeRowVector({numbers, strings}), format);
    InspectableFileDataSource source(
        fileScanConfig, &fileFactory_, nullptr, context_.get(), hiveConfig_);
    source.addDynamicFilter(
        2, std::make_shared<common::BigintRange>(1, 1, false));
    auto add = [&](const auto& file, const std::string& partition) {
      auto split = HiveConnectorSplitBuilder(file->getPath())
                       .connectorId(kHiveConnectorId)
                       .fileFormat(format)
                       .partitionKey("p", partition)
                       .build();
      source.addSplit(split);
    };
    add(missing, "1");
    auto first = next(source);
    first->loadedVector();
    ASSERT_EQ(first->size(), 3);
    ASSERT_TRUE(first->childAt(1)->isNullAt(0));
    const auto firstState = source.scanState();
    // Update while the reader still has an active stripe, then verify that
    // the filter survives replacing the state for subsequent files.
    source.addDynamicFilter(
        0, std::make_shared<common::BigintRange>(2, 3, false));
    ASSERT_EQ(next(source), nullptr);
    add(present, "2");
    while (auto batch = next(source)) {
      ASSERT_EQ(batch->size(), 0);
    }
    add(present, "1");
    ASSERT_NE(source.scanState(), firstState);
    auto last = next(source);
    last->loadedVector();
    ASSERT_EQ(last->size(), 2);
    ASSERT_EQ(next(source), nullptr);
    test::assertEqualVectors(numbers, first->childAt(0));
    test::assertEqualVectors(strings->slice(1, 2), last->childAt(1));
    ASSERT_EQ(fileScanConfig->filters().size(), 0);
  }
}

TEST_F(FileDataSourceTest, dynamicFilterOnPreloadedEmptyRange) {
  auto file = writeScanFile(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      dwio::common::FileFormat::DWRF);
  const auto makeSource = [&] {
    return std::make_unique<FileDataSource>(
        type_,
        makeTableHandle({}, nullptr, "test", type_),
        assignments_,
        &fileFactory_,
        nullptr,
        context_.get(),
        hiveConfig_);
  };
  auto preloaded = makeSource();
  // DWRF's header contains no stripe. Adopting this empty range should keep
  // the destination's filter for the next file without refreshing a reader.
  preloaded->addSplit(HiveConnectorSplitBuilder(file->getPath())
                          .connectorId(kHiveConnectorId)
                          .start(0)
                          .length(1)
                          .build());
  auto source = makeSource();
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(2, 3, false));
  source->setFromDataSource(std::move(preloaded));
  EXPECT_EQ(next(*source), nullptr);
  EXPECT_EQ(source->getCompletedRows(), 0);

  source->addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(*source), {2, 3});
  EXPECT_EQ(next(*source), nullptr);
}

TEST_F(FileDataSourceTest, takeoverPreservesInputsAndDynamicFilters) {
  const auto rowType = ROW({{"x", BIGINT()}, {"y", VARCHAR()}});
  const auto type = ROW({{"c0", BIGINT()}, {"r", rowType}, {"p", BIGINT()}});
  const auto makeSource = [&](const ConnectorQueryCtx* context) {
    std::vector<common::Subfield> subfields;
    subfields.emplace_back("r.y");
    // All handles and expressions are local to this construction. The source
    // must keep their backing configuration alive after preload takeover.
    ColumnHandleMap assignments{
        {"c0", regularColumn("c0", BIGINT())},
        {"r",
         std::make_shared<HiveColumnHandle>(
             "r",
             HiveColumnHandle::ColumnType::kRegular,
             rowType,
             rowType,
             std::move(subfields))},
        {"p", partitionKey("p", BIGINT())}};
    return std::make_unique<InspectableFileDataSource>(
        type,
        makeTableHandle({}, parseExpr("c0 % 2 = 1", type), "test", type),
        assignments,
        &fileFactory_,
        nullptr,
        context,
        hiveConfig_);
  };
  const auto numbers =
      makeFlatVector<int64_t>(20, [](auto row) { return row; });
  const auto strings = makeFlatVector<std::string>(20, [](auto row) {
    return fmt::format("retained nested string value {}", row);
  });
  const auto input = makeRowVector(
      {"c0", "r"}, {numbers, makeRowVector({"x", "y"}, {numbers, strings})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto file = writeScanFile(input, format);
    auto split = HiveConnectorSplitBuilder(file->getPath())
                     .connectorId(kHiveConnectorId)
                     .fileFormat(format)
                     .partitionKey("p", "1")
                     .build();
    auto preloadContext = makeContext();
    auto preloaded = makeSource(preloadContext.get());
    preloaded->addSplit(split);
    preloaded->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(2, 18, false));
    std::weak_ptr<const FileScanConfig> configuration =
        preloaded->configuration();
    std::weak_ptr<FileScanState> state = preloaded->scanState();

    auto source = makeSource(context_.get());
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(5, 15, false));
    // The preloaded partition is already constant. Preserve the existing
    // adaptation behavior here and apply the filter when opening later splits.
    source->addDynamicFilter(
        2, std::make_shared<common::BigintRange>(1, 1, false));
    source->setFromDataSource(std::move(preloaded));
    preloadContext.reset();
    ASSERT_FALSE(configuration.expired());
    ASSERT_EQ(source->scanState(), state.lock());

    std::vector<RowVectorPtr> batches;
    while (auto batch = next(*source, 7)) {
      batch->loadedVector();
      batches.push_back(std::move(batch));
    }
    EXPECT_EQ(source->getCompletedRows(), 20);

    // The same immutable demands and dynamic filters must also work when the
    // adopted state is replaced for another file.
    auto rejected = HiveConnectorSplitBuilder(file->getPath())
                        .connectorId(kHiveConnectorId)
                        .fileFormat(format)
                        .partitionKey("p", "2")
                        .build();
    source->addSplit(rejected);
    while (auto batch = next(*source)) {
      EXPECT_EQ(batch->size(), 0);
    }
    EXPECT_TRUE(state.expired());
    source.reset();
    EXPECT_TRUE(configuration.expired());

    int count = 0;
    for (const auto& batch : batches) {
      DecodedVector values(*batch->childAt(0));
      DecodedVector rowData(*batch->childAt(1));
      const auto* nested = rowData.base()->as<RowVector>();
      DecodedVector text(*nested->childAt(1));
      DecodedVector partitions(*batch->childAt(2));
      for (vector_size_t row = 0; row < batch->size(); ++row) {
        const auto value = 5 + 2 * count++;
        const auto baseRow = rowData.index(row);
        EXPECT_EQ(values.valueAt<int64_t>(row), value);
        EXPECT_EQ(
            text.valueAt<StringView>(baseRow).str(),
            fmt::format("retained nested string value {}", value));
        EXPECT_TRUE(nested->childAt(0)->isNullAt(baseRow));
        EXPECT_EQ(partitions.valueAt<int64_t>(row), 1);
      }
    }
    EXPECT_EQ(count, 6);

    // Preserve transfer of a destination filter to nonconstant fields in the
    // preloaded scan through the existing adaptation path.
    auto rejectedPreload = makeSource(context_.get());
    rejectedPreload->addSplit(split);
    auto rejectingSource = makeSource(context_.get());
    rejectingSource->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(20, 30, false));
    rejectingSource->setFromDataSource(std::move(rejectedPreload));
    while (auto batch = next(*rejectingSource)) {
      EXPECT_EQ(batch->size(), 0);
    }
  }
}

} // namespace
} // namespace facebook::velox::connector::hive

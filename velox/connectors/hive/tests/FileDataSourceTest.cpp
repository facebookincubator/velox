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

#include "velox/connectors/common/FileDataSource.h"

#include <functional>
#include <utility>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

namespace facebook::velox::connector::hive {
namespace {

template <typename DataSourceType>
class InspectableDataSource : public DataSourceType {
 public:
  using DataSourceType::DataSourceType;
  using DataSourceType::remainingFilterExprSet;

  connector::FileScanState* scanState() const {
    return this->fileScanState_.get();
  }

  const std::shared_ptr<const connector::FileScanRequirements>&
  scanRequirements() const {
    return this->fileScanRequirements_;
  }

  const RowTypePtr& readerOutputType() const {
    return this->readerOutputType_;
  }

  const VectorPtr& output() const {
    return this->output_;
  }
};

using InspectableHiveDataSource = InspectableDataSource<HiveDataSource>;
using InspectableFileDataSource =
    InspectableDataSource<connector::FileDataSource>;

class BorrowingFileSplitReader : public connector::FileSplitReader {
 public:
  template <typename... Args>
  explicit BorrowingFileSplitReader(Args&&... args)
      : connector::FileSplitReader(std::forward<Args>(args)...) {}

  BorrowingFileSplitReader(const BorrowingFileSplitReader&) = delete;
  BorrowingFileSplitReader& operator=(const BorrowingFileSplitReader&) = delete;
  BorrowingFileSplitReader(BorrowingFileSplitReader&&) = delete;
  BorrowingFileSplitReader& operator=(BorrowingFileSplitReader&&) = delete;

  ~BorrowingFileSplitReader() override {
    if (onDestroy) {
      onDestroy(subfieldFiltersForValidation_);
    }
  }

  std::function<void(const common::SubfieldFilters*)> onDestroy;
};

class BorrowingFileDataSource : public InspectableFileDataSource {
 public:
  using InspectableFileDataSource::InspectableFileDataSource;

  BorrowingFileSplitReader* reader() const {
    return static_cast<BorrowingFileSplitReader*>(splitReader_.get());
  }

 private:
  std::unique_ptr<connector::FileSplitReader> createSplitReader() override {
    return std::make_unique<BorrowingFileSplitReader>(
        split_,
        tableHandle_,
        &partitionKeys(),
        connectorQueryCtx_,
        fileConfig_,
        readerOutputType_,
        dataIoStats_,
        metadataIoStats_,
        ioStats_,
        fileHandleFactory_,
        ioExecutor_,
        fileScanState_->scanSpec,
        &fileScanState_->filters);
  }
};

class CountingExpressionEvaluator : public exec::SimpleExpressionEvaluator {
 public:
  CountingExpressionEvaluator(
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      int& compilations)
      : SimpleExpressionEvaluator(queryCtx, pool),
        compilations_(compilations) {}

  std::unique_ptr<exec::ExprSet> compile(
      const core::TypedExprPtr& expression) override {
    ++compilations_;
    return SimpleExpressionEvaluator::compile(expression);
  }

 private:
  int& compilations_;
};

class FileDataSourceTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    queryCtx_ = core::QueryCtx::create();
    context_ = makeContext();
  }

  void TearDown() override {
    context_.reset();
    queryCtx_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::unique_ptr<ConnectorQueryCtx> makeContext(int* compilations = nullptr) {
    std::unique_ptr<core::ExpressionEvaluator> evaluator;
    if (compilations) {
      evaluator = std::make_unique<CountingExpressionEvaluator>(
          queryCtx_.get(), pool_.get(), *compilations);
    } else {
      evaluator = std::make_unique<exec::SimpleExpressionEvaluator>(
          queryCtx_.get(), pool_.get());
    }
    return ConnectorQueryCtx::Builder()
        .operatorPool(pool_.get())
        .connectorPool(pool_.get())
        .sessionProperties(config_.get())
        .expressionEvaluator(std::move(evaluator))
        .queryId("FileDataSourceTest")
        .taskId("FileDataSourceTest")
        .planNodeId("scan")
        .build();
  }

  template <typename DataSourceType = InspectableHiveDataSource>
  std::unique_ptr<DataSourceType> makeSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& table,
      const ColumnHandleMap& assignments,
      const ConnectorQueryCtx* context = nullptr) {
    return std::make_unique<DataSourceType>(
        outputType,
        table,
        assignments,
        &fileFactory_,
        nullptr,
        context ? context : context_.get(),
        hiveConfig_);
  }

  std::shared_ptr<exec::test::TempFilePath> writeScanFile(
      const RowVectorPtr& input) {
    auto file = exec::test::TempFilePath::create();
    writeToFile(file->getPath(), input);
    return file;
  }

  static RowVectorPtr next(DataSource& source, uint64_t size = 100) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source.next(size, future);
    VELOX_CHECK(result.has_value());
    VELOX_CHECK(!future.valid());
    return result.value();
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
          std::unordered_map<std::string, std::string>{
              {connector::FileConfig::kUseColumnNamesSession, "true"}});
  const std::shared_ptr<HiveConfig> hiveConfig_ =
      std::make_shared<HiveConfig>(config_);
  FileHandleFactory fileFactory_{
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(0),
      std::make_unique<FileHandleGenerator>()};
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> context_;
};

TEST_F(FileDataSourceTest, scanStateIsolationAndReuse) {
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5}),
       makeConstant<int64_t>(0, 6)});
  const auto schema = asRowType(input->type());
  const auto file = writeScanFile(input);
  const auto filter = std::make_shared<common::BigintRange>(0, 5, false);
  common::SubfieldFilters filters;
  filters.emplace(common::Subfield("c0"), filter);
  const auto table = makeTableHandle(
      std::move(filters),
      parseExpr("c0 > 1 OR c1 > 10", schema),
      "test",
      schema);
  auto first = makeSource(type_, table, assignments_);
  auto second = makeSource(type_, table, assignments_);
  const auto* state = first->scanState();
  const auto scanSpec = state->scanSpec;
  const auto metadata = state->metadataFilter;
  ASSERT_NE(state, second->scanState());
  ASSERT_NE(scanSpec, second->scanState()->scanSpec);
  ASSERT_TRUE(metadata);
  ASSERT_NE(metadata, second->scanState()->metadataFilter);
  EXPECT_EQ(first->getFilters(), &state->filters);
  EXPECT_NE(first->getFilters(), second->getFilters());
  EXPECT_EQ(state->filters.at(common::Subfield("c0")), filter);
  EXPECT_EQ(second->scanState()->filters.at(common::Subfield("c0")), filter);
  ASSERT_GT(scanSpec->childByName("c0")->numMetadataFilters(), 0);
  EXPECT_NE(
      scanSpec->childByName("c0")->metadataFilterNodeAt(0),
      second->scanState()->scanSpec->childByName("c0")->metadataFilterNodeAt(
          0));

  const auto dynamic = std::make_shared<common::BigintRange>(3, 4, false);
  for (int split = 0; split < 2; ++split) {
    first->addSplit(makeHiveConnectorSplit(file->getPath()));
    EXPECT_EQ(first->scanState(), state);
    EXPECT_EQ(first->scanState()->scanSpec, scanSpec);
    EXPECT_EQ(first->scanState()->metadataFilter, metadata);
    if (split == 0) {
      ASSERT_EQ(next(*first, 2)->size(), 0);
      first->addDynamicFilter(0, dynamic);
    }
    EXPECT_EQ(scanSpec->childByName("c0")->filter(), dynamic.get());
    expectValues(next(*first), {3, 4});
    EXPECT_EQ(next(*first), nullptr);
  }
  second->addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(*second), {2, 3, 4, 5});
  EXPECT_EQ(next(*second), nullptr);
}

TEST_F(FileDataSourceTest, bucketConversionReusesNestedAdaptation) {
  const auto values =
      makeFlatVector<int64_t>(100, [](auto row) { return row; });
  const auto nested = makeRowVector({"a", "b"}, {values, values});
  const auto input =
      makeRowVector({"r", "k"}, {nested, makeConstant<int64_t>(0, 100)});
  const auto schema = asRowType(input->type());
  const auto outputType = ROW({{"r", nested->type()}});
  const auto file = writeScanFile(input);

  // Cover both ways bucket conversion changes physical column requirements:
  // adding an unprojected column and expanding a pruned complex column.
  for (bool unprojectedBucket : {true, false}) {
    SCOPED_TRACE(unprojectedBucket);
    ColumnHandleMap assignments;
    if (unprojectedBucket) {
      assignments.emplace("r", regularColumn("r", nested->type()));
    } else {
      assignments.emplace("r", makeColumnHandle("r", nested->type(), {"r.a"}));
    }
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("r.a"),
        std::make_shared<common::BigintRange>(1, 99, false));
    filters.emplace(
        common::Subfield("r.b"),
        std::make_shared<common::BigintRange>(0, 99, false));
    auto source = makeSource(
        outputType,
        makeTableHandle(std::move(filters), nullptr, "test", schema),
        assignments);
    const auto* state = source->scanState();
    const auto initialScanSpec = state->scanSpec;
    const auto add = [&] {
      auto split = makeHiveConnectorSplit(file->getPath());
      split->tableBucketNumber = 0;
      split->bucketConversion = HiveBucketConversion{
          4,
          1,
          {regularColumn(
              unprojectedBucket ? "k" : "r",
              unprojectedBucket ? BIGINT() : nested->type())}};
      source->addSplit(std::move(split));
    };
    add();
    EXPECT_EQ(source->scanState(), state);
    const auto scanSpec = state->scanSpec;
    EXPECT_NE(scanSpec, initialScanSpec);
    auto* nestedFilter = scanSpec->childByName("r")->childByName("a");
    auto result = next(*source);
    ASSERT_TRUE(result);
    result->loadedVector();
    if (unprojectedBucket) {
      test::assertEqualVectors(
          makeRowVector({"r"}, {nested->slice(1, 99)}), result);
    }
    const auto before = nestedFilter->selectivity();
    ASSERT_GT(before.numIn(), 0);
    EXPECT_EQ(next(*source), nullptr);

    add();
    // Check before reading, so new measurements cannot hide a lost history.
    EXPECT_EQ(source->scanState(), state);
    EXPECT_EQ(state->scanSpec, scanSpec);
    const auto& after =
        state->scanSpec->childByName("r")->childByName("a")->selectivity();
    EXPECT_EQ(after.numIn(), before.numIn());
    EXPECT_EQ(after.numOut(), before.numOut());
    EXPECT_EQ(after.timeToDropValue(), before.timeToDropValue());
    while (auto batch = next(*source)) {
      batch->loadedVector();
    }
  }
}

TEST_F(FileDataSourceTest, takeoverTransfersStateButNotExpression) {
  const auto input =
      makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5})});
  const auto file = writeScanFile(input);
  const auto table =
      makeTableHandle({}, parseExpr("c0 % 2 = 0", type_), "test", type_);
  auto source = makeSource(type_, table, assignments_);
  source->addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(*source), {0, 2, 4});
  EXPECT_EQ(next(*source), nullptr);
  const auto* expression = source->remainingFilterExprSet();
  std::weak_ptr<common::ScanSpec> previousScanSpec =
      source->scanState()->scanSpec;
  const auto dynamic = std::make_shared<common::BigintRange>(2, 4, false);
  source->addDynamicFilter(0, dynamic);

  auto preloadContext = makeContext();
  auto preloaded = makeSource(type_, table, assignments_, preloadContext.get());
  preloaded->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(4, 5, false));
  preloaded->addSplit(makeHiveConnectorSplit(file->getPath()));
  const auto* state = preloaded->scanState();
  const auto* filters = preloaded->getFilters();
  const auto metadata = state->metadataFilter;
  ASSERT_NE(expression, preloaded->remainingFilterExprSet());

  source->setFromDataSource(std::move(preloaded));
  preloadContext.reset();
  EXPECT_EQ(source->scanState(), state);
  EXPECT_EQ(source->getFilters(), filters);
  EXPECT_EQ(state->metadataFilter, metadata);
  EXPECT_EQ(source->remainingFilterExprSet(), expression);
  EXPECT_TRUE(previousScanSpec.expired());
  EXPECT_EQ(state->scanSpec->childByName("c0")->filter(), dynamic.get());
  expectValues(next(*source), {2, 4});
  EXPECT_EQ(next(*source), nullptr);
  EXPECT_EQ(source->getCompletedRows(), 12);

  source->addSplit(makeHiveConnectorSplit(file->getPath()));
  EXPECT_EQ(source->scanState(), state);
  expectValues(next(*source), {2, 4});
  EXPECT_EQ(next(*source), nullptr);
}

TEST_F(FileDataSourceTest, readerReleasedBeforeValidationFilters) {
  const auto file =
      writeScanFile(makeRowVector({makeFlatVector<int64_t>({1, 2})}));
  const auto table = makeTableHandle({}, nullptr, "test", type_);
  int destroyedReaders = 0;
  const auto observe = [&](BorrowingFileDataSource& source) {
    source.addSplit(makeHiveConnectorSplit(file->getPath()));
    // This payload is owned only by the state's validation map, not by the
    // table handle or ScanSpec, so it detects the state's release order.
    auto payload = std::make_shared<common::BigintRange>(0, 10, false);
    std::weak_ptr<common::Filter> weak = payload;
    source.scanState()->filters.emplace(
        common::Subfield("c0"), std::move(payload));
    const auto* expected = source.getFilters();
    source.reader()->onDestroy = [&, weak, expected](const auto* filters) {
      EXPECT_EQ(filters, expected);
      ASSERT_FALSE(weak.expired());
      EXPECT_TRUE(filters->at(common::Subfield("c0"))->testInt64(1));
      ++destroyedReaders;
    };
    return weak;
  };
  auto source = makeSource<BorrowingFileDataSource>(type_, table, assignments_);
  auto preloaded =
      makeSource<BorrowingFileDataSource>(type_, table, assignments_);
  const auto previous = observe(*source);
  expectValues(next(*source), {1, 2});
  EXPECT_EQ(next(*source), nullptr);
  const auto adopted = observe(*preloaded);
  const auto* adoptedFilters = preloaded->getFilters();

  source->setFromDataSource(std::move(preloaded));
  EXPECT_EQ(destroyedReaders, 1);
  EXPECT_TRUE(previous.expired());
  EXPECT_FALSE(adopted.expired());
  EXPECT_EQ(source->getFilters(), adoptedFilters);
  expectValues(next(*source), {1, 2});
  source.reset();
  EXPECT_EQ(destroyedReaders, 2);
  EXPECT_TRUE(adopted.expired());
}

TEST_F(FileDataSourceTest, remainingFilterAndOutputReuse) {
  int compilations = 0;
  auto context = makeContext(&compilations);
  const auto numbers =
      makeFlatVector<int64_t>(30, [](auto row) { return row; });
  const auto strings = makeFlatVector<std::string>(30, [](auto row) {
    return fmt::format("out-of-line string value {}", row);
  });
  const auto input = makeRowVector({numbers, strings});
  const auto schema = asRowType(input->type());
  const auto file = writeScanFile(input);
  const auto outputType = ROW({{"value", VARCHAR()}});
  auto source = makeSource(
      outputType,
      makeTableHandle({}, parseExpr("c0 % 100 >= 10", schema), "test", schema),
      {{"value", regularColumn("c1", VARCHAR())}},
      context.get());
  ASSERT_EQ(compilations, 1);
  EXPECT_TRUE(source->readerOutputType()->containsChild("c0"));
  const auto* expression = source->remainingFilterExprSet();
  const auto* state = source->scanState();
  RowVectorPtr retained;
  for (int split = 0; split < 2; ++split) {
    source->addSplit(makeHiveConnectorSplit(file->getPath()));
    ASSERT_EQ(next(*source, 10)->size(), 0);
    const auto* output = source->output().get();
    auto first = next(*source, 10);
    first->loadedVector();
    test::assertEqualVectors(
        makeRowVector({"value"}, {strings->slice(10, 10)}), first);
    EXPECT_EQ(source->output().get(), output);
    retained = first;
    auto second = next(*source, 10);
    second->loadedVector();
    test::assertEqualVectors(
        makeRowVector({"value"}, {strings->slice(20, 10)}), second);
    EXPECT_EQ(next(*source), nullptr);
    EXPECT_EQ(source->scanState(), state);
    EXPECT_EQ(source->remainingFilterExprSet(), expression);
    EXPECT_EQ(compilations, 1);
  }
  source.reset();
  test::assertEqualVectors(
      makeRowVector({"value"}, {strings->slice(10, 10)}), retained);
}

TEST_F(FileDataSourceTest, extractionProducedTypeStaysInState) {
  const auto values = makeFlatVector<int64_t>({1, 2, 3});
  const auto nested = makeRowVector(
      {"x", "y"}, {values, makeFlatVector<std::string>({"a", "b", "c"})});
  const auto input = makeRowVector({"r"}, {nested});
  const auto schema = asRowType(input->type());
  const auto file = writeScanFile(input);
  const auto outputType = ROW({{"value", BIGINT()}});
  const ColumnHandleMap assignments{
      {"value",
       std::make_shared<HiveColumnHandle>(
           "r",
           HiveColumnHandle::ColumnType::kRegular,
           BIGINT(),
           nested->type(),
           std::vector<common::Subfield>{},
           std::vector<connector::NamedExtraction>{
               {"value",
                {connector::ExtractionPathElement::structField("x")},
                BIGINT()}})}};
  auto source = makeSource(
      outputType, makeTableHandle({}, nullptr, "test", schema), assignments);
  const auto* state = source->scanState();
  const auto producedType = state->readerProducedType;
  ASSERT_TRUE(producedType);
  EXPECT_EQ(producedType->findChild("r")->kind(), TypeKind::BIGINT);
  EXPECT_EQ(source->readerOutputType()->findChild("r")->kind(), TypeKind::ROW);
  for (int split = 0; split < 2; ++split) {
    source->addSplit(makeHiveConnectorSplit(file->getPath()));
    test::assertEqualVectors(makeRowVector({"value"}, {values}), next(*source));
    EXPECT_EQ(next(*source), nullptr);
    EXPECT_EQ(source->scanState(), state);
    EXPECT_EQ(state->readerProducedType, producedType);
  }
}

TEST_F(FileDataSourceTest, publishedFilterPayloadIsShared) {
  const auto filter =
      std::make_shared<common::BigintValuesUsingBloomFilter>(8'000'000, false);
  const std::vector<int64_t> values{1, 3, 5};
  for (auto value : values) {
    filter->insert(value);
  }
  const auto file =
      writeScanFile(makeRowVector({makeFlatVector<int64_t>(values)}));
  const common::Subfield field("c0");
  common::SubfieldFilters filters;
  filters.emplace(field.clone(), filter);
  const auto table =
      makeTableHandle(std::move(filters), nullptr, "test", type_);
  auto first = makeSource(type_, table, assignments_);
  auto second = makeSource(type_, table, assignments_);
  EXPECT_EQ(first->getFilters()->at(field), filter);
  EXPECT_EQ(second->getFilters()->at(field), filter);
  EXPECT_EQ(
      first->scanState()->scanSpec->childByName("c0")->filter(), filter.get());
  EXPECT_EQ(
      second->scanState()->scanSpec->childByName("c0")->filter(), filter.get());
  first->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(3, 5, false));
  first->addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(*first), {3, 5});
  second->addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(*second), values);
  EXPECT_EQ(first->getFilters()->at(field), filter);
  EXPECT_EQ(second->getFilters()->at(field), filter);
}

TEST_F(FileDataSourceTest, sharedRequirementsCreateIndependentStates) {
  const auto schema = ROW({{"c0", BIGINT()}, {"c1", BIGINT()}});
  const common::Subfield field("c0");
  const auto filter =
      std::make_shared<common::BigintValuesUsingBloomFilter>(8'000'000, false);
  for (auto value : {1, 3, 5}) {
    filter->insert(value);
  }
  common::SubfieldFilters filters;
  filters.emplace(field.clone(), filter);
  auto source = makeSource(
      type_,
      makeTableHandle(
          std::move(filters),
          parseExpr("c0 > 1 OR c1 > 10", schema),
          "test",
          schema),
      assignments_);
  const auto requirements = source->scanRequirements();
  auto first = connector::FileScanRequirements::createState(
      requirements, context_.get());
  auto second = connector::FileScanRequirements::createState(
      requirements, context_.get());
  EXPECT_EQ(first->scanRequirements, requirements);
  EXPECT_EQ(second->scanRequirements, requirements);
  EXPECT_NE(first->scanSpec, second->scanSpec);
  ASSERT_TRUE(first->metadataFilter);
  EXPECT_NE(first->metadataFilter, second->metadataFilter);
  EXPECT_EQ(requirements->filters().at(field), filter);
  EXPECT_EQ(first->filters.at(field), filter);
  EXPECT_EQ(second->filters.at(field), filter);
  EXPECT_EQ(first->scanSpec->childByName("c0")->filter(), filter.get());
  EXPECT_EQ(second->scanSpec->childByName("c0")->filter(), filter.get());
  ASSERT_GT(first->scanSpec->childByName("c0")->numMetadataFilters(), 0);
  EXPECT_NE(
      first->scanSpec->childByName("c0")->metadataFilterNodeAt(0),
      second->scanSpec->childByName("c0")->metadataFilterNodeAt(0));

  const auto replacement = std::make_shared<common::BigintRange>(3, 5, false);
  first->filters.at(field) = replacement;
  first->scanSpec->childByName("c0")->setFilter(replacement);
  EXPECT_EQ(requirements->filters().at(field), filter);
  EXPECT_EQ(second->filters.at(field), filter);
  EXPECT_EQ(second->scanSpec->childByName("c0")->filter(), filter.get());
  EXPECT_TRUE(requirements->readerOutputType()->containsChild("c1"));
}

TEST_F(FileDataSourceTest, requirementsOwnSubfieldsButNotExecutionContext) {
  const auto nested = ROW({{"a", BIGINT()}, {"b", BIGINT()}});
  const auto schema = ROW({{"r", nested}});
  const auto outputType = ROW({{"value", nested}});
  ColumnHandleMap assignments{
      {"value", makeColumnHandle("r", nested, {"r.a"})}};
  auto table =
      makeTableHandle({}, parseExpr("r.a % 2 = 0", schema), "test", schema);
  std::weak_ptr<const connector::ColumnHandle> columnOwner =
      assignments.at("value");
  std::weak_ptr<const connector::FileTableHandle> tableOwner = table;
  std::weak_ptr<exec::Expr> compiledExpression;
  std::shared_ptr<const connector::FileScanRequirements> requirements;
  {
    auto context = makeContext();
    auto source = makeSource(outputType, table, assignments, context.get());
    compiledExpression = source->remainingFilterExprSet()->expr(0);
    requirements = source->scanRequirements();
  }
  EXPECT_TRUE(compiledExpression.expired());
  assignments.clear();
  table.reset();
  EXPECT_FALSE(columnOwner.expired());
  EXPECT_FALSE(tableOwner.expired());
  EXPECT_EQ(
      requirements->readerOutputType()->names(), std::vector<std::string>{"r"});
  ASSERT_FALSE(requirements->subfields().at("r").empty());
  for (const auto* subfield : requirements->subfields().at("r")) {
    EXPECT_EQ(subfield->toString(), "r.a");
  }

  auto context = makeContext();
  auto state =
      connector::FileScanRequirements::createState(requirements, context.get());
  EXPECT_NE(state->scanSpec->childByName("r")->childByName("a"), nullptr);
  std::weak_ptr<const connector::FileScanRequirements> requirementsOwner =
      requirements;
  requirements.reset();
  EXPECT_FALSE(requirementsOwner.expired());
  EXPECT_FALSE(columnOwner.expired());
  state.reset();
  EXPECT_TRUE(requirementsOwner.expired());
  EXPECT_TRUE(columnOwner.expired());
  EXPECT_TRUE(tableOwner.expired());
}

TEST_F(FileDataSourceTest, bucketConversionPreservesRequirements) {
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5}),
       makeConstant<int64_t>(0, 6)});
  const auto schema = asRowType(input->type());
  const auto file = writeScanFile(input);
  auto source = makeSource(
      type_, makeTableHandle({}, nullptr, "test", schema), assignments_);
  const auto requirements = source->scanRequirements();
  const auto initialScanSpec = source->scanState()->scanSpec;
  std::shared_ptr<common::ScanSpec> adaptedScanSpec;
  for (int i = 0; i < 2; ++i) {
    auto split = makeHiveConnectorSplit(file->getPath());
    split->tableBucketNumber = 0;
    split->bucketConversion =
        HiveBucketConversion{4, 1, {regularColumn("c1", BIGINT())}};
    source->addSplit(std::move(split));
    EXPECT_EQ(source->scanRequirements(), requirements);
    EXPECT_FALSE(requirements->readerOutputType()->containsChild("c1"));
    EXPECT_TRUE(source->readerOutputType()->containsChild("c1"));
    if (i == 0) {
      adaptedScanSpec = source->scanState()->scanSpec;
      EXPECT_NE(adaptedScanSpec, initialScanSpec);
    } else {
      EXPECT_EQ(source->scanState()->scanSpec, adaptedScanSpec);
    }
    expectValues(next(*source), {0, 1, 2, 3, 4, 5});
    EXPECT_EQ(next(*source), nullptr);
  }
  auto fresh = connector::FileScanRequirements::createState(
      requirements, context_.get());
  EXPECT_EQ(
      fresh->scanSpec->children().size(), initialScanSpec->children().size());
  EXPECT_NE(fresh->scanSpec, adaptedScanSpec);
}

TEST_F(
    FileDataSourceTest,
    takeoverRetainsRequirementsOwnersNotCompiledExpression) {
  const auto file =
      writeScanFile(makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})}));
  const auto table =
      makeTableHandle({}, parseExpr("c0 % 2 = 0", type_), "test", type_);
  auto source = makeSource(type_, table, assignments_);
  const auto requirements = source->scanRequirements();
  const auto* expression = source->remainingFilterExprSet();

  auto preloadContext = makeContext();
  auto preloaded = makeSource(type_, table, assignments_, preloadContext.get());
  std::weak_ptr<const connector::FileScanRequirements> preloadedRequirements =
      preloaded->scanRequirements();
  std::weak_ptr<exec::Expr> preloadedExpression =
      preloaded->remainingFilterExprSet()->expr(0);
  preloaded->addSplit(makeHiveConnectorSplit(file->getPath()));
  source->setFromDataSource(std::move(preloaded));
  preloadContext.reset();

  EXPECT_EQ(source->scanRequirements(), requirements);
  EXPECT_EQ(source->remainingFilterExprSet(), expression);
  EXPECT_TRUE(preloadedExpression.expired());
  EXPECT_FALSE(preloadedRequirements.expired());
  EXPECT_EQ(
      source->scanState()->scanRequirements, preloadedRequirements.lock());
  expectValues(next(*source), {0, 2, 4});
  EXPECT_EQ(next(*source), nullptr);
  source.reset();
  EXPECT_TRUE(preloadedRequirements.expired());
}

TEST_F(FileDataSourceTest, inputValidationStillPrecedesCompilation) {
  int compilations = 0;
  auto context = makeContext(&compilations);
  const auto table =
      makeTableHandle({}, parseExpr("c0 % 2 = 0", type_), "test", type_);
  VELOX_ASSERT_THROW(
      makeSource(type_, table, {}, context.get()),
      "ColumnHandle is missing for output column: c0");
  EXPECT_EQ(compilations, 0);

  const auto duplicateOutputs = ROW({{"left", BIGINT()}, {"right", BIGINT()}});
  const ColumnHandleMap duplicateAssignments{
      {"left", regularColumn("c0", BIGINT())},
      {"right", regularColumn("c0", BIGINT())}};
  VELOX_ASSERT_THROW(
      makeSource(duplicateOutputs, table, duplicateAssignments, context.get()),
      "Cannot map from same table column to different outputs in table scan");
  EXPECT_EQ(compilations, 0);
}

TEST_F(FileDataSourceTest, requirementsPreserveExtractedSampling) {
  const auto original = parseExpr("rand() < 0.5 AND rand() < 0.5", type_);
  auto source = makeSource(
      type_, makeTableHandle({}, original, "test", type_), assignments_);
  const auto requirements = source->scanRequirements();
  EXPECT_EQ(requirements->tableHandle()->remainingFilter(), original);
  EXPECT_EQ(requirements->sampleRate(), 0.25);
  EXPECT_EQ(requirements->remainingFilter(), nullptr);
  EXPECT_EQ(source->remainingFilterExprSet(), nullptr);
  EXPECT_TRUE(requirements->remainingFilterColumns().empty());
}

TEST_F(FileDataSourceTest, postProcessorsRemainDataSourceLocal) {
  std::vector<int> calls;
  const auto handle = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{},
      std::vector<connector::NamedExtraction>{},
      HiveColumnHandle::ColumnParseParameters{},
      [count = 0, &calls](VectorPtr&) mutable { calls.push_back(++count); });
  const ColumnHandleMap assignments{{"c0", handle}};
  const auto table = makeTableHandle({}, nullptr, "test", type_);
  const auto file =
      writeScanFile(makeRowVector({makeFlatVector<int64_t>({1, 2})}));
  auto first = makeSource(type_, table, assignments);
  auto second = makeSource(type_, table, assignments);
  for (int i = 0; i < 2; ++i) {
    for (auto* source : {first.get(), second.get()}) {
      source->addSplit(makeHiveConnectorSplit(file->getPath()));
      expectValues(next(*source), {1, 2});
      EXPECT_EQ(next(*source), nullptr);
    }
  }
  EXPECT_EQ(calls, (std::vector<int>{1, 1, 2, 2}));
}

} // namespace
} // namespace facebook::velox::connector::hive

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

#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/common/base/tests/GTestUtils.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/AggregateCompanionSignatures.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/Spill.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SignatureBinder.h"

using namespace facebook::velox::common::testutil;
using facebook::velox::exec::CursorParameters;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;
using facebook::velox::test::VectorMaker;

namespace facebook::velox::functions::aggregate::test {

namespace {
constexpr const char* kHiveConnectorId = "test-hive";

void enableAbandonPartialAggregation(AssertQueryBuilder& queryBuilder) {
  queryBuilder.config(core::QueryConfig::kAbandonPartialAggregationMinRows, 1)
      .config(core::QueryConfig::kAbandonPartialAggregationMinPct, 0)
      .config(core::QueryConfig::kMaxPartialAggregationMemory, 0)
      .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, 0)
      .maxDrivers(1);
}

// Companion function names for an aggregate function.
struct CompanionFunctions {
  std::string partialFunction;
  std::string mergeFunction;
  std::string mergeExtractFunction;
  std::string extractFunction;
};

void testIgnoreNullInputs(
    const std::function<void(PlanBuilder&)>& makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    const std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>&
        assertResults,
    const std::unordered_map<std::string, std::string>& config,
    exec::test::DuckDbQueryRunner& duckDbQueryRunner,
    memory::MemoryPool* pool);

RowVectorPtr makeNullInput(
    const std::vector<core::CallTypedExprPtr>& aggregateCalls,
    const std::vector<std::string>& groupingKeys,
    const RowVectorPtr& sourceInput,
    memory::MemoryPool* pool);

} // namespace

std::vector<RowVectorPtr> AggregationTestBase::makeVectors(
    const RowTypePtr& rowType,
    vector_size_t size,
    int numVectors) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < numVectors; ++i) {
    auto vector = std::dynamic_pointer_cast<RowVector>(
        velox::test::BatchMaker::createBatch(rowType, size, *pool_));
    vectors.push_back(vector);
  }
  return vectors;
}

void AggregationTestBase::SetUp() {
  OperatorTestBase::SetUp();
  filesystems::registerLocalFileSystem();
  connector::hive::HiveConnectorFactory factory;
  auto hiveConnector = factory.newConnector(
      kHiveConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  connector::ConnectorRegistry::global().insert(
      hiveConnector->connectorId(), hiveConnector);
  dwrf::registerDwrfReaderFactory();
}

void AggregationTestBase::TearDown() {
  dwrf::unregisterDwrfReaderFactory();
  connector::ConnectorRegistry::global().erase(kHiveConnectorId);
  OperatorTestBase::TearDown();
}

namespace {
// Returns total spilled bytes inside 'task'.
int64_t spilledBytes(const exec::Task& task) {
  auto stats = task.taskStats();
  int64_t spilledBytes = 0;
  for (auto& pipeline : stats.pipelineStats) {
    for (auto op : pipeline.operatorStats) {
      spilledBytes += op.spilledBytes;
    }
  }
  return spilledBytes;
}

// Returns total spilled rows inside 'task'.
int64_t spilledRows(const exec::Task& task) {
  auto stats = task.taskStats();
  int64_t spilledRows = 0;
  for (auto& pipeline : stats.pipelineStats) {
    for (auto op : pipeline.operatorStats) {
      spilledRows += op.spilledRows;
    }
  }
  return spilledRows;
}

// Returns total spilled input bytes inside 'task'.
int64_t spilledInputBytes(const exec::Task& task) {
  auto stats = task.taskStats();
  int64_t spilledInputBytes = 0;
  for (auto& pipeline : stats.pipelineStats) {
    for (auto op : pipeline.operatorStats) {
      spilledInputBytes += op.spilledInputBytes;
    }
  }
  return spilledInputBytes;
}

// Returns total spilled files inside 'task'.
int64_t spilledFiles(const exec::Task& task) {
  auto stats = task.taskStats();
  int64_t spilledFiles = 0;
  for (auto& pipeline : stats.pipelineStats) {
    for (auto op : pipeline.operatorStats) {
      spilledFiles += op.spilledFiles;
    }
  }
  return spilledFiles;
}

// Number of distinct values in the extra grouping key 'k0' of the companion
// plans.
constexpr int64_t kNumExtraGroupingKeys{4};

// Add a BIGINT column of kNumExtraGroupingKeys distinct values to data.
RowVectorPtr addExtraGroupingKeyImpl(
    const RowVectorPtr& data,
    const std::vector<std::string>& fieldNames,
    VectorMaker& vectorMaker) {
  auto children = data->children();
  children.push_back(vectorMaker.flatVector<int64_t>(
      data->size(), [](auto row) { return row % kNumExtraGroupingKeys; }));
  return vectorMaker.rowVector(fieldNames, children);
}

// Add a BIGINT column of 4 distinct values to every RowVector in data.
std::vector<RowVectorPtr> addExtraGroupingKey(
    const std::vector<RowVectorPtr>& data,
    const std::string& extraKeyName) {
  VELOX_CHECK(!data.empty());
  auto names = asRowType(data[0]->type())->names();
  names.push_back(extraKeyName);

  std::vector<RowVectorPtr> newData{data.size()};
  VectorMaker vectorMaker{data[0]->pool()};
  std::transform(
      data.begin(),
      data.end(),
      newData.begin(),
      [&names, &vectorMaker](const RowVectorPtr& vector) {
        return addExtraGroupingKeyImpl(vector, names, vectorMaker);
      });
  return newData;
}

// Given the name of the original aggregate function and the argument types,
// return the name of the merge_extract companion function with a suffix of the
// result type. Merge_extract function names with suffixes should only be used
// when the original aggregate function has multiple signatures with the same
// intermediate type.
std::string getMergeExtractFunctionNameWithSuffix(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  auto signatures = exec::getAggregateFunctionSignatures(name);
  VELOX_CHECK(signatures.has_value());

  for (const auto& signature : signatures.value()) {
    exec::SignatureBinder binder{*signature, argTypes, TypeCoercer::defaults()};
    if (binder.tryBind()) {
      if (auto resolvedType = binder.tryResolveReturnType()) {
        return exec::CompanionSignatures::mergeExtractFunctionNameWithSuffix(
            name, signature->returnType());
      }
    }
  }
  VELOX_UNREACHABLE("Could not find a function bound to argTypes.");
}

// Given the name of the original aggregation function and the argument types,
// return the name of the extract companion function with a suffix of the result
// type. Extract function names with suffixes should only be used when the
// original aggregation function has multiple signatures with the same
// intermediate type.
std::string getExtractFunctionNameWithSuffix(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  auto signatures = exec::getAggregateFunctionSignatures(name);
  VELOX_CHECK(signatures.has_value());

  for (const auto& signature : signatures.value()) {
    exec::SignatureBinder binder{*signature, argTypes, TypeCoercer::defaults()};
    if (binder.tryBind()) {
      if (auto resultType = binder.tryResolveReturnType()) {
        return exec::CompanionSignatures::extractFunctionNameWithSuffix(
            name, signature->returnType());
      }
    }
  }
  VELOX_UNREACHABLE("Could not find a function bound to argTypes.");
}

// Given the `index`-th aggregation expression, e.g., "avg(c0)", return the
// names of its partial companion aggregation expression, merge companion
// aggregation expression, merge_extract companion expression and extract
// companion expression, e.g., CompanionFunctions{"avg_partial(c0)",
// "avg_merge(a0)", "avg_merge_extract_double(a0)", "avg_extract_double(a0)"}
// when `index` is 0.
CompanionFunctions getCompanionAggregates(
    column_index_t index,
    const exec::CompanionFunctionSignatureMap& companionFunctions,
    const std::string& functionName,
    const std::string& aggregateArgs,
    const std::vector<TypePtr>& argTypes) {
  VELOX_CHECK_EQ(companionFunctions.partial.size(), 1);
  auto partialAggregate = fmt::format(
      "{}({})", companionFunctions.partial.front().functionName, aggregateArgs);

  VELOX_CHECK_EQ(companionFunctions.merge.size(), 1);
  auto mergeAggregate = fmt::format(
      "{}(a{})", companionFunctions.merge.front().functionName, index);

  // Construct the merge_extract expression. Rename the result of the expression
  // to be the same as the original aggregation result, so that post-aggregation
  // projections, if exist, can apply with no change.
  std::string mergeExtractExpression;
  if (companionFunctions.mergeExtract.size() == 1) {
    mergeExtractExpression = fmt::format(
        "{0}(a{1}) as a{1}",
        companionFunctions.mergeExtract.front().functionName,
        index);
  } else {
    VELOX_CHECK_GT(companionFunctions.mergeExtract.size(), 1);
    auto mergeExtractFunctionName =
        getMergeExtractFunctionNameWithSuffix(functionName, argTypes);
    mergeExtractExpression =
        fmt::format("{0}(a{1}) as a{1}", mergeExtractFunctionName, index);
  }

  // Construct the extract expression.
  std::string extractExpression;
  if (companionFunctions.extract.size() == 1) {
    extractExpression = fmt::format(
        "{0}(a{1}) as a{1}",
        companionFunctions.extract.front().functionName,
        index);
  } else {
    VELOX_CHECK_GT(companionFunctions.extract.size(), 1);
    auto extractFunctionName =
        getExtractFunctionNameWithSuffix(functionName, argTypes);
    extractExpression =
        fmt::format("{0}(a{1}) as a{1}", extractFunctionName, index);
  }
  return CompanionFunctions{
      partialAggregate,
      mergeAggregate,
      mergeExtractExpression,
      extractExpression};
}

// Given a list of aggregation expressions, e.g., {"avg(c0)"}, return a list of
// aggregate function names and a list of arguments in these expressions, e.g.,
// {{"avg"}, {"c0"}}.
std::tuple<std::vector<std::string>, std::vector<std::string>>
getFunctionNamesAndArgs(const std::vector<std::string>& aggregates) {
  std::vector<std::string> functionNames;
  std::vector<std::string> aggregateArgs;
  for (const auto& aggregate : aggregates) {
    std::vector<std::string> tokens;
    folly::split('(', aggregate, tokens);
    VELOX_CHECK_EQ(tokens.size(), 2);
    functionNames.push_back(tokens[0]);
    aggregateArgs.push_back(tokens[1].substr(0, tokens[1].find(')')));
  }
  return std::make_pair(functionNames, aggregateArgs);
}

// Given a list of aggregation expressions, e.g., {"avg(c0)",
// "\"$internal$count_distinct\"(c1)"}, fetch the function names from
// AggregationNode of builder.planNode().
std::vector<std::string> getFunctionNames(
    std::function<void(exec::test::PlanBuilder&)> makeSource,
    const std::vector<std::string>& aggregates,
    memory::MemoryPool* pool) {
  std::vector<std::string> functionNames;

  PlanBuilder builder(pool);
  makeSource(builder);

  builder.singleAggregation({}, aggregates);
  auto& aggregationNode =
      static_cast<const core::AggregationNode&>(*builder.planNode());

  for (const auto& aggregate : aggregationNode.aggregates()) {
    const auto& aggregateExpr = aggregate.call;
    const auto& name = aggregateExpr->name();

    functionNames.push_back(name);
  }

  return functionNames;
}

// Given a list of aggregation expressions, check if any of aggregate functions
// are order sensitive with metadata.
bool hasOrderSensitive(
    std::function<void(exec::test::PlanBuilder&)> makeSource,
    const std::vector<std::string>& aggregates,
    memory::MemoryPool* pool) {
  auto functionNames = getFunctionNames(makeSource, aggregates, pool);
  return std::any_of(
      functionNames.begin(), functionNames.end(), [](const auto& functionName) {
        auto* entry = exec::getAggregateFunctionEntry(functionName);
        const auto& metadata = entry->metadata;
        return metadata.orderSensitive;
      });
}
// Same as above, but allows to specify input data instead of a function.
bool hasOrderSensitive(
    const std::vector<RowVectorPtr>& data,
    const std::vector<std::string>& aggregates,
    memory::MemoryPool* pool) {
  return hasOrderSensitive(
      [&](PlanBuilder& builder) { builder.values(data); }, aggregates, pool);
}

// Require every aggregate's fast path so tests can compare partial states.
bool allSupportToIntermediate(
    const core::AggregationNode& node,
    const core::QueryConfig& config) {
  return !node.aggregates().empty() &&
      std::all_of(
          node.aggregates().begin(),
          node.aggregates().end(),
          [&](const auto& aggregate) {
            return exec::Aggregate::create(
                       aggregate.call->name(),
                       core::AggregationNode::Step::kPartial,
                       aggregate.rawInputTypes,
                       aggregate.call->type(),
                       config,
                       {})
                ->supportsToIntermediate();
          });
}

// Runs 'plan', whose root is a partial merge aggregation, with and without
// abandoning partial aggregation and compares the partial states, so a final
// merge cannot hide incorrect empty values such as NULL instead of COUNT's 0.
// Skips merges without the toIntermediate fast path.
void testAbandonedPartialMerge(
    const core::PlanNodePtr& plan,
    const std::unordered_map<std::string, std::string>& config,
    memory::MemoryPool* pool) {
  const auto& mergeNode = dynamic_cast<const core::AggregationNode&>(*plan);
  if (!allSupportToIntermediate(mergeNode, core::QueryConfig(config))) {
    return;
  }

  const auto expected =
      AssertQueryBuilder(plan)
          .configs(config)
          .maxDrivers(1)
          .config(core::QueryConfig::kMaxOutputBatchRows, "2")
          .config(core::QueryConfig::kMaxPartialAggregationMemory, 1ULL << 60)
          .config(
              core::QueryConfig::kMaxExtendedPartialAggregationMemory,
              1ULL << 60)
          .config(core::QueryConfig::kAbandonPartialAggregationMinPct, "101")
          .copyResults(pool);

  // Use a new builder because a run consumes the builder's configs, and a
  // later run would then drop kMaxOutputBatchRows and 'config'.
  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.configs(config).config(
      core::QueryConfig::kMaxOutputBatchRows, "2");
  enableAbandonPartialAggregation(queryBuilder);
  auto task = queryBuilder.assertResults({expected});
  const auto stats = toPlanStats(task->taskStats());
  const auto& mergeStats = stats.at(plan->id());
  // No group reaches the merge, so there is nothing to abandon.
  if (mergeStats.inputRows == 0) {
    return;
  }
  // The first input vector is aggregated before abandoning, so a single vector
  // never reaches the fast path. testCompanionMerge sends at least 3 groups,
  // which 2-row batches split into at least 2 vectors.
  ASSERT_GT(mergeStats.inputVectors, 1)
      << "Merge node received one input vector; the abandoned fast path was "
         "not exercised.";
  EXPECT_GT(
      mergeStats.customStats.at("abandonedPartialAggregationRows").sum, 0);
  EXPECT_GE(
      mergeStats.customStats.at("toIntermediateFastPathCalls").sum,
      mergeNode.aggregates().size() * (mergeStats.inputVectors - 1));
}

// Returns the non-empty vectors of 'inputs' with the first row of each
// intermediate column set to null. The first 'numGroupingKeys' columns are
// grouping keys and stay as they are. Wraps the intermediate columns in
// dictionaries so that the wrapped vectors keep their encodings. Returns one
// empty vector of 'type' when no input has rows, because a Values node takes
// its type from its first vector.
std::vector<RowVectorPtr> nullFirstIntermediateRows(
    const std::vector<RowVectorPtr>& inputs,
    size_t numGroupingKeys,
    const RowTypePtr& type,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> nullableInputs;
  for (const auto& input : inputs) {
    if (input->size() == 0) {
      continue;
    }
    auto children = input->children();
    auto indices = allocateIndices(input->size(), pool);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    std::iota(rawIndices, rawIndices + input->size(), 0);
    auto nulls = allocateNulls(input->size(), pool, bits::kNotNull);
    bits::setNull(nulls->asMutable<uint64_t>(), 0);
    for (auto i = numGroupingKeys; i < children.size(); ++i) {
      children[i] = BaseVector::wrapInDictionary(
          nulls, indices, input->size(), children[i]);
    }
    nullableInputs.push_back(
        std::make_shared<RowVector>(
            pool, input->type(), nullptr, input->size(), std::move(children)));
  }
  if (nullableInputs.empty()) {
    nullableInputs.push_back(BaseVector::create<RowVector>(type, 0, pool));
  }
  return nullableInputs;
}

// Number of copies of the input in the abandoned partial merge checks. Any
// non-empty input then has at least 3 groups, which the 2-row batches of
// testAbandonedPartialMerge send to the merge as at least 2 input vectors.
constexpr int32_t kNumAbandonedMergeCopies{3};

// Name of the column that numbers the input copies of the abandoned partial
// merge checks.
const std::string kCopyKey{"__copy"};

} // namespace

struct AggregationTestBase::CompanionPlanInputs {
  // Builds the inputs for 'aggregates' over 'data' grouped by 'groupingKeys'.
  static CompanionPlanInputs make(
      const std::vector<RowVectorPtr>& data,
      const std::function<void(PlanBuilder&)>& makeSource,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::vector<TypePtr>>& aggregatesArgTypes) {
    CompanionPlanInputs inputs;
    inputs.data = data;
    inputs.makeSource = makeSource;
    inputs.groupingKeys = groupingKeys;
    inputs.partialGroupingKeys = groupingKeys;
    inputs.partialGroupingKeys.push_back("k0");
    inputs.aggregates = aggregates;
    inputs.extractExpressions = groupingKeys;

    auto [functionNames, aggregateArgs] = getFunctionNamesAndArgs(aggregates);
    for (size_t i = 0, size = functionNames.size(); i < size; ++i) {
      const auto& companionSignatures =
          exec::getCompanionFunctionSignatures(functionNames[i]);
      VELOX_CHECK(companionSignatures.has_value());

      const auto& companionFunctions = getCompanionAggregates(
          i,
          *companionSignatures,
          functionNames[i],
          aggregateArgs[i],
          aggregatesArgTypes[i]);
      inputs.partialAggregates.push_back(companionFunctions.partialFunction);
      inputs.mergeAggregates.push_back(companionFunctions.mergeFunction);
      inputs.mergeExtractAggregates.push_back(
          companionFunctions.mergeExtractFunction);
      inputs.extractExpressions.push_back(companionFunctions.extractFunction);
    }
    return inputs;
  }

  // The caller's data, without the extra grouping key.
  std::vector<RowVectorPtr> data;
  // Builds the input of the aggregation, including the extra grouping key k0.
  std::function<void(PlanBuilder&)> makeSource;
  std::vector<std::string> groupingKeys;
  // 'groupingKeys' followed by k0, for the steps that compute partial states.
  std::vector<std::string> partialGroupingKeys;
  std::vector<std::string> aggregates;
  // One companion per aggregate.
  std::vector<std::string> partialAggregates;
  std::vector<std::string> mergeAggregates;
  std::vector<std::string> mergeExtractAggregates;
  // 'groupingKeys' followed by one extract expression per aggregate.
  std::vector<std::string> extractExpressions;
};

void AggregationTestBase::testCompanionExtract(
    const CompanionPlanInputs& inputs,
    const std::vector<std::string>& postAggregationProjections,
    const std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>&
        assertResults,
    const std::unordered_map<std::string, std::string>& config,
    bool streaming) {
  {
    SCOPED_TRACE("Run partial + final");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);
    builder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeAggregates)
        .finalAggregation()
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  if (!inputs.groupingKeys.empty() &&
      !hasOrderSensitive(inputs.data, inputs.aggregates, pool())) {
    SCOPED_TRACE("Run partial + final with spilling");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);

    // Spilling needs at least 2 batches of input. Use round-robin
    // repartitioning to split input into multiple batches.
    core::PlanNodeId partialNodeId;
    core::PlanNodeId finalNodeId;
    builder.localPartitionRoundRobinRow()
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .capturePlanNodeId(partialNodeId)
        .localPartition(inputs.partialGroupingKeys)
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeAggregates)
        .capturePlanNodeId(partialNodeId)
        .localPartition(inputs.groupingKeys)
        .finalAggregation()
        .capturePlanNodeId(finalNodeId)
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    auto spillDirectory = TempDirectoryPath::create();

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config)
        .config(core::QueryConfig::kSpillEnabled, true)
        .config(core::QueryConfig::kAggregationSpillEnabled, true)
        .spillDirectory(spillDirectory->getPath())
        .maxDrivers(4);

    exec::TestScopedSpillInjection scopedSpillInjection(100);
    auto task = assertResults(queryBuilder);

    // Expect > 0 spilled bytes unless there was no input.
    const auto partialInputRows =
        toPlanStats(task->taskStats()).at(partialNodeId).inputRows;
    const auto finalInputRows =
        toPlanStats(task->taskStats()).at(finalNodeId).inputRows;
    if (exec::injectedSpillCount() > 0) {
      EXPECT_LT(0, spilledBytes(*task))
          << "partial inputRows: " << partialInputRows
          << " final inputRows: " << finalInputRows
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
    } else {
      EXPECT_EQ(0, spilledBytes(*task))
          << "partial inputRows: " << partialInputRows
          << " final inputRows: " << finalInputRows
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
    }
  }

  {
    SCOPED_TRACE("Run single");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);
    builder
        .singleAggregation(inputs.partialGroupingKeys, inputs.partialAggregates)
        .singleAggregation(inputs.groupingKeys, inputs.mergeAggregates)
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  {
    SCOPED_TRACE("Run partial + intermediate + final");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);
    builder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .intermediateAggregation()
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeAggregates)
        .intermediateAggregation()
        .finalAggregation()
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  if (!inputs.groupingKeys.empty()) {
    SCOPED_TRACE("Run partial + local exchange + final");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);
    builder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .localPartition(inputs.partialGroupingKeys)
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeAggregates)
        .localPartition(inputs.groupingKeys)
        .finalAggregation()
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config).maxDrivers(4);
    assertResults(queryBuilder);
  }

  {
    SCOPED_TRACE(
        "Run partial + local exchange + intermediate + local exchange + final");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);
    builder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .localPartition(inputs.partialGroupingKeys)
        .intermediateAggregation()
        .localPartition(inputs.partialGroupingKeys)
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeAggregates);

    if (inputs.groupingKeys.empty()) {
      builder.localPartitionRoundRobinRow();
    } else {
      builder.localPartition(inputs.groupingKeys);
    }

    builder.intermediateAggregation()
        .localPartition(inputs.groupingKeys)
        .finalAggregation()
        .project(inputs.extractExpressions);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config).maxDrivers(2);
    assertResults(queryBuilder);
  }

  if (streaming && inputs.groupingKeys.empty() &&
      postAggregationProjections.empty()) {
    {
      SCOPED_TRACE("Streaming partial");
      auto partialResult = validateStreamingInTestAggregations(
          [&](auto& builder) {
            builder.values(addExtraGroupingKey(inputs.data, "k0"));
          },
          inputs.partialAggregates,
          config,
          inputs.extractExpressions);

      // The partial check doesn't run with fewer than 2 rows and returns null.
      if (partialResult != nullptr) {
        validateStreamingInTestAggregations(
            [&](auto& builder) { builder.values({partialResult}); },
            inputs.mergeAggregates,
            config,
            inputs.extractExpressions);
      }
    }
  }
}

void AggregationTestBase::testCompanionMergeExtract(
    const CompanionPlanInputs& inputs,
    const std::vector<std::string>& postAggregationProjections,
    const std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>&
        assertResults,
    const std::unordered_map<std::string, std::string>& config) {
  {
    SCOPED_TRACE("Run partial + intermediate + final merge extract");
    PlanBuilder builder(pool());
    inputs.makeSource(builder);

    builder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .intermediateAggregation()
        .finalAggregation()
        .partialAggregation(inputs.groupingKeys, inputs.mergeExtractAggregates)
        .intermediateAggregation()
        .finalAggregation();

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  const auto functionNames =
      std::get<0>(getFunctionNamesAndArgs(inputs.aggregates));
  bool allFunctionsIgnoreNullInputs{true};
  for (const auto& functionName : functionNames) {
    const auto* functionEntry = exec::getAggregateFunctionEntry(functionName);
    VELOX_CHECK_NOT_NULL(functionEntry);
    if (!functionEntry->metadata.ignoreNullInputs) {
      allFunctionsIgnoreNullInputs = false;
      break;
    }
  }
  if (allFunctionsIgnoreNullInputs) {
    SCOPED_TRACE("Run merge with null inputs");
    // Verify that merge-extract companions ignore null intermediate inputs.
    // Produce real intermediate states from the raw input, add null-bearing
    // copies, then merge and validate using the original result assertion.
    PlanBuilder partialBuilder(pool());
    inputs.makeSource(partialBuilder);
    partialBuilder
        .partialAggregation(
            inputs.partialGroupingKeys, inputs.partialAggregates)
        .finalAggregation();
    const auto intermediateInput = AssertQueryBuilder(partialBuilder.planNode())
                                       .configs(config)
                                       .copyResults(pool());

    PlanBuilder mergeExtractBuilder(pool());
    mergeExtractBuilder.values({intermediateInput})
        .singleAggregation(inputs.groupingKeys, inputs.mergeExtractAggregates);
    const auto& mergeExtractNode = static_cast<const core::AggregationNode&>(
        *mergeExtractBuilder.planNode());

    std::vector<core::CallTypedExprPtr> mergeExtractCalls;
    mergeExtractCalls.reserve(mergeExtractNode.aggregates().size());
    for (const auto& aggregate : mergeExtractNode.aggregates()) {
      mergeExtractCalls.push_back(aggregate.call);
    }

    const auto inputWithNulls = makeNullInput(
        mergeExtractCalls, inputs.groupingKeys, intermediateInput, pool());
    if (inputWithNulls) {
      PlanBuilder builder(pool());
      builder.values({inputWithNulls})
          .singleAggregation(
              inputs.groupingKeys, inputs.mergeExtractAggregates);
      if (!postAggregationProjections.empty()) {
        builder.project(postAggregationProjections);
      }

      AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
      queryBuilder.configs(config);
      assertResults(queryBuilder);
    }
  }
}

void AggregationTestBase::testCompanionMerge(
    const CompanionPlanInputs& inputs,
    const std::unordered_map<std::string, std::string>& config) {
  const auto& partialAggregates = inputs.partialAggregates;
  const auto& mergeAggregates = inputs.mergeAggregates;
  auto groupingKeys = inputs.partialGroupingKeys;
  groupingKeys.push_back(kCopyKey);

  // Returns a builder with the intermediate states of the input copies.
  auto makeIntermediate = [&]() {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    PlanBuilder builder(planNodeIdGenerator, pool());
    inputs.makeSource(builder);

    // Copy every row with a key that numbers the copies, so that the copies
    // stay in separate groups.
    auto copies = makeRowVector(
        {kCopyKey},
        {makeFlatVector<int32_t>(
            kNumAbandonedMergeCopies, [](auto row) { return row; })});
    auto outputLayout = builder.planNode()->outputType()->names();
    outputLayout.push_back(kCopyKey);
    builder
        .nestedLoopJoin(
            PlanBuilder(planNodeIdGenerator, pool())
                .values({copies})
                .planNode(),
            outputLayout)
        .partialAggregation(groupingKeys, partialAggregates)
        .finalAggregation();
    return builder;
  };

  {
    SCOPED_TRACE("Run abandoned partial merge without mask");
    auto builder = makeIntermediate();

    // Group by k0 and the copy key because global partial aggregations
    // cannot be abandoned.
    builder.partialAggregation(groupingKeys, mergeAggregates);
    testAbandonedPartialMerge(builder.planNode(), config, pool());
  }

  for (const auto* mask : {"row_number % 2 = 0", "false"}) {
    SCOPED_TRACE(
        fmt::format("Run abandoned partial merge with mask: {}", mask));
    auto builder = makeIntermediate();
    builder.rowNumber({});

    auto projections = groupingKeys;
    for (auto i = 0; i < mergeAggregates.size(); ++i) {
      projections.push_back(fmt::format("a{}", i));
    }
    projections.push_back(fmt::format("{} AS __merge_mask", mask));
    builder.project(projections);

    // Group by k0 and the copy key because global partial aggregations
    // cannot be abandoned.
    builder.partialAggregation(
        groupingKeys,
        mergeAggregates,
        std::vector<std::string>(mergeAggregates.size(), "__merge_mask"));
    testAbandonedPartialMerge(builder.planNode(), config, pool());
  }

  {
    SCOPED_TRACE("Run abandoned partial merge with selected nulls");
    auto builder = makeIntermediate();

    // Read without copying to keep the encodings that some merges require,
    // such as constant intermediate fields. 'cursor' owns the memory of
    // 'intermediateInputs', so it must outlive the merge run.
    CursorParameters params;
    params.planNode = builder.planNode();
    params.queryConfigs = config;
    params.queryConfigs[core::QueryConfig::kMaxOutputBatchRows] = "2";
    params.copyResult = false;
    auto [cursor, intermediateInputs] = exec::test::readCursor(params);
    PlanBuilder mergeBuilder(pool());
    mergeBuilder.values(nullFirstIntermediateRows(
        intermediateInputs,
        groupingKeys.size(),
        builder.planNode()->outputType(),
        pool()));

    // Group by k0 and the copy key because global partial aggregations
    // cannot be abandoned.
    mergeBuilder.partialAggregation(groupingKeys, mergeAggregates);
    testAbandonedPartialMerge(mergeBuilder.planNode(), config, pool());
  }
}

namespace {

void writeToFile(
    const std::string& path,
    const VectorPtr& vector,
    memory::MemoryPool* pool) {
  dwio::common::WriterOptions options;
  options.formatSpecificOptions = std::make_shared<dwrf::DwrfWriterOptions>();
  options.schema = vector->type();
  options.memoryPool = pool;
  auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
  auto sink =
      std::make_unique<dwio::common::WriteFileSink>(std::move(writeFile), path);
  dwrf::Writer writer(std::move(sink), options);
  writer.write(vector);
  writer.close();
}

bool isTableScanSupported(const TypePtr& type) {
  if (type->kind() == TypeKind::ROW && type->size() == 0) {
    return false;
  }
  if (type->kind() == TypeKind::UNKNOWN) {
    return false;
  }
  if (type->kind() == TypeKind::HUGEINT) {
    return false;
  }
  // Disable testing with TableScan when input contains TIMESTAMP type, due to
  // the issue #8127.
  if (type->kind() == TypeKind::TIMESTAMP) {
    return false;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (!isTableScanSupported(type->childAt(i))) {
      return false;
    }
  }

  return true;
}
} // namespace

void AggregationTestBase::testReadFromFiles(
    std::function<void(exec::test::PlanBuilder&)> makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    std::function<std::shared_ptr<exec::Task>(exec::test::AssertQueryBuilder&)>
        assertResults,
    const std::unordered_map<std::string, std::string>& config) {
  PlanBuilder builder(pool());
  makeSource(builder);

  if (!isTableScanSupported(builder.planNode()->outputType())) {
    return;
  }

  auto input = AssertQueryBuilder(builder.planNode()).copyResults(pool());
  if (input->size() == 0) {
    return;
  }

  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<exec::Split> splits;
  std::vector<VectorPtr> inputs;
  auto writerPool = rootPool_->addAggregateChild("AggregationTestBase.writer");

  // Splits and writes the input vectors into two files, to some extent,
  // involves shuffling of the inputs. So only split input if aggregate
  // is non-orderSensitive. Otherwise, only write into a single file.
  if (!hasOrderSensitive(makeSource, aggregates, pool()) &&
      input->size() >= 2) {
    auto size1 = input->size() / 2;
    auto size2 = input->size() - size1;
    auto input1 = input->slice(0, size1);
    auto input2 = input->slice(size1, size2);
    inputs = {input1, input2};
  } else {
    inputs.push_back(input);
  }

  for (auto& vector : inputs) {
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), vector, writerPool.get());
    files.push_back(file);
    splits.emplace_back(
        std::make_shared<connector::hive::HiveConnectorSplit>(
            kHiveConnectorId, file->getPath(), dwio::common::FileFormat::DWRF));
  }
  // No need to test streaming as the streaming test generates its own inputs,
  // so it would be the same as the original test.
  testAggregationsImpl(
      [&](auto& builder) { builder.tableScan(asRowType(input->type())); },
      groupingKeys,
      aggregates,
      postAggregationProjections,
      [&](auto& builder) { return assertResults(builder.splits(splits)); },
      config,
      /*streaming=*/false);

  for (const auto& file : files) {
    remove(file->getPath().c_str());
  }
}

namespace {

std::vector<VectorPtr> extractArgColumns(
    const core::CallTypedExprPtr& aggregateExpr,
    const RowVectorPtr& input,
    memory::MemoryPool* pool) {
  auto& type = input->type()->asRow();
  std::vector<VectorPtr> columns;
  for (const auto& arg : aggregateExpr->inputs()) {
    if (auto field = core::TypedExprs::asFieldAccess(arg)) {
      columns.push_back(input->childAt(field->name()));
    }
    if (core::TypedExprs::isConstant(arg)) {
      auto constant = core::TypedExprs::asConstant(arg);
      columns.push_back(constant->toConstantVector(pool));
    }
    if (auto lambda = core::TypedExprs::asLambda(arg)) {
      for (const auto& name : lambda->signature()->names()) {
        if (auto captureIndex = type.getChildIdxIfExists(name)) {
          columns.push_back(input->childAt(captureIndex.value()));
        }
      }
    }
  }
  return columns;
}

} // namespace

RowVectorPtr AggregationTestBase::validateStreamingInTestAggregations(
    const std::function<void(PlanBuilder&)>& makeSource,
    const std::vector<std::string>& aggregates,
    const std::unordered_map<std::string, std::string>& config,
    const std::vector<std::string>& projections) {
  PlanBuilder builder(pool());
  makeSource(builder);
  auto input = AssertQueryBuilder(builder.planNode())
                   .configs(config)
                   .copyResults(pool());
  if (input->size() < 2) {
    return nullptr;
  }
  auto size1 = input->size() / 2;
  auto size2 = input->size() - size1;
  builder.singleAggregation({}, aggregates);
  auto expected = AssertQueryBuilder(builder.planNode())
                      .configs(config)
                      .copyResults(pool());
  EXPECT_EQ(expected->size(), 1);
  auto& aggregationNode =
      static_cast<const core::AggregationNode&>(*builder.planNode());
  EXPECT_EQ(expected->childrenSize(), aggregationNode.aggregates().size());
  std::vector<VectorPtr> actualResults1;
  std::vector<VectorPtr> actualResults2;
  for (int i = 0; i < aggregationNode.aggregates().size(); ++i) {
    const auto& aggregate = aggregationNode.aggregates()[i];
    if (aggregate.distinct || !aggregate.sortingKeys.empty() ||
        aggregate.mask != nullptr ||
        std::any_of(
            aggregate.call->inputs().begin(),
            aggregate.call->inputs().end(),
            [](const auto& input) { return input->isLambdaKind(); })) {
      // TODO Add support for all these cases.
      return nullptr;
    }

    const auto& aggregateExpr = aggregate.call;
    const auto& name = aggregateExpr->name();

    SCOPED_TRACE(name);
    std::vector<VectorPtr> rawInput1, rawInput2;
    for (const auto& arg : aggregateExpr->inputs()) {
      VectorPtr column;
      auto channel = exec::exprToChannel(arg.get(), input->type());
      if (channel == kConstantChannel) {
        auto constant = dynamic_cast<const core::ConstantTypedExpr*>(arg.get());
        column = constant->toConstantVector(pool());
      } else {
        column = input->childAt(channel);
      }
      rawInput1.push_back(column->slice(0, size1));
      rawInput2.push_back(column->slice(size1, size2));
    }

    actualResults1.push_back(
        testStreaming(name, true, rawInput1, size1, rawInput2, size2, config));
    actualResults2.push_back(
        testStreaming(name, false, rawInput1, size1, rawInput2, size2, config));
  }

  // Projects in the plan that produced the results, since copying them out of
  // it can change encodings the projections rely on, such as the constant
  // fields of a KLL sketch.
  auto expectedProjected = expected;
  if (!projections.empty()) {
    builder.project(projections);
    expectedProjected = AssertQueryBuilder(builder.planNode())
                            .configs(config)
                            .copyResults(pool());
  }
  const auto& names = asRowType(expected->type())->names();
  for (const auto& actualResults : {actualResults1, actualResults2}) {
    auto actual = makeRowVector(names, actualResults);
    if (!projections.empty()) {
      actual = AssertQueryBuilder(PlanBuilder(pool())
                                      .values({actual})
                                      .project(projections)
                                      .planNode())
                   .configs(config)
                   .copyResults(pool());
    }
    for (auto i = 0; i < actual->childrenSize(); ++i) {
      SCOPED_TRACE(aggregationNode.aggregates()[i].call->name());
      velox::exec::test::assertEqualResults(
          {makeRowVector({expectedProjected->childAt(i)})},
          {makeRowVector({actual->childAt(i)})});
    }
  }
  return expected;
}

void AggregationTestBase::testAggregationsImpl(
    std::function<void(PlanBuilder&)> makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>
        assertResults,
    const std::unordered_map<std::string, std::string>& config,
    bool streaming) {
  {
    SCOPED_TRACE("Run partial + final");
    PlanBuilder builder(pool());
    makeSource(builder);
    builder.partialAggregation(groupingKeys, aggregates).finalAggregation();
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  if (!groupingKeys.empty() &&
      !hasOrderSensitive(makeSource, aggregates, pool())) {
    SCOPED_TRACE("Run partial + final with spilling");
    PlanBuilder builder(pool());
    makeSource(builder);

    // Spilling needs at least 2 batches of input. Use round-robin
    // repartitioning to split input into multiple batches.
    core::PlanNodeId partialNodeId;
    core::PlanNodeId finalNodeId;
    builder.localPartitionRoundRobinRow()
        .partialAggregation(groupingKeys, aggregates)
        .capturePlanNodeId(partialNodeId)
        .localPartition(groupingKeys)
        .finalAggregation()
        .capturePlanNodeId(finalNodeId);

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    auto spillDirectory = TempDirectoryPath::create();

    ASSERT_EQ(memory::spillMemoryPool()->stats().usedBytes, 0);
    const auto peakSpillMemoryUsage =
        memory::spillMemoryPool()->stats().peakBytes;
    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config)
        .config(core::QueryConfig::kSpillEnabled, true)
        .config(core::QueryConfig::kAggregationSpillEnabled, true)
        .spillDirectory(spillDirectory->getPath())
        .maxDrivers(4);

    exec::TestScopedSpillInjection scopedSpillInjection(100);
    auto task = assertResults(queryBuilder);

    // Expect > 0 spilled bytes unless there was no input.
    const auto partialInputRows =
        toPlanStats(task->taskStats()).at(partialNodeId).inputRows;
    const auto finalInputRows =
        toPlanStats(task->taskStats()).at(finalNodeId).inputRows;
    if (exec::injectedSpillCount() > 0) {
      EXPECT_LT(0, spilledBytes(*task))
          << "partial inputRows: " << partialInputRows
          << " final inputRows: " << finalInputRows
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
      ASSERT_EQ(memory::spillMemoryPool()->stats().usedBytes, 0);
      ASSERT_GT(memory::spillMemoryPool()->stats().peakBytes, 0);
      ASSERT_GE(
          memory::spillMemoryPool()->stats().peakBytes, peakSpillMemoryUsage);
    } else {
      EXPECT_EQ(0, spilledBytes(*task))
          << "partial inputRows: " << partialInputRows
          << " final inputRows: " << finalInputRows
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
    }
  }

  if (!groupingKeys.empty()) {
    SCOPED_TRACE("Run partial + final with abandon partial agg");
    PlanBuilder builder(pool());
    makeSource(builder);

    core::PlanNodeId partialNodeId;
    core::PlanNodeId intermediateNodeId;
    builder.partialAggregation(groupingKeys, aggregates)
        .capturePlanNodeId(partialNodeId)
        .intermediateAggregation()
        .capturePlanNodeId(intermediateNodeId)
        .finalAggregation();

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    enableAbandonPartialAggregation(queryBuilder);

    auto task = assertResults(queryBuilder);

    // Expect partial aggregation was turned off if there were more than 1 input
    // batches.
    auto taskStats = toPlanStats(task->taskStats());
    auto inputVectors = taskStats.at(partialNodeId).inputVectors;
    auto partialStats = taskStats.at(partialNodeId).customStats;
    auto intermediateStats = taskStats.at(intermediateNodeId).customStats;
    if (inputVectors > 1) {
      EXPECT_LT(0, partialStats.at("abandonedPartialAggregationRows").sum);
      EXPECT_LT(0, intermediateStats.at("abandonedPartialAggregationRows").sum);
    }
  }

  {
    SCOPED_TRACE("Run single");
    PlanBuilder builder(pool());
    makeSource(builder);
    builder.singleAggregation(groupingKeys, aggregates);
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  if (!groupingKeys.empty() &&
      !hasOrderSensitive(makeSource, aggregates, pool())) {
    SCOPED_TRACE("Run single with spilling");
    PlanBuilder builder(pool());
    makeSource(builder);
    core::PlanNodeId aggregationNodeId;
    builder.singleAggregation(groupingKeys, aggregates)
        .capturePlanNodeId(aggregationNodeId);
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    auto spillDirectory = TempDirectoryPath::create();

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config)
        .config(core::QueryConfig::kSpillEnabled, "true")
        .config(core::QueryConfig::kAggregationSpillEnabled, "true")
        .spillDirectory(spillDirectory->getPath());

    exec::TestScopedSpillInjection scopedSpillInjection(100);
    auto task = assertResults(queryBuilder);
    if (exec::injectedSpillCount() > 0) {
      EXPECT_LT(0, spilledBytes(*task))
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
    } else {
      EXPECT_EQ(0, spilledBytes(*task))
          << " spilledRows: " << spilledRows(*task)
          << " spilledInputBytes: " << spilledInputBytes(*task)
          << " spilledFiles: " << spilledFiles(*task)
          << " injectedSpills: " << exec::injectedSpillCount();
    }
  }

  {
    SCOPED_TRACE("Run partial + intermediate + final");
    PlanBuilder builder(pool());
    makeSource(builder);

    builder.partialAggregation(groupingKeys, aggregates)
        .intermediateAggregation()
        .finalAggregation();

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  if (!groupingKeys.empty()) {
    SCOPED_TRACE("Run partial + local exchange + final");
    PlanBuilder builder(pool());
    makeSource(builder);

    builder.partialAggregation(groupingKeys, aggregates)
        .localPartition(groupingKeys)
        .finalAggregation();

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder.maxDrivers(4));
  }

  {
    SCOPED_TRACE(
        "Run partial + local exchange + intermediate + local exchange + final");
    PlanBuilder builder(pool());
    makeSource(builder);

    builder.partialAggregation(groupingKeys, aggregates);

    if (groupingKeys.empty()) {
      builder.localPartitionRoundRobinRow();
    } else {
      builder.localPartition(groupingKeys);
    }

    builder.intermediateAggregation()
        .localPartition(groupingKeys)
        .finalAggregation();

    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder.maxDrivers(2));
  }

  if (streaming && groupingKeys.empty() && postAggregationProjections.empty()) {
    SCOPED_TRACE("Streaming");
    validateStreamingInTestAggregations(makeSource, aggregates, config);
  }
}

void AggregationTestBase::testStreamingAggregationsImpl(
    std::function<void(PlanBuilder&)> makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>
        assertResults,
    const std::unordered_map<std::string, std::string>& config) {
  {
    SCOPED_TRACE("Test StreamingAggregation: run single");
    PlanBuilder builder(pool());
    makeSource(builder);
    builder.orderBy(groupingKeys, false)
        .streamingAggregation(
            groupingKeys,
            aggregates,
            {},
            core::AggregationNode::Step::kSingle,
            false);
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  {
    SCOPED_TRACE(
        "Test StreamingAggregation: run partial + final StreamingAggregation");
    PlanBuilder builder(pool());
    makeSource(builder);
    builder.orderBy(groupingKeys, false)
        .partialStreamingAggregation(groupingKeys, aggregates)
        .finalAggregation();
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }

  {
    SCOPED_TRACE(
        "Test StreamingAggregation: run partial + intermediate + final");
    PlanBuilder builder(pool());
    makeSource(builder);
    builder.orderBy(groupingKeys, false)
        .partialStreamingAggregation(groupingKeys, aggregates)
        .intermediateAggregation()
        .finalAggregation();
    if (!postAggregationProjections.empty()) {
      builder.project(postAggregationProjections);
    }

    AssertQueryBuilder queryBuilder(builder.planNode(), duckDbQueryRunner_);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }
}

void AggregationTestBase::testAggregations(
    std::function<void(PlanBuilder&)> makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>
        assertResults,
    const std::unordered_map<std::string, std::string>& config,
    bool streaming,
    bool incremental) {
  testAggregationsImpl(
      makeSource,
      groupingKeys,
      aggregates,
      postAggregationProjections,
      assertResults,
      config,
      streaming);

  testIgnoreNullInputs(
      makeSource,
      groupingKeys,
      aggregates,
      postAggregationProjections,
      assertResults,
      config,
      duckDbQueryRunner_,
      pool());

  if (incremental) {
    SCOPED_TRACE("testIncrementalAggregation");
    testIncrementalAggregation(makeSource, aggregates, config);
  }

  if (!hasOrderSensitive(makeSource, aggregates, pool()) &&
      !groupingKeys.empty()) {
    testStreamingAggregationsImpl(
        makeSource,
        groupingKeys,
        aggregates,
        postAggregationProjections,
        assertResults,
        config);
  }

  {
    SCOPED_TRACE("Test reading input from table scan");
    testReadFromFiles(
        makeSource,
        groupingKeys,
        aggregates,
        postAggregationProjections,
        assertResults,
        config);
  }
}

VectorPtr AggregationTestBase::testStreaming(
    const std::string& functionName,
    bool testGlobal,
    const std::vector<VectorPtr>& rawInput1,
    const std::vector<VectorPtr>& rawInput2,
    const std::unordered_map<std::string, std::string>& config) {
  VELOX_CHECK(!rawInput1.empty());
  VELOX_CHECK(!rawInput2.empty());
  return testStreaming(
      functionName,
      testGlobal,
      rawInput1,
      rawInput1[0]->size(),
      rawInput2,
      rawInput2[0]->size(),
      config);
}

namespace {

std::unique_ptr<exec::Aggregate> createAggregateFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& inputTypes,
    const std::unordered_map<std::string, std::string>& config,
    const std::vector<VectorPtr>& rawInputs = {}) {
  auto finalType = exec::resolveResultType(functionName, inputTypes);
  auto intermediateType =
      exec::resolveIntermediateType(functionName, inputTypes);
  core::QueryConfig queryConfig({config});
  // Pass constant inputs so aggregates can read constant arguments at
  // initialization time (e.g., ignoreNulls flag for collect_set).
  std::vector<VectorPtr> constantInputs(rawInputs.size());
  for (size_t i = 0; i < rawInputs.size(); ++i) {
    if (rawInputs[i] && rawInputs[i]->isConstantEncoding()) {
      constantInputs[i] = rawInputs[i];
    }
  }
  return exec::Aggregate::create(
      functionName,
      core::AggregationNode::Step::kSingle,
      inputTypes,
      finalType,
      queryConfig,
      constantInputs);
}

// Makes a container for the groups of 'func' and points 'func' at it. The
// container destroys the accumulators of its rows when it goes away, so it
// must not outlive 'func'.
std::unique_ptr<exec::RowContainer> makeGroupRows(
    exec::Aggregate& func,
    const TypePtr& intermediateType,
    memory::MemoryPool* pool) {
  auto rows = std::make_unique<exec::RowContainer>(
      /*keyTypes=*/std::vector<TypePtr>{},
      std::vector<exec::Accumulator>{
          exec::Accumulator{&func, intermediateType}},
      /*dependentTypes=*/std::vector<TypePtr>{},
      pool,
      exec::RowContainer::Options{});
  func.setAllocator(&rows->stringAllocator());
  func.setOffsets(rows->columnAt(0), rows->rowSizeOffset());
  return rows;
}

// Copies every source row twice. Each even row has one non-grouping column
// argument of every aggregate set to null in round-robin order, while each odd
// row preserves the original input.
RowVectorPtr makeNullInput(
    const std::vector<core::CallTypedExprPtr>& aggregateCalls,
    const std::vector<std::string>& groupingKeys,
    const RowVectorPtr& sourceInput,
    memory::MemoryPool* pool) {
  std::vector<std::vector<std::string>> columnArgumentsPerAggregate;
  columnArgumentsPerAggregate.reserve(aggregateCalls.size());
  for (const auto& aggregateCall : aggregateCalls) {
    std::vector<std::string> columnArguments;
    for (const auto& argument : aggregateCall->inputs()) {
      if (const auto field = core::TypedExprs::asFieldAccess(argument); field &&
          std::find(groupingKeys.begin(), groupingKeys.end(), field->name()) ==
              groupingKeys.end()) {
        columnArguments.push_back(field->name());
      }
    }
    if (columnArguments.empty()) {
      return nullptr;
    }
    columnArgumentsPerAggregate.push_back(std::move(columnArguments));
  }
  if (columnArgumentsPerAggregate.empty() || sourceInput->size() == 0) {
    return nullptr;
  }

  const auto size = 2 * sourceInput->size();
  auto indices = allocateIndices(size, pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t row = 0; row < size; ++row) {
    rawIndices[row] = row / 2;
  }

  const auto& sourceType = sourceInput->type()->asRow();
  auto names = sourceType.names();
  auto types = sourceType.children();
  std::vector<VectorPtr> children;
  children.reserve(sourceInput->childrenSize());
  for (size_t column = 0; column < sourceInput->childrenSize(); ++column) {
    BufferPtr nulls;
    for (vector_size_t row = 0; row < size; ++row) {
      if (row % 2 == 0) {
        const auto sourceRow = row / 2;
        for (const auto& columnArguments : columnArgumentsPerAggregate) {
          if (columnArguments[sourceRow % columnArguments.size()] ==
              names[column]) {
            if (!nulls) {
              nulls = allocateNulls(size, pool, bits::kNotNull);
            }
            bits::setNull(nulls->asMutable<uint64_t>(), row);
            break;
          }
        }
      }
    }
    children.push_back(
        BaseVector::wrapInDictionary(
            nulls, indices, size, sourceInput->childAt(column)));
  }

  return std::make_shared<RowVector>(
      pool,
      ROW(std::move(names), std::move(types)),
      nullptr,
      size,
      std::move(children));
}

void testIgnoreNullInputs(
    const std::function<void(PlanBuilder&)>& makeSource,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& postAggregationProjections,
    const std::function<std::shared_ptr<exec::Task>(AssertQueryBuilder&)>&
        assertResults,
    const std::unordered_map<std::string, std::string>& config,
    exec::test::DuckDbQueryRunner& duckDbQueryRunner,
    memory::MemoryPool* pool) {
  PlanBuilder sourceBuilder(pool);
  makeSource(sourceBuilder);
  const auto sourceInput = AssertQueryBuilder(sourceBuilder.planNode())
                               .configs(config)
                               .copyResults(pool);

  PlanBuilder builder(pool);
  makeSource(builder);
  builder.singleAggregation(groupingKeys, aggregates);
  const auto& aggregationNode =
      static_cast<const core::AggregationNode&>(*builder.planNode());

  std::vector<core::CallTypedExprPtr> aggregateCalls;
  aggregateCalls.reserve(aggregationNode.aggregates().size());
  for (const auto& aggregate : aggregationNode.aggregates()) {
    const auto& call = aggregate.call;
    const auto& functionName = call->name();
    if (functionName.starts_with("$internal$")) {
      aggregateCalls.clear();
      break;
    }
    const auto* functionEntry = exec::getAggregateFunctionEntry(functionName);
    VELOX_CHECK_NOT_NULL(functionEntry);
    if (!functionEntry->metadata.ignoreNullInputs ||
        aggregate.rawInputTypes.empty()) {
      aggregateCalls.clear();
      break;
    }
    aggregateCalls.push_back(call);
  }

  if (aggregateCalls.empty()) {
    return;
  }

  const auto nullInput =
      makeNullInput(aggregateCalls, groupingKeys, sourceInput, pool);
  if (nullInput) {
    SCOPED_TRACE("raw input");
    PlanBuilder actualBuilder(pool);
    actualBuilder.values({nullInput})
        .singleAggregation(groupingKeys, aggregates);
    if (!postAggregationProjections.empty()) {
      actualBuilder.project(postAggregationProjections);
    }
    AssertQueryBuilder queryBuilder(
        actualBuilder.planNode(), duckDbQueryRunner);
    queryBuilder.configs(config);
    assertResults(queryBuilder);
  }
}

} // namespace

void AggregationTestBase::testIncrementalAggregation(
    const std::function<void(exec::test::PlanBuilder&)>& makeSource,
    const std::vector<std::string>& aggregates,
    const std::unordered_map<std::string, std::string>& config) {
  PlanBuilder builder(pool());
  makeSource(builder);
  auto data = AssertQueryBuilder(builder.planNode())
                  .configs(config)
                  .copyResults(pool());
  auto inputSize = data->size();
  if (inputSize == 0) {
    return;
  }

  auto& aggregationNode = static_cast<const core::AggregationNode&>(
      *builder.singleAggregation({}, aggregates).planNode());
  for (int i = 0; i < aggregationNode.aggregates().size(); ++i) {
    const auto& aggregate = aggregationNode.aggregates()[i];
    const auto& aggregateExpr = aggregate.call;
    const auto& functionName = aggregateExpr->name();
    auto input = extractArgColumns(aggregateExpr, data, pool());

    std::vector<core::LambdaTypedExprPtr> lambdas;
    for (const auto& arg : aggregate.call->inputs()) {
      if (auto lambda =
              std::dynamic_pointer_cast<const core::LambdaTypedExpr>(arg)) {
        lambdas.push_back(lambda);
      }
    }
    auto queryCtxConfig = config;
    auto func = createAggregateFunction(
        functionName, aggregate.rawInputTypes, config, input);
    auto intermediateType =
        exec::resolveIntermediateType(functionName, aggregate.rawInputTypes);
    auto queryCtx =
        core::QueryCtx::create(nullptr, core::QueryConfig{queryCtxConfig});

    std::shared_ptr<core::ExpressionEvaluator> expressionEvaluator;
    if (!lambdas.empty()) {
      expressionEvaluator = std::make_shared<exec::SimpleExpressionEvaluator>(
          queryCtx.get(), pool());
      func->setLambdaExpressions(lambdas, expressionEvaluator);
    }

    auto groupRows = makeGroupRows(*func, intermediateType, pool());
    char* group = groupRows->newRow();
    std::vector<char*> groups(inputSize, group);
    std::vector<vector_size_t> indices(1, 0);
    func->initializeNewGroups(groups.data(), indices);
    func->addSingleGroupRawInput(
        group, SelectivityVector(inputSize), input, false);

    // Extract intermediate result from the same accumulator twice and expect
    // results to be the same.
    auto intermediateResult1 = BaseVector::create(intermediateType, 1, pool());
    auto intermediateResult2 = BaseVector::create(intermediateType, 1, pool());
    func->extractAccumulators(groups.data(), 1, &intermediateResult1);
    func->extractAccumulators(groups.data(), 1, &intermediateResult2);
    velox::test::assertEqualVectors(intermediateResult1, intermediateResult2);

    // Extract values from the same accumulator twice and expect results to be
    // the same.
    auto result1 = BaseVector::create(func->resultType(), 1, pool());
    auto result2 = BaseVector::create(func->resultType(), 1, pool());
    func->extractValues(groups.data(), 1, &result1);
    func->extractValues(groups.data(), 1, &result2);

    velox::test::assertEqualVectors(result1, result2);
  }
}

VectorPtr AggregationTestBase::testStreaming(
    const std::string& functionName,
    bool testGlobal,
    const std::vector<VectorPtr>& rawInput1,
    vector_size_t rawInput1Size,
    const std::vector<VectorPtr>& rawInput2,
    vector_size_t rawInput2Size,
    const std::unordered_map<std::string, std::string>& config) {
  std::vector<TypePtr> rawInputTypes(rawInput1.size());
  std::transform(
      rawInput1.begin(),
      rawInput1.end(),
      rawInputTypes.begin(),
      [](const VectorPtr& vec) { return vec->type(); });

  auto intermediateType =
      exec::resolveIntermediateType(functionName, rawInputTypes);
  int maxRowCount = std::max(rawInput1Size, rawInput2Size);
  std::vector<vector_size_t> indices(maxRowCount, 0);

  auto intermediate = BaseVector::create(intermediateType, 1, pool());
  {
    auto func =
        createAggregateFunction(functionName, rawInputTypes, config, rawInput1);
    auto groupRows = makeGroupRows(*func, intermediateType, pool());
    char* group = groupRows->newRow();
    std::vector<char*> groups(maxRowCount, group);
    func->initializeNewGroups(groups.data(), indices);
    if (testGlobal) {
      func->addSingleGroupRawInput(
          group, SelectivityVector(rawInput1Size), rawInput1, false);
    } else {
      func->addRawInput(
          groups.data(), SelectivityVector(rawInput1Size), rawInput1, false);
    }
    func->extractAccumulators(groups.data(), 1, &intermediate);
  }

  // Create a new function picking up the intermediate result.
  auto func2 =
      createAggregateFunction(functionName, rawInputTypes, config, rawInput1);
  auto groupRows = makeGroupRows(*func2, intermediateType, pool());
  char* group = groupRows->newRow();
  std::vector<char*> groups(maxRowCount, group);
  func2->initializeNewGroups(groups.data(), indices);
  if (testGlobal) {
    func2->addSingleGroupIntermediateResults(
        group, SelectivityVector(1), {intermediate}, false);
  } else {
    func2->addIntermediateResults(
        groups.data(), SelectivityVector(1), {intermediate}, false);
  }
  // Add more raw input.
  if (testGlobal) {
    func2->addSingleGroupRawInput(
        group, SelectivityVector(rawInput2Size), rawInput2, false);
  } else {
    func2->addRawInput(
        groups.data(), SelectivityVector(rawInput2Size), rawInput2, false);
  }
  auto result = BaseVector::create(func2->resultType(), 1, pool());
  func2->extractValues(groups.data(), 1, &result);
  return result;
}

void AggregationTestBase::testFailingAggregations(
    const std::vector<RowVectorPtr>& data,
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::string& expectedMessage,
    const std::unordered_map<std::string, std::string>& config,
    bool streaming) {
  {
    SCOPED_TRACE("Run single");
    auto builder = PlanBuilder(pool()).values(data);
    builder.singleAggregation(groupingKeys, aggregates);
    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  {
    SCOPED_TRACE("Run partial + final");
    auto builder = PlanBuilder(pool()).values(data);
    builder.partialAggregation(groupingKeys, aggregates).finalAggregation();
    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  {
    SCOPED_TRACE("Run partial + final with abandon partial agg");
    auto builder = PlanBuilder(pool()).values(data);
    builder.partialAggregation(groupingKeys, aggregates)
        .intermediateAggregation()
        .finalAggregation();
    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    enableAbandonPartialAggregation(queryBuilder);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  {
    SCOPED_TRACE("Run partial + intermediate + final");
    auto builder = PlanBuilder(pool()).values(data);
    builder.partialAggregation(groupingKeys, aggregates)
        .intermediateAggregation()
        .finalAggregation();
    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  if (!groupingKeys.empty()) {
    SCOPED_TRACE("Run partial + local exchange + final");
    auto builder = PlanBuilder(pool()).values(data);
    builder.partialAggregation(groupingKeys, aggregates)
        .localPartition(groupingKeys)
        .finalAggregation();
    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  {
    SCOPED_TRACE(
        "Run partial + local exchange + intermediate + local exchange + final");
    auto builder = PlanBuilder(pool()).values(data);
    builder.partialAggregation(groupingKeys, aggregates);

    if (groupingKeys.empty()) {
      builder.localPartitionRoundRobinRow();
    } else {
      builder.localPartition(groupingKeys);
    }

    builder.intermediateAggregation()
        .localPartition(groupingKeys)
        .finalAggregation();

    AssertQueryBuilder queryBuilder(builder.planNode());
    queryBuilder.configs(config);
    VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), expectedMessage);
  }

  // The streaming check splits the input in two, so it doesn't run with fewer
  // than 2 rows.
  vector_size_t numRows{0};
  for (const auto& vector : data) {
    numRows += vector->size();
  }
  if (streaming && groupingKeys.empty() && numRows >= 2) {
    SCOPED_TRACE("Streaming");
    auto makeSource = [&](PlanBuilder& builder) { builder.values(data); };
    VELOX_ASSERT_THROW(
        validateStreamingInTestAggregations(makeSource, aggregates, config),
        expectedMessage);
  }
}

void AggregationTestBase::AggregationTester::assertResults(
    const std::string& duckDbSql) {
  SCOPED_TRACE(duckDbSql);
  run([&](auto& builder) { return builder.assertResults(duckDbSql); });
}

void AggregationTestBase::AggregationTester::assertResults(
    const std::vector<RowVectorPtr>& expected) {
  run([&](auto& builder) { return builder.assertResults(expected); });
}

void AggregationTestBase::AggregationTester::assertResults(
    const RowVectorPtr& expected) {
  assertResults(std::vector<RowVectorPtr>{expected});
}

void AggregationTestBase::AggregationTester::assertResults(
    const AssertResults& assertResults) {
  run(assertResults);
}

void AggregationTestBase::AggregationTester::assertError(
    const std::string& message) {
  VELOX_CHECK(
      !preAggregation_ && projections_.empty() &&
          companionPlans_ == CompanionPlans::kNone,
      "assertError supports only the input, grouping keys, aggregates and "
      "configs");
  test_.testFailingAggregations(
      inputs_, groupingKeys_, aggregates_, message, config_, streaming_);
}

void AggregationTestBase::AggregationTester::run(
    const AssertResults& assertResults) {
  VELOX_CHECK(!inputs_.empty(), "The inputs are not set");
  VELOX_CHECK(!aggregates_.empty(), "The aggregates are not set");
  test_.testAggregations(
      [&](PlanBuilder& builder) { addSource(builder); },
      groupingKeys_,
      aggregates_,
      projections_,
      assertResults,
      config_,
      streaming_,
      incremental_);

  if (companionPlans_ != CompanionPlans::kNone) {
    SCOPED_TRACE("Companion functions");
    const auto companionInputs = CompanionPlanInputs::make(
        inputs_,
        [&](PlanBuilder& builder) {
          addSource(builder);
          // Add the extra grouping key after the pre-aggregation nodes, so a
          // projection there doesn't need to carry it.
          auto projections = builder.planNode()->outputType()->names();
          projections.push_back(
              fmt::format("row_number % {} AS k0", kNumExtraGroupingKeys));
          builder.rowNumber({}).project(projections);
        },
        groupingKeys_,
        aggregates_,
        aggregateArgTypes());
    if (companionPlans_ == CompanionPlans::kAll) {
      test_.testCompanionExtract(
          companionInputs, projections_, assertResults, config_, streaming_);
      test_.testCompanionMergeExtract(
          companionInputs, projections_, assertResults, config_);
    }
    test_.testCompanionMerge(companionInputs, config_);
  }
}

void AggregationTestBase::AggregationTester::addSource(
    PlanBuilder& builder) const {
  builder.values(inputs_);
  if (preAggregation_) {
    preAggregation_(builder);
  }
}

std::vector<std::vector<TypePtr>>
AggregationTestBase::AggregationTester::aggregateArgTypes() const {
  PlanBuilder builder(test_.pool());
  addSource(builder);
  builder.singleAggregation(groupingKeys_, aggregates_);
  const auto& aggregationNode =
      dynamic_cast<const core::AggregationNode&>(*builder.planNode());

  std::vector<std::vector<TypePtr>> argTypes;
  argTypes.reserve(aggregationNode.aggregates().size());
  for (const auto& aggregate : aggregationNode.aggregates()) {
    argTypes.push_back(aggregate.rawInputTypes);
  }
  return argTypes;
}

uint32_t aggregateAndReadRowSize(
    memory::MemoryPool* pool,
    const std::string& name,
    const TypePtr& valueType,
    const VectorPtr& values,
    const VectorPtr& comparisons) {
  core::QueryConfig queryConfig({});
  auto fn = exec::Aggregate::create(
      name,
      core::AggregationNode::Step::kSingle,
      std::vector<TypePtr>{valueType, comparisons->type()},
      valueType,
      queryConfig,
      /*constantInputs=*/{});

  auto groupRows = makeGroupRows(*fn, valueType, pool);

  const auto size = values->size();
  char* group = groupRows->newRow();
  std::vector<char*> groups(size, group);
  std::vector<vector_size_t> indices{0};
  fn->initializeNewGroups(groups.data(), indices);

  SelectivityVector rows{size};
  fn->addRawInput(groups.data(), rows, {values, comparisons}, false);

  return groupRows->rowSize(group) - groupRows->fixedRowSize();
}
} // namespace facebook::velox::functions::aggregate::test

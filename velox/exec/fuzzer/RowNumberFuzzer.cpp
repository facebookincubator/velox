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

#include "velox/exec/fuzzer/RowNumberFuzzer.h"
#include <boost/random/uniform_int_distribution.hpp>
#include <utility>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int32(steps, 10, "Number of plans to generate and test.");

DEFINE_int32(
    duration_sec,
    0,
    "For how long it should run (in seconds). If zero, "
    "it executes exactly --steps iterations and exits.");

DEFINE_int32(
    batch_size,
    100,
    "The number of elements on each generated vector.");

DEFINE_int32(num_batches, 10, "The number of generated vectors.");

DEFINE_double(
    null_ratio,
    0.1,
    "Chance of adding a null value in a vector "
    "(expressed as double from 0 to 1).");

DEFINE_bool(enable_spill, true, "Whether to test plans with spilling enabled.");

DEFINE_bool(
    enable_oom_injection,
    false,
    "When enabled OOMs will randomly be triggered while executing query "
    "plans. The goal of this mode is to ensure unexpected exceptions "
    "aren't thrown and the process isn't killed in the process of cleaning "
    "up after failures. Therefore, results are not compared when this is "
    "enabled. Note that this option only works in debug builds.");

namespace facebook::velox::exec::test {
namespace {

class RowNumberFuzzer {
 public:
  explicit RowNumberFuzzer(
      size_t initialSeed,
      std::unique_ptr<ReferenceQueryRunner>);

  void go();

  struct PlanWithSplits {
    core::PlanNodePtr plan;
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;

    explicit PlanWithSplits(
        core::PlanNodePtr _plan,
        const std::vector<std::shared_ptr<connector::ConnectorSplit>>& _splits =
            {})
        : plan(std::move(_plan)), splits(_splits) {}
  };

 private:
  static VectorFuzzer::Options getFuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_batch_size;
    opts.stringVariableLength = true;
    opts.stringLength = 100;
    opts.nullRatio = FLAGS_null_ratio;
    return opts;
  }

  static inline const std::string kHiveConnectorId = "test-hive";

  // Makes a connector split from a file path on storage.
  static std::shared_ptr<connector::ConnectorSplit> makeSplit(
      const std::string& filePath);

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

  void reSeed() {
    seed(rng_());
  }

  // Runs one test iteration from query plans generations, executions and result
  // verifications.
  void verify();

  int32_t randInt(int32_t min, int32_t max) {
    return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
  }

  std::pair<std::vector<std::string>, std::vector<TypePtr>>
  generatePartitionKeys();

  std::vector<RowVectorPtr> generateInput(
      const std::vector<std::string>& keyNames,
      const std::vector<TypePtr>& keyTypes);

  std::optional<MaterializedRowMultiset> computeReferenceResults(
      core::PlanNodePtr& plan,
      const std::vector<RowVectorPtr>& input);

  RowVectorPtr execute(const PlanWithSplits& plan, bool injectSpill);

  void addPlansWithTableScan(
      const std::string& tableDir,
      const std::vector<std::string>& partitionKeys,
      const std::vector<RowVectorPtr>& input,
      std::vector<PlanWithSplits>& altPlans);

  // Makes the query plan with default settings in RowNumberFuzzer and value
  // inputs for both probe and build sides.
  //
  // NOTE: 'input' could either input rows with lazy
  // vectors or flatten ones.
  static PlanWithSplits makeDefaultPlan(
      const std::vector<std::string>& partitionKeys,
      const std::vector<RowVectorPtr>& input);

  static PlanWithSplits makePlanWithTableScan(
      const RowTypePtr& type,
      const std::vector<std::string>& partitionKeys,
      const std::vector<std::shared_ptr<connector::ConnectorSplit>>& splits);

  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool(
          "rowNumberFuzzer",
          memory::kMaxMemory,
          memory::MemoryReclaimer::create())};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild(
      "rowNumberFuzzerLeaf",
      true,
      exec::MemoryReclaimer::create())};
  std::shared_ptr<memory::MemoryPool> writerPool_{rootPool_->addAggregateChild(
      "rowNumberFuzzerWriter",
      exec::MemoryReclaimer::create())};
  VectorFuzzer vectorFuzzer_;
  std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner_;
};

RowNumberFuzzer::RowNumberFuzzer(
    size_t initialSeed,
    std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner)
    : vectorFuzzer_{getFuzzerOptions(), pool_.get()},
      referenceQueryRunner_{std::move(referenceQueryRunner)} {
  filesystems::registerLocalFileSystem();

  // Make sure not to run out of open file descriptors.
  const std::unordered_map<std::string, std::string> hiveConfig = {
      {connector::hive::HiveConfig::kNumCacheFileHandles, "1000"}};
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(
              kHiveConnectorId, std::make_shared<core::MemConfig>(hiveConfig));
  connector::registerConnector(hiveConnector);

  seed(initialSeed);
}

void writeToFile(
    const std::string& path,
    const VectorPtr& vector,
    memory::MemoryPool* pool) {
  dwrf::WriterOptions options;
  options.schema = vector->type();
  options.memoryPool = pool;
  auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
  auto sink =
      std::make_unique<dwio::common::WriteFileSink>(std::move(writeFile), path);
  dwrf::Writer writer(std::move(sink), options);
  writer.write(vector);
  writer.close();
}

// static
std::shared_ptr<connector::ConnectorSplit> RowNumberFuzzer::makeSplit(
    const std::string& filePath) {
  return std::make_shared<connector::hive::HiveConnectorSplit>(
      kHiveConnectorId, filePath, dwio::common::FileFormat::DWRF);
}

template <typename T>
bool isDone(size_t i, T startTime) {
  if (FLAGS_duration_sec > 0) {
    std::chrono::duration<double> elapsed =
        std::chrono::system_clock::now() - startTime;
    return elapsed.count() >= FLAGS_duration_sec;
  }
  return i >= FLAGS_steps;
}

std::vector<RowVectorPtr> flatten(const std::vector<RowVectorPtr>& vectors) {
  std::vector<RowVectorPtr> flatVectors;
  for (const auto& vector : vectors) {
    auto flat = BaseVector::create<RowVector>(
        vector->type(), vector->size(), vector->pool());
    flat->copy(vector.get(), 0, 0, vector->size());
    flatVectors.push_back(flat);
  }

  return flatVectors;
}

std::pair<std::vector<std::string>, std::vector<TypePtr>>
RowNumberFuzzer::generatePartitionKeys() {
  const auto numKeys = randInt(1, 3);
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < numKeys; ++i) {
    names.push_back(fmt::format("c{}", i));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/1));
  }
  return std::make_pair(names, types);
}

std::vector<RowVectorPtr> RowNumberFuzzer::generateInput(
    const std::vector<std::string>& keyNames,
    const std::vector<TypePtr>& keyTypes) {
  std::vector<std::string> names = keyNames;
  std::vector<TypePtr> types = keyTypes;
  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("c{}", i + keyNames.size()));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/2));
  }

  const auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  input.reserve(FLAGS_num_batches);
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
  }

  return input;
}

RowNumberFuzzer::PlanWithSplits RowNumberFuzzer::makeDefaultPlan(
    const std::vector<std::string>& partitionKeys,
    const std::vector<RowVectorPtr>& input) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<std::string> projectFields = partitionKeys;
  projectFields.emplace_back("row_number");
  auto plan = PlanBuilder()
                  .values(input)
                  .rowNumber(partitionKeys)
                  .project(projectFields)
                  .planNode();
  return PlanWithSplits{std::move(plan)};
}

bool containsType(const TypePtr& type, const TypePtr& search) {
  if (type->equivalent(*search)) {
    return true;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (containsType(type->childAt(i), search)) {
      return true;
    }
  }
  return false;
}

bool containsTypeKind(const TypePtr& type, const TypeKind& search) {
  if (type->kind() == search) {
    return true;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (containsTypeKind(type->childAt(i), search)) {
      return true;
    }
  }

  return false;
}

bool containsUnsupportedTypes(const TypePtr& type) {
  // Skip queries that use Timestamp, Varbinary, and IntervalDayTime types.
  // DuckDB doesn't support nanosecond precision for timestamps or casting from
  // Bigint to Interval.
  // TODO Investigate mismatches reported when comparing Varbinary.
  return containsTypeKind(type, TypeKind::TIMESTAMP) ||
      containsTypeKind(type, TypeKind::VARBINARY) ||
      containsType(type, INTERVAL_DAY_TIME());
}

std::optional<MaterializedRowMultiset> RowNumberFuzzer::computeReferenceResults(
    core::PlanNodePtr& plan,
    const std::vector<RowVectorPtr>& input) {
  if (containsUnsupportedTypes(input[0]->type())) {
    return std::nullopt;
  }

  if (auto sql = referenceQueryRunner_->toSql(plan)) {
    return referenceQueryRunner_->execute(
        sql.value(), input, plan->outputType());
  }

  LOG(INFO) << "Query not supported by the reference DB";
  return std::nullopt;
}

RowVectorPtr RowNumberFuzzer::execute(
    const PlanWithSplits& plan,
    bool injectSpill) {
  LOG(INFO) << "Executing query plan: " << plan.plan->toString(true, true);

  AssertQueryBuilder builder(plan.plan);
  if (!plan.splits.empty()) {
    builder.splits(plan.splits);
  }

  std::shared_ptr<TempDirectoryPath> spillDirectory;
  int32_t spillPct{0};
  if (injectSpill) {
    spillDirectory = exec::test::TempDirectoryPath::create();
    builder.config(core::QueryConfig::kSpillEnabled, true)
        .config(core::QueryConfig::kRowNumberSpillEnabled, true)
        .spillDirectory(spillDirectory->getPath());
    spillPct = 10;
  }

  ScopedOOMInjector oomInjector(
      []() -> bool { return folly::Random::oneIn(10); },
      10); // Check the condition every 10 ms.
  if (FLAGS_enable_oom_injection) {
    oomInjector.enable();
  }

  // Wait for the task to be destroyed before start next query execution to
  // avoid the potential interference of the background activities across query
  // executions.
  auto stopGuard = folly::makeGuard([&]() { waitForAllTasksToBeDeleted(); });

  TestScopedSpillInjection scopedSpillInjection(spillPct);
  RowVectorPtr result;
  try {
    result = builder.copyResults(pool_.get());
  } catch (VeloxRuntimeError& e) {
    if (FLAGS_enable_oom_injection &&
        e.errorCode() == facebook::velox::error_code::kMemCapExceeded &&
        e.message() == ScopedOOMInjector::kErrorMessage) {
      // If we enabled OOM injection we expect the exception thrown by the
      // ScopedOOMInjector.
      return nullptr;
    }

    throw e;
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << std::endl << result->toString(0, result->size());
  }

  return result;
}

RowNumberFuzzer::PlanWithSplits RowNumberFuzzer::makePlanWithTableScan(
    const RowTypePtr& type,
    const std::vector<std::string>& partitionKeys,
    const std::vector<std::shared_ptr<connector::ConnectorSplit>>& splits) {
  std::vector<std::string> projectFields = partitionKeys;
  projectFields.emplace_back("row_number");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .tableScan(type)
                  .rowNumber(partitionKeys)
                  .project(projectFields)
                  .planNode();
  return PlanWithSplits{plan, splits};
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

void RowNumberFuzzer::addPlansWithTableScan(
    const std::string& tableDir,
    const std::vector<std::string>& partitionKeys,
    const std::vector<RowVectorPtr>& input,
    std::vector<PlanWithSplits>& altPlans) {
  VELOX_CHECK(!tableDir.empty());

  if (!isTableScanSupported(input[0]->type())) {
    return;
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> inputSplits;
  for (auto i = 0; i < input.size(); ++i) {
    const std::string filePath = fmt::format("{}/row_number/{}", tableDir, i);
    writeToFile(filePath, input[i], writerPool_.get());
    inputSplits.push_back(makeSplit(filePath));
  }

  altPlans.push_back(makePlanWithTableScan(
      asRowType(input[0]->type()), partitionKeys, inputSplits));
}

void RowNumberFuzzer::verify() {
  const auto [keyNames, keyTypes] = generatePartitionKeys();
  const auto input = generateInput(keyNames, keyTypes);
  // Flatten inputs.
  const auto flatInput = flatten(input);

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Input: " << input[0]->toString();
    for (const auto& v : flatInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }
  }

  auto defaultPlan = makeDefaultPlan(keyNames, input);
  const auto expected = execute(defaultPlan, /*injectSpill=*/false);

  if (expected != nullptr) {
    if (const auto referenceResult =
            computeReferenceResults(defaultPlan.plan, input)) {
      VELOX_CHECK(
          assertEqualResults(
              referenceResult.value(),
              defaultPlan.plan->outputType(),
              {expected}),
          "Velox and Reference results don't match");
    }
  }

  std::vector<PlanWithSplits> altPlans;
  altPlans.push_back(std::move(defaultPlan));

  const auto tableScanDir = exec::test::TempDirectoryPath::create();
  addPlansWithTableScan(tableScanDir->getPath(), keyNames, input, altPlans);

  for (auto i = 0; i < altPlans.size(); ++i) {
    LOG(INFO) << "Testing plan #" << i;
    auto actual = execute(altPlans[i], /*injectSpill=*/false);
    if (actual != nullptr && expected != nullptr) {
      VELOX_CHECK(
          assertEqualResults({expected}, {actual}),
          "Logically equivalent plans produced different results");
    } else {
      VELOX_CHECK(
          FLAGS_enable_oom_injection, "Got unexpected nullptr for results");
    }

    if (FLAGS_enable_spill) {
      LOG(INFO) << "Testing plan #" << i << " with spilling";
      actual = execute(altPlans[i], /*=injectSpill=*/true);
      if (actual != nullptr && expected != nullptr) {
        try {
          VELOX_CHECK(
              assertEqualResults({expected}, {actual}),
              "Logically equivalent plans produced different results");
        } catch (const VeloxException&) {
          LOG(ERROR) << "Expected\n"
                     << expected->toString(0, expected->size()) << "\nActual\n"
                     << actual->toString(0, actual->size());
          throw;
        }
      } else {
        VELOX_CHECK(
            FLAGS_enable_oom_injection, "Got unexpected nullptr for results");
      }
    }
  }
}

void RowNumberFuzzer::go() {
  VELOX_USER_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.")
  VELOX_USER_CHECK_GE(FLAGS_batch_size, 10, "Batch size must be at least 10.");

  const auto startTime = std::chrono::system_clock::now();
  size_t iteration = 0;

  while (!isDone(iteration, startTime)) {
    LOG(INFO) << "==============================> Started iteration "
              << iteration << " (seed: " << currentSeed_ << ")";
    verify();
    LOG(INFO) << "==============================> Done with iteration "
              << iteration;

    reSeed();
    ++iteration;
  }
}
} // namespace

void rowNumberFuzzer(
    size_t seed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner) {
  RowNumberFuzzer(seed, std::move(referenceQueryRunner)).go();
}
} // namespace facebook::velox::exec::test

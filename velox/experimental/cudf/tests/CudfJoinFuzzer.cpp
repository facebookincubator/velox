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

#include "velox/experimental/cudf/tests/CudfJoinFuzzer.h"

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <boost/random/uniform_int_distribution.hpp>

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

namespace facebook::velox::cudf_velox::test {

namespace {

using namespace facebook::velox::exec;

class CudfJoinFuzzer {
 public:
  CudfJoinFuzzer(
      size_t initialSeed,
      std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner);

  void go();

 private:
  static VectorFuzzer::Options getFuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_batch_size;
    opts.stringVariableLength = true;
    opts.stringLength = 100;
    opts.nullRatio = FLAGS_null_ratio;
    opts.useRandomNullPattern = true;
    opts.timestampPrecision =
        VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
    return opts;
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

  void reSeed() {
    seed(rng_());
  }

  // Randomly pick a join type to test.
  core::JoinType pickJoinType();

  // Returns the name of the join type for logging.
  static std::string joinTypeName(core::JoinType joinType);

  // Runs one test iteration from query plans generations, executions and result
  // verifications.
  void verify(core::JoinType joinType);

  // Returns a list of randomly generated join key types.
  std::vector<TypePtr> generateJoinKeyTypes(int32_t numKeys);

  // Returns randomly generated probe input with upto 3 additional payload
  // columns.
  std::vector<RowVectorPtr> generateProbeInput(
      const std::vector<std::string>& keyNames,
      const std::vector<TypePtr>& keyTypes);

  // Same as generateProbeInput() but copies over 10% of the input in the probe
  // columns to ensure some matches during joining. Also generates an empty
  // input with a 10% chance.
  std::vector<RowVectorPtr> generateBuildInput(
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<std::string>& probeKeys,
      const std::vector<std::string>& buildKeys);

  RowVectorPtr execute(const core::PlanNodePtr& plan);

  std::optional<exec::test::MaterializedRowMultiset> computeReferenceResults(
      const core::PlanNodePtr& plan,
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<RowVectorPtr>& buildInput);

  int32_t randInt(int32_t min, int32_t max) {
    return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
  }

  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool(
          "cudfJoinFuzzer",
          memory::kMaxMemory,
          memory::MemoryReclaimer::create())};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild(
      "cudfJoinFuzzerLeaf",
      true,
      exec::MemoryReclaimer::create())};

  VectorFuzzer vectorFuzzer_;
  std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner_;

  struct Stats {
    size_t numIterations{0};
    size_t numVerified{0};

    std::string toString() const {
      std::stringstream out;
      out << "\nTotal iterations tested: " << numIterations << std::endl;
      out << "Total iterations verified against reference DB: " << numVerified
          << " (" << (double)numVerified / numIterations * 100 << "%)"
          << std::endl;
      return out.str();
    }
  };

  Stats stats_;
};

CudfJoinFuzzer::CudfJoinFuzzer(
    size_t initialSeed,
    std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner)
    : vectorFuzzer_{getFuzzerOptions(), pool_.get()},
      referenceQueryRunner_{std::move(referenceQueryRunner)} {
  filesystems::registerLocalFileSystem();

  std::unordered_map<std::string, std::string> hiveConfig = {
      {connector::hive::HiveConfig::kNumCacheFileHandles, "1000"}};

  connector::hive::HiveConnectorFactory factory;
  auto hiveConnector = factory.newConnector(
      exec::test::kHiveConnectorId,
      std::make_shared<config::ConfigBase>(std::move(hiveConfig)));
  if (!connector::hasConnector(exec::test::kHiveConnectorId)) {
    connector::registerConnector(hiveConnector);
  }

  seed(initialSeed);
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

core::JoinType CudfJoinFuzzer::pickJoinType() {
  // Join types that are both:
  // 1. Supported by CudfHashJoinProbe::isSupportedJoinType()
  // 2. Can be converted to SQL by DuckDB reference runner
  // Note: kRight and kRightSemiFilter are supported by cuDF but not by DuckDB's
  // SQL conversion, so we exclude them here.
  static std::vector<core::JoinType> kJoinTypes = {
      core::JoinType::kInner,
      core::JoinType::kLeft,
      core::JoinType::kAnti,
      core::JoinType::kLeftSemiFilter,
  };

  const size_t idx = randInt(0, kJoinTypes.size() - 1);
  return kJoinTypes[idx];
}

std::string CudfJoinFuzzer::joinTypeName(core::JoinType joinType) {
  switch (joinType) {
    case core::JoinType::kInner:
      return "INNER";
    case core::JoinType::kLeft:
      return "LEFT";
    case core::JoinType::kRight:
      return "RIGHT";
    case core::JoinType::kAnti:
      return "ANTI";
    case core::JoinType::kLeftSemiFilter:
      return "LEFT SEMI FILTER";
    case core::JoinType::kRightSemiFilter:
      return "RIGHT SEMI FILTER";
    default:
      return "UNKNOWN";
  }
}

std::vector<TypePtr> CudfJoinFuzzer::generateJoinKeyTypes(int32_t numKeys) {
  // Types supported by both cuDF and DuckDB for join keys.
  // cuDF has limited type support compared to CPU Velox.
  static std::vector<TypePtr> kSupportedTypes = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
  };

  std::vector<TypePtr> types;
  types.reserve(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    types.push_back(vectorFuzzer_.randType(kSupportedTypes, /*maxDepth=*/0));
  }
  return types;
}

std::vector<RowVectorPtr> CudfJoinFuzzer::generateProbeInput(
    const std::vector<std::string>& keyNames,
    const std::vector<TypePtr>& keyTypes) {
  std::vector<std::string> names = keyNames;
  std::vector<TypePtr> types = keyTypes;

  bool keyTypesAllBool = true;
  for (const auto& type : keyTypes) {
    if (!type->isBoolean()) {
      keyTypesAllBool = false;
      break;
    }
  }

  // Types supported by cuDF for payload columns.
  static std::vector<TypePtr> kSupportedTypes = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
  };

  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("tp{}", i + keyNames.size()));
    types.push_back(vectorFuzzer_.randType(kSupportedTypes, /*maxDepth=*/0));
  }

  const auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    if (keyTypesAllBool) {
      // Joining on just boolean keys creates so many hits it explodes the
      // output size, reduce the batch size to 10% to control the output size.
      input.push_back(
          vectorFuzzer_.fuzzRow(inputType, FLAGS_batch_size / 10, false));
    } else {
      input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
    }
  }
  return input;
}

std::vector<RowVectorPtr> CudfJoinFuzzer::generateBuildInput(
    const std::vector<RowVectorPtr>& probeInput,
    const std::vector<std::string>& probeKeys,
    const std::vector<std::string>& buildKeys) {
  std::vector<std::string> names = buildKeys;
  std::vector<TypePtr> types;
  for (const auto& key : probeKeys) {
    types.push_back(asRowType(probeInput[0]->type())->findChild(key));
  }

  // Types supported by cuDF for payload columns.
  static std::vector<TypePtr> kSupportedTypes = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
  };

  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("bp{}", i + buildKeys.size()));
    types.push_back(vectorFuzzer_.randType(kSupportedTypes, /*maxDepth=*/0));
  }

  const auto rowType = ROW(std::move(names), std::move(types));

  // 1 in 10 times use empty build.
  if (vectorFuzzer_.coinToss(0.1)) {
    return {BaseVector::create<RowVector>(rowType, 0, pool_.get())};
  }

  // To ensure there are some matches, sample with replacement 10% of probe join
  // keys and use these as 80% of build keys.
  std::vector<RowVectorPtr> input;
  for (const auto& probe : probeInput) {
    auto numRows = 1 + probe->size() / 8;
    auto build = vectorFuzzer_.fuzzRow(rowType, numRows, false);

    // Pick probe side rows to copy.
    std::vector<vector_size_t> rowNumbers(numRows);
    SelectivityVector rows(numRows, false);
    for (auto i = 0; i < numRows; ++i) {
      if (vectorFuzzer_.coinToss(0.8) && probe->size() > 0) {
        rowNumbers[i] = randInt(0, probe->size() - 1);
        rows.setValid(i, true);
      }
    }

    for (auto i = 0; i < probeKeys.size(); ++i) {
      build->childAt(i)->resize(numRows);
      build->childAt(i)->copy(probe->childAt(i).get(), rows, rowNumbers.data());
    }

    for (auto i = 0; i < numPayload; ++i) {
      auto column = i + probeKeys.size();
      build->childAt(column) =
          vectorFuzzer_.fuzz(rowType->childAt(column), numRows);
    }

    input.push_back(build);
  }

  return input;
}

RowVectorPtr CudfJoinFuzzer::execute(const core::PlanNodePtr& plan) {
  LOG(INFO) << "Executing query plan: " << std::endl
            << plan->toString(true, true);

  exec::test::AssertQueryBuilder builder(plan);

  RowVectorPtr result = builder.maxDrivers(2).copyResults(pool_.get());

  LOG(INFO) << "Results: " << result->toString();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << std::endl << result->toString(0, result->size());
  }

  exec::test::waitForAllTasksToBeDeleted();
  return result;
}

std::optional<exec::test::MaterializedRowMultiset>
CudfJoinFuzzer::computeReferenceResults(
    const core::PlanNodePtr& plan,
    const std::vector<RowVectorPtr>& probeInput,
    const std::vector<RowVectorPtr>& buildInput) {
  if (referenceQueryRunner_->runnerType() ==
      exec::test::ReferenceQueryRunner::RunnerType::kDuckQueryRunner) {
    VELOX_CHECK(!exec::test::containsUnsupportedTypes(probeInput[0]->type()));
    VELOX_CHECK(!exec::test::containsUnsupportedTypes(buildInput[0]->type()));
  }

  auto result = referenceQueryRunner_->execute(plan);
  VELOX_CHECK_NE(
      result.second, exec::test::ReferenceQueryErrorCode::kReferenceQueryFail);
  return result.first;
}

void CudfJoinFuzzer::verify(core::JoinType joinType) {
  const int numKeys = randInt(1, 5);
  std::vector<TypePtr> keyTypes = generateJoinKeyTypes(numKeys);

  std::vector<std::string> probeKeys = exec::test::makeNames("t", numKeys);
  std::vector<std::string> buildKeys = exec::test::makeNames("u", numKeys);

  auto probeInput = generateProbeInput(probeKeys, keyTypes);
  auto buildInput = generateBuildInput(probeInput, probeKeys, buildKeys);

  auto [convertedProbeInput, probeProjections] =
      referenceQueryRunner_->inputProjections(probeInput);
  auto [convertedBuildInput, buildProjections] =
      referenceQueryRunner_->inputProjections(buildInput);

  VELOX_CHECK(!convertedProbeInput.empty());
  VELOX_CHECK(!convertedBuildInput.empty());

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Probe input: " << convertedProbeInput[0]->toString();
    for (const auto& v : convertedProbeInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }

    VLOG(1) << "Build input: " << convertedBuildInput[0]->toString();
    for (const auto& v : convertedBuildInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }
  }

  // Determine output columns based on join type.
  std::vector<std::string> outputColumns;
  if (core::isLeftSemiFilterJoin(joinType) ||
      core::isRightSemiFilterJoin(joinType) || core::isAntiJoin(joinType)) {
    // Semi and anti joins only output probe columns.
    if (core::isRightSemiFilterJoin(joinType)) {
      outputColumns = asRowType(buildInput[0]->type())->names();
    } else {
      outputColumns = asRowType(probeInput[0]->type())->names();
    }
  } else {
    // Other joins output both probe and build columns.
    auto combinedType = exec::test::concat(
        asRowType(probeInput[0]->type()), asRowType(buildInput[0]->type()));
    outputColumns = combinedType->names();
  }

  // Shuffle output columns.
  std::shuffle(outputColumns.begin(), outputColumns.end(), rng_);

  // Remove some output columns.
  const auto numOutput = randInt(1, outputColumns.size());
  outputColumns.resize(numOutput);

  // Build the hash join plan.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                  .values(convertedProbeInput)
                  .hashJoin(
                      probeKeys,
                      buildKeys,
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values(convertedBuildInput)
                          .planNode(),
                      /*filter=*/"",
                      outputColumns,
                      joinType)
                  .planNode();

  const auto result = execute(plan);

  // Verify against reference.
  if (auto referenceResult = computeReferenceResults(
          plan, convertedProbeInput, convertedBuildInput)) {
    VELOX_CHECK(
        exec::test::assertEqualResults(
            referenceResult.value(), plan->outputType(), {result}),
        "Velox and Reference results don't match");

    LOG(INFO) << "Result matches with reference DB.";
    stats_.numVerified++;
  }
}

void CudfJoinFuzzer::go() {
  VELOX_USER_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.");
  VELOX_USER_CHECK_GE(FLAGS_batch_size, 10, "Batch size must be at least 10.");

  const auto startTime = std::chrono::system_clock::now();

  while (!isDone(stats_.numIterations, startTime)) {
    const auto joinType = pickJoinType();

    LOG(WARNING) << "==============================> Started iteration "
                 << stats_.numIterations << " (seed: " << currentSeed_
                 << ", join type: " << joinTypeName(joinType) << ")";

    verify(joinType);

    LOG(WARNING) << "==============================> Done with iteration "
                 << stats_.numIterations;

    reSeed();
    ++stats_.numIterations;
  }

  LOG(INFO) << stats_.toString();
}

} // namespace

void cudfJoinFuzzer(
    size_t seed,
    std::unique_ptr<exec::test::ReferenceQueryRunner> referenceQueryRunner) {
  CudfJoinFuzzer(seed, std::move(referenceQueryRunner)).go();
}

} // namespace facebook::velox::cudf_velox::test

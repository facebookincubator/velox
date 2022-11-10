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
#include "velox/exec/tests/AggregationFuzzer.h"
#include <boost/random/uniform_int_distribution.hpp>
#include "velox/common/base/Fs.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/tests/ArgumentTypeFuzzer.h"
#include "velox/expression/tests/FuzzerToolkit.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int32(steps, 10, "Number of plans to generate and execute.");

DEFINE_int32(
    duration_sec,
    0,
    "For how long it should run (in seconds). If zero, "
    "it executes exactly --steps iterations and exits.");

DEFINE_int32(
    batch_size,
    100,
    "The number of elements on each generated vector.");

DEFINE_int32(
    max_num_varargs,
    5,
    "The maximum number of variadic arguments fuzzer will generate for "
    "functions that accept variadic arguments. Fuzzer will generate up to "
    "max_num_varargs arguments for the variadic list in addition to the "
    "required arguments by the function.");

DEFINE_double(
    null_ratio,
    0.1,
    "Chance of adding a null constant to the plan, or null value in a vector "
    "(expressed as double from 0 to 1).");

DEFINE_string(
    repro_persist_path,
    "",
    "Directory path for persistence of data and SQL when fuzzer fails for "
    "future reproduction. Empty string disables this feature.");

DEFINE_bool(
    persist_and_run_once,
    false,
    "Persist repro info before evaluation and only run one iteration. "
    "This is to rerun with the seed number and persist repro info upon a "
    "crash failure. Only effective if repro_persist_path is set.");

using facebook::velox::test::CallableSignature;
using facebook::velox::test::SignatureTemplate;

namespace facebook::velox::exec::test {
namespace {

struct ResultOrError {
  RowVectorPtr result;
  std::exception_ptr exceptionPtr;
};

class AggregationFuzzer {
 public:
  AggregationFuzzer(
      AggregateFunctionSignatureMap signatureMap,
      size_t seed,
      const std::unordered_set<std::string>& orderDependentFunctions);

  void go();

 private:
  struct Stats {
    // Names of functions that were tested.
    std::unordered_set<std::string> functionNames;

    // Number of iterations using masked aggregation.
    size_t numMask{0};

    // Number of iterations using group-by aggregation.
    size_t numGroupBy{0};

    // Number of iterations using global aggregation.
    size_t numGlobal{0};

    // Number of iterations where results were verified against DuckDB,
    size_t numDuckDbVerified{0};

    // Number of iterations where aggregation failed.
    size_t numFailed{0};

    void print(size_t numIterations) const;
  };

  static VectorFuzzer::Options getFuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_batch_size;
    opts.stringVariableLength = true;
    opts.stringLength = 100;
    opts.nullRatio = FLAGS_null_ratio;
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

  CallableSignature pickSignature();

  void verify(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& masks,
      const std::vector<RowVectorPtr>& input,
      bool orderDependent);

  std::optional<MaterializedRowMultiset> computeDuckDbResult(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& masks,
      const std::vector<RowVectorPtr>& input,
      const core::PlanNodePtr& plan);

  ResultOrError execute(const core::PlanNodePtr& plan);

  const std::unordered_set<std::string> orderDependentFunctions_;
  const bool persistAndRunOnce_;
  const std::string reproPersistPath_;

  std::unordered_set<std::string> duckDbFunctionNames_;

  std::vector<CallableSignature> signatures_;
  std::vector<SignatureTemplate> signatureTemplates_;

  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::unique_ptr<memory::MemoryPool> pool_{
      memory::getDefaultScopedMemoryPool()};
  VectorFuzzer vectorFuzzer_;

  Stats stats_;
};
} // namespace

void aggregateFuzzer(
    AggregateFunctionSignatureMap signatureMap,
    size_t seed,
    const std::unordered_set<std::string>& orderDependentFunctions) {
  AggregationFuzzer(std::move(signatureMap), seed, orderDependentFunctions)
      .go();
}

namespace {

std::string toPercentageString(size_t n, size_t total) {
  return fmt::format("{:.2f}%", (double)n / total * 100);
}

void printStats(
    size_t numFunctions,
    size_t numSignatures,
    size_t numSupportedFunctions,
    size_t numSupportedSignatures) {
  LOG(INFO) << fmt::format(
      "Total functions: {} ({} signatures)", numFunctions, numSignatures);
  LOG(INFO) << fmt::format(
      "Functions with at least one supported signature: {} ({})",
      numSupportedFunctions,
      toPercentageString(numSupportedFunctions, numFunctions));

  size_t numNotSupportedFunctions = numFunctions - numSupportedFunctions;
  LOG(INFO) << fmt::format(
      "Functions with no supported signature: {} ({})",
      numNotSupportedFunctions,
      toPercentageString(numNotSupportedFunctions, numFunctions));
  LOG(INFO) << fmt::format(
      "Supported function signatures: {} ({})",
      numSupportedSignatures,
      toPercentageString(numSupportedSignatures, numSignatures));

  size_t numNotSupportedSignatures = numSignatures - numSupportedSignatures;
  LOG(INFO) << fmt::format(
      "Unsupported function signatures: {} ({})",
      numNotSupportedSignatures,
      toPercentageString(numNotSupportedSignatures, numSignatures));
}

std::unordered_set<std::string> getDuckDbFunctions() {
  std::string sql =
      "SELECT distinct on(function_name) function_name "
      "FROM duckdb_functions() "
      "WHERE function_type = 'aggregate'";

  DuckDbQueryRunner queryRunner;
  auto result = queryRunner.executeOrdered(sql, ROW({VARCHAR()}));

  std::unordered_set<std::string> names;
  for (const auto& row : result) {
    names.insert(row[0].value<std::string>());
  }

  return names;
}

AggregationFuzzer::AggregationFuzzer(
    AggregateFunctionSignatureMap signatureMap,
    size_t initialSeed,
    const std::unordered_set<std::string>& orderDependentFunctions)
    : orderDependentFunctions_{orderDependentFunctions},
      persistAndRunOnce_{FLAGS_persist_and_run_once},
      reproPersistPath_{FLAGS_repro_persist_path},
      vectorFuzzer_{getFuzzerOptions(), pool_.get()} {
  seed(initialSeed);
  VELOX_CHECK(!signatureMap.empty(), "No function signatures available.");

  if (persistAndRunOnce_ && reproPersistPath_.empty()) {
    std::cout
        << "--repro_persist_path must be specified if --persist_and_run_once is specified"
        << std::endl;
    exit(1);
  }

  duckDbFunctionNames_ = getDuckDbFunctions();

  size_t numFunctions = 0;
  size_t numSignatures = 0;
  size_t numSupportedFunctions = 0;
  size_t numSupportedSignatures = 0;

  for (auto& [name, signatures] : signatureMap) {
    ++numFunctions;
    bool hasSupportedSignature = false;
    for (auto& signature : signatures) {
      ++numSignatures;

      if (signature->variableArity()) {
        LOG(WARNING) << "Skipping variadic function signature: " << name
                     << signature->toString();
        continue;
      }

      if (!signature->typeVariableConstraints().empty()) {
        bool skip = false;
        std::unordered_set<std::string> typeVariables;
        for (auto& constraint : signature->typeVariableConstraints()) {
          if (constraint.isIntegerParameter()) {
            LOG(WARNING) << "Skipping generic function signature: " << name
                         << signature->toString();
            skip = true;
            break;
          }

          typeVariables.insert(constraint.name());
        }
        if (skip) {
          continue;
        }

        signatureTemplates_.push_back(
            {name, signature.get(), std::move(typeVariables)});
      } else {
        CallableSignature callable{
            .name = name,
            .args = {},
            .returnType =
                SignatureBinder::tryResolveType(signature->returnType(), {})};
        VELOX_CHECK_NOT_NULL(callable.returnType);

        // Process each argument and figure out its type.
        for (const auto& arg : signature->argumentTypes()) {
          auto resolvedType = SignatureBinder::tryResolveType(arg, {});
          VELOX_CHECK_NOT_NULL(resolvedType);

          callable.args.emplace_back(resolvedType);
        }

        signatures_.emplace_back(callable);
      }

      ++numSupportedSignatures;
      hasSupportedSignature = true;
    }
    if (hasSupportedSignature) {
      ++numSupportedFunctions;
    }
  }

  printStats(
      numFunctions,
      numSignatures,
      numSupportedFunctions,
      numSupportedSignatures);

  sortCallableSignatures(signatures_);
  sortSignatureTemplates(signatureTemplates_);
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

std::string makeFunctionCall(
    const std::string& name,
    const std::vector<std::string>& argNames) {
  std::ostringstream call;
  call << name << "(";
  for (auto i = 0; i < argNames.size(); ++i) {
    if (i > 0) {
      call << ", ";
    }
    call << argNames[i];
  }
  call << ")";

  return call.str();
}

std::vector<std::string> makeNames(size_t n) {
  std::vector<std::string> names;
  for (auto i = 0; i < n; ++i) {
    names.push_back(fmt::format("c{}", i));
  }
  return names;
}

void persistReproInfo(
    const std::vector<RowVectorPtr>& input,
    const core::PlanNodePtr& plan,
    const std::string& basePath) {
  std::string inputPath;

  if (!common::generateFileDirectory(basePath.c_str())) {
    return;
  }

  // Save input vector.
  auto inputPathOpt = generateFilePath(basePath.c_str(), "vector");
  if (!inputPathOpt.has_value()) {
    inputPath = "Failed to create file for saving input vector.";
  } else {
    inputPath = inputPathOpt.value();
    try {
      // TODO Save all input vectors.
      saveVectorToFile(input.front().get(), inputPath.c_str());
    } catch (std::exception& e) {
      inputPath = e.what();
    }
  }

  // Save plan.
  std::string planPath;
  auto planPathOpt = generateFilePath(basePath.c_str(), "plan");
  if (!planPathOpt.has_value()) {
    planPath = "Failed to create file for saving SQL.";
  } else {
    planPath = planPathOpt.value();
    try {
      saveStringToFile(
          plan->toString(true /*detailed*/, true /*recursive*/),
          planPath.c_str());
    } catch (std::exception& e) {
      planPath = e.what();
    }
  }

  LOG(INFO) << "Persisted input: " << inputPath << " and plan: " << planPath;
}

CallableSignature AggregationFuzzer::pickSignature() {
  size_t idx = boost::random::uniform_int_distribution<uint32_t>(
      0, signatures_.size() + signatureTemplates_.size() - 1)(rng_);
  CallableSignature signature;
  if (idx < signatures_.size()) {
    signature = signatures_[idx];
  } else {
    const auto& signatureTemplate =
        signatureTemplates_[idx - signatures_.size()];
    signature.name = signatureTemplate.name;
    velox::test::ArgumentTypeFuzzer typeFuzzer(
        *signatureTemplate.signature, rng_);
    VELOX_CHECK(typeFuzzer.fuzzArgumentTypes(FLAGS_max_num_varargs));
    signature.args = typeFuzzer.argumentTypes();
  }

  return signature;
}

void AggregationFuzzer::go() {
  VELOX_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.")

  auto startTime = std::chrono::system_clock::now();
  size_t iteration = 0;

  while (!isDone(iteration, startTime)) {
    LOG(INFO) << "==============================> Started iteration "
              << iteration << " (seed: " << currentSeed_ << ")";

    // Pick a random signature.
    CallableSignature signature = pickSignature();
    stats_.functionNames.insert(signature.name);

    const bool orderDependent =
        orderDependentFunctions_.count(signature.name) != 0;

    std::vector<TypePtr> argTypes = signature.args;
    std::vector<std::string> argNames = makeNames(argTypes.size());
    auto call = makeFunctionCall(signature.name, argNames);

    // 20% of times use mask.
    std::vector<std::string> masks;
    if (vectorFuzzer_.coinToss(0.2)) {
      ++stats_.numMask;

      masks.push_back("m0");
      argTypes.push_back(BOOLEAN());
      argNames.push_back(masks.back());
    }

    // 10% of times use global aggregation (no grouping keys).
    std::vector<std::string> groupingKeys;
    if (vectorFuzzer_.coinToss(0.1)) {
      ++stats_.numGlobal;
    } else {
      ++stats_.numGroupBy;

      auto numGroupingKeys =
          boost::random::uniform_int_distribution<uint32_t>(1, 5)(rng_);
      for (auto i = 0; i < numGroupingKeys; ++i) {
        groupingKeys.push_back(fmt::format("g{}", i));

        // Pick random scalar type.
        argTypes.push_back(vectorFuzzer_.randType(0 /*maxDepth*/));
        argNames.push_back(groupingKeys.back());
      }
    }

    // Generate random input data.
    auto inputType = ROW(std::move(argNames), std::move(argTypes));
    std::vector<RowVectorPtr> input;
    for (auto i = 0; i < 10; ++i) {
      input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
    }

    verify(groupingKeys, {call}, masks, input, orderDependent);

    LOG(INFO) << "==============================> Done with iteration "
              << iteration;

    if (persistAndRunOnce_) {
      LOG(WARNING)
          << "Iteration succeeded with --persist_and_run_once flag enabled "
             "(expecting crash failure)";
      exit(0);
    }

    reSeed();
    ++iteration;
  }

  stats_.print(iteration);
}

ResultOrError AggregationFuzzer::execute(const core::PlanNodePtr& plan) {
  LOG(INFO) << "Executing query plan: " << std::endl
            << plan->toString(true, true);

  ResultOrError resultOrError;
  try {
    resultOrError.result =
        AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool_.get());
    LOG(INFO) << resultOrError.result->toString();
  } catch (VeloxUserError& e) {
    // NOTE: velox user exception is accepted as it is caused by the invalid
    // fuzzer test inputs.
    resultOrError.exceptionPtr = std::current_exception();
  }

  return resultOrError;
}

// Generate SELECT <keys>, <aggregates> FROM tmp GROUP BY <keys>.
std::string makeDuckDbSql(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks) {
  std::stringstream sql;
  sql << "SELECT " << folly::join(", ", groupingKeys);

  if (!groupingKeys.empty()) {
    sql << ", ";
  }

  for (auto i = 0; i < aggregates.size(); ++i) {
    if (i > 0) {
      sql << ", ";
    }
    sql << aggregates[i];
    if (masks.size() > i && !masks[i].empty()) {
      sql << " filter (where " << masks[i] << ")";
    }
  }

  sql << " FROM tmp";

  if (!groupingKeys.empty()) {
    sql << " GROUP BY " << folly::join(", ", groupingKeys);
  }

  return sql.str();
}

std::optional<MaterializedRowMultiset> AggregationFuzzer::computeDuckDbResult(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& input,
    const core::PlanNodePtr& plan) {
  // Check if DuckDB supports specified aggregate functions.
  for (const auto& agg :
       dynamic_cast<const core::AggregationNode*>(plan.get())->aggregates()) {
    if (duckDbFunctionNames_.count(agg->name()) == 0) {
      return std::nullopt;
    }
  }

  const auto& outputType = plan->outputType();

  // Skip queries that use Timestamp type.
  // DuckDB doesn't support nanosecond precision for timestamps.
  for (auto i = 0; i < input[0]->type()->size(); ++i) {
    if (input[0]->type()->childAt(i)->isTimestamp()) {
      return std::nullopt;
    }
  }

  for (auto i = 0; i < outputType->size(); ++i) {
    if (outputType->childAt(i)->isTimestamp()) {
      return std::nullopt;
    }
  }

  DuckDbQueryRunner queryRunner;
  queryRunner.createTable("tmp", {input});
  return queryRunner.execute(
      makeDuckDbSql(groupingKeys, aggregates, masks), outputType);
}

std::vector<core::PlanNodePtr> makeAlternativePlans(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& inputVectors) {
  std::vector<core::PlanNodePtr> plans;

  // Partial -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .values(inputVectors)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .finalAggregation()
                      .planNode());

  // Partial -> intermediate -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .values(inputVectors)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .intermediateAggregation()
                      .finalAggregation()
                      .planNode());

  // Partial -> local exchange -> final aggregation plan.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  for (const auto& vector : inputVectors) {
    sources.push_back(PlanBuilder(planNodeIdGenerator)
                          .values({vector})
                          .partialAggregation(groupingKeys, aggregates, masks)
                          .planNode());
  }
  plans.push_back(PlanBuilder(planNodeIdGenerator)
                      .localPartition(groupingKeys, sources)
                      .finalAggregation()
                      .planNode());

  return plans;
}

void AggregationFuzzer::verify(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& input,
    bool orderDependent) {
  auto plan = PlanBuilder()
                  .values(input)
                  .singleAggregation(groupingKeys, aggregates, masks)
                  .planNode();

  if (persistAndRunOnce_) {
    persistReproInfo(input, plan, reproPersistPath_);
  }

  try {
    auto resultOrError = execute(plan);
    if (resultOrError.exceptionPtr) {
      ++stats_.numFailed;
    }

    std::optional<MaterializedRowMultiset> expectedResult;
    try {
      if (!orderDependent) {
        expectedResult =
            computeDuckDbResult(groupingKeys, aggregates, masks, input, plan);
        ++stats_.numDuckDbVerified;
      }
    } catch (std::exception& e) {
      LOG(WARNING) << "Couldn't get results from DuckDB";
    }

    if (expectedResult && resultOrError.result) {
      VELOX_CHECK(
          assertEqualResults(expectedResult.value(), {resultOrError.result}),
          "Velox and DuckDB results don't match");
    }

    auto altPlans =
        makeAlternativePlans(groupingKeys, aggregates, masks, input);

    for (const auto& altPlan : altPlans) {
      auto altResultOrError = execute(altPlan);

      // Compare results or exceptions (if any). Fail is anything is different.
      if (resultOrError.exceptionPtr || altResultOrError.exceptionPtr) {
        // Throws in case exceptions are not compatible.
        velox::test::compareExceptions(
            resultOrError.exceptionPtr, altResultOrError.exceptionPtr);
      } else if (!orderDependent) {
        VELOX_CHECK(
            assertEqualResults(
                {resultOrError.result}, {altResultOrError.result}),
            "Logically equivalent plans produced different results");
      }
    }
  } catch (...) {
    if (!reproPersistPath_.empty()) {
      persistReproInfo(input, plan, reproPersistPath_);
    }
    throw;
  }
}

void AggregationFuzzer::Stats::print(size_t numIterations) const {
  LOG(INFO) << "Total functions tested: " << functionNames.size();
  LOG(INFO) << "Total masked aggregations: " << numMask << " ("
            << toPercentageString(numMask, numIterations) << ")";
  LOG(INFO) << "Total global aggregations: " << numGlobal << " ("
            << toPercentageString(numGlobal, numIterations) << ")";
  LOG(INFO) << "Total group-by aggregations: " << numGroupBy << " ("
            << toPercentageString(numGroupBy, numIterations) << ")";
  LOG(INFO) << "Total aggregations verified against DuckDB: "
            << numDuckDbVerified << " ("
            << toPercentageString(numDuckDbVerified, numIterations) << ")";
  LOG(INFO) << "Total failed aggregations: " << numFailed << " ("
            << toPercentageString(numFailed, numIterations) << ")";
}

} // namespace
} // namespace facebook::velox::exec::test

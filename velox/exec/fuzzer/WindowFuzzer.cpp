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

#include "velox/exec/fuzzer/WindowFuzzer.h"

#include <boost/random/uniform_int_distribution.hpp>
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

DEFINE_bool(
    enable_window_reference_verification,
    false,
    "When true, the results of the window aggregation are compared to reference DB results");

namespace facebook::velox::exec::test {

namespace {

void logVectors(const std::vector<RowVectorPtr>& vectors) {
  for (auto i = 0; i < vectors.size(); ++i) {
    VLOG(1) << "Input batch " << i << ":";
    for (auto j = 0; j < vectors[i]->size(); ++j) {
      VLOG(1) << "\tRow " << j << ": " << vectors[i]->toString(j);
    }
  }
}

bool supportIgnoreNulls(const std::string& name) {
  static std::unordered_set<std::string> supportFunctions{
      "first_value",
      "last_value",
      "nth_value",
      "lead",
      "lag",
  };
  return supportFunctions.count(name) > 0;
}

} // namespace

void WindowFuzzer::addWindowFunctionSignatures(
    const WindowFunctionMap& signatureMap) {
  for (const auto& [name, entry] : signatureMap) {
    ++functionsStats.numFunctions;
    bool hasSupportedSignature = false;
    for (auto& signature : entry.signatures) {
      hasSupportedSignature |= addSignature(name, signature);
    }
    if (hasSupportedSignature) {
      ++functionsStats.numSupportedFunctions;
    }
  }
}

void WindowFuzzer::go() {
  VELOX_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.")

  auto startTime = std::chrono::system_clock::now();
  size_t iteration = 0;

  while (!isDone(iteration, startTime)) {
    LOG(INFO) << "==============================> Started iteration "
              << iteration << " (seed: " << currentSeed_ << ")";

    auto signatureWithStats = pickSignature();
    signatureWithStats.second.numRuns++;

    auto signature = signatureWithStats.first;
    stats_.functionNames.insert(signature.name);

    const bool customVerification =
        customVerificationFunctions_.count(signature.name) != 0;
    const bool requireSortedInput =
        orderDependentFunctions_.count(signature.name) != 0;

    std::vector<TypePtr> argTypes = signature.args;
    std::vector<std::string> argNames = makeNames(argTypes.size());

    bool ignoreNulls =
        supportIgnoreNulls(signature.name) && vectorFuzzer_.coinToss(0.5);
    auto call = makeFunctionCall(signature.name, argNames, false, ignoreNulls);

    // std::vector<std::string> sortingKeys;
    std::vector<SortingKeyAndOrder> sortingKeysAndOrders;
    // 50% chance without order-by clause.
    if (vectorFuzzer_.coinToss(0.5)) {
      // sortingKeys = generateSortingKeys("s", argNames, argTypes);
      sortingKeysAndOrders = generateSortingKeysAndOrders("s", argNames, argTypes);
    }
    auto partitionKeys = generateSortingKeys("p", argNames, argTypes);
    auto frameClause = generateFrameClause();
    auto input = generateInputDataWithRowNumber(
        argNames, argTypes, partitionKeys, signature);
    // auto input = generateInputDataWithRowNumber(argNames, argTypes, signature);
    // If the function is order-dependent, sort all input rows by row_number
    // additionally.
    if (requireSortedInput) {
      // sortingKeys.push_back("row_number");
      sortingKeysAndOrders.push_back(SortingKeyAndOrder("row_number", "asc", "nulls last"));
      ++stats_.numSortedInputs;
    }

    logVectors(input);

    bool failed = verifyWindow(
        partitionKeys,
        // sortingKeys,
        sortingKeysAndOrders,
        frameClause,
        call,
        input,
        customVerification,
        FLAGS_enable_window_reference_verification);
    if (failed) {
      signatureWithStats.second.numFailed++;
    }

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
  printSignatureStats();
}

void WindowFuzzer::go(const std::string& /*planPath*/) {
  // TODO: allow running window fuzzer with saved plans and splits.
  VELOX_NYI();
}

void WindowFuzzer::updateReferenceQueryStats(
    AggregationFuzzerBase::ReferenceQueryErrorCode ec) {
  if (ec == ReferenceQueryErrorCode::kReferenceQueryFail) {
    ++stats_.numReferenceQueryFailed;
  } else if (ec == ReferenceQueryErrorCode::kReferenceQueryUnsupported) {
    ++stats_.numVerificationNotSupported;
  }
}

const std::string WindowFuzzer::generateFrameClause() {
  auto frameType = [](int value) -> const std::string {
    switch (value) {
      case 0:
        return "ROWS";
      case 1:
        return "RANGE";
      default:
        VELOX_UNREACHABLE("Unknown value for frame bound generation");
    }
  };
  auto frameTypeValue =
      boost::random::uniform_int_distribution<uint32_t>(0, 1)(rng_);
  auto frameTypeString = frameType(frameTypeValue);

  auto frameBound = [&](int value, bool start) -> const std::string {
    // Generating only constant bounded k PRECEDING/FOLLOWING frames for now.
    auto kValue =
        boost::random::uniform_int_distribution<uint32_t>(1, 10)(rng_);
    switch (value) {
      case 0:
        return "CURRENT ROW";
      case 1:
        return start ? "UNBOUNDED PRECEDING" : "UNBOUNDED FOLLOWING";
      case 2:
        return fmt::format("{} FOLLOWING", kValue);
      case 3:
        return fmt::format("{} PRECEDING", kValue);
      default:
        VELOX_UNREACHABLE("Unknown option for frame clause generation");
    }
  };

  // Generating k PRECEDING and k FOLLOWING frames only for ROWS type.
  // k RANGE frames require more work as we have to generate columns with the
  // frame bound values.
  auto startValue = boost::random::uniform_int_distribution<uint32_t>(
      0, frameTypeValue == 0 ? 3 : 1)(rng_);
  auto startBound = frameBound(startValue, true);
  // Frame end has additional limitation that if the frameStart is k FOLLOWING
  // or CURRENT ROW then the frameEnd cannot be k PRECEDING or CURRENT ROW.
  auto startLimit =
      frameTypeValue == 0 ? ((startValue == 2) | (startValue == 0)) ? 1 : 0 : 0;
  auto endLimit =
      frameTypeValue == 0 ? ((startValue == 2) | (startValue == 0)) ? 2 : 3 : 1;
  auto endBound = frameBound(
      boost::random::uniform_int_distribution<uint32_t>(
          startLimit, endLimit)(rng_),
      false);

  return frameTypeString + " BETWEEN " + startBound + " AND " + endBound;
}

const std::string WindowFuzzer::generateOrderByClause(
    // const std::vector<std::string>& sortingKeys
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders) {
  VELOX_CHECK(!sortingKeysAndOrders.empty());
  std::stringstream frame;
  frame << " order by ";
  for (auto i = 0; i < sortingKeysAndOrders.size(); ++i) {
    if (i != 0) {
      frame << ", ";
    }
    frame << sortingKeysAndOrders[i].key_ << " "
          << sortingKeysAndOrders[i].order_ << " "
          << sortingKeysAndOrders[i].nullsOrder_;
    /*auto asc = boost::random::uniform_int_distribution<uint32_t>(0, 1)(rng_);
    if (asc == 0) {
      frame << "asc ";
    } else {
      frame << "desc ";
    }
    auto nullsFirst =
        boost::random::uniform_int_distribution<uint32_t>(0, 1)(rng_);
    if (nullsFirst == 0) {
      frame << "nulls first";
    } else {
      frame << "nulls last";
    }*/
  }
  return frame.str();
}

std::string WindowFuzzer::getFrame(
    const std::vector<std::string>& partitionKeys,
    // const std::vector<std::string>& sortingKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frameClause) {
  // TODO: allow randomly generated frames.
  std::stringstream frame;
  VELOX_CHECK(!partitionKeys.empty());
  frame << "partition by " << folly::join(", ", partitionKeys);
  if (!sortingKeysAndOrders.empty()) {
    // frame << " order by " << folly::join(", ", sortingKeys);
    frame << generateOrderByClause(sortingKeysAndOrders);
  }
  frame << " " << frameClause;
  return frame.str();
}

std::vector<RowVectorPtr> WindowFuzzer::generateInputDataWithRowNumber(
    std::vector<std::string>& names,
    std::vector<TypePtr>& types,
    const std::vector<std::string>& partitionKeys,
    const CallableSignature& signature) {
  names.push_back("row_number");
  types.push_back(BIGINT());

  std::unordered_set<std::string> partitionSet;
  partitionSet.reserve(partitionKeys.size());
  for (const auto& key : partitionKeys) {
    partitionSet.insert(key);
  }

  auto generator = findInputGenerator(signature);

  std::vector<RowVectorPtr> input;
  auto size = vectorFuzzer_.getOptions().vectorSize;
  velox::test::VectorMaker vectorMaker{pool_.get()};
  int64_t rowNumber = 0;

  auto partitionNumRows =
      boost::random::uniform_int_distribution<uint32_t>(1, 5)(rng_) * size / 10;
  partitionNumRows = partitionNumRows > 0 ? partitionNumRows : 1;
  for (auto j = 0; j < FLAGS_num_batches; ++j) {
    std::vector<VectorPtr> children;

    if (generator != nullptr) {
      children =
          generator->generate(signature.args, vectorFuzzer_, rng_, pool_.get());
    }

    for (auto i = children.size(); i < types.size() - 1; ++i) {
      if (partitionSet.count(names[i]) != 0) {
        // The partition keys are built with a dictionary over a smaller set of
        // values. This is done to introduce some repetition of key values for
        // windowing.
        auto baseVector = vectorFuzzer_.fuzz(types[i], partitionNumRows);
        children.push_back(vectorFuzzer_.fuzzDictionary(baseVector, size));
      } else {
        children.push_back(vectorFuzzer_.fuzz(types[i], size));
      }
    }
    children.push_back(vectorMaker.flatVector<int64_t>(
        size, [&](auto /*row*/) { return rowNumber++; }));
    input.push_back(vectorMaker.rowVector(names, children));
  }

  if (generator != nullptr) {
    generator->reset();
  }

  return input;
}

void WindowFuzzer::testAlternativePlans(
    const std::vector<std::string>& partitionKeys,
    // const std::vector<std::string>& sortingKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frame,
    const std::string& functionCall,
    const std::vector<RowVectorPtr>& input,
    bool customVerification,
    const velox::test::ResultOrError& expected) {
  std::vector<AggregationFuzzerBase::PlanWithSplits> plans;

  std::vector<std::string> allKeys;
  for (const auto& key : partitionKeys) {
    allKeys.push_back(key + " NULLS FIRST");
  }
  for (const auto& keyAndOrder : sortingKeysAndOrders) {
    allKeys.push_back(folly::to<std::string>(
        keyAndOrder.key_, " ", keyAndOrder.order_, " ", keyAndOrder.nullsOrder_));
  }
  // allKeys.insert(allKeys.end(), sortingKeys.begin(), sortingKeys.end());

  // Streaming window from values.
  if (!allKeys.empty()) {
    plans.push_back(
        {PlanBuilder()
             .values(input)
             .orderBy(allKeys, false)
             .streamingWindow(
                 {fmt::format("{} over ({})", functionCall, frame)})
             .planNode(),
         {}});
  }

  // With TableScan.
  auto directory = exec::test::TempDirectoryPath::create();
  const auto inputRowType = asRowType(input[0]->type());
  if (isTableScanSupported(inputRowType)) {
    auto splits = makeSplits(input, directory->path);

    plans.push_back(
        {PlanBuilder()
             .tableScan(inputRowType)
             .localPartition(partitionKeys)
             .window({fmt::format("{} over ({})", functionCall, frame)})
             .planNode(),
         splits});

    if (!allKeys.empty()) {
      plans.push_back(
          {PlanBuilder()
               .tableScan(inputRowType)
               .orderBy(allKeys, false)
               .streamingWindow(
                   {fmt::format("{} over ({})", functionCall, frame)})
               .planNode(),
           splits});
    }
  }

  for (const auto& plan : plans) {
    testPlan(
        plan,
        false,
        false,
        customVerification,
        /*customVerifiers*/ {},
        expected);
  }
}

bool WindowFuzzer::verifyWindow(
    const std::vector<std::string>& partitionKeys,
    // const std::vector<std::string>& sortingKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frameClause,
    const std::string& functionCall,
    const std::vector<RowVectorPtr>& input,
    bool customVerification,
    bool enableWindowVerification) {
  auto frame = getFrame(partitionKeys, sortingKeysAndOrders, frameClause);
  auto plan = PlanBuilder()
                  .values(input)
                  .window({fmt::format("{} over ({})", functionCall, frame)})
                  .planNode();
  if (persistAndRunOnce_) {
    persistReproInfo({{plan, {}}}, reproPersistPath_);
  }

  velox::test::ResultOrError resultOrError;
  try {
    resultOrError = execute(plan);
    if (resultOrError.exceptionPtr) {
      ++stats_.numFailed;
    }

    if (!customVerification && enableWindowVerification) {
      if (resultOrError.result) {
        auto referenceResult = computeReferenceResults(plan, input);
        updateReferenceQueryStats(referenceResult.second);
        if (auto expectedResult = referenceResult.first) {
          ++stats_.numVerified;
          VELOX_CHECK(
              assertEqualResults(
                  expectedResult.value(),
                  plan->outputType(),
                  {resultOrError.result}),
              "Velox and reference DB results don't match");
          LOG(INFO) << "Verified results against reference DB";
        }
      }
    } else {
      // TODO: support custom verification.
      LOG(INFO) << "Verification skipped";
      ++stats_.numVerificationSkipped;
    }

    testAlternativePlans(
        partitionKeys,
        sortingKeysAndOrders,
        frame,
        functionCall,
        input,
        customVerification,
        resultOrError);

    if (resultOrError.exceptionPtr != nullptr) {
      return true;
    }
  } catch (...) {
    if (!reproPersistPath_.empty()) {
      persistReproInfo({{plan, {}}}, reproPersistPath_);
    }
    throw;
  }
  return false;
}

void windowFuzzer(
    AggregateFunctionSignatureMap aggregationSignatureMap,
    WindowFunctionMap windowSignatureMap,
    size_t seed,
    const std::unordered_map<std::string, std::shared_ptr<ResultVerifier>>&
        customVerificationFunctions,
    const std::unordered_map<std::string, std::shared_ptr<InputGenerator>>&
        customInputGenerators,
    const std::unordered_set<std::string>& orderDependentFunctions,
    VectorFuzzer::Options::TimestampPrecision timestampPrecision,
    const std::unordered_map<std::string, std::string>& queryConfigs,
    const std::optional<std::string>& planPath,
    std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner) {
  auto windowFuzzer = WindowFuzzer(
      std::move(aggregationSignatureMap),
      std::move(windowSignatureMap),
      seed,
      customVerificationFunctions,
      customInputGenerators,
      orderDependentFunctions,
      timestampPrecision,
      queryConfigs,
      std::move(referenceQueryRunner));
  planPath.has_value() ? windowFuzzer.go(planPath.value()) : windowFuzzer.go();
}

void WindowFuzzer::Stats::print(size_t numIterations) const {
  LOG(INFO) << "Total functions tested: " << functionNames.size();
  LOG(INFO) << "Total functions requiring sorted inputs: "
            << printPercentageStat(numSortedInputs, numIterations);
  LOG(INFO) << "Total functions verified against reference DB: "
            << printPercentageStat(numVerified, numIterations);
  LOG(INFO)
      << "Total functions not verified (verification skipped / not supported by reference DB / reference DB failed): "
      << printPercentageStat(numVerificationSkipped, numIterations) << " / "
      << printPercentageStat(numVerificationNotSupported, numIterations)
      << " / " << printPercentageStat(numReferenceQueryFailed, numIterations);
  LOG(INFO) << "Total failed functions: "
            << printPercentageStat(numFailed, numIterations);
}

} // namespace facebook::velox::exec::test

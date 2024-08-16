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
#include "velox/common/base/Portability.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

DEFINE_bool(
    enable_window_reference_verification,
    false,
    "When true, the results of the window aggregation are compared to reference DB results");

namespace facebook::velox::exec::test {

namespace {

bool supportIgnoreNulls(const std::string& name) {
  // Below are all functions that support ignore nulls. Aggregation functions in
  // window operations do not support ignore nulls.
  // https://github.com/prestodb/presto/issues/21304.
  static std::unordered_set<std::string> supportedFunctions{
      "first_value",
      "last_value",
      "nth_value",
      "lead",
      "lag",
  };
  return supportedFunctions.count(name) > 0;
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

std::string WindowFuzzer::generateFrameClause(
    std::vector<std::string>& argNames,
    std::vector<TypePtr>& argTypes,
    bool& isRowsFrame) {
  auto frameType = [](int value) -> const std::string {
    switch (value) {
      case 0:
        return "RANGE";
      case 1:
        return "ROWS";
      default:
        VELOX_UNREACHABLE("Unknown value for frame type generation");
    }
  };
  isRowsFrame = boost::random::uniform_int_distribution<uint32_t>(0, 1)(rng_);
  auto frameTypeString = frameType(isRowsFrame);

  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  // For frames with kPreceding, kFollowing bounds, pick a valid k, in the range
  // of 1 to 10, 70% of times. Test for random k values remaining times.
  int64_t minKValue, maxKValue;
  if (vectorFuzzer_.coinToss(0.7)) {
    minKValue = 1;
    maxKValue = 10;
  } else {
    minKValue = kMin;
    maxKValue = kMax;
  }

  // Generating column bounded frames 50% of the time and constant
  // bounded the rest of the times. The column bounded frames are of
  // type INTEGER and only have positive values as Window functions
  // reject negative values for them.
  auto frameBound = [this, minKValue, maxKValue, &argNames, &argTypes](
                        core::WindowNode::BoundType boundType,
                        const std::string& colName) -> const std::string {
    auto kValue = boost::random::uniform_int_distribution<int64_t>(
        minKValue, maxKValue)(rng_);
    switch (boundType) {
      case core::WindowNode::BoundType::kUnboundedPreceding:
        return "UNBOUNDED PRECEDING";
      case core::WindowNode::BoundType::kPreceding:
        if (vectorFuzzer_.coinToss(0.5)) {
          argTypes.push_back(INTEGER());
          argNames.push_back(colName);
          return fmt::format("{} PRECEDING", colName);
        }
        return fmt::format("{} PRECEDING", kValue);
      case core::WindowNode::BoundType::kCurrentRow:
        return "CURRENT ROW";
      case core::WindowNode::BoundType::kFollowing:
        if (vectorFuzzer_.coinToss(0.5)) {
          argTypes.push_back(INTEGER());
          argNames.push_back(colName);
          return fmt::format("{} FOLLOWING", colName);
        }
        return fmt::format("{} FOLLOWING", kValue);
      case core::WindowNode::BoundType::kUnboundedFollowing:
        return "UNBOUNDED FOLLOWING";
      default:
        VELOX_UNREACHABLE("Unknown option for frame clause generation");
    }
  };

  // Generating k PRECEDING and k FOLLOWING frames only for ROWS type.
  // k RANGE frames require more work as we have to generate columns with the
  // frame bound values.
  std::vector<core::WindowNode::BoundType> startBoundOptions, endBoundOptions;
  if (isRowsFrame) {
    startBoundOptions = {
        core::WindowNode::BoundType::kUnboundedPreceding,
        core::WindowNode::BoundType::kPreceding,
        core::WindowNode::BoundType::kCurrentRow,
        core::WindowNode::BoundType::kFollowing};
    endBoundOptions = {
        core::WindowNode::BoundType::kPreceding,
        core::WindowNode::BoundType::kCurrentRow,
        core::WindowNode::BoundType::kFollowing,
        core::WindowNode::BoundType::kUnboundedFollowing};
  } else {
    startBoundOptions = {
        core::WindowNode::BoundType::kUnboundedPreceding,
        core::WindowNode::BoundType::kCurrentRow};
    endBoundOptions = {
        core::WindowNode::BoundType::kCurrentRow,
        core::WindowNode::BoundType::kUnboundedFollowing};
  }

  // End bound option should not be greater than start bound option as this
  // would result in an invalid frame.
  auto startBoundIndex = boost::random::uniform_int_distribution<uint32_t>(
      0, startBoundOptions.size() - 1)(rng_);
  auto endBoundMinIdx = std::max(0, static_cast<int>(startBoundIndex) - 1);
  auto endBoundIndex = boost::random::uniform_int_distribution<uint32_t>(
      endBoundMinIdx, endBoundOptions.size() - 1)(rng_);
  auto frameStartBound = frameBound(startBoundOptions[startBoundIndex], "k0");
  auto frameEndBound = frameBound(endBoundOptions[endBoundIndex], "k1");

  return frameTypeString + " BETWEEN " + frameStartBound + " AND " +
      frameEndBound;
}

std::string WindowFuzzer::generateOrderByClause(
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders) {
  std::stringstream frame;
  frame << " order by ";
  for (auto i = 0; i < sortingKeysAndOrders.size(); ++i) {
    if (i != 0) {
      frame << ", ";
    }
    frame << sortingKeysAndOrders[i].key_ << " "
          << sortingKeysAndOrders[i].sortOrder_.toString();
  }
  return frame.str();
}

std::string WindowFuzzer::getFrame(
    const std::vector<std::string>& partitionKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frameClause) {
  std::stringstream frame;
  VELOX_CHECK(!partitionKeys.empty());
  frame << "partition by " << folly::join(", ", partitionKeys);
  if (!sortingKeysAndOrders.empty()) {
    frame << generateOrderByClause(sortingKeysAndOrders);
  }
  frame << " " << frameClause;
  return frame.str();
}

std::vector<SortingKeyAndOrder> WindowFuzzer::generateSortingKeysAndOrders(
    const std::string& prefix,
    std::vector<std::string>& names,
    std::vector<TypePtr>& types) {
  auto keys = generateSortingKeys(prefix, names, types);
  std::vector<SortingKeyAndOrder> results;
  for (auto i = 0; i < keys.size(); ++i) {
    auto asc = vectorFuzzer_.coinToss(0.5);
    auto nullsFirst = vectorFuzzer_.coinToss(0.5);
    results.emplace_back(keys[i], core::SortOrder(asc, nullsFirst));
  }
  return results;
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

    const auto signature = signatureWithStats.first;
    stats_.functionNames.insert(signature.name);

    const bool customVerification =
        customVerificationFunctions_.count(signature.name) != 0;
    std::shared_ptr<ResultVerifier> customVerifier = nullptr;
    if (customVerification) {
      customVerifier = customVerificationFunctions_.at(signature.name);
    }
    const bool requireSortedInput =
        orderDependentFunctions_.count(signature.name) != 0;

    std::vector<TypePtr> argTypes = signature.args;
    std::vector<std::string> argNames = makeNames(argTypes.size());

    const bool ignoreNulls =
        supportIgnoreNulls(signature.name) && vectorFuzzer_.coinToss(0.5);
    const auto call =
        makeFunctionCall(signature.name, argNames, false, false, ignoreNulls);

    std::vector<SortingKeyAndOrder> sortingKeysAndOrders;
    // 50% chance without order-by clause.
    if (vectorFuzzer_.coinToss(0.5)) {
      sortingKeysAndOrders =
          generateSortingKeysAndOrders("s", argNames, argTypes);
    }
    const auto partitionKeys = generateSortingKeys("p", argNames, argTypes);
    bool isRowsFrame = false;
    const auto frameClause =
        generateFrameClause(argNames, argTypes, isRowsFrame);
    const auto input = generateInputDataWithRowNumber(
        argNames, argTypes, partitionKeys, signature);
    // If the function is order-dependent or uses "rows" frame, sort all input
    // rows by row_number additionally.
    if (requireSortedInput || isRowsFrame) {
      sortingKeysAndOrders.emplace_back("row_number", core::kAscNullsLast);
      ++stats_.numSortedInputs;
    }

    logVectors(input);

    bool failed = verifyWindow(
        partitionKeys,
        sortingKeysAndOrders,
        frameClause,
        call,
        input,
        customVerification,
        customVerifier,
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

void WindowFuzzer::testAlternativePlans(
    const std::vector<std::string>& partitionKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frame,
    const std::string& functionCall,
    const std::vector<RowVectorPtr>& input,
    bool customVerification,
    const std::shared_ptr<ResultVerifier>& customVerifier,
    const velox::fuzzer::ResultOrError& expected) {
  std::vector<AggregationFuzzerBase::PlanWithSplits> plans;

  std::vector<std::string> allKeys;
  for (const auto& key : partitionKeys) {
    allKeys.emplace_back(key + " NULLS FIRST");
  }
  for (const auto& keyAndOrder : sortingKeysAndOrders) {
    allKeys.emplace_back(fmt::format(
        "{} {}", keyAndOrder.key_, keyAndOrder.sortOrder_.toString()));
  }

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
    auto splits = makeSplits(input, directory->getPath(), writerPool_);

// There is a known issue where LocalPartition will send DictionaryVectors
// with the same underlying base Vector to multiple threads.  This triggers
// TSAN to report data races, particularly if that base Vector is from the
// TableScan and reused.  Don't run these tests when TSAN is enabled to avoid
// the false negatives.
#ifndef TSAN_BUILD
    plans.push_back(
        {PlanBuilder()
             .tableScan(inputRowType)
             .localPartition(partitionKeys)
             .window({fmt::format("{} over ({})", functionCall, frame)})
             .planNode(),
         splits});
#endif

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
        plan, false, false, customVerification, {customVerifier}, expected);
  }
}

namespace {
void initializeVerifier(
    const core::PlanNodePtr& plan,
    const std::shared_ptr<ResultVerifier>& customVerifier,
    const std::vector<RowVectorPtr>& input,
    const std::vector<std::string>& partitionKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frame) {
  const auto& windowNode =
      std::dynamic_pointer_cast<const core::WindowNode>(plan);
  customVerifier->initializeWindow(
      input,
      partitionKeys,
      sortingKeysAndOrders,
      windowNode->windowFunctions()[0],
      frame,
      "w0");
}
} // namespace

bool WindowFuzzer::verifyWindow(
    const std::vector<std::string>& partitionKeys,
    const std::vector<SortingKeyAndOrder>& sortingKeysAndOrders,
    const std::string& frameClause,
    const std::string& functionCall,
    const std::vector<RowVectorPtr>& input,
    bool customVerification,
    const std::shared_ptr<ResultVerifier>& customVerifier,
    bool enableWindowVerification) {
  SCOPE_EXIT {
    if (customVerifier) {
      customVerifier->reset();
    }
  };

  auto frame = getFrame(partitionKeys, sortingKeysAndOrders, frameClause);
  auto plan = PlanBuilder()
                  .values(input)
                  .window({fmt::format("{} over ({})", functionCall, frame)})
                  .planNode();

  if (persistAndRunOnce_) {
    persistReproInfo({{plan, {}}}, reproPersistPath_);
  }

  velox::fuzzer::ResultOrError resultOrError;
  try {
    resultOrError = execute(plan);
    if (resultOrError.exceptionPtr) {
      ++stats_.numFailed;
    }

    if (!customVerification) {
      if (resultOrError.result && enableWindowVerification) {
        auto referenceResult =
            computeReferenceResults(plan, input, referenceQueryRunner_.get());
        stats_.updateReferenceQueryStats(referenceResult.second);
        if (auto expectedResult = referenceResult.first) {
          ++stats_.numVerified;
          stats_.verifiedFunctionNames.insert(
              retrieveWindowFunctionName(plan)[0]);
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
      LOG(INFO) << "Verification through custom verifier";
      ++stats_.numVerificationSkipped;

      if (customVerifier && resultOrError.result) {
        VELOX_CHECK(
            customVerifier->supportsVerify(),
            "Window fuzzer only uses custom verify() methods.");
        initializeVerifier(
            plan,
            customVerifier,
            input,
            partitionKeys,
            sortingKeysAndOrders,
            frame);
        customVerifier->verify(resultOrError.result);
      }
    }

    testAlternativePlans(
        partitionKeys,
        sortingKeysAndOrders,
        frame,
        functionCall,
        input,
        customVerification,
        customVerifier,
        resultOrError);

    return resultOrError.exceptionPtr != nullptr;
  } catch (...) {
    if (!reproPersistPath_.empty()) {
      persistReproInfo({{plan, {}}}, reproPersistPath_);
    }
    throw;
  }
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
    bool orderableGroupKeys,
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
      orderableGroupKeys,
      std::move(referenceQueryRunner));
  planPath.has_value() ? windowFuzzer.go(planPath.value()) : windowFuzzer.go();
}

void WindowFuzzer::Stats::print(size_t numIterations) const {
  AggregationFuzzerBase::Stats::print(numIterations);
  LOG(INFO) << "Total functions verified in reference DB: "
            << verifiedFunctionNames.size();
}

} // namespace facebook::velox::exec::test

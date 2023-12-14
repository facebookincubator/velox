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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/VectorSaver.h"

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

    auto call = makeFunctionCall(signature.name, argNames, false);

    std::vector<std::string> sortingKeys;
    // 50% chance without order-by clause.
    if (vectorFuzzer_.coinToss(0.5) || requireSortedInput) {
      auto sortingKeys = generateSortingKeys("s", argNames, argTypes);
    }
    auto partitionKeys = generateKeys("p", argNames, argTypes);
    auto input = generateInputDataWithRowNumber(argNames, argTypes, signature);
    if (requireSortedInput) {
      sortingKeys.push_back("row_number");
      ++stats_.numSortedInputs;
    }

    logVectors(input);

    bool failed = verifyWindow(
        partitionKeys,
        sortingKeys,
        {call},
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

void WindowFuzzer::go(const std::string& planPath) {
  Type::registerSerDe();
  connector::hive::HiveTableHandle::registerSerDe();
  connector::hive::LocationHandle::registerSerDe();
  connector::hive::HiveColumnHandle::registerSerDe();
  connector::hive::HiveInsertTableHandle::registerSerDe();
  core::ITypedExpr::registerSerDe();
  core::PlanNode::registerSerDe();

  LOG(INFO) << "Attempting to use serialized plan at: " << planPath;
  auto planString = restoreStringFromFile(planPath.c_str());
  auto parsedPlans = folly::parseJson(planString);
  VELOX_CHECK_EQ(parsedPlans.size(), 1, "Expected exactly one plan");
  PlanWithSplits plan = deserialize(parsedPlans.at(0));

  verifyWindow(plan);
}

std::string makeFunctionCallString(const core::WindowNode::Function& call) {
  std::ostringstream result;
  result << call.functionCall->name() << "(";
  int i = 0;
  for (const auto& input : call.functionCall->inputs()) {
    if (i > 0) {
      result << ", ";
    }
    auto inputExpr =
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(input);
    VELOX_CHECK_NOT_NULL(inputExpr);
    result << inputExpr->name();
    ++i;
  }
  result << ")";
  return result.str();
}

void WindowFuzzer::verifyWindow(const PlanWithSplits& plan) {
  const auto node = dynamic_cast<const core::WindowNode*>(plan.plan.get());
  VELOX_CHECK_NOT_NULL(node);

  std::vector<std::string> partitionKeys;
  for (const auto& key : node->partitionKeys()) {
    partitionKeys.push_back(key->name());
  }

  std::vector<std::string> sortingKeys;
  for (const auto& key : node->sortingKeys()) {
    sortingKeys.push_back(key->name());
  }

  std::vector<std::string> functionCalls;
  bool customVerification = false;
  for (const auto& call : node->windowFunctions()) {
    functionCalls.push_back(makeFunctionCallString(call));
    customVerification =
        customVerificationFunctions_.count(call.functionCall->name()) != 0;
  }

  std::vector<RowVectorPtr> input;
  for (auto source : node->sources()) {
    auto valueNode = dynamic_cast<const core::ValuesNode*>(source.get());
    VELOX_CHECK_NOT_NULL(valueNode);
    auto values = valueNode->values();
    input.insert(input.end(), values.begin(), values.end());
  }

  verifyWindow(
      partitionKeys,
      sortingKeys,
      functionCalls,
      input,
      customVerification,
      FLAGS_enable_window_reference_verification);
}

void WindowFuzzer::updateReferenceQueryStats(
    AggregationFuzzerBase::ReferenceQueryErrorCode ec) {
  if (ec == ReferenceQueryErrorCode::kReferenceQueryFail) {
    ++stats_.numReferenceQueryFailed;
  } else if (ec == ReferenceQueryErrorCode::kReferenceQueryUnsupported) {
    ++stats_.numVerificationNotSupported;
  }
}

bool WindowFuzzer::verifyWindow(
    const std::vector<std::string>& partitionKeys,
    const std::vector<std::string>& sortingKeys,
    const std::vector<std::string>& functionCalls,
    const std::vector<RowVectorPtr>& input,
    bool customVerification,
    bool enableWindowVerification) {
  std::stringstream frame;
  VELOX_CHECK(!partitionKeys.empty());
  frame << "partition by " << folly::join(", ", partitionKeys);
  if (!sortingKeys.empty()) {
    frame << " order by " << folly::join(", ", sortingKeys);
  }
  // TODO: allow randomly generated frames.
  auto plan =
      PlanBuilder()
          .values(input)
          .window({fmt::format("{} over ({})", functionCalls[0], frame.str())})
          .planNode();
  if (persistAndRunOnce_) {
    persistReproInfo({{plan, {}}}, reproPersistPath_);
  }
  try {
    auto resultOrError = execute(plan);
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
      LOG(INFO) << "Verification skipped";
      ++stats_.numVerificationSkipped;
    }

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

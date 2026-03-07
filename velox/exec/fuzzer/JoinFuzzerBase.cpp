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

#include "velox/exec/fuzzer/JoinFuzzerBase.h"

#include <boost/random/uniform_int_distribution.hpp>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"

DECLARE_int32(steps);
DECLARE_int32(duration_sec);
DECLARE_int32(batch_size);
DECLARE_int32(num_batches);
DECLARE_double(null_ratio);

namespace facebook::velox::exec {

namespace {

template <typename T>
bool isDone(size_t i, T startTime) {
  if (FLAGS_duration_sec > 0) {
    std::chrono::duration<double> elapsed =
        std::chrono::system_clock::now() - startTime;
    return elapsed.count() >= FLAGS_duration_sec;
  }
  return i >= FLAGS_steps;
}

std::string makePercentageString(size_t value, size_t total) {
  return fmt::format("{} ({:.2f}%)", value, (double)value / total * 100);
}

} // namespace

JoinFuzzerBase::JoinFuzzerBase(
    size_t initialSeed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner,
    const std::string& poolName)
    : rootPool_(
          memory::memoryManager()->addRootPool(
              poolName,
              memory::kMaxMemory,
              memory::MemoryReclaimer::create())),
      pool_(rootPool_->addLeafChild(
          poolName + "Leaf",
          true,
          exec::MemoryReclaimer::create())),
      vectorFuzzer_{getFuzzerOptions(), pool_.get()},
      referenceQueryRunner_{std::move(referenceQueryRunner)} {
  filesystems::registerLocalFileSystem();

  std::unordered_map<std::string, std::string> hiveConfig = {
      {connector::hive::HiveConfig::kNumCacheFileHandles, "1000"}};

  if (!connector::hasConnector(test::kHiveConnectorId)) {
    connector::hive::HiveConnectorFactory factory;
    auto hiveConnector = factory.newConnector(
        test::kHiveConnectorId,
        std::make_shared<config::ConfigBase>(std::move(hiveConfig)));
    connector::registerConnector(hiveConnector);
  }

  seed(initialSeed);
}

VectorFuzzer::Options JoinFuzzerBase::getFuzzerOptions() {
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

void JoinFuzzerBase::seed(size_t seed) {
  currentSeed_ = seed;
  vectorFuzzer_.reSeed(seed);
  rng_.seed(currentSeed_);
}

void JoinFuzzerBase::reSeed() {
  seed(rng_());
}

int32_t JoinFuzzerBase::randInt(int32_t min, int32_t max) {
  return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
}

core::JoinType JoinFuzzerBase::pickJoinType() {
  const auto& joinTypes = getSupportedJoinTypes();
  const size_t idx = randInt(0, joinTypes.size() - 1);
  return joinTypes[idx];
}

std::string JoinFuzzerBase::joinTypeName(core::JoinType joinType) const {
  switch (joinType) {
    case core::JoinType::kInner:
      return "INNER";
    case core::JoinType::kLeft:
      return "LEFT";
    case core::JoinType::kRight:
      return "RIGHT";
    case core::JoinType::kFull:
      return "FULL";
    case core::JoinType::kLeftSemiFilter:
      return "LEFT SEMI FILTER";
    case core::JoinType::kLeftSemiProject:
      return "LEFT SEMI PROJECT";
    case core::JoinType::kRightSemiFilter:
      return "RIGHT SEMI FILTER";
    case core::JoinType::kRightSemiProject:
      return "RIGHT SEMI PROJECT";
    case core::JoinType::kAnti:
      return "ANTI";
    default:
      return "UNKNOWN";
  }
}

std::vector<TypePtr> JoinFuzzerBase::generateJoinKeyTypes(int32_t numKeys) {
  const auto& supportedTypes = getSupportedTypes();
  std::vector<TypePtr> types;
  types.reserve(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    types.push_back(vectorFuzzer_.randType(supportedTypes, /*maxDepth=*/0));
  }
  return types;
}

std::vector<RowVectorPtr> JoinFuzzerBase::generateProbeInput(
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

  const auto& supportedTypes = getSupportedTypes();

  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("tp{}", i + keyNames.size()));
    types.push_back(vectorFuzzer_.randType(supportedTypes, /*maxDepth=*/0));
  }

  const auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    if (keyTypesAllBool) {
      // Joining on just boolean keys creates so many hits it explodes the
      // output size, reduce the batch size to 10% to control the output size
      // while still covering this case.
      input.push_back(
          vectorFuzzer_.fuzzRow(inputType, FLAGS_batch_size / 10, false));
    } else {
      input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
    }
  }
  return input;
}

std::vector<RowVectorPtr> JoinFuzzerBase::generateBuildInput(
    const std::vector<RowVectorPtr>& probeInput,
    const std::vector<std::string>& probeKeys,
    const std::vector<std::string>& buildKeys) {
  std::vector<std::string> names = buildKeys;
  std::vector<TypePtr> types;
  for (const auto& key : probeKeys) {
    types.push_back(asRowType(probeInput[0]->type())->findChild(key));
  }

  const auto& supportedTypes = getSupportedTypes();

  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("bp{}", i + buildKeys.size()));
    types.push_back(vectorFuzzer_.randType(supportedTypes, /*maxDepth=*/0));
  }

  const auto rowType = ROW(std::move(names), std::move(types));

  // 1 in 10 times use empty build.
  if (vectorFuzzer_.coinToss(0.1)) {
    return {BaseVector::create<RowVector>(rowType, 0, pool_.get())};
  }

  // To ensure there are some matches, sample with replacement 10% of probe join
  // keys and use these as 80% of build keys. The rest build keys are randomly
  // generated. This allows the build side to have unmatched rows that should
  // appear in right join and full join.
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

std::optional<test::MaterializedRowMultiset>
JoinFuzzerBase::computeReferenceResults(
    const core::PlanNodePtr& plan,
    const std::vector<RowVectorPtr>& probeInput,
    const std::vector<RowVectorPtr>& buildInput) {
  if (referenceQueryRunner_->runnerType() ==
      test::ReferenceQueryRunner::RunnerType::kDuckQueryRunner) {
    VELOX_CHECK(!test::containsUnsupportedTypes(probeInput[0]->type()));
    VELOX_CHECK(!test::containsUnsupportedTypes(buildInput[0]->type()));
  }

  auto result = referenceQueryRunner_->execute(plan);
  VELOX_CHECK_NE(
      result.second, test::ReferenceQueryErrorCode::kReferenceQueryFail);
  return result.first;
}

std::string JoinFuzzerBase::Stats::toString() const {
  std::stringstream out;
  out << "\nTotal iterations tested: " << numIterations << std::endl;
  out << "Total iterations verified against reference DB: "
      << makePercentageString(numVerified, numIterations) << std::endl;
  return out.str();
}

void JoinFuzzerBase::go() {
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

} // namespace facebook::velox::exec

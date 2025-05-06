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

#include "velox/exec/MergeBuffer.h"

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace facebook::velox::functions::test {
namespace {
// Class to write runtime stats in the tests to the stats container.
class TestRuntimeStatWriter : public BaseRuntimeStatWriter {
 public:
  explicit TestRuntimeStatWriter(
      std::unordered_map<std::string, RuntimeMetric>& stats)
      : stats_{stats} {}

  void addRuntimeStat(const std::string& name, const RuntimeCounter& value)
      override {
    addOperatorRuntimeStats(name, value, stats_);
  }

 private:
  std::unordered_map<std::string, RuntimeMetric>& stats_;
};
} // namespace

class MergeBufferTest : public OperatorTestBase,
                        public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    statWriter_ = std::make_unique<TestRuntimeStatWriter>(stats_);
    setThreadLocalRunTimeStatWriter(statWriter_.get());
  }

  void TearDown() override {
    pool_.reset();
    rootPool_.reset();
    OperatorTestBase::TearDown();
  }

  const RowTypePtr inputType_ = ROW(
      {{"c0", BIGINT()},
       {"c1", INTEGER()},
       {"c2", SMALLINT()},
       {"c3", REAL()},
       {"c4", DOUBLE()},
       {"c5", VARCHAR()}});

  std::vector<std::pair<column_index_t, CompareFlags>> sortingKeys_{
      {0, {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}}};

  const std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  std::unordered_map<std::string, RuntimeMetric> stats_;
  std::unique_ptr<TestRuntimeStatWriter> statWriter_;
};

TEST_F(MergeBufferTest, spill) {
  auto spillDirectory = exec::test::TempDirectoryPath::create();
  auto spillConfig = common::SpillConfig(
      [&]() -> const std::string& { return spillDirectory->getPath(); },
      [&](uint64_t) {},
      "0.0.0",
      400,
      0,
      1 << 20,
      executor_.get(),
      100,
      100,
      0,
      0,
      0,
      0,
      0,
      "none",
      std::nullopt);
  folly::Synchronized<common::SpillStats> spillStats;
  const auto mergeBuffer = std::make_unique<MergeBuffer>(
      inputType_, pool_.get(), sortingKeys_, &spillConfig, &spillStats);

  const RowVectorPtr data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 8, 10, 12, 15}),
       makeFlatVector<int32_t>(
           {17, 16, 15, 14, 13, 10, 8, 7, 4, 3}), // sorted column
       makeFlatVector<int16_t>({1, 2, 3, 4, 5, 6, 8, 10, 12, 15}),
       makeFlatVector<float>(
           {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1}),
       makeFlatVector<double>(
           {1.1, 2.2, 2.2, 5.5, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1}),
       makeFlatVector<std::string>(
           {"hello",
            "world",
            "today",
            "is",
            "great",
            "hello",
            "world",
            "is",
            "great",
            "today"})});

  mergeBuffer->addInput(data);
  mergeBuffer->addInput(data);
  mergeBuffer->finishSpill(false);
  mergeBuffer->addInput(data);
  mergeBuffer->addInput(data);
  mergeBuffer->finishSpill(true);
  ASSERT_EQ(mergeBuffer->getOutput(20)->size(), 20);
  ASSERT_EQ(mergeBuffer->getOutput(20)->size(), 20);
  ASSERT_EQ(mergeBuffer->getOutput(10), nullptr);
  stats_.clear();
}

} // namespace facebook::velox::functions::test

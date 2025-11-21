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

#include "velox/exec/NonMaterializedSortBuffer.h"
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

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

class NonMaterializedSortBufferTest : public OperatorTestBase,
                                      public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    rng_.seed(123);
    statWriter_ = std::make_unique<TestRuntimeStatWriter>(stats_);
    setThreadLocalRunTimeStatWriter(statWriter_.get());
  }

  void TearDown() override {
    pool_.reset();
    rootPool_.reset();
    OperatorTestBase::TearDown();
    setThreadLocalRunTimeStatWriter(nullptr);
  }

  const bool enableSpillPrefixSort_{GetParam()};
  const velox::common::PrefixSortConfig prefixSortConfig_ =
      velox::common::PrefixSortConfig{
          std::numeric_limits<uint32_t>::max(),
          GetParam() ? 8 : std::numeric_limits<uint32_t>::max(),
          12};

  const RowTypePtr inputType_ = ROW(
      {{"c0", BIGINT()},
       {"c1", INTEGER()},
       {"c2", SMALLINT()},
       {"c3", REAL()},
       {"c4", DOUBLE()},
       {"c5", VARCHAR()}});
  // Specifies the sort columns ["c4", "c1"].
  std::vector<column_index_t> sortColumnIndices_{4, 1};
  std::vector<CompareFlags> sortCompareFlags_{
      {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue},
      {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};

  const std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  tsan_atomic<bool> nonReclaimableSection_{false};
  folly::Random::DefaultGenerator rng_;
  std::unordered_map<std::string, RuntimeMetric> stats_;
  std::unique_ptr<TestRuntimeStatWriter> statWriter_;
  const std::shared_ptr<memory::MemoryPool> fuzzerPool_ =
      memory::memoryManager()->addLeafPool("NonMaterializedSortBufferTest");
};

TEST_P(NonMaterializedSortBufferTest, singleKey) {
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

  struct {
    std::vector<CompareFlags> sortCompareFlags;
    std::vector<int32_t> expectedResult;

    std::string debugString() const {
      const std::string expectedResultStr = folly::join(",", expectedResult);
      std::stringstream sortCompareFlagsStr;
      for (const auto sortCompareFlag : sortCompareFlags) {
        sortCompareFlagsStr << sortCompareFlag.toString();
      }
      return fmt::format(
          "sortCompareFlags:{}, expectedResult:{}",
          sortCompareFlagsStr.str(),
          expectedResultStr);
    }
  } testSettings[] = {
      {{{true,
         true,
         false,
         CompareFlags::NullHandlingMode::kNullAsValue}}, // Ascending
       {3, 4, 7, 8, 10, 13, 14, 15, 16, 17}},
      {{{true,
         false,
         false,
         CompareFlags::NullHandlingMode::kNullAsValue}}, // Descending
       {17, 16, 15, 14, 13, 10, 8, 7, 4, 3}}};

  // Specifies the sort columns ["c1"].
  sortColumnIndices_ = {1};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto sortBuffer = std::make_unique<NonMaterializedSortBuffer>(
        inputType_,
        sortColumnIndices_,
        testData.sortCompareFlags,
        pool_.get(),
        &nonReclaimableSection_,
        prefixSortConfig_);

    sortBuffer->addInput(data);
    sortBuffer->noMoreInput();
    auto output = sortBuffer->getOutput(10000);
    ASSERT_EQ(output->size(), 10);
    int resultIndex = 0;
    for (int expectedValue : testData.expectedResult) {
      ASSERT_EQ(
          output->childAt(1)->asFlatVector<int32_t>()->valueAt(resultIndex++),
          expectedValue);
    }
    if (GetParam()) {
      ASSERT_EQ(
          stats_.at(PrefixSort::kNumPrefixSortKeys).sum,
          sortColumnIndices_.size());
      ASSERT_EQ(
          stats_.at(PrefixSort::kNumPrefixSortKeys).max,
          sortColumnIndices_.size());
      ASSERT_EQ(
          stats_.at(PrefixSort::kNumPrefixSortKeys).min,
          sortColumnIndices_.size());
    } else {
      ASSERT_EQ(stats_.count(PrefixSort::kNumPrefixSortKeys), 0);
    }
    stats_.clear();
  }
}

TEST_P(NonMaterializedSortBufferTest, multipleKeys) {
  auto sortBuffer = std::make_unique<NonMaterializedSortBuffer>(
      inputType_,
      sortColumnIndices_,
      sortCompareFlags_,
      pool_.get(),
      &nonReclaimableSection_,
      prefixSortConfig_);

  RowVectorPtr data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 8, 10, 12, 15}),
       makeFlatVector<int32_t>(
           {15, 12, 9, 8, 7, 6, 5, 4, 3, 1}), // sorted-2 column
       makeFlatVector<int16_t>({1, 2, 3, 4, 5, 6, 8, 10, 12, 15}),
       makeFlatVector<float>(
           {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1}),
       makeFlatVector<double>(
           {1.1, 2.2, 2.2, 5.5, 5.5, 7.7, 8.1, 8, 8.1, 8.1, 10.0}), // sorted-1
                                                                    // column
       makeFlatVector<std::string>(
           {"hello",
            "world",
            "today",
            "is",
            "great",
            "hello",
            "world",
            "is",
            "sort",
            "sorted"})});

  sortBuffer->addInput(data);
  sortBuffer->noMoreInput();
  auto output = sortBuffer->getOutput(10000);
  ASSERT_EQ(output->size(), 10);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(0), 15);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(1), 9);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(2), 12);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(3), 7);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(4), 8);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(5), 6);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(6), 4);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(7), 1);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(8), 3);
  ASSERT_EQ(output->childAt(1)->asFlatVector<int32_t>()->valueAt(9), 5);
  if (GetParam()) {
    ASSERT_EQ(
        stats_.at(PrefixSort::kNumPrefixSortKeys).sum,
        sortColumnIndices_.size());
    ASSERT_EQ(
        stats_.at(PrefixSort::kNumPrefixSortKeys).max,
        sortColumnIndices_.size());
    ASSERT_EQ(
        stats_.at(PrefixSort::kNumPrefixSortKeys).min,
        sortColumnIndices_.size());
  } else {
    ASSERT_EQ(stats_.count(PrefixSort::kNumPrefixSortKeys), 0);
  }
}

TEST_P(NonMaterializedSortBufferTest, batchOutput) {
  struct {
    std::vector<size_t> numInputRows;
    size_t maxOutputRows;
    std::vector<size_t> expectedOutputRowCount;

    std::string debugString() const {
      const std::string numInputRowsStr = folly::join(",", numInputRows);
      const std::string expectedOutputRowCountStr =
          folly::join(",", expectedOutputRowCount);
      return fmt::format(
          "numInputRows:{}, maxOutputRows:{}, expectedOutputRowCount:{}",
          numInputRowsStr,
          maxOutputRows,
          expectedOutputRowCountStr);
    }
  } testSettings[] = {
      {{2, 3, 3}, 1, {1, 1, 1, 1, 1, 1, 1, 1}},
      {{2, 3, 3}, 1, {1, 1, 1, 1, 1, 1, 1, 1}},
      {{2000, 2000}, 10000, {4000}},
      {{2000, 2000}, 10000, {4000}},
      {{2000, 2000}, 2000, {2000, 2000}},
      {{2000, 2000}, 2000, {2000, 2000}},
      {{1024, 1024, 1024}, 1000, {1000, 1000, 1000, 72}},
      {{1024, 1024, 1024}, 1000, {1000, 1000, 1000, 72}}};

  TestScopedSpillInjection scopedSpillInjection(100);
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto sortBuffer = std::make_unique<NonMaterializedSortBuffer>(
        inputType_,
        sortColumnIndices_,
        sortCompareFlags_,
        pool_.get(),
        &nonReclaimableSection_,
        prefixSortConfig_,
        nullptr,
        nullptr);
    ASSERT_FALSE(sortBuffer->canSpill());
    std::vector<RowVectorPtr> inputVectors;
    inputVectors.reserve(testData.numInputRows.size());
    uint64_t totalNumInput = 0;
    for (size_t inputRows : testData.numInputRows) {
      VectorFuzzer fuzzer({.vectorSize = inputRows}, fuzzerPool_.get());
      RowVectorPtr input = fuzzer.fuzzRow(inputType_);
      sortBuffer->addInput(input);
      inputVectors.push_back(input);
      totalNumInput += inputRows;
    }
    sortBuffer->noMoreInput();
    int expectedOutputBufferIndex = 0;
    RowVectorPtr output = sortBuffer->getOutput(testData.maxOutputRows);
    while (output != nullptr) {
      ASSERT_EQ(
          output->size(),
          testData.expectedOutputRowCount[expectedOutputBufferIndex++]);
      output = sortBuffer->getOutput(testData.maxOutputRows);
    }
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    NonMaterializedSortBufferTest,
    NonMaterializedSortBufferTest,
    testing::ValuesIn({false, true}));
} // namespace facebook::velox::functions::test

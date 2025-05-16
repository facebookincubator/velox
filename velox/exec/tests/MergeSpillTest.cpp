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


#include "velox/common/file/FileSystems.h"
#include "velox/exec/SortBuffer.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/exec/MergeSpill.h"

#include <gtest/gtest.h>

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace facebook::velox::exec::test {

class MergeSpillTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
  }

  std::vector<RowVectorPtr> generateSortedVectors(
      const int32_t numVectors,
      const size_t maxOutputRows) {
    const VectorFuzzer::Options fuzzerOpts{.vectorSize = maxOutputRows};
    const auto vectors = createVectors(numVectors, inputType_, fuzzerOpts);
    const auto sortBuffer = std::make_unique<SortBuffer>(
        inputType_,
        sortColumnIndices_,
        sortCompareFlags_,
        pool_.get(),
        &nonReclaimableSection_,
        common::PrefixSortConfig{},
        nullptr,
        nullptr);
    for (const auto& vector : vectors) {
      sortBuffer->addInput(vector);
    }
    sortBuffer->noMoreInput();
    std::vector<RowVectorPtr> sortedVectors;
    sortedVectors.reserve(numVectors);
    for (auto i = 0; i < numVectors; ++i) {
      sortedVectors.emplace_back(sortBuffer->getOutput(maxOutputRows));
    }
    return sortedVectors;
  }

  std::vector<RowVectorPtr> makeExpectedResults(
      const std::vector<RowVectorPtr>& vectors,
      size_t maxOutputRows) {
    const auto sortBuffer = std::make_unique<SortBuffer>(
        inputType_,
        sortColumnIndices_,
        sortCompareFlags_,
        pool_.get(),
        &nonReclaimableSection_,
        common::PrefixSortConfig{},
        nullptr,
        nullptr);
    for (const auto& vector : vectors) {
      sortBuffer->addInput(vector);
    }
    sortBuffer->noMoreInput();
    std::vector<RowVectorPtr> sortedVectors;
    sortedVectors.reserve(vectors.size());
    for (auto i = 0; i < vectors.size(); ++i) {
      sortedVectors.emplace_back(sortBuffer->getOutput(maxOutputRows));
    }
    return sortedVectors;
  }

 protected:
  const RowTypePtr inputType_ = ROW(
      {{"c0", BIGINT()},
       {"c1", INTEGER()},
       {"c2", SMALLINT()},
       {"c3", VARCHAR()}});
  const std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  const std::vector<column_index_t> sortColumnIndices_{0, 2};
  const std::vector<CompareFlags> sortCompareFlags_{
      CompareFlags{},
      CompareFlags{.ascending = false}};
  const std::vector<SpillSortKey> sortingKeys_ =
      SpillState::makeSortingKeys(sortColumnIndices_, sortCompareFlags_);
  const std::shared_ptr<TempDirectoryPath> spillDirectory_ =
      exec::test::TempDirectoryPath::create();
  const common::SpillConfig spillConfig_{
      [&]() -> const std::string& { return spillDirectory_->getPath(); },
      [&](uint64_t) {},
      "0.0.0",
      0,
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
      std::nullopt};

  folly::Synchronized<common::SpillStats> spillStats_;
  tsan_atomic<bool> nonReclaimableSection_{false};
};
} // namespace facebook::velox::exec::test

TEST_F(MergeSpillTest, basic) {
  const auto mergeSpiller = std::make_unique<MergeSpiller>(
      inputType_, pool_.get(), sortingKeys_, &spillConfig_, &spillStats_);
  for (auto i = 0; i < 7; ++i) {
    const auto vectors = generateSortedVectors(i + 1, 10);
    for (const auto& vector : vectors) {
      mergeSpiller->addInput(vector);
    }
    mergeSpiller->finishSpill(i == 6);
  }
  ASSERT_EQ(mergeSpiller->spillFiles().size(), 7);
  ASSERT_EQ(mergeSpiller->spillFiles()[0].size(), 1);
  ASSERT_EQ(mergeSpiller->spillFiles()[3].size(), 1);
  const auto mergeBuffer = std::make_unique<MergeBuffer>(
      inputType_,
      pool_.get(),
      mergeSpiller->spillFiles(),
      mergeSpiller->numSpillRows(),
      1 << 20,
      &spillStats_);
  for (auto i = 0; i < mergeSpiller->numSpillRows() / 10; ++i) {
    std::cout << mergeBuffer->getOutput(10)->toString(0, 10) << std::endl;
  }
  ASSERT_EQ(mergeBuffer->getOutput(10), nullptr);
}

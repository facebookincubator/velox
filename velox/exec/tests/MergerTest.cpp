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
#include "velox/exec/Merge.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/SortBuffer.h"
#include "velox/exec/Spill.h"
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

namespace facebook::velox::exec::test {

class MergerTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
  }

  std::vector<RowVectorPtr> generateSortedVectors(
      const int32_t numVectors,
      const size_t vectorSize) {
    const VectorFuzzer::Options fuzzerOpts{.vectorSize = vectorSize};
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
      sortedVectors.emplace_back(sortBuffer->getOutput(vectorSize));
    }
    return sortedVectors;
  }

  SpillFiles generateSortedSpillFiles(
      const std::vector<RowVectorPtr>& sortedVectors) {
    const auto spiller = std::make_unique<MergeSpiller>(
        inputType_,
        std::nullopt,
        HashBitRange{},
        sortingKeys_,
        &spillConfig_,
        spillStats_.get());
    for (const auto& vector : sortedVectors) {
      spiller->spill(SpillPartitionId(0), vector);
    }
    SpillPartitionSet spillPartitionSet;
    spiller->finishSpill(spillPartitionSet);
    EXPECT_EQ(spillPartitionSet.size(), 1);
    return spillPartitionSet.cbegin()->second->files();
  }

  std::pair<
      std::vector<std::vector<RowVectorPtr>>,
      std::vector<std::vector<std::unique_ptr<SpillReadFile>>>>
  generateInputs(size_t numStreams, size_t vectorSize) {
    std::vector<std::vector<RowVectorPtr>> totalVectors;
    std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
        spillReadFilesGroups;
    for (auto i = 1; i <= numStreams; ++i) {
      const auto vectors = generateSortedVectors(i * 3 + 1, vectorSize);
      totalVectors.push_back(vectors);
      const auto spillFiles = generateSortedSpillFiles(vectors);
      EXPECT_EQ(spillFiles.size(), vectors.size());
      std::vector<std::unique_ptr<SpillReadFile>> spillReadFiles;
      spillReadFiles.reserve(spillFiles.size());
      for (const auto& spillFile : spillFiles) {
        spillReadFiles.emplace_back(SpillReadFile::create(
            spillFile,
            spillConfig_.readBufferSize,
            pool_.get(),
            spillStats_.get()));
      }
      spillReadFilesGroups.emplace_back(std::move(spillReadFiles));
    }
    return std::make_pair(
        std::move(totalVectors), std::move(spillReadFilesGroups));
  }

  std::vector<RowVectorPtr> makeExpectedResults(
      const std::vector<std::vector<RowVectorPtr>>& inputs,
      size_t vectorSize) {
    std::vector<RowVectorPtr> flatInputs;
    for (const auto& vectors : inputs) {
      for (const auto& vector : vectors) {
        flatInputs.emplace_back(vector);
      }
    }
    const auto sortBuffer = std::make_unique<SortBuffer>(
        inputType_,
        sortColumnIndices_,
        sortCompareFlags_,
        pool_.get(),
        &nonReclaimableSection_,
        common::PrefixSortConfig{},
        nullptr,
        nullptr);
    for (const auto& vector : flatInputs) {
      sortBuffer->addInput(vector);
    }
    sortBuffer->noMoreInput();
    std::vector<RowVectorPtr> sortedVectors;
    sortedVectors.reserve(flatInputs.size());
    for (auto i = 0; i < flatInputs.size(); ++i) {
      sortedVectors.emplace_back(sortBuffer->getOutput(vectorSize));
    }
    return sortedVectors;
  }

  std::unique_ptr<SourceMerger> createSourceMerger(
      const std::vector<std::shared_ptr<MergeSource>>& sources,
      uint64_t outputBatchSize) const {
    std::vector<std::unique_ptr<SourceStream>> sourceStreams;
    for (const auto& source : sources) {
      sourceStreams.push_back(std::make_unique<SourceStream>(
          source.get(), sortingKeys_, outputBatchSize));
    }
    return std::make_unique<SourceMerger>(
        inputType_, outputBatchSize, std::move(sourceStreams), pool());
  }

  static std::vector<std::shared_ptr<MergeSource>> createMergeSources(int num) {
    std::vector<std::shared_ptr<MergeSource>> sources;
    sources.reserve(num);
    for (auto i = 0; i < num; ++i) {
      sources.push_back(MergeSource::createLocalMergeSource());
    }
    for (const auto& source : sources) {
      source->start();
    }
    return sources;
  }

  void produceAsync(
      MergeSource* mergeSource,
      const std::vector<RowVectorPtr>& vectors,
      size_t index = 0) const {
    ContinueFuture future;
    if (index >= vectors.size()) {
      const auto reason = mergeSource->enqueue(nullptr, &future);
      EXPECT_EQ(reason, BlockingReason::kNotBlocked);
      return;
    }

    mergeSource->enqueue(vectors[index], &future);
    std::move(future)
        .via(executor_.get())
        .thenValue([this, mergeSource, &vectors, index](folly::Unit) {
          produceAsync(mergeSource, vectors, index + 1);
        })
        .thenError(folly::tag_t<std::exception>{}, [](const std::exception& e) {
          VELOX_FAIL(e.what());
        });
  }

  void createProducers(
      int num,
      const std::vector<std::vector<RowVectorPtr>>& inputs,
      const std::vector<std::shared_ptr<MergeSource>>& sources) const {
    for (auto i = 0; i < inputs.size(); ++i) {
      executor_->add([&, i]() { produceAsync(sources[i].get(), inputs[i]); });
    }
  }

  static std::vector<RowVectorPtr> getOutputFromSourceMerger(
      SourceMerger* sourceMerger) {
    std::vector<ContinueFuture> sourceBlockingFutures;
    std::vector<RowVectorPtr> results;
    for (;;) {
      sourceMerger->isBlocked(sourceBlockingFutures);
      if (!sourceBlockingFutures.empty()) {
        auto future = std::move(sourceBlockingFutures.back());
        sourceBlockingFutures.pop_back();
        future.wait();
        continue;
      }
      bool atEnd = false;
      auto output = sourceMerger->getOutput(sourceBlockingFutures, atEnd);
      if (output != nullptr) {
        results.emplace_back(std::move(output));
      }

      if (atEnd) {
        break;
      }
    }
    return results;
  }

  std::shared_ptr<SpillMerger> createSpillMerger(
      std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
          spillReadFilesGroups,
      vector_size_t outputBatchSize) const {
    return std::make_shared<SpillMerger>(
        sortingKeys_,
        inputType_,
        outputBatchSize,
        std::move(spillReadFilesGroups),
        &spillConfig_,
        spillStats_,
        pool());
  }

  static std::vector<RowVectorPtr> getOutputFromSpillMerger(
      SpillMerger* spillMerger) {
    std::vector<ContinueFuture> sourceBlockingFutures;
    std::vector<RowVectorPtr> results;
    for (;;) {
      bool atEnd = false;
      auto output = spillMerger->getOutput(sourceBlockingFutures, atEnd);
      if (output != nullptr) {
        results.emplace_back(std::move(output));
      }

      if (atEnd) {
        break;
      }

      while (!sourceBlockingFutures.empty()) {
        auto future = std::move(sourceBlockingFutures.back());
        sourceBlockingFutures.pop_back();
        future.wait();
      }
    }
    return results;
  }

  static void checkResults(
      std::vector<RowVectorPtr> expectedResults,
      std::vector<RowVectorPtr> actualResults) {
    ASSERT_TRUE(assertEqualResults(expectedResults, actualResults));
    const auto& actual = actualResults[0];
    std::for_each(
        std::next(actualResults.begin()),
        actualResults.end(),
        [&](const auto& ele) { actual->append(ele.get()); });
    const auto& expect = expectedResults[0];
    std::for_each(
        std::next(expectedResults.begin()),
        expectedResults.end(),
        [&](const auto& ele) { expect->append(ele.get()); });
    facebook::velox::test::assertEqualVectors(expect, actual);
  }

 private:
  const RowTypePtr inputType_ = ROW({{"c0", BIGINT()}, {"c1", SMALLINT()}});
  const std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  const std::vector<column_index_t> sortColumnIndices_{0, 1};
  const std::vector<CompareFlags> sortCompareFlags_{
      CompareFlags{.ascending = true},
      CompareFlags{.ascending = false}};
  const std::vector<SpillSortKey> sortingKeys_ =
      SpillState::makeSortingKeys(sortColumnIndices_, sortCompareFlags_);
  const std::shared_ptr<TempDirectoryPath> spillDirectory_ =
      exec::test::TempDirectoryPath::create();
  const common::SpillConfig spillConfig_{
      [&]() -> const std::string& { return spillDirectory_->getPath(); },
      [&](uint64_t) {},
      "0.0.0",
      10, // Force to create a file per spill to mock multiple files per stream
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

  std::shared_ptr<folly::Synchronized<common::SpillStats>> spillStats_ =
      std::make_shared<folly::Synchronized<common::SpillStats>>();
  tsan_atomic<bool> nonReclaimableSection_{false};
};
} // namespace facebook::velox::exec::test

TEST_F(MergerTest, sourceMerger) {
  struct {
    size_t maxOutputRows;
    size_t numSources;

    std::string debugString() const {
      return fmt::format(
          "maxOutputRows:{} numStreams:{}", maxOutputRows, numSources);
    }
  } testSettings[] = {
      {1, 1},
      {1, 3},
      {1, 8},
      {7, 1},
      {7, 3},
      {7, 8},
      {16, 1},
      {16, 3},
      {16, 8},
      {32, 3},
      {1024, 8}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    std::vector<std::vector<RowVectorPtr>> inputs;
    for (auto i = 1; i <= testData.numSources; ++i) {
      inputs.emplace_back(generateSortedVectors(i, 16));
    }
    const auto sources = createMergeSources(testData.numSources);
    const auto sourceMerger =
        createSourceMerger(sources, testData.maxOutputRows);
    createProducers(testData.numSources, inputs, sources);
    const auto results = getOutputFromSourceMerger(sourceMerger.get());
    const auto expectedResults = makeExpectedResults(inputs, 16);
    checkResults(expectedResults, results);
  }
}

TEST_F(MergerTest, sourceMergerWithEmptySources) {
  std::vector<std::vector<RowVectorPtr>> inputs;
  for (auto i = 0; i < 10; ++i) {
    const auto numVectors = (i % 2 == 0) ? 0 : i;
    inputs.emplace_back(generateSortedVectors(numVectors, 16));
  }
  const auto sources = createMergeSources(10);
  const auto sourceMerger = createSourceMerger(sources, 32);
  createProducers(10, inputs, sources);
  const auto results = getOutputFromSourceMerger(sourceMerger.get());
  const auto expectedResults = makeExpectedResults(inputs, 16);
  checkResults(expectedResults, results);
}

TEST_F(MergerTest, spillMerger) {
  struct {
    size_t maxOutputRows;
    size_t numSources;

    std::string debugString() const {
      return fmt::format(
          "maxOutputRows:{} numStreams:{}", maxOutputRows, numSources);
    }
  } testSettings[] = {
      {1, 1},
      {1, 3},
      {1, 8},
      {7, 1},
      {7, 3},
      {7, 8},
      {16, 1},
      {16, 3},
      {16, 8},
      {32, 3},
      {1024, 8}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto sources = createMergeSources(testData.numSources);
    auto [inputs, filesGroup] = generateInputs(testData.numSources, 16);
    ASSERT_EQ(filesGroup.size(), testData.numSources);
    const auto spillMerger =
        createSpillMerger(std::move(filesGroup), testData.maxOutputRows);
    spillMerger->start();
    const auto results = getOutputFromSpillMerger(spillMerger.get());
    const auto expectedResults = makeExpectedResults(inputs, 16);
    checkResults(expectedResults, results);
  }
}

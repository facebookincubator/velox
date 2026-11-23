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
#include "velox/exec/tests/utils/MergeTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class TreeOfLosersTest : public testing::Test, public MergeTestBase {
 protected:
  void SetUp() override {
    seed(1);
  }

  void testBoth(int32_t numValues, int32_t numStreams) {
    TestData testData = makeTestData(numValues, numStreams);
    test<TreeOfLosers<TestingStream>>(testData, true);
    test<MergeArray<TestingStream>>(testData, true);
  }
};

TEST_F(TreeOfLosersTest, merge) {
  testBoth(11, 2);
  testBoth(16, 32);
  testBoth(17, 17);
  testBoth(0, 9);
  testBoth(5000000, 37);
  testBoth(500, 1);
}

TEST_F(TreeOfLosersTest, allDuplicates) {
  const int kNumsPerStream = 40;
  const int kNumStreams = 20;
  const uint32_t kValue = 10;
  std::vector<std::unique_ptr<TestingStream>> mergeStreams;
  for (int i = 0; i < kNumStreams; ++i) {
    std::vector<uint32_t> streamNumbers;
    for (int j = 0; j < kNumsPerStream; ++j) {
      streamNumbers.push_back(kValue);
    }
    mergeStreams.push_back(
        std::make_unique<TestingStream>(std::move(streamNumbers)));
  }
  const int expectedNumMergeStreams = mergeStreams.size();
  TreeOfLosers<TestingStream> merge(std::move(mergeStreams));
  ASSERT_EQ(merge.numStreams(), expectedNumMergeStreams);
  for (auto i = 0; i < kNumStreams * kNumsPerStream; ++i) {
    auto* stream = merge.next();
    ASSERT_TRUE(stream != nullptr) << i;
    ASSERT_EQ(stream->current()->value(), kValue) << i;
    stream->pop();
  }
}

TEST_F(TreeOfLosersTest, allSorted) {
  const int kNumsPerStream = 40;
  const int kNumStreams = 20;
  const uint32_t kStartValue = 10;
  std::vector<std::unique_ptr<TestingStream>> mergeStreams;
  for (int i = 0; i < kNumStreams; ++i) {
    std::vector<uint32_t> streamNumbers;
    for (int j = 0; j < kNumsPerStream; ++j) {
      streamNumbers.push_back(kStartValue + i * kNumsPerStream + j);
    }
    std::reverse(streamNumbers.begin(), streamNumbers.end());
    mergeStreams.push_back(
        std::make_unique<TestingStream>(std::move(streamNumbers)));
  }
  const int expectedNumMergeStreams = mergeStreams.size();
  TreeOfLosers<TestingStream> merge(std::move(mergeStreams));
  ASSERT_EQ(merge.numStreams(), expectedNumMergeStreams);
  for (auto i = 0; i < kNumStreams * kNumsPerStream; ++i) {
    auto* stream = merge.next();
    ASSERT_TRUE(stream != nullptr) << i;
    EXPECT_EQ(stream->current()->value(), kStartValue + i) << i;
    stream->pop();
  }
}

TEST_F(TreeOfLosersTest, allEmpty) {
  for (int numStreams : {0, 1, 5, 100}) {
    std::vector<std::unique_ptr<TestingStream>> mergeStreams;
    for (int i = 0; i < numStreams; ++i) {
      mergeStreams.push_back(
          std::make_unique<TestingStream>(std::vector<uint32_t>{}));
    }
    if (numStreams == 0) {
      EXPECT_ANY_THROW(
          TreeOfLosers<TestingStream> merge(std::move(mergeStreams)));
      continue;
    }
    const int expectedNumMergeStreams = mergeStreams.size();
    TreeOfLosers<TestingStream> merge(std::move(mergeStreams));
    ASSERT_EQ(merge.numStreams(), expectedNumMergeStreams);
    ASSERT_TRUE(merge.next() == nullptr);
  }
}

TEST_F(TreeOfLosersTest, randomWithDuplicates) {
  rng_.seed(1);
  for (int iter = 0; iter < 10; ++iter) {
    const int numCount = std::max<int>(1, folly::Random::rand32(1000'000));
    const int numStreams = std::max<int>(3, folly::Random::rand32(100));
    SCOPED_TRACE(
        fmt::format(
            "iter: {}, numCount: {}, numStreams: {}",
            iter,
            numCount,
            numStreams));
    std::vector<std::vector<uint32_t>> streamNumVectors(numStreams);
    for (int i = 0; i < numCount; ++i) {
      const int streamIndex = folly::Random::rand32(numStreams);
      streamNumVectors[streamIndex].push_back(numCount - i);
      streamNumVectors[(streamIndex + 1) % numStreams].push_back(numCount - i);
      streamNumVectors[(streamIndex + 2) % numStreams].push_back(numCount - i);
    }
    std::vector<std::unique_ptr<TestingStream>> mergeStreams;
    for (int i = 0; i < numStreams; ++i) {
      mergeStreams.push_back(
          std::make_unique<TestingStream>(std::move(streamNumVectors[i])));
    }
    const int expectedNumMergeStreams = mergeStreams.size();
    TreeOfLosers<TestingStream> merge(std::move(mergeStreams));
    ASSERT_EQ(merge.numStreams(), expectedNumMergeStreams);
    for (auto i = 3; i <= 3 * numCount; ++i) {
      auto* stream = merge.next();
      ASSERT_TRUE(stream != nullptr);
      ASSERT_EQ(stream->current()->value(), i / 3);
      stream->pop();
    }
  }
}

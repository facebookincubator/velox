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

#include "velox/exec/MergingVector.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec::test {

class MergingVectorTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    mergingVector_ = std::make_shared<MergingVector>(
        1024, 10UL << 20, 16, 10UL << 14, true, true, pool());
  }

  std::shared_ptr<MergingVector> mergingVector_;
};

TEST_F(MergingVectorTest, bufferSmallVector) {
  auto count = 0;
  std::vector<RowVectorPtr> vectors;
  vector_size_t batchSize;
  while (count <= 1024) {
    batchSize = folly::Random::rand32() % 14 + 1;
    count += batchSize;
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](auto row) { return batchSize + row; },
        VectorTestBase::nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        batchSize, [&](auto row) { return 123; }, nullEvery(5));

    auto indices = makeIndices(c1->size(), [](auto row) { return row; });
    auto c1Dict = wrapInDictionary(indices, c1->size(), c1);

    auto c2 = makeFlatVector<double>(
        batchSize, [](auto row) { return row * 0.1; }, nullEvery(11));
    auto c3 = makeFlatVector<StringView>(batchSize, [](auto row) {
      return StringView::makeInline(std::to_string(row));
    });
    vectors.push_back(makeRowVector({c0, c1Dict, c2, c3}));
  }

  EXPECT_EQ(mergingVector_->getOutput(false), nullptr);

  for (auto i = 0; i < vectors.size(); i++) {
    mergingVector_->addInput(vectors[i]);
  }

  auto firstMergedVector = mergingVector_->getOutput(false);
  EXPECT_EQ(mergingVector_->getOutput(false), nullptr);
  auto firstMergedVectorSie = firstMergedVector->size();
  mergingVector_->reuseIfNeed(firstMergedVector);
  EXPECT_EQ(firstMergedVector, nullptr);

  auto secondMergedVector = mergingVector_->getOutput(true);
  EXPECT_EQ(firstMergedVectorSie + secondMergedVector->size(), count);
  EXPECT_EQ(mergingVector_->getOutput(true), nullptr);
}

TEST_F(MergingVectorTest, bufferLargeVector) {
  vector_size_t batchSize = 20;
  RowVectorPtr vector;
  auto c0 = makeFlatVector<int64_t>(
      batchSize,
      [&](auto row) { return batchSize * 1 + row; },
      VectorTestBase::nullEvery(5));
  auto c1 = makeFlatVector<int64_t>(
      batchSize, [&](auto row) { return row; }, nullEvery(5));
  auto c2 = makeFlatVector<double>(
      batchSize, [](auto row) { return row * 0.1; }, nullEvery(11));
  auto c3 = makeFlatVector<StringView>(batchSize, [](auto row) {
    return StringView::makeInline(std::to_string(row));
  });

  EXPECT_EQ(mergingVector_->getOutput(false), nullptr);

  RowVectorPtr vector1 = makeRowVector({c0, c1, c2, c3});
  mergingVector_->addInput(vector1);

  auto mergedVector = mergingVector_->getOutput(false);
  EXPECT_EQ(mergedVector->size(), batchSize);
  EXPECT_EQ(mergingVector_->getOutput(true), nullptr);
}
} // namespace facebook::velox::exec::test

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

#include "velox/exec/MergingVectorOutput.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec::test {

class MergingVectorOutputTest : public testing::Test,
                                public velox::test::VectorTestBase {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::addDefaultLeafMemoryPool();
    mergingVectorOutput_ = std::make_shared<MergingVectorOutput>(
        pool(), 10UL << 20, 1024, 10UL << 16, 16);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<MergingVectorOutput> mergingVectorOutput_;
};

TEST_F(MergingVectorOutputTest, bufferSmallVector) {
  vector_size_t batchSize = 10;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 110; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](auto row) { return batchSize * i + row; },
        VectorTestBase::nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        batchSize, [&](auto row) { return row; }, nullEvery(5));
    auto c2 = makeFlatVector<double>(
        batchSize, [](auto row) { return row * 0.1; }, nullEvery(11));
    auto c3 = makeFlatVector<StringView>(batchSize, [](auto row) {
      return StringView::makeInline(std::to_string(row));
    });
    vectors.push_back(makeRowVector({c0, c1, c2, c3}));
  }

  EXPECT_EQ(mergingVectorOutput_->needsInput(), true);
  EXPECT_EQ(mergingVectorOutput_->getOutput(), nullptr);

  for (auto i = 0; i < vectors.size(); i++) {
    mergingVectorOutput_->addVector(vectors[i]);
  }

  EXPECT_EQ(mergingVectorOutput_->needsInput(), false);
  EXPECT_EQ(mergingVectorOutput_->getOutput()->size(), 1020);
  EXPECT_EQ(mergingVectorOutput_->getOutput(), nullptr);

  mergingVectorOutput_->noMoreInput();
  EXPECT_EQ(mergingVectorOutput_->isFinished(), false);
  EXPECT_EQ(mergingVectorOutput_->needsInput(), false);
  EXPECT_EQ(mergingVectorOutput_->getOutput()->size(), 80);
  EXPECT_EQ(mergingVectorOutput_->getOutput(), nullptr);
  EXPECT_EQ(mergingVectorOutput_->isFinished(), true);
}

TEST_F(MergingVectorOutputTest, bufferLargeVector) {
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

  EXPECT_EQ(mergingVectorOutput_->needsInput(), true);
  EXPECT_EQ(mergingVectorOutput_->getOutput(), nullptr);

  mergingVectorOutput_->addVector(makeRowVector({c0, c1, c2, c3}));

  EXPECT_EQ(mergingVectorOutput_->needsInput(), false);
  EXPECT_EQ(mergingVectorOutput_->getOutput()->size(), batchSize);
  EXPECT_EQ(mergingVectorOutput_->getOutput(), nullptr);
}
} // namespace facebook::velox::exec::test

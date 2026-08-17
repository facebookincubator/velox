/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <gtest/gtest.h>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"

using namespace ::facebook;

class RleTests : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(RleTests, EmptyInput) {
  std::vector<int32_t> data;
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  EXPECT_TRUE(runLengths.empty());
  EXPECT_TRUE(runValues.empty());
}

TEST_F(RleTests, SingleElement) {
  std::vector<int32_t> data{42};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{1};
  const std::vector<int32_t> expectedValues{42};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, AllSameValues) {
  std::vector<int32_t> data{5, 5, 5, 5, 5};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{5};
  const std::vector<int32_t> expectedValues{5};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, AllDifferentValues) {
  std::vector<int32_t> data{1, 2, 3, 4, 5};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{1, 1, 1, 1, 1};
  const std::vector<int32_t> expectedValues{1, 2, 3, 4, 5};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, MixedRuns) {
  std::vector<int32_t> data{1, 1, 1, 2, 2, 3, 3, 3, 3};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{3, 2, 4};
  const std::vector<int32_t> expectedValues{1, 2, 3};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, NegativeValues) {
  std::vector<int32_t> data{-1, -1, 0, 0, 0, 1, 1};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{2, 3, 2};
  const std::vector<int32_t> expectedValues{-1, 0, 1};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, Int64Type) {
  std::vector<int64_t> data{100, 100, 200, 200, 200};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int64_t> runValues(pool_.get());

  nimble::rle::computeRuns<int64_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{2, 3};
  const std::vector<int64_t> expectedValues{100, 200};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int64_t>(runValues.begin(), runValues.end()), expectedValues);
}

TEST_F(RleTests, UnsignedType) {
  std::vector<uint32_t> data{0, 0, 1, 2, 2, 2};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<uint32_t> runValues(pool_.get());

  nimble::rle::computeRuns<uint32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{2, 1, 3};
  const std::vector<uint32_t> expectedValues{0, 1, 2};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<uint32_t>(runValues.begin(), runValues.end()),
      expectedValues);
}

TEST_F(RleTests, AlternatingValues) {
  std::vector<int32_t> data{1, 2, 1, 2, 1, 2};
  nimble::Vector<uint32_t> runLengths(pool_.get());
  nimble::Vector<int32_t> runValues(pool_.get());

  nimble::rle::computeRuns<int32_t>(data, &runLengths, &runValues);

  const std::vector<uint32_t> expectedLengths{1, 1, 1, 1, 1, 1};
  const std::vector<int32_t> expectedValues{1, 2, 1, 2, 1, 2};
  EXPECT_EQ(
      std::vector<uint32_t>(runLengths.begin(), runLengths.end()),
      expectedLengths);
  EXPECT_EQ(
      std::vector<int32_t>(runValues.begin(), runValues.end()), expectedValues);
}

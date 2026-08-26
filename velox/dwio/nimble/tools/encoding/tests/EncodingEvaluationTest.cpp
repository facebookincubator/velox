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

#include "velox/dwio/nimble/tools/encoding/EncodingEvaluation.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::selection {
namespace {

class EncodingEvaluationTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("encoding_eval_test");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  velox::VectorPtr makeConstantColumn(int64_t value, int32_t size) {
    return vectorMaker_->flatVector<int64_t>(
        size, [value](velox::vector_size_t) { return value; });
  }

  velox::VectorPtr makeVariedColumn(int32_t size) {
    return vectorMaker_->flatVector<int64_t>(
        size, [](velox::vector_size_t i) { return i * 17 + 3; });
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(EncodingEvaluationTest, evaluateCandidatesReturnsResultsForCompatible) {
  const std::vector<velox::VectorPtr> vectors{makeConstantColumn(42, 1'024)};
  const auto candidates = buildEncodingCandidates({
      nimble::EncodingType::Trivial,
      nimble::EncodingType::FixedBitWidth,
  });

  EvaluationOptions opts;
  opts.iterations = 1;

  const auto results =
      evaluateCandidates(vectors, candidates, opts, pool_.get());

  ASSERT_EQ(results.size(), 2u);
  ASSERT_TRUE(results[0].has_value());
  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[0]->type, nimble::EncodingType::Trivial);
  EXPECT_EQ(results[1]->type, nimble::EncodingType::FixedBitWidth);
  EXPECT_GT(results[0]->encodedBytes, 0u);
  EXPECT_GT(results[1]->encodedBytes, 0u);
}

TEST_F(EncodingEvaluationTest, evaluateCandidatesReturnsNulloptOnIncompatible) {
  const std::vector<velox::VectorPtr> vectors{makeVariedColumn(1'024)};
  const auto candidates = buildEncodingCandidates({
      nimble::EncodingType::Trivial,
      nimble::EncodingType::Constant,
  });

  EvaluationOptions opts;
  opts.iterations = 1;

  const auto results =
      evaluateCandidates(vectors, candidates, opts, pool_.get());

  ASSERT_EQ(results.size(), 2u);
  EXPECT_TRUE(results[0].has_value());
  EXPECT_FALSE(results[1].has_value());
}

TEST_F(EncodingEvaluationTest, evaluateCandidatesThrowsOnEmptyVectors) {
  const std::vector<velox::VectorPtr> vectors;
  const auto candidates = buildEncodingCandidates({
      nimble::EncodingType::Trivial,
  });
  EvaluationOptions opts;

  EXPECT_THROW(
      evaluateCandidates(vectors, candidates, opts, pool_.get()),
      nimble::NimbleUserError);
}

TEST_F(EncodingEvaluationTest, rankResultsThrowsWithoutTrivialBaseline) {
  std::vector<std::optional<EvaluationResult>> results;
  results.emplace_back(
      EvaluationResult{
          .type = nimble::EncodingType::FixedBitWidth,
          .encodedBytes = 100,
          .encodeNanos = 100,
          .decodeNanos = 100,
      });

  ScoreWeights weights;
  EXPECT_THROW(rankResults(results, weights), nimble::NimbleUserError);
}

TEST_F(EncodingEvaluationTest, rankResultsSortsAscendingByScore) {
  std::vector<std::optional<EvaluationResult>> results;
  results.emplace_back(
      EvaluationResult{
          .type = nimble::EncodingType::Trivial,
          .encodedBytes = 1'000,
          .encodeNanos = 1'000,
          .decodeNanos = 1'000,
      });
  results.emplace_back(
      EvaluationResult{
          .type = nimble::EncodingType::FixedBitWidth,
          .encodedBytes = 500,
          .encodeNanos = 2'000,
          .decodeNanos = 2'000,
      });
  results.emplace_back(std::nullopt);
  results.emplace_back(
      EvaluationResult{
          .type = nimble::EncodingType::Constant,
          .encodedBytes = 10,
          .encodeNanos = 10'000,
          .decodeNanos = 10'000,
      });

  ScoreWeights sizeOnly;
  const auto rankedBySize = rankResults(results, sizeOnly);
  ASSERT_EQ(rankedBySize.size(), 3u);
  EXPECT_EQ(rankedBySize[0].type, nimble::EncodingType::Constant);
  EXPECT_EQ(rankedBySize[1].type, nimble::EncodingType::FixedBitWidth);
  EXPECT_EQ(rankedBySize[2].type, nimble::EncodingType::Trivial);

  ScoreWeights decodeOnly{.encodeSize = 0.0, .decodeTime = 1.0};
  const auto rankedByDecode = rankResults(results, decodeOnly);
  ASSERT_EQ(rankedByDecode.size(), 3u);
  EXPECT_EQ(rankedByDecode[0].type, nimble::EncodingType::Trivial);
}

TEST_F(EncodingEvaluationTest, getOptimalEncodingPicksConstantForConstantData) {
  const std::vector<velox::VectorPtr> vectors{makeConstantColumn(7, 1'024)};
  const auto candidates = buildEncodingCandidates({
      nimble::EncodingType::Trivial,
      nimble::EncodingType::FixedBitWidth,
      nimble::EncodingType::Constant,
  });

  EvaluationOptions opts;
  opts.iterations = 1;

  const auto winningTree =
      getOptimalEncoding(vectors, candidates, opts, pool_.get());

  ASSERT_EQ(winningTree.schemaKind(), nimble::Kind::Scalar);
  const auto* layout = winningTree.encodingLayout(
      nimble::EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->encodingType(), nimble::EncodingType::Constant);
}

TEST_F(EncodingEvaluationTest, getOptimalEncodingSkipsIncompatibleCandidates) {
  const std::vector<velox::VectorPtr> vectors{makeVariedColumn(1'024)};
  const auto candidates = buildEncodingCandidates({
      nimble::EncodingType::Trivial,
      nimble::EncodingType::Constant,
  });
  EvaluationOptions opts;
  opts.iterations = 1;

  const auto winningTree =
      getOptimalEncoding(vectors, candidates, opts, pool_.get());
  ASSERT_EQ(winningTree.schemaKind(), nimble::Kind::Scalar);
  const auto* layout = winningTree.encodingLayout(
      nimble::EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->encodingType(), nimble::EncodingType::Trivial);
}

} // namespace
} // namespace facebook::nimble::selection

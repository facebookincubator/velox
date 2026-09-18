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

#include "velox/dwio/nimble/index/VectorIndexUtility.h"

#include <array>

#include <gtest/gtest.h>

namespace facebook::nimble::index::test {
namespace {

TEST(VectorIndexUtilityTest, metricConversions) {
  struct TestCase {
    VectorDistanceMetric metric;
    faiss::MetricType faissMetric;
    serialization::VectorDistanceMetric serializedMetric;
  };
  constexpr std::array testCases{
      TestCase{
          VectorDistanceMetric::kL2,
          faiss::METRIC_L2,
          serialization::VectorDistanceMetric_L2,
      },
      TestCase{
          VectorDistanceMetric::kCosine,
          faiss::METRIC_INNER_PRODUCT,
          serialization::VectorDistanceMetric_Cosine,
      },
      TestCase{
          VectorDistanceMetric::kDotProduct,
          faiss::METRIC_INNER_PRODUCT,
          serialization::VectorDistanceMetric_DotProduct,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        ::testing::Message() << "metric=" << static_cast<int>(testCase.metric));
    EXPECT_EQ(toFaissMetric(testCase.metric), testCase.faissMetric);
    EXPECT_EQ(toSerializedMetric(testCase.metric), testCase.serializedMetric);
    EXPECT_EQ(
        fromSerializedMetric(static_cast<int8_t>(testCase.serializedMetric)),
        testCase.metric);
  }
}

TEST(VectorIndexUtilityTest, indexTypeConversions) {
  struct TestCase {
    VectorIndexType indexType;
    serialization::VectorIndexType serializedIndexType;
  };
  constexpr std::array testCases{
      TestCase{
          VectorIndexType::kIvfFlat,
          serialization::VectorIndexType_IVF_FLAT,
      },
      TestCase{
          VectorIndexType::kIvfSq8,
          serialization::VectorIndexType_IVF_SQ8,
      },
      TestCase{
          VectorIndexType::kIvfPq,
          serialization::VectorIndexType_IVF_PQ,
      },
      TestCase{
          VectorIndexType::kIvfRaBitQ,
          serialization::VectorIndexType_IVF_RABITQ,
      },
      TestCase{
          VectorIndexType::kHnswSq8,
          serialization::VectorIndexType_HNSW_SQ8,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        ::testing::Message()
        << "indexType=" << static_cast<int>(testCase.indexType));
    EXPECT_EQ(
        toSerializedIndexType(testCase.indexType),
        testCase.serializedIndexType);
    EXPECT_EQ(
        fromSerializedIndexType(
            static_cast<int8_t>(testCase.serializedIndexType)),
        testCase.indexType);
  }
}

TEST(VectorIndexUtilityTest, normalizeVectors) {
  const std::array<float, 4> input{3, 4, 0, 0};
  for (const auto metric :
       {VectorDistanceMetric::kL2, VectorDistanceMetric::kDotProduct}) {
    SCOPED_TRACE(::testing::Message() << "metric=" << static_cast<int>(metric));
    auto vectors = input;
    normalizeVectors(
        metric, /*numVectors=*/2, /*dimensions=*/2, vectors.data());
    EXPECT_EQ(vectors, input);
  }

  auto vectors = input;
  normalizeVectors(
      VectorDistanceMetric::kCosine,
      /*numVectors=*/2,
      /*dimensions=*/2,
      vectors.data());
  EXPECT_FLOAT_EQ(vectors[0], 0.6f);
  EXPECT_FLOAT_EQ(vectors[1], 0.8f);
  EXPECT_FLOAT_EQ(vectors[2], 0);
  EXPECT_FLOAT_EQ(vectors[3], 0);
}

} // namespace
} // namespace facebook::nimble::index::test

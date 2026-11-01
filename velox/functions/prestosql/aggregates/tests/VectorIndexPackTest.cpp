/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <numeric>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {
namespace {

class VectorIndexPackTest : public AggregationTestBase {
 protected:
  // The codec of 2-dimensional unit-length centroids given in cluster-id
  // order.
  static std::string codec(const std::vector<float>& centroids) {
    std::vector<int64_t> clusterIds(centroids.size() / 2);
    std::iota(clusterIds.begin(), clusterIds.end(), 0);
    std::string result;
    common::vectorindex::LocalVectorIndex::build(
        common::vectorindex::VectorIndexKind::kFlat,
        common::vectorindex::VectorIndexMetric::kCosine,
        common::vectorindex::LocalVectorIndexOptions{},
        common::vectorindex::Embeddings(centroids, /*dimension=*/2),
        clusterIds,
        [&](std::string_view chunk) { result.append(chunk); });
    return result;
  }

  // Runs vector_index_pack over 'clusterIds' and 'centroids' in one step.
  // 'metricSql' is the metric argument as a SQL expression.
  RowVectorPtr pack(
      const VectorPtr& clusterIds,
      const VectorPtr& centroids,
      std::string_view metricSql = "'cosine'") {
    return pack(makeRowVector({clusterIds, centroids}), metricSql);
  }

  // Runs vector_index_pack(c0, c1, 'metricSql') over 'data' in one step.
  RowVectorPtr pack(const RowVectorPtr& data, std::string_view metricSql) {
    auto plan =
        PlanBuilder()
            .values({data})
            .singleAggregation(
                {}, {fmt::format("vector_index_pack(c0, c1, {})", metricSql)})
            .planNode();
    return AssertQueryBuilder(plan).copyResults(pool());
  }
};

TEST_F(VectorIndexPackTest, global) {
  // Rows arrive out of cluster-id order; the codec places each centroid at its
  // id. Centroids are normalized for cosine, so (3, 0) packs as (1, 0).
  auto data = makeRowVector({
      makeFlatVector<int64_t>({2, 0, 1}),
      makeArrayVector<float>({{-1, 0}, {3, 0}, {0, 1}}),
  });

  auto expected = makeRowVector({
      makeFlatVector<std::string>({codec({1, 0, 0, 1, -1, 0})}, VARBINARY()),
  });

  testAggregations(
      {data}, {}, {"vector_index_pack(c0, c1, 'cosine')"}, {expected});
}

TEST_F(VectorIndexPackTest, groupBy) {
  // One codec per partition: 'a' and 'b' number the same two directions in
  // opposite orders.
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "b", "a", "b"}),
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeArrayVector<float>({{1, 0}, {0, 1}, {0, 1}, {1, 0}}),
  });

  auto expected = makeRowVector({
      makeFlatVector<std::string>({"a", "b"}),
      makeFlatVector<std::string>(
          {codec({1, 0, 0, 1}), codec({0, 1, 1, 0})}, VARBINARY()),
  });

  // The incremental check aggregates all rows into one group, where each
  // cluster id appears twice.
  disableTestIncremental();
  testAggregations(
      {data}, {"c0"}, {"vector_index_pack(c1, c2, 'cosine')"}, {expected});
}

TEST_F(VectorIndexPackTest, errors) {
  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0, 0}),
          makeArrayVector<float>({{1, 0}, {0, 1}})),
      "Cluster id must appear once: 0");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0, 2}),
          makeArrayVector<float>({{1, 0}, {0, 1}})),
      "Cluster ids must be in [0, number of clusters) range");

  VELOX_ASSERT_THROW(
      pack(makeFlatVector<int64_t>({-1}), makeArrayVector<float>({{1, 0}})),
      "Cluster id must not be negative");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0, 1}),
          makeArrayVector<float>({{1, 0}, {0, 1, 0}})),
      "All cluster centroids must have the same length");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0}),
          makeArrayVectorFromJson<float>({"[1, null]"})),
      "Cluster centroid must not contain nulls");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0}),
          makeArrayVector<float>(
              {{1, std::numeric_limits<float>::quiet_NaN()}})),
      "Cluster centroid must not contain NaN or infinity");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0}),
          makeArrayVector<float>({{1, 0}}),
          "'dot'"),
      "Unsupported vector index metric: dot");

  VELOX_ASSERT_THROW(
      pack(
          makeFlatVector<int64_t>({0}),
          makeArrayVector<float>({{1, 0}}),
          "null::varchar"),
      "Metric must not be null");

  VELOX_ASSERT_THROW(
      pack(
          makeRowVector({
              makeFlatVector<int64_t>({0}),
              makeArrayVector<float>({{1, 0}}),
              makeFlatVector<std::string>({"cosine"}),
          }),
          "c2"),
      "Metric must be a constant");
}

TEST_F(VectorIndexPackTest, mergeExtractErrors) {
  // vector_index_pack_merge_extract takes intermediate results directly:
  // metric, cluster ids and centroids.
  auto testFailingWithMetrics =
      [&](const std::vector<std::optional<int32_t>>& metrics,
          const std::vector<std::string>& idsJson,
          const std::vector<std::string>& centroidsJson,
          const std::string& expectedMessage) {
        SCOPED_TRACE(expectedMessage);
        testFailingAggregations(
            {makeRowVector({makeRowVector({
                makeNullableFlatVector<int32_t>(metrics),
                makeArrayVectorFromJson<int64_t>(idsJson),
                makeArrayVectorFromJson<float>(centroidsJson),
            })})},
            /*groupingKeys=*/{},
            {"vector_index_pack_merge_extract(c0)"},
            expectedMessage);
      };
  auto testFailing = [&](const std::vector<std::string>& idsJson,
                         const std::vector<std::string>& centroidsJson,
                         const std::string& expectedMessage) {
    testFailingWithMetrics({0, 0}, idsJson, centroidsJson, expectedMessage);
  };

  testFailingWithMetrics(
      {0, std::nullopt},
      {"[0]", "[1]"},
      {"[1, 0]", "[0, 1]"},
      "Intermediate vector index codec must include the metric");
  testFailingWithMetrics(
      {0, 9},
      {"[0]", "[1]"},
      {"[1, 0]", "[0, 1]"},
      "Invalid vector index metric value: 9");

  testFailing(
      {"[0]", "[-1]"},
      {"[1, 0]", "[0, 1]"},
      "Cluster ids must be in [0, number of clusters) range");
  testFailing(
      {"[0]", "null"},
      {"[1, 0]", "[0, 1]"},
      "Intermediate vector index codec must include the cluster ids");
  testFailing(
      {"[0]", "[1]"},
      {"[1, 0]", "null"},
      "Intermediate vector index codec must include the centroids");
  testFailing(
      {"[0]", "[]"},
      {"[1, 0]", "[]"},
      "Intermediate vector index codec must include a cluster");
  testFailing(
      {"[0]", "[1, 2]"},
      {"[1, 0]", "[0, 1, 0]"},
      "Intermediate vector index codec must have the same number of values per centroid");
  testFailing(
      {"[0]", "[null]"},
      {"[1, 0]", "[0, 1]"},
      "Intermediate cluster ids must not be null");
  testFailing(
      {"[0]", "[1]"},
      {"[1, 0]", "[0, null]"},
      "Intermediate cluster centroids must not contain nulls");
  testFailingAggregations(
      {makeRowVector({makeRowVector({
          makeFlatVector<int32_t>({0, 0}),
          makeArrayVector<int64_t>({{0}, {1}}),
          makeArrayVector<float>(
              {{1, 0}, {0, std::numeric_limits<float>::infinity()}}),
      })})},
      /*groupingKeys=*/{},
      {"vector_index_pack_merge_extract(c0)"},
      "Cluster centroid must not contain NaN or infinity");
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql

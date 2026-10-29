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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfVectorJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox {
namespace {

using Element = std::optional<float>;
using Vector = std::optional<std::vector<Element>>;

struct Match {
  int64_t probeId;
  int64_t buildId;
  float score;

  bool operator<(const Match& other) const {
    return std::tie(probeId, buildId, score) <
        std::tie(other.probeId, other.buildId, other.score);
  }
};

bool largerIsBetter(VectorJoinMetric metric) {
  return metric != VectorJoinMetric::kL2Squared;
}

bool usable(const Vector& values, VectorJoinMetric metric) {
  if (!values.has_value() || values->empty()) {
    return false;
  }
  float squares = 0;
  for (const auto& element : *values) {
    if (!element.has_value()) {
      return false;
    }
    squares += element.value() * element.value();
  }
  return metric != VectorJoinMetric::kCosine || squares > 0;
}

float exactScore(
    const Vector& probe,
    const Vector& build,
    VectorJoinMetric metric) {
  float dot = 0;
  float leftSquares = 0;
  float rightSquares = 0;
  float squared = 0;
  for (size_t i = 0; i < probe->size(); ++i) {
    const auto left = probe->at(i).value();
    const auto right = build->at(i).value();
    const auto difference = left - right;
    dot += left * right;
    leftSquares += left * left;
    rightSquares += right * right;
    squared += difference * difference;
  }
  if (metric == VectorJoinMetric::kCosine) {
    return dot / std::sqrt(leftSquares * rightSquares);
  }
  return metric == VectorJoinMetric::kInnerProduct ? dot : squared;
}

std::vector<Match> bruteForce(
    const std::vector<int64_t>& probeIds,
    const std::vector<Vector>& probes,
    const std::vector<int64_t>& buildIds,
    const std::vector<Vector>& builds,
    VectorJoinMetric metric,
    int32_t k) {
  std::vector<Match> matches;
  for (size_t probe = 0; probe < probes.size(); ++probe) {
    if (!usable(probes[probe], metric)) {
      continue;
    }
    std::vector<Match> hits;
    for (size_t build = 0; build < builds.size(); ++build) {
      if (!usable(builds[build], metric) ||
          builds[build]->size() != probes[probe]->size()) {
        continue;
      }
      hits.push_back(
          Match{
              probeIds[probe],
              buildIds[build],
              exactScore(probes[probe], builds[build], metric)});
    }
    const auto better = largerIsBetter(metric);
    std::sort(
        hits.begin(), hits.end(), [&](const Match& left, const Match& right) {
          if (left.score == right.score) {
            return left.buildId < right.buildId;
          }
          return better ? left.score > right.score : left.score < right.score;
        });
    if (static_cast<int32_t>(hits.size()) > k) {
      hits.resize(k);
    }
    matches.insert(matches.end(), hits.begin(), hits.end());
  }
  std::sort(matches.begin(), matches.end());
  return matches;
}

class VectorJoinTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }

  RowVectorPtr makeSide(
      const std::string& idName,
      const std::string& vectorName,
      const std::vector<int64_t>& ids,
      const std::vector<Vector>& values) {
    return makeRowVector(
        {idName, vectorName},
        {makeFlatVector<int64_t>(ids), makeNullableArrayVector<float>(values)});
  }

  core::PlanNodePtr makePlan(
      const std::vector<RowVectorPtr>& probes,
      const std::vector<RowVectorPtr>& builds,
      VectorJoinMetric metric,
      int32_t k,
      int32_t nLists = 0,
      int32_t nProbes = 0,
      int32_t candidateMultiplier = 0,
      bool roundRobin = false) const {
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    auto build = PlanBuilder(ids).values(builds).planNode();
    PlanBuilder builder(ids);
    builder.values(probes);
    if (roundRobin) {
      builder.localPartitionRoundRobinRow();
    }
    return builder
        .addNode([&](core::PlanNodeId id, core::PlanNodePtr input) {
          return std::make_shared<CudfVectorJoinNode>(
              id,
              "p_vec",
              "b_vec",
              "b_id",
              "score",
              metric,
              k,
              candidateMultiplier,
              nLists,
              nProbes,
              0,
              0,
              std::move(input),
              build,
              ROW({"p_id", "b_id", "score"}, {BIGINT(), BIGINT(), REAL()}));
        })
        .planNode();
  }

  RowVectorPtr run(
      const core::PlanNodePtr& plan,
      int32_t maxDrivers = 1,
      std::optional<int32_t> probeBatchRows = std::nullopt,
      std::optional<int64_t> maxShardRows = std::nullopt) const {
    AssertQueryBuilder builder(plan);
    builder.maxDrivers(maxDrivers);
    if (probeBatchRows.has_value()) {
      builder.config(CudfConfig::kVectorJoinProbeBatchRows, *probeBatchRows);
    }
    if (maxShardRows.has_value()) {
      builder.config(CudfConfig::kVectorJoinMaxShardRows, *maxShardRows);
    }
    return builder.copyResults(pool());
  }

  std::vector<Match> matchesOf(const RowVectorPtr& result) const {
    std::vector<Match> matches;
    if (result == nullptr || result->size() == 0) {
      return matches;
    }
    auto probeIds = result->childAt(0)->asFlatVector<int64_t>();
    auto buildIds = result->childAt(1)->asFlatVector<int64_t>();
    auto scores = result->childAt(2)->asFlatVector<float>();
    VELOX_CHECK_NOT_NULL(probeIds);
    VELOX_CHECK_NOT_NULL(buildIds);
    VELOX_CHECK_NOT_NULL(scores);
    matches.reserve(result->size());
    for (auto i = 0; i < result->size(); ++i) {
      matches.push_back(
          Match{
              probeIds->valueAt(i), buildIds->valueAt(i), scores->valueAt(i)});
    }
    std::sort(matches.begin(), matches.end());
    return matches;
  }

  void expectMatches(
      const std::vector<Match>& actual,
      const std::vector<Match>& expected) const {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i].probeId, expected[i].probeId) << i;
      EXPECT_EQ(actual[i].buildId, expected[i].buildId) << i;
      EXPECT_NEAR(actual[i].score, expected[i].score, 1e-3) << i;
    }
  }
};

TEST_F(VectorJoinTest, metricsMatchExactRerank) {
  const std::vector<int64_t> probeIds{1, 2, 3};
  const std::vector<Vector> probes{
      Vector{{1.f, 0.f}}, Vector{{5.f, 5.f}}, Vector{{0.f, 1.f}}};
  const std::vector<int64_t> buildIds{10, 20, 30, 40};
  const std::vector<Vector> builds{
      Vector{{1.f, 0.f}},
      Vector{{0.f, 1.f}},
      Vector{{5.f, 4.f}},
      Vector{{5.f, 6.f}}};
  auto probe = makeSide("p_id", "p_vec", probeIds, probes);
  auto first = makeSide(
      "b_id", "b_vec", {buildIds[0], buildIds[1]}, {builds[0], builds[1]});
  auto second = makeSide(
      "b_id", "b_vec", {buildIds[2], buildIds[3]}, {builds[2], builds[3]});
  for (const auto metric :
       {VectorJoinMetric::kL2Squared,
        VectorJoinMetric::kInnerProduct,
        VectorJoinMetric::kCosine}) {
    auto result = run(makePlan({probe}, {first, second}, metric, 2));
    expectMatches(
        matchesOf(result),
        bruteForce(probeIds, probes, buildIds, builds, metric, 2));
  }
}

TEST_F(VectorJoinTest, skipsNullEmptyAndZeroNormRows) {
  const std::vector<int64_t> probeIds{1, 2, 3, 4};
  const std::vector<Vector> probes{
      Vector{{1.f, 0.f}},
      std::nullopt,
      Vector{std::vector<Element>{}},
      Vector{{0.f, 0.f}}};
  const std::vector<int64_t> buildIds{10, 20, 30, 40};
  const std::vector<Vector> builds{
      Vector{{1.f, 0.f}},
      Vector{{std::optional<float>{1.f}, std::nullopt}},
      Vector{std::vector<Element>{}},
      Vector{{0.f, 2.f}}};
  auto probe = makeSide("p_id", "p_vec", probeIds, probes);
  auto build = makeSide("b_id", "b_vec", buildIds, builds);
  for (const auto metric :
       {VectorJoinMetric::kL2Squared,
        VectorJoinMetric::kInnerProduct,
        VectorJoinMetric::kCosine}) {
    auto result = run(makePlan({probe}, {build}, metric, 2));
    expectMatches(
        matchesOf(result),
        bruteForce(probeIds, probes, buildIds, builds, metric, 2));
  }
}

TEST_F(VectorJoinTest, emptyBuildProducesNoRows) {
  auto probe = makeSide("p_id", "p_vec", {1}, {Vector{{1.f, 0.f}}});
  auto build = makeSide("b_id", "b_vec", {}, {});
  auto result =
      run(makePlan({probe}, {build}, VectorJoinMetric::kL2Squared, 1));
  EXPECT_EQ(result->size(), 0);
}

TEST_F(VectorJoinTest, kLargerThanShardReturnsEveryBuildRow) {
  const std::vector<int64_t> probeIds{1};
  const std::vector<Vector> probes{Vector{{1.f, 1.f}}};
  const std::vector<int64_t> buildIds{4, 7};
  const std::vector<Vector> builds{Vector{{0.f, 1.f}}, Vector{{1.f, 0.f}}};
  auto result = run(makePlan(
      {makeSide("p_id", "p_vec", probeIds, probes)},
      {makeSide("b_id", "b_vec", buildIds, builds)},
      VectorJoinMetric::kL2Squared,
      5));
  expectMatches(
      matchesOf(result),
      bruteForce(
          probeIds, probes, buildIds, builds, VectorJoinMetric::kL2Squared, 5));
}

TEST_F(VectorJoinTest, tiesKeepTheSmallerBuildId) {
  const std::vector<int64_t> probeIds{1};
  const std::vector<Vector> probes{Vector{{2.f, 2.f}}};
  const std::vector<int64_t> buildIds{9, 3, 5};
  const std::vector<Vector> builds{
      Vector{{2.f, 2.f}}, Vector{{2.f, 2.f}}, Vector{{2.f, 2.f}}};
  for (const auto metric :
       {VectorJoinMetric::kL2Squared,
        VectorJoinMetric::kInnerProduct,
        VectorJoinMetric::kCosine}) {
    auto result = run(makePlan(
        {makeSide("p_id", "p_vec", probeIds, probes)},
        {makeSide("b_id", "b_vec", buildIds, builds)},
        metric,
        1));
    expectMatches(
        matchesOf(result),
        bruteForce(probeIds, probes, buildIds, builds, metric, 1));
    ASSERT_EQ(matchesOf(result).size(), 1);
    EXPECT_EQ(matchesOf(result)[0].buildId, 3);
  }
}

TEST_F(VectorJoinTest, dimensionMismatchIsAUserError) {
  auto probe = makeSide("p_id", "p_vec", {1}, {Vector{{1.f, 0.f}}});
  auto build = makeSide("b_id", "b_vec", {2}, {Vector{{1.f, 0.f, 0.f}}});
  VELOX_ASSERT_USER_THROW(
      run(makePlan({probe}, {build}, VectorJoinMetric::kL2Squared, 1)),
      "one dimension");
}

TEST_F(VectorJoinTest, shardRowLimitAsksForRepartition) {
  auto probe = makeSide("p_id", "p_vec", {1}, {Vector{{1.f, 0.f}}});
  auto build = makeSide(
      "b_id", "b_vec", {2, 3}, {Vector{{1.f, 0.f}}, Vector{{0.f, 1.f}}});
  VELOX_ASSERT_USER_THROW(
      run(makePlan({probe}, {build}, VectorJoinMetric::kL2Squared, 1),
          1,
          std::nullopt,
          1),
      "Repartition");
}

TEST_F(VectorJoinTest, probeBatchesStayExact) {
  const std::vector<int64_t> probeIds{1, 2, 3, 4};
  const std::vector<Vector> probes{
      Vector{{1.f, 0.f}},
      Vector{{0.f, 1.f}},
      Vector{{4.f, 4.f}},
      Vector{{0.5f, 0.5f}}};
  const std::vector<int64_t> buildIds{8, 6, 7};
  const std::vector<Vector> builds{
      Vector{{1.f, 0.f}}, Vector{{0.f, 1.f}}, Vector{{4.f, 0.f}}};
  auto result =
      run(makePlan(
              {makeSide("p_id", "p_vec", probeIds, probes)},
              {makeSide("b_id", "b_vec", buildIds, builds)},
              VectorJoinMetric::kInnerProduct,
              2),
          1,
          1);
  expectMatches(
      matchesOf(result),
      bruteForce(
          probeIds,
          probes,
          buildIds,
          builds,
          VectorJoinMetric::kInnerProduct,
          2));
}

TEST_F(VectorJoinTest, concurrentProbesMatchOneDriver) {
  const std::vector<int64_t> probeIds{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<Vector> probes;
  for (int32_t i = 0; i < 8; ++i) {
    probes.push_back(Vector{{static_cast<float>(i), 1.f}});
  }
  const std::vector<int64_t> buildIds{10, 11, 12, 13};
  const std::vector<Vector> builds{
      Vector{{0.f, 1.f}},
      Vector{{3.f, 1.f}},
      Vector{{6.f, 1.f}},
      Vector{{8.f, 1.f}}};
  auto probe = makeSide("p_id", "p_vec", probeIds, probes);
  auto build = makeSide("b_id", "b_vec", buildIds, builds);
  auto expected = bruteForce(
      probeIds, probes, buildIds, builds, VectorJoinMetric::kL2Squared, 2);
  auto single =
      run(makePlan({probe}, {build}, VectorJoinMetric::kL2Squared, 2));
  auto concurrent =
      run(makePlan(
              {probe}, {build}, VectorJoinMetric::kL2Squared, 2, 0, 0, 0, true),
          4);
  expectMatches(matchesOf(single), expected);
  expectMatches(matchesOf(concurrent), expected);
}

TEST_F(VectorJoinTest, repeatedRunsDoNotGrowDeviceMemory) {
  auto probe = makeSide(
      "p_id",
      "p_vec",
      {1, 2, 3},
      {Vector{{1.f, 0.f}}, Vector{{0.f, 1.f}}, Vector{{2.f, 2.f}}});
  auto build = makeSide(
      "b_id",
      "b_vec",
      {4, 5, 6},
      {Vector{{1.f, 0.f}}, Vector{{0.f, 1.f}}, Vector{{2.f, 1.f}}});
  auto plan = makePlan({probe}, {build}, VectorJoinMetric::kL2Squared, 2);
  ASSERT_GT(run(plan)->size(), 0);
  cudaDeviceSynchronize();
  const auto afterFirst = cudfAllocatedBytes();
  ASSERT_GT(run(plan)->size(), 0);
  cudaDeviceSynchronize();
  const auto afterSecond = cudfAllocatedBytes();
  EXPECT_LE(afterSecond, afterFirst + (1 << 20));
}

TEST_F(VectorJoinTest, clusteredRecall) {
  constexpr int32_t kClusters = 4;
  constexpr int32_t kPerCluster = 20;
  constexpr int32_t kDimension = 4;
  std::vector<int64_t> ids;
  std::vector<Vector> values;
  ids.reserve(kClusters * kPerCluster);
  for (int32_t cluster = 0; cluster < kClusters; ++cluster) {
    for (int32_t point = 0; point < kPerCluster; ++point) {
      std::vector<Element> coordinates(kDimension, 0.f);
      coordinates[0] = static_cast<float>(cluster * 100);
      coordinates[1] = static_cast<float>(point);
      ids.push_back(cluster * kPerCluster + point);
      values.push_back(Vector{std::move(coordinates)});
    }
  }
  auto table = makeSide("p_id", "p_vec", ids, values);
  auto build = makeSide("b_id", "b_vec", ids, values);
  auto result = run(makePlan(
      {table},
      {build},
      VectorJoinMetric::kL2Squared,
      1,
      /*nLists=*/4,
      /*nProbes=*/1,
      /*candidateMultiplier=*/kPerCluster));
  const auto matches = matchesOf(result);
  ASSERT_EQ(matches.size(), ids.size());
  int32_t found = 0;
  for (const auto& match : matches) {
    found += match.probeId == match.buildId ? 1 : 0;
  }
  EXPECT_GE(found, static_cast<int32_t>(ids.size() * 0.9));
}

TEST_F(VectorJoinTest, shardTopKMergesToGlobalTopK) {
  const std::vector<int64_t> probeIds{1, 2};
  const std::vector<Vector> probes{Vector{{0.f, 0.f}}, Vector{{10.f, 0.f}}};
  const std::vector<int64_t> buildIds{1, 2, 3, 4, 5, 6};
  const std::vector<Vector> builds{
      Vector{{0.f, 1.f}},
      Vector{{0.f, 3.f}},
      Vector{{1.f, 0.f}},
      Vector{{10.f, 1.f}},
      Vector{{12.f, 0.f}},
      Vector{{9.f, 0.f}}};
  auto probe = makeSide("p_id", "p_vec", probeIds, probes);
  auto left = makeSide(
      "b_id",
      "b_vec",
      {buildIds[0], buildIds[1], buildIds[2]},
      {builds[0], builds[1], builds[2]});
  auto right = makeSide(
      "b_id",
      "b_vec",
      {buildIds[3], buildIds[4], buildIds[5]},
      {builds[3], builds[4], builds[5]});
  for (const auto metric :
       {VectorJoinMetric::kL2Squared, VectorJoinMetric::kCosine}) {
    constexpr int32_t k = 2;
    auto localLeft = run(makePlan({probe}, {left}, metric, k));
    auto localRight = run(makePlan({probe}, {right}, metric, k));
    const std::vector<std::string> order =
        metric == VectorJoinMetric::kL2Squared
        ? std::vector<std::string>{"score", "b_id"}
        : std::vector<std::string>{"score DESC", "b_id"};
    auto merged = AssertQueryBuilder(
                      PlanBuilder()
                          .values({localLeft, localRight})
                          .topNRowNumber({"p_id"}, order, k, false)
                          .planNode())
                      .copyResults(pool());
    auto global = run(makePlan(
        {probe}, {makeSide("b_id", "b_vec", buildIds, builds)}, metric, k));
    expectMatches(matchesOf(merged), matchesOf(global));
    expectMatches(
        matchesOf(global),
        bruteForce(probeIds, probes, buildIds, builds, metric, k));
  }
}

TEST_F(VectorJoinTest, planRoundTripsThroughSerde) {
  core::PlanNode::registerSerDe();
  auto probe = makeSide("p_id", "p_vec", {1}, {Vector{{1.f, 2.f}}});
  auto build = makeSide("b_id", "b_vec", {2}, {Vector{{3.f, 4.f}}});
  auto plan = makePlan({probe}, {build}, VectorJoinMetric::kInnerProduct, 3);
  auto restored =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  auto* join = dynamic_cast<const CudfVectorJoinNode*>(restored.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->k(), 3);
  EXPECT_EQ(join->metric(), VectorJoinMetric::kInnerProduct);
  EXPECT_EQ(join->probeVector(), "p_vec");
  EXPECT_EQ(join->buildId(), "b_id");
  EXPECT_EQ(join->sources().size(), 2);
}

} // namespace
} // namespace facebook::velox::cudf_velox

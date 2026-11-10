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
#include <gmock/gmock.h>

#include <map>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/vectorindex/FaissLocalVectorIndex.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {
namespace {

using common::vectorindex::Embeddings;
using common::vectorindex::FaissLocalVectorIndex;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::VectorIndexMetric;

using TrainStats = common::vectorindex::FaissLocalVectorIndex::TrainStats;

class VectorIndexTrainTest : public AggregationTestBase {
 protected:
  // Runs vector_index_train(c0, c1, 'numClusters', 'cosine', <no options>)
  // over 'data',
  // grouped by c2 if 'groupBy', in every plan shape AggregationTester covers.
  // Checks that each shape gives the same codec for each group and passes
  // each group's codec to 'verify', with the group's key, null without
  // grouping. Returns the codecs by group key.
  std::map<Variant, std::string> testTrain(
      const RowVectorPtr& data,
      int64_t numClusters,
      bool groupBy,
      const std::function<void(const Variant&, const FaissLocalVectorIndex&)>&
          verify) {
    std::optional<std::map<Variant, std::string>> firstCodecs;
    AggregationTester(*this)
        .inputs(data)
        .groupBy(
            groupBy ? std::vector<std::string>{"c2"}
                    : std::vector<std::string>{})
        .aggregates({aggregate(numClusters)})
        .assertResults([&](AssertQueryBuilder& builder) {
          std::shared_ptr<exec::Task> task;
          auto result = builder.copyResults(pool(), task);
          std::map<Variant, std::string> codecs;
          for (auto row = 0; row < result->size(); ++row) {
            const auto key = groupBy ? result->childAt(0)->variantAt(row)
                                     : Variant::null(TypeKind::BIGINT);
            const auto bytes = result->childAt(groupBy ? 1 : 0)
                                   ->variantAt(row)
                                   .value<TypeKind::VARBINARY>();
            verify(
                key,
                FaissLocalVectorIndex(
                    bytes,
                    VectorIndexMetric::kCosine,
                    LocalVectorIndexOptions{}));
            codecs.emplace(key, bytes);
          }
          if (firstCodecs.has_value()) {
            EXPECT_EQ(*firstCodecs, codecs);
          } else {
            firstCodecs = std::move(codecs);
          }
          return task;
        });
    return std::move(firstCodecs).value();
  }

  static std::string aggregate(int64_t numClusters) {
    return fmt::format(
        "vector_index_train(c0, c1, {}, 'cosine', map()::map(varchar, varchar))",
        numClusters);
  }

  // The cluster nearest to 'query' and its similarity.
  static std::pair<int64_t, float> nearest(
      const FaissLocalVectorIndex& codec,
      std::vector<float> query) {
    int64_t id{-1};
    float score{0};
    codec.search(
        Embeddings(query, query.size()),
        /*k=*/1,
        std::span(&id, 1),
        std::span(&score, 1));
    return {id, score};
  }
};

TEST_F(VectorIndexTrainTest, clusters) {
  // Two groups of directions. (3, 0) and (0, 5) normalize to (1, 0) and
  // (0, 1), so each cluster's centroid is exactly its direction. The same
  // rows in reverse order give the same codec.
  auto data = makeRowVector({
      makeArrayVector<float>({{1, 0}, {0, 1}, {3, 0}, {0, 5}, {1, 0}, {0, 1}}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
  });
  auto reversed = makeRowVector({
      makeArrayVector<float>({{0, 1}, {1, 0}, {0, 5}, {3, 0}, {0, 1}, {1, 0}}),
      makeFlatVector<int64_t>({60, 50, 40, 30, 20, 10}),
  });

  const auto verify = [&](const auto&, const auto& clusters) {
    EXPECT_EQ(2, clusters.size());
    const auto [x, xScore] = nearest(clusters, {1, 0});
    const auto [y, yScore] = nearest(clusters, {0, 1});
    EXPECT_THAT(
        std::vector<int64_t>({x, y}), testing::UnorderedElementsAre(0, 1));
    EXPECT_FLOAT_EQ(1, xScore);
    EXPECT_FLOAT_EQ(1, yScore);
  };
  EXPECT_EQ(
      testTrain(data, /*numClusters=*/2, /*groupBy=*/false, verify),
      testTrain(reversed, /*numClusters=*/2, /*groupBy=*/false, verify));
}

// A group of more embeddings than one 64KB chunk of the aggregate's state
// holds gives the same codec whatever the order of its rows.
TEST_F(VectorIndexTrainTest, manyEmbeddings) {
  constexpr int32_t kSize{20'000};
  // Embedding i points near (1, 0) for even i and near (0, 1) for odd i.
  const auto makeData = [&](bool reverse) {
    const auto rowAt = [&](auto row) {
      return reverse ? kSize - 1 - row : row;
    };
    return makeRowVector({
        makeArrayVector<float>(
            kSize,
            [](auto) { return 2; },
            [&](auto index) {
              const auto i = rowAt(index / 2);
              const float small = 0.001f * (i % 7);
              return (index % 2 == 0) == (i % 2 == 0) ? 1.0f : small;
            }),
        makeFlatVector<int64_t>(kSize, rowAt),
    });
  };
  const auto verify = [&](const auto&, const auto& clusters) {
    EXPECT_EQ(2, clusters.size());
  };
  EXPECT_EQ(
      testTrain(makeData(false), /*numClusters=*/2, /*groupBy=*/false, verify),
      testTrain(makeData(true), /*numClusters=*/2, /*groupBy=*/false, verify));
}

TEST_F(VectorIndexTrainTest, fewerEmbeddingsThanClusters) {
  // A group of 2 embeddings gets 2 clusters, one at each.
  auto data = makeRowVector({
      makeArrayVector<float>({{1, 0}, {0, 1}}),
      makeFlatVector<int64_t>({10, 20}),
  });

  testTrain(
      data,
      /*numClusters=*/5,
      /*groupBy=*/false,
      [&](const auto&, const auto& clusters) {
        EXPECT_EQ(2, clusters.size());
        EXPECT_FLOAT_EQ(1, nearest(clusters, {1, 0}).second);
        EXPECT_FLOAT_EQ(1, nearest(clusters, {0, 1}).second);
      });
}

TEST_F(VectorIndexTrainTest, groupBy) {
  // Each group gets the clusters of its own embeddings. Rows with a null
  // embedding or id are ignored.
  auto data = makeRowVector({
      makeArrayVectorFromJson<float>({
          "[1, 0]",
          "[0, 1]",
          "null",
          "[0, 1]",
          "[1, 1]",
          "[1, 0]",
      }),
      makeNullableFlatVector<int64_t>({10, 20, 30, 40, std::nullopt, 60}),
      makeFlatVector<int64_t>({1, 2, 1, 2, 2, 1}),
  });

  testTrain(
      data,
      /*numClusters=*/4,
      /*groupBy=*/true,
      [&](const auto& key, const auto& clusters) {
        // Each group has 2 valid embeddings, so 2 clusters.
        EXPECT_EQ(2, clusters.size());
        const std::vector<float> direction =
            key.template value<TypeKind::BIGINT>() == 1
            ? std::vector<float>{1, 0}
            : std::vector<float>{0, 1};
        EXPECT_FLOAT_EQ(1, nearest(clusters, direction).second);
      });
}

// The codec options reach training through the intermediate results. Training
// reports the embeddings it trains on, its threads from the session property,
// its levels and its wall time.
TEST_F(VectorIndexTrainTest, codecOptions) {
  auto data = makeRowVector({
      makeArrayVector<float>({{1, 0}, {0, 1}, {3, 0}, {0, 5}}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });
  const auto train = [&](int64_t maxFlatClusters, int32_t numThreads) {
    core::PlanNodeId finalId;
    auto plan = PlanBuilder(pool())
                    .values({data})
                    .partialAggregation(
                        {},
                        {fmt::format(
                            "vector_index_train(c0, c1, 2, 'cosine', "
                            "map(array['train.maxFlatClusters'], array['{}']))",
                            maxFlatClusters)})
                    .finalAggregation()
                    .capturePlanNodeId(finalId)
                    .planNode();
    std::shared_ptr<exec::Task> task;
    const auto result =
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kVectorIndexTrainThreads, numThreads)
            .copyResults(pool(), task);
    EXPECT_EQ(
        2,
        FaissLocalVectorIndex(
            result->childAt(0)->variantAt(0).value<TypeKind::VARBINARY>(),
            VectorIndexMetric::kCosine,
            LocalVectorIndexOptions{})
            .size());
    return exec::PlanNodeStats::extract(task->taskStats(), finalId).customStats;
  };
  const auto stat = [](const auto& stats, std::string_view name) {
    return stats.at(std::string(name));
  };

  {
    const auto stats = train(/*maxFlatClusters=*/1, /*numThreads=*/2);
    EXPECT_EQ(4, stat(stats, TrainStats::kEmbeddings).sum);
    EXPECT_EQ(2, stat(stats, TrainStats::kThreads).max);
    EXPECT_EQ(2, stat(stats, TrainStats::kLevels).max);
    EXPECT_EQ(1, stat(stats, TrainStats::kWallNanos).count);
  }

  {
    const auto stats = train(/*maxFlatClusters=*/2, /*numThreads=*/1);
    EXPECT_EQ(1, stat(stats, TrainStats::kThreads).max);
    EXPECT_EQ(1, stat(stats, TrainStats::kLevels).max);
  }
}

TEST_F(VectorIndexTrainTest, errors) {
  auto data = makeRowVector({
      makeArrayVector<float>({{1, 0}, {0, 1}}),
      makeFlatVector<int64_t>({10, 20}),
      makeFlatVector<int64_t>({2, 2}),
      makeFlatVector<std::string>({"cosine", "cosine"}),
  });
  auto testFailingCall = [&](const std::string& call,
                             const std::string& expectedMessage) {
    SCOPED_TRACE(call);
    AggregationTester(*this).inputs(data).aggregates({call}).assertError(
        expectedMessage);
  };

  testFailingCall(
      "vector_index_train(c0, c1, 0, 'cosine', map()::map(varchar, varchar))",
      "Number of clusters must be positive");
  testFailingCall(
      "vector_index_train(c0, c1, 2, 'l2', map()::map(varchar, varchar))",
      "Unsupported vector index metric: l2");
  testFailingCall(
      "vector_index_train(c0, c1, c2, 'cosine', map()::map(varchar, varchar))",
      "Number of clusters must be a constant");
  testFailingCall(
      "vector_index_train(c0, c1, 2, c3, map()::map(varchar, varchar))",
      "Metric must be a constant");
  testFailingCall(
      "vector_index_train(c0, c1, null::bigint, 'cosine', map()::map(varchar, varchar))",
      "Number of clusters must not be null");
  testFailingCall(
      "vector_index_train(c0, c1, 2, null::varchar, map()::map(varchar, varchar))",
      "Metric must not be null");
  testFailingCall(
      "vector_index_train(c0, c1, 2, 'cosine', null::map(varchar, varchar))",
      "Options must not be null");
  testFailingCall(
      "vector_index_train(c0, c1, 2, 'cosine', map(array['ivf.nlist'], array['4']))",
      "Unknown vector index codec option: ivf.nlist");
  testFailingCall(
      "vector_index_train(c0, c1, 2, 'cosine', map(array['train.maxFlatClusters'], array['0']))",
      "Vector index codec option must be positive: train.maxFlatClusters");

  AggregationTester(*this)
      .inputs(data)
      .config(core::QueryConfig::kVectorIndexTrainThreads, "0")
      .aggregates({aggregate(2)})
      .assertError(
          "Session property must be positive: vector_index_train_threads");

  AggregationTester(*this)
      .inputs(makeRowVector({
          makeArrayVector<float>({{1, 0}, {0, 1, 0}}),
          makeFlatVector<int64_t>({10, 20}),
      }))
      .aggregates({aggregate(2)})
      .assertError("All embeddings must have the same length");
}

TEST_F(VectorIndexTrainTest, mergeExtractErrors) {
  // vector_index_train_merge_extract takes intermediate results directly:
  // metric, number of clusters, most clusters trained in one level, ids and
  // embeddings.
  auto testFailing =
      [&](const std::vector<std::optional<int32_t>>& metrics,
          const std::vector<std::optional<int64_t>>& numClusters,
          const std::vector<std::optional<int64_t>>& maxFlatClusters,
          const std::vector<std::string>& idsJson,
          const std::vector<std::string>& embeddingsJson,
          const std::string& expectedMessage) {
        SCOPED_TRACE(expectedMessage);
        AggregationTester(*this)
            .inputs(makeRowVector({makeRowVector({
                makeNullableFlatVector<int32_t>(metrics),
                makeNullableFlatVector<int64_t>(numClusters),
                makeNullableFlatVector<int64_t>(maxFlatClusters),
                makeArrayVectorFromJson<int64_t>(idsJson),
                makeArrayVectorFromJson<float>(embeddingsJson),
            })}))
            .aggregates({"vector_index_train_merge_extract(c0)"})
            .assertError(expectedMessage);
      };
  const std::vector<std::string> ids{"[10]", "[20]"};
  const std::vector<std::string> embeddings{"[1, 0]", "[0, 1]"};

  testFailing(
      {std::nullopt, 0},
      {2, 2},
      {5, 5},
      ids,
      embeddings,
      "Intermediate vector index training must include the metric");
  testFailing(
      {0, 0},
      {std::nullopt, 2},
      {5, 5},
      ids,
      embeddings,
      "Intermediate vector index training must include the number of clusters");
  testFailing(
      {0, 0},
      {2, 2},
      {std::nullopt, 5},
      ids,
      embeddings,
      "Intermediate vector index training must include the codec options");
  testFailing(
      {0, 0},
      {2, 2},
      {5, 5},
      {"[10]", "null"},
      embeddings,
      "Intermediate vector index training must include the ids");
  testFailing(
      {0, 0},
      {2, 2},
      {5, 5},
      ids,
      {"[1, 0]", "null"},
      "Intermediate vector index training must include the embeddings");
  testFailing(
      {0, 0},
      {2, 0},
      {5, 5},
      ids,
      embeddings,
      "Number of clusters must be positive");
  testFailing(
      {0, 0},
      {2, 2},
      {5, 0},
      ids,
      embeddings,
      "Vector index codec option must be positive: train.maxFlatClusters");
  testFailing(
      {0, 9},
      {2, 2},
      {5, 5},
      ids,
      embeddings,
      "Invalid vector index metric value: 9");
  testFailing(
      {0, 0},
      {2, 3},
      {5, 5},
      ids,
      embeddings,
      "Intermediate vector index trainings of a group must have the same settings");
  testFailing(
      {0, 0},
      {2, 2},
      {5, 6},
      ids,
      embeddings,
      "Intermediate vector index trainings of a group must have the same settings");
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql

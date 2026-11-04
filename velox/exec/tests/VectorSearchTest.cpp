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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <set>

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalFixedPoint.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
namespace {

using namespace facebook::velox::exec::test;
using common::testutil::TempDirectoryPath;

class VectorSearchTest : public HiveConnectorTestBase {
 protected:
  // The tables of a vector index: the codec table (codec, ds, codec_id), one
  // codec per partition, and the index table (ds, codec_id, cluster_id,
  // group_id, local_index), local indexes of at most 4 rows.
  struct VectorIndex {
    RowVectorPtr codecs;
    RowVectorPtr indexes;
  };

  // Returns a plan of spherical k-means over 'points' (ds, id, embedding),
  // which produces the centroids (ds, cluster_id, centroid). It uses
  // relational plans only, as a library of algorithms would provide it.
  //  - Seeds: the first point of each group. PALM seeds with random points
  //    instead, which can put two seeds in one group, split it across
  //    clusters and make a search that probes one cluster miss nearest
  //    points.
  //  - Iteration: assigns each point to the most similar centroid of its
  //    partition, then averages and normalizes each cluster's points. A
  //    cluster with no points keeps its centroid.
  //  - Stops when no centroid moves.
  PlanBuilder kMeans(const RowVectorPtr& points) {
    PlanBuilder plan(pool());
    // 'movement' is how far the centroid moved in the last iteration: one
    // minus the cosine similarity of its old and new positions.
    auto centroidsType = ROW({
        {"ds", VARCHAR()},
        {"cluster_id", BIGINT()},
        {"centroid", ARRAY(REAL())},
        {"movement", REAL()},
    });
    const auto normalize = [](const std::string& column) {
      return fmt::format(
          "transform({0}, x -> (x / l2_norm({0})::REAL))", column);
    };

    // The points are a state entry because the iteration's plan must start
    // with a state source.
    return plan
        .fixedPoint(
            {
                std::make_shared<core::VectorStateDeclaration>(
                    "points",
                    points->rowType(),
                    plan.newPlan()
                        .values({points})
                        .project({
                            "ds",
                            "id",
                            normalize("embedding") + " AS embedding",
                        })
                        .planNode()),
                std::make_shared<core::VectorStateDeclaration>(
                    "centroids",
                    centroidsType,
                    plan.newPlan()
                        .values({points})
                        .filter("id % 10 = 0")
                        .project({
                            "ds",
                            "id % 100 / 10 AS cluster_id",
                            normalize("embedding") + " AS centroid",
                            "0::REAL AS movement",
                        })
                        .planNode()),
            },
            plan.newPlan()
                .stateSource("points", points->rowType())
                .hashJoin(
                    {"ds"},
                    {"c_ds"},
                    plan.newPlan()
                        .stateSource("centroids", centroidsType)
                        .project({"ds AS c_ds", "cluster_id", "centroid"})
                        .planNode(),
                    /*filter=*/"",
                    {"ds", "id", "embedding", "cluster_id", "centroid"})
                .project({
                    "ds",
                    "id",
                    "embedding",
                    "cluster_id",
                    "cosine_similarity(embedding, centroid) AS score",
                })
                .singleAggregation(
                    {"ds", "id"},
                    {
                        "max_by(cluster_id, score) AS cluster_id",
                        "arbitrary(embedding) AS embedding",
                    })
                .singleAggregation(
                    {"ds", "cluster_id"}, {"vector_sum(embedding) AS sum"})
                .project({
                    "ds AS new_ds",
                    "cluster_id AS new_cluster_id",
                    normalize("sum") + " AS new_centroid",
                })
                // The RIGHT join keeps the clusters that drew no points.
                .hashJoin(
                    {"new_ds", "new_cluster_id"},
                    {"ds", "cluster_id"},
                    plan.newPlan()
                        .stateSource("centroids", centroidsType)
                        .project(
                            {"ds", "cluster_id", "centroid AS old_centroid"})
                        .planNode(),
                    /*filter=*/"",
                    {"ds", "cluster_id", "old_centroid", "new_centroid"},
                    core::JoinType::kRight)
                .project({
                    "ds",
                    "cluster_id",
                    "coalesce(new_centroid, old_centroid) AS centroid",
                    "coalesce(1::REAL - "
                    "cosine_similarity(old_centroid, new_centroid), "
                    "0::REAL) AS movement",
                }),
            core::ConvergenceConfig::converging(
                plan.newPlan()
                    .stateSource("centroids", centroidsType)
                    .singleAggregation({}, {"max(movement)"})
                    .project({"a0 <= 1e-6 AS converged"})
                    .planNode(),
                /*maxIterations=*/20),
            "centroids")
        .project({"ds", "cluster_id", "centroid"});
  }

  // Runs the plans of CREATE VECTOR INDEX ... AS SELECT over points given by
  // their partitions 'ds', 'ids' and 'embeddings'. Two writes:
  //  - the codec table: kMeans() with its centroids packed into a codec per
  //    partition, with a new codec id;
  //  - the index table: each point assigned to the nearest cluster of its
  //    partition's codec, then a local index per group of a cluster's points.
  VectorIndex createVectorIndex(
      const std::vector<std::string>& ds,
      const std::vector<int64_t>& ids,
      const std::vector<std::vector<float>>& embeddings) {
    const auto points = makeNamedRowVector({
        {"ds", makeFlatVector(ds)},
        {"id", makeFlatVector(ids)},
        {"embedding", makeArrayVector(embeddings)},
    });
    auto codecs = run(
        kMeans(points)
            .singleAggregation(
                {"ds"},
                {"vector_index_pack(cluster_id, centroid, 'cosine') AS codec"})
            .rowNumber({})
            .project({"codec", "ds", "row_number AS codec_id"}));

    PlanBuilder plan(pool());
    auto indexes =
        run(plan.values({points})
                .hashJoin(
                    {"ds"},
                    {"c_ds"},
                    plan.newPlan()
                        .values({codecs})
                        .project({"ds AS c_ds", "codec_id"})
                        .planNode(),
                    /*filter=*/"",
                    {"ds", "id", "embedding", "codec_id"})
                .vectorSearchJoin(
                    /*leftKeys=*/{"ds", "codec_id"},
                    /*rightKeys=*/{"c_ds", "c_codec_id"},
                    codecBuild(plan, codecs),
                    /*probeEmbedding=*/"embedding",
                    /*buildIndex=*/"codec",
                    /*k=*/1,
                    /*outputLayout=*/
                    {"ds", "codec_id", "id", "embedding", "cluster_id"},
                    /*id=*/"cluster_id")
                .rowNumber({"ds", "cluster_id"})
                .project(
                    {"ds",
                     "codec_id",
                     "cluster_id",
                     "(row_number - 1) / 4 AS group_id",
                     "id",
                     "embedding"})
                .singleAggregation(
                    {"ds", "codec_id", "cluster_id", "group_id"},
                    {
                        "vector_index_build(embedding, id, 'flat', 'cosine', "
                        "map()::map(varchar, varchar)) AS local_index",
                    }));
    return {std::move(codecs), std::move(indexes)};
  }

  // Returns the start of the plan of SELECT ... FROM vector_search(...) for
  // the query embeddings 'queries': assigns each query to its 'numProbes'
  // nearest clusters of each partition's codec. Produces (qid, query, ds,
  // codec_id, cluster_id), where 'qid' is the query's position in 'queries'.
  PlanBuilder assignQueries(
      const VectorIndex& index,
      const std::vector<std::vector<float>>& queries,
      int64_t numProbes) {
    const auto queryRows = makeNamedRowVector({
        {"qid", makeFlatIdentityVector<int64_t>(queries.size())},
        {"query", makeArrayVector(queries)},
    });

    PlanBuilder plan(pool());
    return plan.newPlan()
        .values({queryRows})
        .nestedLoopJoin(
            plan.newPlan()
                .values({index.codecs})
                .project({"ds", "codec_id"})
                .planNode(),
            {"qid", "query", "ds", "codec_id"})
        .vectorSearchJoin(
            /*leftKeys=*/{"ds", "codec_id"},
            /*rightKeys=*/{"c_ds", "c_codec_id"},
            codecBuild(plan, index.codecs),
            /*probeEmbedding=*/"query",
            /*buildIndex=*/"codec",
            numProbes,
            /*outputLayout=*/{"qid", "query", "ds", "codec_id", "cluster_id"},
            /*id=*/"cluster_id");
  }

  // Adds to 'probe', queries assigned to clusters, the search of those
  // clusters' local indexes in 'indexes', the index table. Produces (qid, id,
  // score), the 'k' nearest points per query of each local index.
  static PlanBuilder&
  searchLocalIndexes(PlanBuilder& probe, PlanBuilder& indexes, int64_t k) {
    return probe.vectorSearchJoin(
        /*leftKeys=*/{"ds", "codec_id", "cluster_id"},
        /*rightKeys=*/{"i_ds", "i_codec_id", "i_cluster_id"},
        indexes
            .project(
                {"local_index",
                 "ds AS i_ds",
                 "codec_id AS i_codec_id",
                 "cluster_id AS i_cluster_id"})
            .planNode(),
        /*probeEmbedding=*/"query",
        /*buildIndex=*/"local_index",
        k,
        /*outputLayout=*/{"qid", "id", "score"},
        /*id=*/"id",
        /*score=*/"score");
  }

  // Adds to 'builder', which produces (qid, id, score), keeping the 'k'
  // nearest points per query. Produces (qid, id).
  static PlanBuilder& keepNearest(PlanBuilder& builder, int64_t k) {
    return builder
        .topNRowNumber(
            /*partitionKeys=*/{"qid"},
            /*sortingKeys=*/{"score DESC"},
            k,
            /*generateRowNumber=*/false)
        .project({"qid", "id"});
  }

  // Runs the plan of SELECT ... FROM vector_search(...) for the query
  // embeddings 'queries': assigns each query to its 'numProbes' nearest
  // clusters of each partition's codec, searches those clusters' local
  // indexes, and returns (qid, id), the 'k' nearest points per query, where
  // 'qid' is the query's position in 'queries'.
  RowVectorPtr vectorSearch(
      const VectorIndex& index,
      const std::vector<std::vector<float>>& queries,
      int64_t k,
      int64_t numProbes) {
    auto plan = assignQueries(index, queries, numProbes);
    auto indexes = plan.newPlan().values({index.indexes});
    return run(keepNearest(searchLocalIndexes(plan, indexes, k), k));
  }

  // Points in two partitions, each of three groups of 10 unit-length points.
  // In partition p, the groups are around p * 60 + 0, 120 and 240 degrees,
  // and the ids of group g are p * 100 + g * 10 to p * 100 + g * 10 + 9.
  struct Points {
    std::vector<std::string> ds;
    std::vector<int64_t> ids;
    std::vector<std::vector<float>> embeddings;
  };

  static Points makePoints() {
    constexpr int kNumGroups = 3;
    constexpr int kGroupSize = 10;
    const std::vector<std::string> partitions{"2026-10-27", "2026-10-28"};
    Points points;
    for (int partition = 0; partition < partitions.size(); ++partition) {
      for (int group = 0; group < kNumGroups; ++group) {
        for (int i = 0; i < kGroupSize; ++i) {
          points.ds.push_back(partitions[partition]);
          points.ids.push_back(partition * 100 + group * kGroupSize + i);
          points.embeddings.push_back(
              unitVector(partition * 60 + group * 120 + (i - 4.5) * 3));
        }
      }
    }
    return points;
  }

  // Query embeddings near the middle of groups of both partitions.
  static std::vector<std::vector<float>> makeQueries() {
    return {unitVector(1), unitVector(61), unitVector(121), unitVector(241)};
  }

  // Returns (qid, id), the 'k' points nearest to each of 'queries' by
  // comparing with every point.
  RowVectorPtr nearestByBruteForce(
      const Points& points,
      const std::vector<std::vector<float>>& queries,
      int64_t k) {
    std::vector<int64_t> qids;
    std::vector<int64_t> ids;
    for (int64_t qid = 0; qid < queries.size(); ++qid) {
      std::vector<size_t> nearest(points.ids.size());
      std::iota(nearest.begin(), nearest.end(), 0);
      const auto similarity = [&](size_t row) {
        return points.embeddings[row][0] * queries[qid][0] +
            points.embeddings[row][1] * queries[qid][1];
      };
      std::partial_sort(
          nearest.begin(),
          nearest.begin() + k,
          nearest.end(),
          [&](size_t left, size_t right) {
            return similarity(left) > similarity(right);
          });
      for (int64_t i = 0; i < k; ++i) {
        qids.push_back(qid);
        ids.push_back(points.ids[nearest[i]]);
      }
    }
    return makeRowVector({makeFlatVector(qids), makeFlatVector(ids)});
  }

  RowVectorPtr run(const PlanBuilder& plan) {
    return AssertQueryBuilder(plan.planNode())
        .fixedPointOptions(&LocalFixedPoint::options())
        .copyResults(pool());
  }

 private:
  // The unit vector at 'degrees' degrees.
  static std::vector<float> unitVector(double degrees) {
    const double radians = degrees * M_PI / 180;
    return {float(std::cos(radians)), float(std::sin(radians))};
  }

  // The codec table as the build side of a vector search join on (c_ds,
  // c_codec_id) in 'plan'.
  static core::PlanNodePtr codecBuild(
      const PlanBuilder& plan,
      const RowVectorPtr& codecs) {
    return plan.newPlan()
        .values({codecs})
        .project({"codec", "ds AS c_ds", "codec_id AS c_codec_id"})
        .planNode();
  }
};

// Runs the plans of a vector index's SQL workflow, one statement at a time,
// with each table write materialized as a vector:
//
//   CREATE VECTOR INDEX product_index
//       (id BIGINT, embedding ARRAY(REAL), ds VARCHAR)
//   WITH (distance_metric = 'cosine', index_type = 'flat',
//         partitioned_by = ARRAY['ds'])
//   AS SELECT id, embedding, ds FROM points;
//
//   SELECT * FROM TABLE(vector_search(
//       queries => TABLE(SELECT id, embedding FROM queries),
//       index => TABLE(SELECT * FROM product_index),
//       k => 3,
//       num_probes => 1));
//
// The search must return what a brute-force comparison with every point
// returns.
TEST_F(VectorSearchTest, endToEnd) {
  const auto points = makePoints();
  const auto queries = makeQueries();
  const auto index =
      createVectorIndex(points.ds, points.ids, points.embeddings);
  assertEqualResults(
      {nearestByBruteForce(points, queries, /*k=*/3)},
      {vectorSearch(index, queries, /*k=*/3, /*numProbes=*/1)});
}

// Runs the search of local indexes as a bucketed join. The queries assigned
// to clusters and the index table are both bucketed on cluster_id, and each
// bucket runs as a split group of grouped execution, which builds and probes
// only that bucket's local indexes.
TEST_F(VectorSearchTest, bucketedSearch) {
  constexpr int kNumBuckets = 2;
  const auto points = makePoints();
  const auto queries = makeQueries();
  const auto index =
      createVectorIndex(points.ds, points.ids, points.embeddings);

  const auto assigned = run(assignQueries(index, queries, /*numProbes=*/1));

  // Writes 'rows' to a table bucketed on cluster_id and returns a split per
  // file in the split group of its bucket. A bucket's file name starts with
  // its number.
  std::vector<std::shared_ptr<TempDirectoryPath>> tables;
  const auto writeBucketed = [&](const RowVectorPtr& rows) {
    tables.push_back(TempDirectoryPath::create());
    const auto& directory = tables.back()->getPath();
    run(PlanBuilder(pool())
            .values({rows})
            .startTableWriter()
            .outputDirectoryPath(directory)
            .bucketCount(kNumBuckets)
            .bucketedBy({"cluster_id"})
            .endTableWriter());
    std::vector<Split> splits;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
      splits.emplace_back(
          makeHiveConnectorSplit(file.path().string()),
          std::stoi(file.path().filename().string()));
    }
    return splits;
  };

  core::PlanNodeId queriesScanId;
  core::PlanNodeId indexesScanId;
  PlanBuilder probe(pool());
  probe.tableScan(assigned->rowType()).capturePlanNodeId(queriesScanId);
  auto indexes = probe.newPlan()
                     .tableScan(index.indexes->rowType())
                     .capturePlanNodeId(indexesScanId);
  const auto queriesSplits = writeBucketed(assigned);
  std::set<int32_t> groups;
  for (const auto& split : queriesSplits) {
    groups.insert(split.groupId);
  }
  ASSERT_EQ(kNumBuckets, groups.size());

  const auto matches =
      AssertQueryBuilder(searchLocalIndexes(probe, indexes, /*k=*/3).planNode())
          .splits(queriesScanId, queriesSplits)
          .splits(indexesScanId, writeBucketed(index.indexes))
          .executionStrategy(core::ExecutionStrategy::kGrouped)
          .groupedExecutionLeafNodeIds({queriesScanId, indexesScanId})
          .numSplitGroups(kNumBuckets)
          .numConcurrentSplitGroups(1)
          .copyResults(pool());

  auto nearest = PlanBuilder(pool()).values({matches});
  assertEqualResults(
      {nearestByBruteForce(points, queries, /*k=*/3)},
      {run(keepNearest(nearest, /*k=*/3))});
}

} // namespace
} // namespace facebook::velox::exec

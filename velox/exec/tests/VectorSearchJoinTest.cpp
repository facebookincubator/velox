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

#include <atomic>
#include <numeric>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/vectorindex/FaissLocalVectorIndex.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
namespace {

using namespace facebook::velox::exec::test;

using common::vectorindex::Embeddings;
using common::vectorindex::FaissLocalVectorIndex;
using common::vectorindex::LocalVectorIndex;
using common::vectorindex::LocalVectorIndexFormat;
using common::vectorindex::LocalVectorIndexFormats;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::LocalVectorIndexReader;
using common::vectorindex::VectorIndexKind;
using common::vectorindex::VectorIndexMetric;

class VectorSearchJoinTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto formats = std::make_shared<LocalVectorIndexFormats>();
    formats->insert(
        std::string(FaissLocalVectorIndex::kFormat),
        FaissLocalVectorIndex::format());
    queryCtx_ = core::QueryCtx::create(driverExecutor_.get());
    queryCtx_->setRegistry(LocalVectorIndexFormat::kRegistryKey, formats);
  }

  void TearDown() override {
    queryCtx_.reset();
    OperatorTestBase::TearDown();
  }

  // Returns a builder that runs 'plan' with the query's local vector index
  // formats.
  AssertQueryBuilder queryBuilder(const core::PlanNodePtr& plan) const {
    AssertQueryBuilder builder(plan);
    builder.queryCtx(queryCtx_);
    return builder;
  }

  // Packs 2-dimensional unit-length centroids given in cluster-id order into a
  // codec, as vector_index_train returns it.
  static std::string pack(const std::vector<float>& centroids) {
    std::vector<int64_t> clusterIds(centroids.size() / 2);
    std::iota(clusterIds.begin(), clusterIds.end(), 0);
    return build(centroids, clusterIds, VectorIndexKind::kFlat);
  }

  // Builds a local index over unit-length embeddings of 'dimension' values.
  static std::string build(
      const std::vector<float>& embeddings,
      const std::vector<int64_t>& ids,
      VectorIndexKind kind = VectorIndexKind::kIvfFlat,
      int32_t dimension = 2) {
    std::string bytes;
    FaissLocalVectorIndex::build(
        kind,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings, dimension),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    return bytes;
  }

  // The join that preserves the query rows: left with the indexes on the
  // build side, right with them on the probe side.
  static core::JoinType outerJoinType(bool indexOnProbe) {
    return indexOnProbe ? core::JoinType::kRight : core::JoinType::kLeft;
  }

  // Probe rows (t0, ds, embedding) and codec rows (c_ds, codec).
  core::PlanNodePtr makeCodecPlan(
      const RowVectorPtr& probe,
      const std::vector<RowVectorPtr>& codecs,
      core::JoinType joinType = core::JoinType::kInner,
      const std::vector<std::string>& outputLayout = {
          "t0",
          "cluster_id",
          "score"}) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({probe})
        .vectorSearchJoin(
            {"ds"},
            {"c_ds"},
            PlanBuilder(planNodeIdGenerator).values(codecs).planNode(),
            "embedding",
            "codec",
            /*k=*/2,
            outputLayout,
            "cluster_id",
            "score",
            joinType)
        .planNode();
  }

  // Partition 1 has clusters (1, 0), (0.6, 0.8) and (0, 1); partition 2 has
  // one cluster, (0, 1). The rows with a null partition or a null codec match
  // nothing.
  std::vector<RowVectorPtr> makeCodecs() {
    return {
        makeRowVector(
            {"c_ds", "codec"},
            {
                makeNullableFlatVector<int64_t>({1, std::nullopt}),
                makeFlatVector<std::string>(
                    {pack({1, 0, 0.6, 0.8, 0, 1}), pack({1, 0})}, VARBINARY()),
            }),
        makeRowVector(
            {"c_ds", "codec"},
            {
                makeFlatVector<int64_t>({2, 3}),
                makeNullableFlatVector<std::string>(
                    {pack({0, 1}), std::nullopt}, VARBINARY()),
            }),
    };
  }

  // Row 1's embedding is normalized before the search. Partition 2 has fewer
  // clusters than k. Partition 3 has no codec.
  RowVectorPtr makeCodecProbe() {
    return makeRowVector(
        {"t0", "ds", "embedding"},
        {
            makeFlatVector<int64_t>({0, 1, 2, 3}),
            makeFlatVector<int64_t>({1, 1, 2, 3}),
            makeArrayVector<float>({{1, 0}, {0, 2}, {1, 0}, {1, 0}}),
        });
  }

  // The results of joining makeCodecProbe() with makeCodecs().
  RowVectorPtr makeCodecResults() {
    return makeRowVector({
        makeFlatVector<int64_t>({0, 0, 1, 1, 2}),
        makeFlatVector<int64_t>({0, 1, 2, 1, 0}),
        makeFlatVector<float>({1, 0.6, 1, 0.8, 0}),
    });
  }

  // Joins index rows (i_cluster_id, local_index, ...) and query rows (t0,
  // cluster_id, embedding) on the cluster, with the id of each embedding found
  // as 'id'. With 'indexOnProbe', the index rows are on the probe side. With
  // 'parallelProbe' or 'parallelBuild', each driver of that side reads all its
  // rows.
  static core::PlanNodePtr makeLocalIndexPlan(
      const RowVectorPtr& indexes,
      const RowVectorPtr& queries,
      bool indexOnProbe,
      core::JoinType joinType,
      int64_t k,
      const std::vector<std::string>& outputLayout,
      bool parallelProbe,
      bool parallelBuild,
      bool byCluster) {
    std::vector<std::string> leftKeys;
    std::vector<std::string> rightKeys;
    if (byCluster) {
      leftKeys = {indexOnProbe ? "i_cluster_id" : "cluster_id"};
      rightKeys = {indexOnProbe ? "cluster_id" : "i_cluster_id"};
    }
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({indexOnProbe ? indexes : queries}, parallelProbe)
        .vectorSearchJoin(
            leftKeys,
            rightKeys,
            PlanBuilder(planNodeIdGenerator)
                .values({indexOnProbe ? queries : indexes}, parallelBuild)
                .planNode(),
            "embedding",
            "local_index",
            k,
            outputLayout,
            "id",
            /*score=*/std::nullopt,
            joinType,
            {{std::string(LocalVectorIndexOptions::kNprobe), "1"}})
        .planNode();
  }

  // The context of the test's queries, which read the "faiss" format.
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

TEST_F(VectorSearchJoinTest, codec) {
  queryBuilder(makeCodecPlan(makeCodecProbe(), makeCodecs()))
      .assertResults(makeCodecResults());
}

TEST_F(VectorSearchJoinTest, driversAndBatches) {
  // The build rows come from several build drivers, and the results fill
  // several output batches.
  queryBuilder(makeCodecPlan(makeCodecProbe(), makeCodecs()))
      .maxDrivers(4)
      .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
      .assertResults(makeCodecResults());
}

TEST_F(VectorSearchJoinTest, noKeys) {
  auto probe = makeRowVector(
      {"t0", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeArrayVector<float>({{1, 0}, {0, 1}}),
      });
  auto codecs = makeRowVector(
      {"codec"},
      {makeFlatVector<std::string>({pack({1, 0, 0, 1})}, VARBINARY())});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .vectorSearchJoin(
              /*leftKeys=*/{},
              /*rightKeys=*/{},
              PlanBuilder(planNodeIdGenerator).values({codecs}).planNode(),
              "embedding",
              "codec",
              /*k=*/1,
              {"t0", "cluster_id"},
              "cluster_id")
          .planNode();

  queryBuilder(plan).assertResults(makeRowVector({
      makeFlatVector<int64_t>({0, 1}),
      makeFlatVector<int64_t>({0, 1}),
  }));

  // Every query searches every local index, on either side of the join. The
  // last query has a null embedding.
  auto indexes = makeRowVector(
      {"i_cluster_id", "local_index"},
      {
          makeFlatVector<int64_t>({7, 8}),
          makeFlatVector<std::string>(
              {build({1, 0}, {10}), build({0, 1}, {20})}, VARBINARY()),
      });
  auto queries = makeRowVector(
      {"t0", "cluster_id", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1, 2}),
          makeFlatVector<int64_t>({7, 8, 9}),
          makeNullableArrayVector<float>({{{1, 0}}, {{0, 1}}, std::nullopt}),
      });
  // Repeats each embedding found 'numCopies' times, and for an outer join, the
  // query that finds none 'numNullCopies' times.
  const auto makeResults = [&](int32_t numCopies, int32_t numNullCopies) {
    std::vector<std::optional<int64_t>> ids;
    std::vector<int64_t> rows;
    for (const int64_t row : {0, 1}) {
      for (const int64_t id : {10, 20}) {
        for (int32_t i = 0; i < numCopies; ++i) {
          ids.emplace_back(id);
          rows.push_back(row);
        }
      }
    }
    for (int32_t i = 0; i < numNullCopies; ++i) {
      ids.emplace_back(std::nullopt);
      rows.push_back(2);
    }
    return makeRowVector({
        makeNullableFlatVector<int64_t>(ids),
        makeFlatVector<int64_t>(rows),
    });
  };

  for (const bool indexOnProbe : {false, true}) {
    for (const bool isOuterJoin : {false, true}) {
      const auto joinType =
          isOuterJoin ? outerJoinType(indexOnProbe) : core::JoinType::kInner;
      SCOPED_TRACE(
          fmt::format(
              "{} join, index on {}",
              core::JoinTypeName::toName(joinType),
              indexOnProbe ? "probe" : "build"));
      const auto makePlan = [&](bool parallelBuild) {
        return makeLocalIndexPlan(
            indexes,
            queries,
            indexOnProbe,
            joinType,
            /*k=*/1,
            {"id", "t0"},
            /*parallelProbe=*/false,
            parallelBuild,
            /*byCluster=*/false);
      };

      queryBuilder(makePlan(false))
          .assertResults(makeResults(1, isOuterJoin ? 1 : 0));
      // Each search takes one query and the matches are listed one at a time.
      queryBuilder(makePlan(false))
          .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
          .assertResults(makeResults(1, isOuterJoin ? 1 : 0));
      // Each of 3 build drivers reads all build rows. Repeated query rows on
      // the build side repeat the query that finds nothing too.
      const int32_t numNullCopies = isOuterJoin ? (indexOnProbe ? 3 : 1) : 0;
      queryBuilder(makePlan(true))
          .maxDrivers(3)
          .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
          .assertResults(makeResults(3, numNullCopies));
    }
  }
}

TEST_F(VectorSearchJoinTest, probeEncodings) {
  // A constant null embedding matches nothing.
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({1, 1}),
          BaseVector::createNullConstant(ARRAY(REAL()), 2, pool()),
      });
  queryBuilder(makeCodecPlan(probe, makeCodecs(), core::JoinType::kLeft))
      .assertResults(makeRowVector({
          makeFlatVector<int64_t>({0, 1}),
          makeNullConstant(TypeKind::BIGINT, 2),
          makeNullConstant(TypeKind::REAL, 2),
      }));

  // A dictionary over the embeddings.
  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({1, 1}),
          wrapInDictionary(
              makeIndices({1, 0}), makeArrayVector<float>({{1, 0}, {0, 1}})),
      });
  queryBuilder(makeCodecPlan(probe, makeCodecs()))
      .assertResults(makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({2, 1, 0, 1}),
          makeFlatVector<float>({1, 0.8, 1, 0.6}),
      }));

  // The same embeddings with a dictionary over the elements.
  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({1, 1}),
          makeArrayVector(
              {0, 2},
              wrapInDictionary(
                  makeIndices({2, 3, 0, 1}),
                  makeFlatVector<float>({1, 0, 0, 1}))),
      });
  queryBuilder(makeCodecPlan(probe, makeCodecs()))
      .assertResults(makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({2, 1, 0, 1}),
          makeFlatVector<float>({1, 0.8, 1, 0.6}),
      }));
}

TEST_F(VectorSearchJoinTest, leftJoin) {
  // Row 1 has a null embedding, row 2 a null key and row 3 no codec.
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<int64_t>({2, 1, std::nullopt, 3}),
          makeNullableArrayVector<float>(
              {{{1, 0}}, std::nullopt, {{1, 0}}, {{1, 0}}}),
      });

  queryBuilder(makeCodecPlan(probe, makeCodecs(), core::JoinType::kLeft))
      .assertResults(makeRowVector({
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {0, std::nullopt, std::nullopt, std::nullopt}),
          makeNullableFlatVector<float>(
              {0, std::nullopt, std::nullopt, std::nullopt}),
      }));
}

TEST_F(VectorSearchJoinTest, emptyBuild) {
  // Every build row has a null key or a null codec.
  auto codecs = makeRowVector(
      {"c_ds", "codec"},
      {
          makeNullableFlatVector<int64_t>({std::nullopt, 1}),
          makeNullableFlatVector<std::string>(
              {pack({1, 0}), std::nullopt}, VARBINARY()),
      });
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({1, 1}),
          makeArrayVector<float>({{1, 0}, {0, 1}}),
      });
  const std::vector<std::string> outputLayout{
      "t0", "c_ds", "cluster_id", "score"};

  queryBuilder(
      makeCodecPlan(probe, {codecs}, core::JoinType::kInner, outputLayout))
      .assertEmptyResults();

  queryBuilder(
      makeCodecPlan(probe, {codecs}, core::JoinType::kLeft, outputLayout))
      .assertResults(makeRowVector({
          makeFlatVector<int64_t>({0, 1}),
          makeNullConstant(TypeKind::BIGINT, 2),
          makeNullConstant(TypeKind::BIGINT, 2),
          makeNullConstant(TypeKind::REAL, 2),
      }));
}

// The index rows give the same results on either side of the join.
TEST_F(VectorSearchJoinTest, localIndexes) {
  // Cluster 7 has two local indexes, cluster 8 one, and cluster 9 one with a
  // null index. The last row has a null cluster. Each search finds up to k
  // entries in each local index of the cluster.
  auto indexes = makeRowVector(
      {"i_cluster_id", "local_index", "group_id"},
      {
          makeNullableFlatVector<int64_t>({7, 7, 8, 9, std::nullopt}),
          makeNullableFlatVector<std::string>(
              {build({1, 0, 0, 1}, {10, 20}),
               build({0.6, 0.8}, {30}),
               build({1, 0}, {40}),
               std::nullopt,
               build({1, 0}, {50})},
              VARBINARY()),
          makeFlatVector<int64_t>({0, 1, 0, 0, 0}),
      });
  // Rows 0 and 4 search cluster 7. Row 2 has a null embedding and row 3 a null
  // cluster.
  auto queries = makeRowVector(
      {"t0", "cluster_id", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
          makeNullableFlatVector<int64_t>({7, 8, 9, std::nullopt, 7}),
          makeNullableArrayVector<float>(
              {{{1, 0}}, {{0, 1}}, std::nullopt, {{1, 0}}, {{0, 1}}}),
      });
  const auto innerResults = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 20, 10, 30}),
      makeFlatVector<int64_t>({0, 0, 0, 1, 4, 4, 4}),
      makeFlatVector<int64_t>({0, 0, 1, 0, 0, 0, 1}),
  });
  // An outer join also outputs the queries that find nothing.
  const auto outerResults = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {10, 20, 30, 40, 20, 10, 30, std::nullopt, std::nullopt}),
      makeFlatVector<int64_t>({0, 0, 0, 1, 4, 4, 4, 2, 3}),
      makeNullableFlatVector<int64_t>(
          {0, 0, 1, 0, 0, 0, 1, std::nullopt, std::nullopt}),
  });

  for (const bool indexOnProbe : {false, true}) {
    for (const bool isOuterJoin : {false, true}) {
      const auto joinType =
          isOuterJoin ? outerJoinType(indexOnProbe) : core::JoinType::kInner;
      // A k larger than every index finds all of its entries.
      for (const int64_t k :
           {int64_t{2}, std::numeric_limits<int64_t>::max()}) {
        SCOPED_TRACE(
            fmt::format(
                "{} join, index on {}, k: {}",
                core::JoinTypeName::toName(joinType),
                indexOnProbe ? "probe" : "build",
                k));
        const auto plan = makeLocalIndexPlan(
            indexes,
            queries,
            indexOnProbe,
            joinType,
            k,
            {"id", "t0", "group_id"},
            /*parallelProbe=*/false,
            /*parallelBuild=*/false,
            /*byCluster=*/true);
        const auto& expected = isOuterJoin ? outerResults : innerResults;

        queryBuilder(plan).assertResults(expected);
        // Each search takes one query, so an index is searched in several
        // steps, and the results fill several output batches.
        queryBuilder(plan)
            .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
            .assertResults(expected);
      }
    }
  }
}

// Several drivers on either side give the same results: the build drivers'
// tables are merged, and with the query rows on the build side, a right join
// outputs a query row that no probe driver finds embeddings for once.
TEST_F(VectorSearchJoinTest, drivers) {
  auto indexes = makeRowVector(
      {"i_cluster_id", "local_index"},
      {
          makeFlatVector<int64_t>({7, 8}),
          makeFlatVector<std::string>(
              {build({1, 0}, {10}), build({0, 1}, {20})}, VARBINARY()),
      });
  // Row 2 searches a cluster with no index and row 3 has a null embedding.
  auto queries = makeRowVector(
      {"t0", "cluster_id", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeFlatVector<int64_t>({7, 8, 9, 7}),
          makeNullableArrayVector<float>(
              {{{1, 0}}, {{0, 1}}, {{1, 0}}, std::nullopt}),
      });
  // Each of 3 drivers of one side reads all its rows. Repeated query rows
  // repeat every result; repeated index rows repeat only the embeddings found.
  const auto repeatedQueries = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {10,
           10,
           10,
           20,
           20,
           20,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt}),
  });
  const auto repeatedIndexes = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 1, 2, 3}),
      makeNullableFlatVector<int64_t>(
          {10, 10, 10, 20, 20, 20, std::nullopt, std::nullopt}),
  });

  for (const bool indexOnProbe : {false, true}) {
    for (const bool parallelProbe : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "index on {}, parallel {}",
              indexOnProbe ? "probe" : "build",
              parallelProbe ? "probe" : "build"));
      const auto plan = makeLocalIndexPlan(
          indexes,
          queries,
          indexOnProbe,
          outerJoinType(indexOnProbe),
          /*k=*/1,
          {"t0", "id"},
          parallelProbe,
          /*parallelBuild=*/!parallelProbe,
          /*byCluster=*/true);
      const bool queriesRepeated = indexOnProbe != parallelProbe;
      queryBuilder(plan).maxDrivers(3).assertResults(
          queriesRepeated ? repeatedQueries : repeatedIndexes);
    }
  }
}

// A search reads its indexes with the format it names, among the query's
// formats. The format checks the search's settings before reading any index. A
// format the query does not have fails.
TEST_F(VectorSearchJoinTest, format) {
  // Reads indexes as "faiss" does and counts them.
  class CountingReader : public LocalVectorIndexReader {
   public:
    CountingReader(
        std::unique_ptr<LocalVectorIndexReader> faiss,
        std::atomic<int32_t>& numOpened)
        : faiss_{std::move(faiss)}, numOpened_{numOpened} {}

    std::unique_ptr<LocalVectorIndex> open(
        std::string_view bytes) const override {
      ++numOpened_;
      return faiss_->open(bytes);
    }

   private:
    const std::unique_ptr<LocalVectorIndexReader> faiss_;
    std::atomic<int32_t>& numOpened_;
  };

  class CountingFormat : public LocalVectorIndexFormat {
   public:
    std::unique_ptr<LocalVectorIndexReader> makeReader(
        VectorIndexMetric metric,
        const LocalVectorIndex::SearchOptions& searchOptions) const override {
      return std::make_unique<CountingReader>(
          FaissLocalVectorIndex::format()->makeReader(metric, searchOptions),
          numOpened);
    }

    mutable std::atomic<int32_t> numOpened{0};
  };
  const auto counting = std::make_shared<CountingFormat>();
  queryCtx_
      ->registry<LocalVectorIndexFormats>(LocalVectorIndexFormat::kRegistryKey)
      ->insert("counting", counting);

  const auto makePlan = [&](const std::string& format,
                            LocalVectorIndex::SearchOptions searchOptions) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({makeCodecProbe()})
        .vectorSearchJoin(
            {"ds"},
            {"c_ds"},
            PlanBuilder(planNodeIdGenerator).values(makeCodecs()).planNode(),
            "embedding",
            "codec",
            /*k=*/2,
            {"t0", "cluster_id", "score"},
            "cluster_id",
            "score",
            core::JoinType::kInner,
            std::move(searchOptions),
            VectorIndexMetric::kCosine,
            format)
        .planNode();
  };

  // The codecs of partitions 1 and 2 can match; the others have a null
  // partition or a null codec.
  const auto expected = queryBuilder(makePlan("faiss", {})).copyResults(pool());
  queryBuilder(makePlan("counting", {})).assertResults(expected);
  EXPECT_EQ(2, counting->numOpened);

  VELOX_ASSERT_USER_THROW(
      queryBuilder(makePlan("counting", {{"ivf.nlist", "4"}}))
          .copyResults(pool()),
      "Vector index option is not a search setting: ivf.nlist");
  EXPECT_EQ(2, counting->numOpened);

  VELOX_ASSERT_USER_THROW(
      queryBuilder(makePlan("no_such_format", {})).copyResults(pool()),
      "Local vector index format is not registered: no_such_format");
}

TEST_F(VectorSearchJoinTest, errors) {
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{1, 0, 0}}),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, makeCodecs())).copyResults(pool()),
      "Query embedding length must match the vector index's");

  // A query that matches no index is checked too: partition 3 has no codec.
  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({3}),
          makeArrayVector<float>({{1, 0, 0}}),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, makeCodecs())).copyResults(pool()),
      "Query embedding length must match the vector index's");

  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({1, 3}),
          makeArrayVector<float>({{1, 0}, {1, 0, 0}}),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, makeCodecs())).copyResults(pool()),
      "Query embeddings must have the same length");

  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeNullableArrayVector<float>(
              std::vector<std::optional<std::vector<std::optional<float>>>>{
                  {{1, std::nullopt}}}),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, makeCodecs())).copyResults(pool()),
      "Query embedding must not contain nulls");

  // A null element under a dictionary.
  const auto nullUnderDictionary = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector(
              {0},
              wrapInDictionary(
                  makeIndices({0, 1}),
                  makeNullableFlatVector<float>({1, std::nullopt}))),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(nullUnderDictionary, makeCodecs()))
          .copyResults(pool()),
      "Query embedding must not contain nulls");

  for (const float value :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity()}) {
    const auto nonFinite = makeRowVector(
        {"t0", "ds", "embedding"},
        {
            makeFlatVector<int64_t>({0}),
            makeFlatVector<int64_t>({1}),
            makeArrayVector<float>({{1, value}}),
        });
    VELOX_ASSERT_THROW(
        queryBuilder(makeCodecPlan(nonFinite, makeCodecs()))
            .copyResults(pool()),
        "Query embedding must not contain NaN or infinity");
  }

  auto codecs = makeRowVector(
      {"c_ds", "codec"},
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<std::string>({"not a codec"}, VARBINARY()),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, {codecs})).copyResults(pool()),
      "Invalid local vector index");

  codecs = makeRowVector(
      {"c_ds", "codec"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<std::string>(
              {pack({1, 0}), build({1, 0, 0}, {0}, VectorIndexKind::kFlat, 3)},
              VARBINARY()),
      });
  VELOX_ASSERT_THROW(
      queryBuilder(makeCodecPlan(probe, {codecs})).copyResults(pool()),
      "Vector search join indexes must have the same dimension");

  // With the query rows on the build side, they must all have the same length.
  const auto makeIndexOnProbePlan = [&](const RowVectorPtr& queries) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({codecs})
        .vectorSearchJoin(
            {"c_ds"},
            {"ds"},
            PlanBuilder(planNodeIdGenerator).values({queries}).planNode(),
            "embedding",
            "codec",
            /*k=*/1,
            {"t0", "cluster_id"},
            "cluster_id")
        .planNode();
  };
  VELOX_ASSERT_THROW(
      queryBuilder(makeIndexOnProbePlan(makeRowVector(
                       {"t0", "ds", "embedding"},
                       {
                           makeFlatVector<int64_t>({0, 1}),
                           makeFlatVector<int64_t>({1, 2}),
                           makeArrayVector<float>({{1, 0}, {1, 0, 0}}),
                       })))
          .copyResults(pool()),
      "Query embeddings must have the same length");

  // A query of another length than an index it matches fails.
  VELOX_ASSERT_THROW(
      queryBuilder(makeIndexOnProbePlan(makeRowVector(
                       {"t0", "ds", "embedding"},
                       {
                           makeFlatVector<int64_t>({0}),
                           makeFlatVector<int64_t>({1}),
                           makeArrayVector<float>({{1, 0, 0}}),
                       })))
          .copyResults(pool()),
      "Query embedding length must match the vector index's");
}

TEST_F(VectorSearchJoinTest, planValidation) {
  auto probe = makeRowVector(
      {"t0", "embedding", "doubles"},
      {
          makeFlatVector<int64_t>({0}),
          makeArrayVector<float>({{1, 0}}),
          makeArrayVector<double>({{1, 0}}),
      });
  auto codecs = makeRowVector(
      {"c0", "codec"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<std::string>({pack({1, 0})}, VARBINARY()),
      });

  struct Case {
    std::vector<std::string> leftKeys;
    std::string embedding;
    std::string index;
    int64_t k;
    std::vector<std::string> outputLayout;
    core::JoinType joinType;
    std::string error;
  };
  const std::vector<Case> cases = {
      {{"t0"},
       "embedding",
       "codec",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kFull,
       "The join type is not supported by vector search join: FULL"},
      {{"t0"},
       "embedding",
       "codec",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kRight,
       "Vector search join may not preserve the index side: RIGHT"},
      {{},
       "embedding",
       "codec",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kInner,
       "JoinNode requires same number of join keys on left and right sides"},
      {{"t0"},
       "doubles",
       "codec",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kInner,
       "Embedding must be ARRAY(REAL)"},
      {{"t0"},
       "embedding",
       "c0",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kInner,
       "Index must be VARBINARY"},
      {{"t0"},
       "embedding",
       "codec",
       0,
       {"t0", "cluster_id"},
       core::JoinType::kInner,
       "Vector search join requires k of at least 1"},
      {{"t0"},
       "embedding",
       "codec",
       1,
       {"t0", "codec", "cluster_id"},
       core::JoinType::kInner,
       "Vector search join output may not include the index"},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.error);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    VELOX_ASSERT_THROW(
        PlanBuilder(planNodeIdGenerator)
            .values({probe})
            .vectorSearchJoin(
                testCase.leftKeys,
                {"c0"},
                PlanBuilder(planNodeIdGenerator).values({codecs}).planNode(),
                testCase.embedding,
                testCase.index,
                testCase.k,
                testCase.outputLayout,
                "cluster_id",
                /*score=*/std::nullopt,
                testCase.joinType),
        testCase.error);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  VELOX_ASSERT_THROW(
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .vectorSearchJoin(
              {"t0"},
              {"c0"},
              PlanBuilder(planNodeIdGenerator).values({codecs}).planNode(),
              "embedding",
              "codec",
              /*k=*/1,
              {"t0"},
              /*id=*/"t0"),
      "Vector search join id or score name is also a left side column: t0");

  VELOX_ASSERT_THROW(
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .vectorSearchJoin(
              {"t0"},
              {"c0"},
              PlanBuilder(planNodeIdGenerator).values({codecs}).planNode(),
              "embedding",
              "codec",
              /*k=*/1,
              {"t0"},
              /*id=*/"cluster_id"),
      "Vector search join id or score is not an output column: cluster_id");
}

} // namespace
} // namespace facebook::velox::exec

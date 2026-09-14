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
  // codec, as vector_index_pack does.
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
      .config(core::QueryConfig::kMaxOutputBatchRows, "1")
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

TEST_F(VectorSearchJoinTest, localIndexes) {
  // Cluster 7 has two local indexes, cluster 8 one. Each search finds up to k
  // entries in each local index of the cluster.
  auto indexes = makeRowVector(
      {"i_cluster_id", "local_index", "group_id"},
      {
          makeFlatVector<int64_t>({7, 7, 8}),
          makeFlatVector<std::string>(
              {build({1, 0, 0, 1}, {10, 20}),
               build({0.6, 0.8}, {30}),
               build({1, 0}, {40})},
              VARBINARY()),
          makeFlatVector<int64_t>({0, 1, 0}),
      });
  auto probe = makeRowVector(
      {"t0", "cluster_id", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({7, 8}),
          makeArrayVector<float>({{1, 0}, {0, 1}}),
      });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .vectorSearchJoin(
              {"cluster_id"},
              {"i_cluster_id"},
              PlanBuilder(planNodeIdGenerator).values({indexes}).planNode(),
              "embedding",
              "local_index",
              /*k=*/2,
              {"id", "t0", "group_id"},
              "id",
              /*score=*/std::nullopt,
              core::JoinType::kInner,
              {{std::string(LocalVectorIndexOptions::kNprobe), "1"}})
          .planNode();

  queryBuilder(plan).assertResults(makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeFlatVector<int64_t>({0, 0, 0, 1}),
      makeFlatVector<int64_t>({0, 0, 1, 0}),
  }));
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
    std::string probeEmbedding;
    std::string buildIndex;
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
       "Probe embedding must be ARRAY(REAL)"},
      {{"t0"},
       "embedding",
       "c0",
       1,
       {"t0", "cluster_id"},
       core::JoinType::kInner,
       "Build index must be VARBINARY"},
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
       "Vector search join output may not include the build index"},
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
                testCase.probeEmbedding,
                testCase.buildIndex,
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

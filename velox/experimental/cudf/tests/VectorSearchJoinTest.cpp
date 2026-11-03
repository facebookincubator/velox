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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/vectorindex/GpuLocalVectorIndex.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/vector/DecodedVector.h"

#include <folly/ScopeGuard.h>
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <numeric>
#include <random>

namespace facebook::velox::cudf_velox {
namespace {

using namespace facebook::velox::exec::test;

using common::vectorindex::Embeddings;
using common::vectorindex::LocalVectorIndex;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::VectorIndexMetric;
using common::vectorindex::VectorIndexKind;

class CudfVectorSearchJoinTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    OperatorTestBase::TearDown();
  }

  static std::string pack(const std::vector<float>& centroids) {
    std::vector<int64_t> clusterIds(centroids.size() / 2);
    std::iota(clusterIds.begin(), clusterIds.end(), 0);
    return build(centroids, clusterIds, VectorIndexKind::kFlat);
  }

  static std::string build(
      const std::vector<float>& embeddings,
      const std::vector<int64_t>& ids,
      VectorIndexKind type = VectorIndexKind::kIvfFlat,
      int32_t dimension = 2) {
    std::string bytes;
    LocalVectorIndex::build(
        type,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings, dimension),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    return bytes;
  }

  core::PlanNodePtr makeCodecPlan(
      const RowVectorPtr& probe,
      const std::vector<RowVectorPtr>& codecs,
      core::JoinType joinType = core::JoinType::kInner,
      const std::vector<std::string>& outputLayout =
          {"t0", "cluster_id", "score"},
      int64_t k = 2) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({probe})
        .vectorSearchJoin(
            {"ds"},
            {"c_ds"},
            PlanBuilder(planNodeIdGenerator).values(codecs).planNode(),
            "embedding",
            "codec",
            k,
            outputLayout,
            "cluster_id",
            "score",
            joinType)
        .planNode();
  }

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

  RowVectorPtr makeCodecProbe() {
    return makeRowVector(
        {"t0", "ds", "embedding"},
        {
            makeFlatVector<int64_t>({0, 1, 2, 3}),
            makeFlatVector<int64_t>({1, 1, 2, 3}),
            makeArrayVector<float>({{1, 0}, {0, 2}, {1, 0}, {1, 0}}),
        });
  }

  RowVectorPtr makeCodecResults() {
    return makeRowVector({
        makeFlatVector<int64_t>({0, 0, 1, 1, 2}),
        makeFlatVector<int64_t>({0, 1, 2, 1, 0}),
        makeFlatVector<float>({1, 0.6, 1, 0.8, 0}),
    });
  }

  // Runs 'plan' and checks that the join ran as the cuDF operators.
  void assertGpuResults(
      const core::PlanNodePtr& plan,
      const RowVectorPtr& expected,
      int32_t maxDrivers = 1) {
    assertGpuJoin(
        AssertQueryBuilder(plan).maxDrivers(maxDrivers).assertResults(
            expected));
  }

  static void assertGpuJoin(const std::shared_ptr<exec::Task>& task) {
    bool build = false;
    bool probe = false;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        EXPECT_NE(op.operatorType, "VectorSearchJoinBuild");
        EXPECT_NE(op.operatorType, "VectorSearchJoinProbe");
        build |= op.operatorType == "CudfVectorSearchJoinBuild";
        probe |= op.operatorType == "CudfVectorSearchJoinProbe";
      }
    }
    EXPECT_TRUE(build);
    EXPECT_TRUE(probe);
  }
};

TEST_F(CudfVectorSearchJoinTest, codec) {
  assertGpuResults(
      makeCodecPlan(makeCodecProbe(), makeCodecs()), makeCodecResults());
}

TEST_F(CudfVectorSearchJoinTest, drivers) {
  assertGpuResults(
      makeCodecPlan(makeCodecProbe(), makeCodecs()),
      makeCodecResults(),
      /*maxDrivers=*/4);
}

TEST_F(CudfVectorSearchJoinTest, noKeys) {
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

  assertGpuResults(
      plan,
      makeRowVector({
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({0, 1}),
      }));
}

TEST_F(CudfVectorSearchJoinTest, leftJoin) {
  // Row 1 has a null embedding, row 2 a null key and row 3 no codec.
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<int64_t>({2, 1, std::nullopt, 3}),
          makeNullableArrayVector<float>(
              {{{1, 0}}, std::nullopt, {{1, 0}}, {{1, 0}}}),
      });

  assertGpuResults(
      makeCodecPlan(probe, makeCodecs(), core::JoinType::kLeft),
      makeRowVector({
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {0, std::nullopt, std::nullopt, std::nullopt}),
          makeNullableFlatVector<float>(
              {0, std::nullopt, std::nullopt, std::nullopt}),
      }));
}

TEST_F(CudfVectorSearchJoinTest, emptyBuild) {
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

  assertGpuJoin(
      AssertQueryBuilder(
          makeCodecPlan(probe, {codecs}, core::JoinType::kInner, outputLayout))
          .assertEmptyResults());

  assertGpuResults(
      makeCodecPlan(probe, {codecs}, core::JoinType::kLeft, outputLayout),
      makeRowVector({
          makeFlatVector<int64_t>({0, 1}),
          makeNullConstant(TypeKind::BIGINT, 2),
          makeNullConstant(TypeKind::BIGINT, 2),
          makeNullConstant(TypeKind::REAL, 2),
      }));
}

TEST_F(CudfVectorSearchJoinTest, localIndexes) {
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

  assertGpuResults(
      plan,
      makeRowVector({
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({0, 0, 0, 1}),
          makeFlatVector<int64_t>({0, 0, 1, 0}),
      }));
}

TEST_F(CudfVectorSearchJoinTest, hnswTooSmallForCagra) {
  // Three embeddings make no CAGRA graph; the index is searched exactly.
  auto indexes = makeRowVector(
      {"c_ds", "codec"},
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<std::string>(
              {build({1, 0, 0.6, 0.8, 0, 1}, {5, 6, 7}, VectorIndexKind::kHnswSq)},
              VARBINARY()),
      });
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{0, 1}}),
      });
  // The index is too small for a CAGRA graph, so the GPU searches an exact
  // flat index over the reconstructed embeddings. Reconstruction does not
  // reproduce the original unit vectors, so the scores differ in the third
  // digit; the neighbor order is what this checks.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .vectorSearchJoin(
              {"ds"},
              {"c_ds"},
              PlanBuilder(planNodeIdGenerator).values({indexes}).planNode(),
              "embedding",
              "codec",
              /*k=*/2,
              {"t0", "cluster_id"},
              "cluster_id")
          .planNode();
  assertGpuResults(
      plan,
      makeRowVector({
          makeFlatVector<int64_t>({0, 0}),
          makeFlatVector<int64_t>({7, 6}),
      }));
}

TEST_F(CudfVectorSearchJoinTest, probeEncodings) {
  // The cuDF scan flattens constant and dictionary encodings of scalar
  // columns. Dictionary and constant encodings of the ARRAY embedding are not
  // accepted by that export, so the embedding stays flat here.
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0, 1}),
          makeConstant<int64_t>(1, 2),
          makeArrayVector<float>({{1, 0}, {0, 1}}),
      });
  assertGpuResults(
      makeCodecPlan(probe, makeCodecs()),
      makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({0, 1, 2, 1}),
          makeFlatVector<float>({1, 0.6, 1, 0.8}),
      }));

  // Indices {1, 0} select probe values 20 then 10, and key 1 for both rows.
  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          wrapInDictionary(
              makeIndices({1, 0}), makeFlatVector<int64_t>({10, 20})),
          wrapInDictionary(
              makeIndices({0, 0}), makeFlatVector<int64_t>({1, 9})),
          makeArrayVector<float>({{0, 1}, {1, 0}}),
      });
  assertGpuResults(
      makeCodecPlan(probe, makeCodecs()),
      makeRowVector({
          makeFlatVector<int64_t>({20, 20, 10, 10}),
          makeFlatVector<int64_t>({2, 1, 0, 1}),
          makeFlatVector<float>({1, 0.8, 1, 0.6}),
      }));

  // Index 1 selects the single embedding (0, 1).
  auto codecs = makeRowVector(
      {"c_ds", "codec"},
      {
          makeFlatVector<int64_t>({1}),
          wrapInDictionary(
              makeIndices({1}),
              makeFlatVector<std::string>(
                  {pack({1, 0}), pack({0, 1})}, VARBINARY())),
      });
  probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{0, 1}}),
      });
  assertGpuResults(
      makeCodecPlan(probe, {codecs}),
      makeRowVector({
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
          makeFlatVector<float>({1}),
      }));
}

TEST_F(CudfVectorSearchJoinTest, probeBatches) {
  // The build index stays on the device across probe batches.
  auto first = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{1, 0}}),
      });
  auto second = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{0, 1}}),
      });
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({first, second})
          .vectorSearchJoin(
              {"ds"},
              {"c_ds"},
              PlanBuilder(planNodeIdGenerator)
                  .values(makeCodecs())
                  .planNode(),
              "embedding",
              "codec",
              /*k=*/2,
              {"t0", "cluster_id"},
              "cluster_id")
          .planNode();
  assertGpuResults(
      plan,
      makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({0, 1, 2, 1}),
      }));
}

namespace {

// The ids a result vector reports, one (probe row, embedding id) pair each,
// in probe-row order then id order.
std::vector<std::pair<int64_t, int64_t>> resultIds(const RowVectorPtr& result) {
  DecodedVector probe(*result->childAt(0));
  DecodedVector id(*result->childAt(1));
  std::vector<std::pair<int64_t, int64_t>> ids;
  ids.reserve(result->size());
  for (vector_size_t row = 0; row < result->size(); ++row) {
    ids.emplace_back(probe.valueAt<int64_t>(row), id.valueAt<int64_t>(row));
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

} // namespace

TEST_F(CudfVectorSearchJoinTest, faissGpuIndexTypes) {
  // One local index per Faiss type the GPU searches, including the ANN types.
  // The device search has to find the same embeddings as Faiss GPU searching
  // the same bytes on the host. CAGRA is built by cuVS; the others are built
  // by Faiss on the CPU and cloned to the GPU.
  constexpr int32_t kDimension = 8;
  constexpr int32_t kEmbeddings = 40;
  constexpr int64_t kResults = 4;
  std::mt19937 random(11);
  std::normal_distribution<float> value;
  std::vector<float> embeddings(kEmbeddings * kDimension);
  for (auto& x : embeddings) {
    x = value(random);
  }
  for (int32_t row = 0; row < kEmbeddings; ++row) {
    common::vectorindex::EmbeddingNormalizer::normalize(
        VectorIndexMetric::kCosine,
        {embeddings.data() + row * kDimension, kDimension});
  }
  std::vector<int64_t> ids(kEmbeddings);
  std::iota(ids.begin(), ids.end(), 100);

  std::vector<std::vector<float>> queries(6);
  for (auto& query : queries) {
    query.resize(kDimension);
    for (auto& x : query) {
      x = value(random);
    }
  }
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>(queries.size(), [](auto row) { return row; }),
          makeFlatVector<int64_t>(
              queries.size(), [](auto /*row*/) { return 1; }),
          makeArrayVector<float>(queries),
      });

  const std::vector<VectorIndexKind> types = {
      VectorIndexKind::kIvfFlat,
      VectorIndexKind::kIvfSq,
      VectorIndexKind::kCagra,
  };
  for (const auto type : types) {
    SCOPED_TRACE(static_cast<int>(type));
    // CAGRA has no CPU build. Registering cuDF again afterwards puts the
    // search accelerator back the way SetUp left it.
    bool restoreBuildAccelerator = false;
    if (type == VectorIndexKind::kCagra) {
      unregisterCudf();
      common::vectorindex::setLocalVectorIndexAccelerator(
          common::vectorindex::makeGpuLocalVectorIndexAccelerator(
              {.device = 0, .build = true}));
      restoreBuildAccelerator = true;
    }
    SCOPE_EXIT {
      if (restoreBuildAccelerator) {
        common::vectorindex::setLocalVectorIndexAccelerator(nullptr);
        registerCudf();
      }
    };
    auto index = makeRowVector(
        {"c_ds", "codec"},
        {
            makeFlatVector<int64_t>({1}),
            makeFlatVector<std::string>(
                {build(embeddings, ids, type, kDimension)}, VARBINARY()),
        });
    if (restoreBuildAccelerator) {
      common::vectorindex::setLocalVectorIndexAccelerator(nullptr);
      registerCudf();
      restoreBuildAccelerator = false;
    }

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan =
        PlanBuilder(planNodeIdGenerator)
            .values({probe})
            .vectorSearchJoin(
                {"ds"},
                {"c_ds"},
                PlanBuilder(planNodeIdGenerator).values({index}).planNode(),
                "embedding",
                "codec",
                kResults,
                {"t0", "cluster_id"},
                "cluster_id")
            .planNode();

    auto gpu = AssertQueryBuilder(plan).copyResults(pool());
    unregisterCudf();
    common::vectorindex::setLocalVectorIndexAccelerator(
        common::vectorindex::makeGpuLocalVectorIndexAccelerator(
            {.device = 0}));
    auto faiss = AssertQueryBuilder(plan).copyResults(pool());
    common::vectorindex::setLocalVectorIndexAccelerator(nullptr);
    registerCudf();

    const auto gpuIds = resultIds(gpu);
    ASSERT_EQ(gpuIds, resultIds(faiss));
    std::vector<int32_t> hits(queries.size());
    for (const auto& [probeRow, id] : gpuIds) {
      ASSERT_GE(probeRow, 0);
      ASSERT_LT(probeRow, static_cast<int64_t>(queries.size()));
      ASSERT_GE(id, ids.front());
      ASSERT_LT(id, ids.front() + kEmbeddings);
      ++hits[probeRow];
    }
    for (const auto count : hits) {
      ASSERT_EQ(count, kResults);
    }
  }
}

TEST_F(CudfVectorSearchJoinTest, matchesCpu) {
  // Exact flat indexes over random embeddings, several per key, searched by
  // many probe rows: the GPU join finds what the CPU join finds.
  constexpr int32_t kDimension = 16;
  constexpr int32_t kNumKeys = 5;
  constexpr int32_t kIndexesPerKey = 3;
  constexpr int32_t kEmbeddingsPerIndex = 200;
  constexpr int32_t kNumProbeRows = 3'000;
  std::mt19937 random(7);
  std::normal_distribution<float> value;

  std::vector<int64_t> keys;
  std::vector<std::string> codecs;
  int64_t nextId = 0;
  for (int32_t key = 0; key < kNumKeys; ++key) {
    for (int32_t i = 0; i < kIndexesPerKey; ++i) {
      std::vector<float> embeddings(kEmbeddingsPerIndex * kDimension);
      for (auto& x : embeddings) {
        x = value(random);
      }
      for (int32_t row = 0; row < kEmbeddingsPerIndex; ++row) {
        common::vectorindex::EmbeddingNormalizer::normalize(
            VectorIndexMetric::kCosine,
            {embeddings.data() + row * kDimension, kDimension});
      }
      std::vector<int64_t> ids(kEmbeddingsPerIndex);
      std::iota(ids.begin(), ids.end(), nextId);
      nextId += kEmbeddingsPerIndex;
      keys.push_back(key);
      codecs.push_back(
          build(embeddings, ids, VectorIndexKind::kFlat, kDimension));
    }
  }
  auto buildRows = makeRowVector(
      {"c_ds", "codec"},
      {makeFlatVector<int64_t>(keys),
       makeFlatVector<std::string>(codecs, VARBINARY())});

  std::vector<std::vector<float>> queries(kNumProbeRows);
  for (auto& query : queries) {
    query.resize(kDimension);
    for (auto& x : query) {
      x = value(random);
    }
  }
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>(kNumProbeRows, [](auto row) { return row; }),
          makeFlatVector<int64_t>(
              kNumProbeRows, [](auto row) { return row % (kNumKeys + 1); }),
          makeArrayVector<float>(queries),
      });
  auto plan = makeCodecPlan(
      probe,
      {buildRows},
      core::JoinType::kLeft,
      {"t0", "c_ds", "cluster_id", "score"},
      /*k=*/10);

  unregisterCudf();
  auto expected = AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  EXPECT_EQ(
      expected->size(),
      (kNumProbeRows / (kNumKeys + 1)) * kNumKeys * kIndexesPerKey * 10 +
          kNumProbeRows / (kNumKeys + 1));

  assertGpuResults(plan, expected, /*maxDrivers=*/2);
}

TEST_F(CudfVectorSearchJoinTest, errors) {
  auto probe = makeRowVector(
      {"t0", "ds", "embedding"},
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
          makeArrayVector<float>({{1, 0, 0}}),
      });
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(makeCodecPlan(probe, makeCodecs()))
          .copyResults(pool()),
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
      AssertQueryBuilder(makeCodecPlan(probe, makeCodecs()))
          .copyResults(pool()),
      "Query embedding must not contain nulls");

  auto codecs = makeRowVector(
      {"c_ds", "codec"},
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<std::string>({"not a codec"}, VARBINARY()),
      });
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(makeCodecPlan(probe, {codecs})).copyResults(pool()),
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
      AssertQueryBuilder(makeCodecPlan(probe, {codecs})).copyResults(pool()),
      "Vector search join indexes must have the same dimension");
}

} // namespace
} // namespace facebook::velox::cudf_velox

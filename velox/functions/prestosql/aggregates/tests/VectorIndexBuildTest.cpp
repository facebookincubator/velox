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

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {
namespace {

class VectorIndexBuildTest : public AggregationTestBase {
 protected:
  static constexpr std::string_view kNoOptions{"map()::map(varchar, varchar)"};

  static std::string aggregate(
      std::string_view indexType,
      std::string_view optionsSql) {
    return fmt::format(
        "vector_index_build(c0, c1, '{}', 'cosine', {})",
        indexType,
        optionsSql);
  }

  // Runs vector_index_build over 'embeddings' and 'ids' in every plan shape
  // testAggregations covers and passes each resulting index to 'verify'.
  // 'optionsSql' is the options argument as a SQL expression.
  void testBuild(
      const VectorPtr& embeddings,
      const VectorPtr& ids,
      std::string_view indexType,
      std::string_view optionsSql,
      const std::function<void(const faiss::Index&)>& verify) {
    testAggregations(
        [&](PlanBuilder& builder) {
          builder.values({makeRowVector({embeddings, ids})});
        },
        /*groupingKeys=*/{},
        {aggregate(indexType, optionsSql)},
        /*postAggregationProjections=*/{},
        [&](AssertQueryBuilder& builder) {
          std::shared_ptr<exec::Task> task;
          auto result = builder.copyResults(pool(), task);
          verify(*read(
              result->childAt(0)->variantAt(0).value<TypeKind::VARBINARY>()));
          return task;
        });
  }

  void testBuildFails(
      const VectorPtr& embeddings,
      const VectorPtr& ids,
      std::string_view indexType,
      std::string_view optionsSql,
      const std::string& expectedMessage) {
    testFailingAggregations(
        {makeRowVector({embeddings, ids})},
        /*groupingKeys=*/{},
        {aggregate(indexType, optionsSql)},
        expectedMessage);
  }

  static std::unique_ptr<faiss::Index> read(const std::string& bytes) {
    faiss::VectorIOReader reader;
    reader.data.assign(bytes.begin(), bytes.end());
    return std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
  }

  // Returns 'index' as a 'T'. Fails if it is another kind of FAISS index.
  template <typename T>
  static const T& as(const faiss::Index& index) {
    const auto* result = dynamic_cast<const T*>(&index);
    VELOX_CHECK_NOT_NULL(result, "Unexpected kind of FAISS index");
    return *result;
  }

  // The ids of the 'k' embeddings nearest to 'query', nearest first.
  static std::vector<int64_t>
  search(const faiss::Index& index, std::vector<float> query, int64_t k) {
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> ids(k);
    index.search(1, query.data(), k, distances.data(), ids.data());
    return {ids.begin(), ids.end()};
  }
};

TEST_F(VectorIndexBuildTest, indexTypes) {
  // Without normalization, (1, 1, 0) would be nearest to the long (10, 1, 0).
  auto embeddings = makeArrayVector<float>({
      {10, 1, 0},
      {1, 1, 0},
      {0, 0, 4},
  });
  auto ids = makeFlatVector<int64_t>({10, 20, 30});

  for (const auto* indexType : {"flat", "ivf_flat", "ivf_sq", "hnsw_sq"}) {
    SCOPED_TRACE(indexType);
    testBuild(embeddings, ids, indexType, kNoOptions, [](const auto& index) {
      EXPECT_EQ(3, index.ntotal);
      EXPECT_THAT(search(index, {1, 1, 0}, 2), testing::ElementsAre(20, 10));
      EXPECT_THAT(search(index, {0, 0, 1}, 1), testing::ElementsAre(30));
    });
  }
}

TEST_F(VectorIndexBuildTest, groupBy) {
  auto data = makeRowVector({
      makeArrayVector<float>({{1, 0}, {1, 0}, {0, 1}, {0, 1}}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeFlatVector<int64_t>({1, 2, 1, 2}),
  });
  testAggregations(
      [&](PlanBuilder& builder) { builder.values({data}); },
      /*groupingKeys=*/{"c2"},
      {aggregate("flat", kNoOptions)},
      /*postAggregationProjections=*/{},
      [&](AssertQueryBuilder& builder) {
        std::shared_ptr<exec::Task> task;
        auto result = builder.copyResults(pool(), task);
        EXPECT_EQ(2, result->size());
        for (auto row = 0; row < result->size(); ++row) {
          const auto group =
              result->childAt(0)->variantAt(row).value<TypeKind::BIGINT>();
          const auto indexBytes =
              result->childAt(1)->variantAt(row).value<TypeKind::VARBINARY>();
          auto index = read(indexBytes);
          EXPECT_THAT(
              search(*index, {1, 0}, 2),
              group == 1 ? testing::ElementsAre(10, 30)
                         : testing::ElementsAre(20, 40));
        }
        return task;
      });
}

TEST_F(VectorIndexBuildTest, nullRows) {
  auto embeddings = makeArrayVectorFromJson<float>({
      "[1, 0]",
      "null",
      "[0, 1]",
      "[1, 1]",
  });
  auto ids = makeNullableFlatVector<int64_t>({10, 20, 30, std::nullopt});

  testBuild(embeddings, ids, "flat", kNoOptions, [](const auto& index) {
    EXPECT_EQ(2, index.ntotal);
    EXPECT_THAT(search(index, {1, 0}, 2), testing::ElementsAre(10, 30));
  });
}

TEST_F(VectorIndexBuildTest, options) {
  // 80 embeddings give each of 2 lists at least 39 embeddings.
  auto embeddings = makeArrayVector<float>(
      80, [](auto /*row*/) { return 2; }, [](auto row) { return row % 7 + 1; });
  auto ids = makeFlatIdentityVector<int64_t>(80);

  auto checkIvf = [](int32_t nlist, int32_t nprobe) {
    return [nlist, nprobe](const faiss::Index& index) {
      const auto& ivf = as<faiss::IndexIVF>(index);
      EXPECT_EQ(nlist, ivf.nlist);
      EXPECT_EQ(nprobe, ivf.nprobe);
    };
  };

  // HNSW keeps 2 * M neighbors per embedding on the graph's bottom level.
  auto checkHnsw = [](int32_t m, int32_t efConstruction, int32_t efSearch) {
    return [=](const faiss::Index& index) {
      const auto& hnsw =
          as<faiss::IndexHNSWSQ>(*as<faiss::IndexIDMap>(index).index).hnsw;
      EXPECT_EQ(2 * m, hnsw.nb_neighbors(0));
      EXPECT_EQ(efConstruction, hnsw.efConstruction);
      EXPECT_EQ(efSearch, hnsw.efSearch);
    };
  };

  testBuild(
      embeddings,
      ids,
      "ivf_flat",
      "map(array['ivf.nlist', 'ivf.nprobe'], array['2', '2'])",
      checkIvf(2, 2));

  // The default 'ivf.nlist', 35 for 80 embeddings, is reduced to give each list
  // at least 39 embeddings. 'ivf.nprobe' defaults to a tenth of the lists,
  // rounded up.
  testBuild(embeddings, ids, "ivf_flat", kNoOptions, checkIvf(2, 1));

  // The build reduces 'ivf.nlist' even when set, and caps 'ivf.nprobe' at the
  // number of lists.
  testBuild(
      embeddings,
      ids,
      "ivf_sq",
      "map(array['ivf.nlist', 'ivf.nprobe'], array['100', '50'])",
      checkIvf(2, 2));

  // A group of fewer than 39 embeddings gets one list.
  testBuild(
      makeArrayVector<float>({{1, 0}, {0, 1}}),
      makeFlatVector<int64_t>({10, 20}),
      "ivf_flat",
      "map(array['ivf.nlist'], array['4'])",
      checkIvf(1, 1));

  testBuild(
      embeddings,
      ids,
      "hnsw_sq",
      "map(array['hnsw.M', 'hnsw.efConstruction', 'hnsw.efSearch'], array['8', '40', '20'])",
      checkHnsw(8, 40, 20));

  testBuild(embeddings, ids, "hnsw_sq", kNoOptions, checkHnsw(32, 200, 64));
}

TEST_F(VectorIndexBuildTest, errors) {
  auto embeddings = makeArrayVector<float>({{1, 0}, {0, 1}});
  auto ids = makeFlatVector<int64_t>({10, 20});

  testBuildFails(
      embeddings,
      ids,
      "ivf_pq",
      kNoOptions,
      "Unsupported vector index type: ivf_pq");

  testBuildFails(
      embeddings,
      makeFlatVector<int64_t>({10, -1}),
      "flat",
      kNoOptions,
      "Local vector index embedding id must not be -1");

  // The constant arguments must be non-null constants. c2 and c3 are columns.
  auto testFailingCall = [&](const std::string& call,
                             const std::string& expectedMessage) {
    SCOPED_TRACE(call);
    testFailingAggregations(
        {makeRowVector({
            embeddings,
            ids,
            makeFlatVector<std::string>({"flat", "flat"}),
            makeMapVectorFromJson<std::string, std::string>({"{}", "{}"}),
        })},
        /*groupingKeys=*/{},
        {call},
        expectedMessage);
  };
  testFailingCall(
      "vector_index_build(c0, c1, 'flat', 'l2', map()::map(varchar, varchar))",
      "Unsupported vector index metric: l2");
  testFailingCall(
      "vector_index_build(c0, c1, c2, 'cosine', map()::map(varchar, varchar))",
      "Index type must be a constant");
  testFailingCall(
      "vector_index_build(c0, c1, 'flat', c2, map()::map(varchar, varchar))",
      "Metric must be a constant");
  testFailingCall(
      "vector_index_build(c0, c1, 'flat', 'cosine', c3)",
      "Options must be a constant");
  testFailingCall(
      "vector_index_build(c0, c1, null::varchar, 'cosine', map()::map(varchar, varchar))",
      "Index type must not be null");
  testFailingCall(
      "vector_index_build(c0, c1, 'flat', null::varchar, map()::map(varchar, varchar))",
      "Metric must not be null");
  testFailingCall(
      "vector_index_build(c0, c1, 'flat', 'cosine', null::map(varchar, varchar))",
      "Options must not be null");

  testBuildFails(
      embeddings,
      ids,
      "flat",
      "map(array['nlist'], array['2'])",
      "Unknown vector index option: nlist");
  testBuildFails(
      embeddings,
      ids,
      "flat",
      "map(array['ivf.nlist'], array['2'])",
      "Vector index option does not apply to the index type. Option: ivf.nlist. Index type: flat");
  testBuildFails(
      embeddings,
      ids,
      "ivf_flat",
      "map(array['ivf.nlist'], array['two'])",
      "Vector index option must be an integer: ivf.nlist=two");
  testBuildFails(
      embeddings,
      ids,
      "ivf_flat",
      "map(array['ivf.nlist'], array['0'])",
      "Vector index option must be positive: ivf.nlist");
  testBuildFails(
      embeddings,
      ids,
      "hnsw_sq",
      "map(array['hnsw.M'], array['1'])",
      "Vector index option is too small: hnsw.M");
  testBuildFails(
      embeddings,
      ids,
      "hnsw_sq",
      "map(array['hnsw.M'], array['1000'])",
      "Vector index option is too large: hnsw.M");
  testBuildFails(
      embeddings,
      ids,
      "ivf_flat",
      "map(array['ivf.nlist', 'ivf.nprobe'], array['2', null])",
      "Option values must not be null");

  testBuildFails(
      makeArrayVectorFromJson<float>({"[1, 0]", "[1, null]"}),
      ids,
      "flat",
      kNoOptions,
      "Embedding must not contain nulls");
  testBuildFails(
      makeArrayVector<float>(
          {{1, 0}, {std::numeric_limits<float>::quiet_NaN(), 0}}),
      ids,
      "flat",
      kNoOptions,
      "Embedding must not contain NaN or infinity");
  testBuildFails(
      makeArrayVector<float>({{1, 0}, {1, 0, 0}}),
      ids,
      "flat",
      kNoOptions,
      "All embeddings must have the same length");
  testBuildFails(
      makeArrayVector<float>({{1, 0}, {}}),
      ids,
      "flat",
      kNoOptions,
      "Embedding must not be empty");
}

TEST_F(VectorIndexBuildTest, mergeExtract) {
  // Each group builds with its own settings. Group 2's settings come from
  // before the options existed and leave them unset.
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeRowVector({
          makeArrayVectorFromJson<int32_t>({
              "[1, 0, null, 3, null, null, null]",
              "[0, 0]",
          }),
          makeArrayVectorFromJson<int64_t>({"[10, 20]", "[30, 40]"}),
          makeArrayVectorFromJson<float>({"[1, 0, 0, 1]", "[1, 0, 0, 1]"}),
      }),
  });
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation({"c0"}, {"vector_index_build_merge_extract(c1)"})
          .orderBy({"c0"}, false)
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(2, result->size());

  auto ivf =
      read(result->childAt(1)->variantAt(0).value<TypeKind::VARBINARY>());
  EXPECT_THAT(
      search(as<faiss::IndexIVF>(*ivf), {1, 0}, 2),
      testing::ElementsAre(10, 20));

  auto flat =
      read(result->childAt(1)->variantAt(1).value<TypeKind::VARBINARY>());
  EXPECT_THAT(
      search(as<faiss::IndexIDMap>(*flat), {1, 0}, 2),
      testing::ElementsAre(30, 40));
}

TEST_F(VectorIndexBuildTest, mergeExtractErrors) {
  // vector_index_build_merge_extract takes intermediate results directly:
  // settings (index type, metric, then the options), ids and embeddings.
  auto testFailing = [&](const std::vector<std::string>& settingsJson,
                         const std::vector<std::string>& idsJson,
                         const std::vector<std::string>& embeddingsJson,
                         const std::string& expectedMessage) {
    SCOPED_TRACE(expectedMessage);
    testFailingAggregations(
        {makeRowVector({makeRowVector({
            makeArrayVectorFromJson<int32_t>(settingsJson),
            makeArrayVectorFromJson<int64_t>(idsJson),
            makeArrayVectorFromJson<float>(embeddingsJson),
        })})},
        /*groupingKeys=*/{},
        {"vector_index_build_merge_extract(c0)"},
        expectedMessage);
  };

  const std::string flat{"[0, 0, null, null, null, null, null]"};
  auto testFailingSettings = [&](const std::string& settingsJson,
                                 const std::string& expectedMessage) {
    testFailing(
        {settingsJson, settingsJson},
        {"[10]", "[20]"},
        {"[1, 0]", "[0, 1]"},
        expectedMessage);
  };

  testFailingSettings(
      "[0]", "Too few entries in intermediate vector index settings");
  testFailingSettings(
      "[0, 0, null, null, null, null, null, null]",
      "Too many entries in intermediate vector index settings");
  testFailingSettings(
      "[null, 0, null, null, null, null, null]",
      "Intermediate vector index settings must include the index type");
  testFailingSettings(
      "[0, null, null, null, null, null, null]",
      "Intermediate vector index settings must include the metric");
  testFailingSettings(
      "[9, 0, null, null, null, null, null]",
      "Invalid vector index kind value: 9");
  testFailingSettings(
      "[0, 9, null, null, null, null, null]",
      "Invalid vector index metric value: 9");
  testFailingSettings(
      "[1, 0, 0, null, null, null, null]",
      "Vector index option must be positive: ivf.nlist");
  testFailingSettings(
      "[3, 0, null, null, 1, null, null]",
      "Vector index option is too small: hnsw.M");
  testFailingSettings(
      "[0, 0, 2, null, null, null, null]",
      "Vector index option does not apply to the index type. Option: ivf.nlist. Index type: flat");

  testFailing(
      {flat, "[1, 0, null, null, null, null, null]"},
      {"[10]", "[20]"},
      {"[1, 0]", "[0, 1]"},
      "Intermediate vector index results of a group must have the same settings");
  testFailing(
      {flat, flat},
      {"[10]", "[]"},
      {"[1, 0]", "[]"},
      "Intermediate vector index result must include an embedding");
  testFailing(
      {flat, flat},
      {"[10]", "[20, 30]"},
      {"[1, 0]", "[0, 1, 0]"},
      "Intermediate vector index result must have the same number of values per embedding");
  testFailing(
      {flat, flat},
      {"[10]", "[null]"},
      {"[1, 0]", "[0, 1]"},
      "Intermediate vector index ids must not be null");
  testFailing(
      {flat, flat},
      {"[10]", "[20]"},
      {"[1, 0]", "[0, null]"},
      "Intermediate vector index embeddings must not contain nulls");
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql

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

#include "velox/common/vectorindex/LocalVectorIndex.h"

#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::common::vectorindex {
namespace {

class LocalVectorIndexTest : public testing::Test {
 protected:
  // Builds a local index of 'kind' over 2-dimensional unit-length embeddings
  // (1, 0), (0, 1) and (-1, 0) with ids 10, 20 and 30.
  static std::string build(VectorIndexKind kind) {
    const std::vector<float> embeddings{1, 0, 0, 1, -1, 0};
    const std::vector<int64_t> ids{10, 20, 30};
    std::string bytes;
    LocalVectorIndex::build(
        kind,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings, /*dimension=*/2),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    return bytes;
  }
};

TEST_F(LocalVectorIndexTest, search) {
  for (const auto kind :
       {VectorIndexKind::kFlat,
        VectorIndexKind::kIvfFlat,
        VectorIndexKind::kIvfSq,
        VectorIndexKind::kHnswSq}) {
    SCOPED_TRACE(VectorIndexKindName::toName(kind));
    const auto bytes = build(kind);
    const LocalVectorIndex index(
        bytes, VectorIndexMetric::kCosine, /*searchOptions=*/{});
    EXPECT_EQ(kind, index.kind());
    EXPECT_EQ(2, index.dimension());
    EXPECT_EQ(3, index.size());
    EXPECT_EQ(3, LocalVectorIndex::size(bytes));

    // Nearest (0.8, 0.6) are ids 10, then 20. 'k' exceeds the size of the
    // index, so the last entry is missing.
    const std::vector<float> query{0.8, 0.6};
    std::vector<int64_t> ids(4);
    std::vector<float> scores(4);
    index.search(Embeddings(query, 2), /*k=*/4, ids, scores);
    EXPECT_THAT(ids, testing::ElementsAre(10, 20, 30, -1));
    EXPECT_NEAR(0.8, scores[0], 0.01);
    EXPECT_NEAR(0.6, scores[1], 0.01);
  }
}

TEST_F(LocalVectorIndexTest, searchOptions) {
  const auto ivfBytes = build(VectorIndexKind::kIvfFlat);
  const LocalVectorIndex ivf(
      ivfBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.nprobe = 5});
  const auto hnswBytes = build(VectorIndexKind::kHnswSq);
  const LocalVectorIndex hnsw(
      hnswBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.efSearch = 100});
  const std::vector<float> query{1, 0};
  std::vector<int64_t> ids(1);
  std::vector<float> scores(1);
  ivf.search(Embeddings(query, 2), 1, ids, scores);
  EXPECT_THAT(ids, testing::ElementsAre(10));
  hnsw.search(Embeddings(query, 2), 1, ids, scores);
  EXPECT_THAT(ids, testing::ElementsAre(10));
}

TEST_F(LocalVectorIndexTest, errors) {
  VELOX_ASSERT_THROW(
      LocalVectorIndex("not an index", VectorIndexMetric::kCosine, {}),
      "Invalid local vector index");

  // A flat index without the ids build() keeps.
  faiss::IndexFlatIP bare(2);
  const std::vector<float> embedding{1, 0};
  bare.add(1, embedding.data());
  faiss::VectorIOWriter writer;
  faiss::write_index(&bare, &writer);
  VELOX_ASSERT_THROW(
      LocalVectorIndex(
          std::string(writer.data.begin(), writer.data.end()),
          VectorIndexMetric::kCosine,
          {}),
      "Invalid local vector index: unsupported FAISS index");
  VELOX_ASSERT_THROW(
      LocalVectorIndex::size(
          std::string(writer.data.begin(), writer.data.end())),
      "Invalid local vector index: unsupported FAISS index");
  VELOX_ASSERT_THROW(
      LocalVectorIndex::size("IxMp"), "Invalid local vector index: too short");
  auto negative = build(VectorIndexKind::kFlat);
  const int64_t numEmbeddings{-1};
  std::memcpy(negative.data() + 8, &numEmbeddings, sizeof(numEmbeddings));
  VELOX_ASSERT_THROW(
      LocalVectorIndex::size(negative),
      "Local vector index must have at least one embedding");

  const auto flat = build(VectorIndexKind::kFlat);
  VELOX_ASSERT_THROW(
      LocalVectorIndex(
          flat,
          VectorIndexMetric::kCosine,
          LocalVectorIndexOptions{.nprobe = 5}),
      "Vector index option does not apply to the index type. Option: ivf.nprobe. Index type: flat");

  // Id -1 means a missing result to FAISS.
  const std::vector<float> embeddings{1, 0};
  const std::vector<int64_t> missingId{-1};
  VELOX_ASSERT_THROW(
      LocalVectorIndex::build(
          VectorIndexKind::kFlat,
          VectorIndexMetric::kCosine,
          LocalVectorIndexOptions{},
          Embeddings(embeddings, 2),
          missingId,
          [](std::string_view) {}),
      "Local vector index embedding id must not be -1");

  // Indexes in the layouts build() writes, written with id -1 by FAISS.
  const auto write = [](faiss::Index& index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return std::string(writer.data.begin(), writer.data.end());
  };
  faiss::IndexFlatIP flatWithIds(2);
  faiss::IndexIDMap idMap(&flatWithIds);
  idMap.add_with_ids(1, embeddings.data(), missingId.data());
  VELOX_ASSERT_THROW(
      LocalVectorIndex(write(idMap), VectorIndexMetric::kCosine, {}),
      "Local vector index embedding id must not be -1");
  faiss::IndexFlatIP quantizer(2);
  faiss::IndexIVFFlat ivf(&quantizer, 2, 1, faiss::METRIC_INNER_PRODUCT);
  ivf.train(1, embeddings.data());
  ivf.add_with_ids(1, embeddings.data(), missingId.data());
  VELOX_ASSERT_THROW(
      LocalVectorIndex(write(ivf), VectorIndexMetric::kCosine, {}),
      "Local vector index embedding id must not be -1");

  const LocalVectorIndex index(flat, VectorIndexMetric::kCosine, {});
  const std::vector<float> query{1, 0, 0};
  std::vector<int64_t> ids(1);
  std::vector<float> scores(1);
  VELOX_ASSERT_THROW(
      index.search(Embeddings(query, 3), 1, ids, scores),
      "Query embedding length must match the vector index's");
}

TEST_F(LocalVectorIndexTest, parseSearchOptions) {
  using Options = LocalVectorIndexOptions;
  EXPECT_EQ(
      (Options{.nprobe = 4, .efSearch = 100}),
      Options::parseSearch(
          {{Options::kNprobe, "4"}, {Options::kEfSearch, "100"}}));
  VELOX_ASSERT_THROW(
      Options::parseSearch({{Options::kNlist, "5"}}),
      "Vector index option is not a search setting: ivf.nlist");
  VELOX_ASSERT_THROW(
      Options::parseSearch({{"foo", "5"}}), "Unknown vector index option: foo");
  VELOX_ASSERT_THROW(
      Options::parseSearch({{Options::kNprobe, "x"}}),
      "Vector index option must be an integer: ivf.nprobe=x");
  VELOX_ASSERT_THROW(
      Options::parseSearch({{Options::kEfSearch, "0"}}),
      "Vector index option must be positive: hnsw.efSearch");
}

} // namespace
} // namespace facebook::velox::common::vectorindex

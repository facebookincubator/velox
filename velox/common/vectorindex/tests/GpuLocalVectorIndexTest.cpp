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

#include "velox/common/vectorindex/GpuLocalVectorIndex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/zerocopy_io.h>
#include <faiss/index_io.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"

namespace facebook::velox::common::vectorindex {
namespace {

constexpr int32_t kDimension{32};
constexpr int64_t kNumEmbeddings{4'000};
constexpr int64_t kNumQueries{64};
constexpr int64_t kK{10};

class GpuLocalVectorIndexTest : public testing::Test {
 protected:
  void SetUp() override {
    if (gpuLocalVectorIndexDevices() == 0) {
      GTEST_SKIP() << "No CUDA device";
    }
    std::mt19937 random(7);
    embeddings_ = randomUnitEmbeddings(kNumEmbeddings, random);
    queries_ = randomUnitEmbeddings(kNumQueries, random);
    ids_.resize(kNumEmbeddings);
    for (int64_t i = 0; i < kNumEmbeddings; ++i) {
      ids_[i] = 1'000 + 3 * i;
    }
  }

  void TearDown() override {
    setLocalVectorIndexAccelerator(nullptr);
  }

  static std::vector<float> randomUnitEmbeddings(
      int64_t count,
      std::mt19937& random) {
    std::normal_distribution<float> normal;
    std::vector<float> values(count * kDimension);
    for (int64_t i = 0; i < count; ++i) {
      float norm = 0;
      for (int32_t j = 0; j < kDimension; ++j) {
        values[i * kDimension + j] = normal(random);
        norm += values[i * kDimension + j] * values[i * kDimension + j];
      }
      norm = std::sqrt(norm);
      for (int32_t j = 0; j < kDimension; ++j) {
        values[i * kDimension + j] /= norm;
      }
    }
    return values;
  }

  std::string build(VectorIndexType type) const {
    std::string bytes;
    LocalVectorIndex::build(
        type,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings_, kDimension),
        ids_,
        [&](std::string_view chunk) { bytes.append(chunk); });
    return bytes;
  }

  static void useGpu(GpuLocalVectorIndexOptions options = {}) {
    setLocalVectorIndexAccelerator(makeGpuLocalVectorIndexAccelerator(options));
  }

  struct Results {
    std::vector<int64_t> ids;
    std::vector<float> scores;
  };

  Results search(const LocalVectorIndex& index, int64_t k = kK) const {
    Results results{
        std::vector<int64_t>(kNumQueries * k),
        std::vector<float>(kNumQueries * k)};
    index.search(
        Embeddings(queries_, kDimension), k, results.ids, results.scores);
    return results;
  }

  // The exact 'kK' nearest ids of each query.
  std::vector<int64_t> exactIds() const {
    std::vector<int64_t> exact;
    std::vector<std::pair<float, int64_t>> scored(kNumEmbeddings);
    for (int64_t q = 0; q < kNumQueries; ++q) {
      for (int64_t e = 0; e < kNumEmbeddings; ++e) {
        float dot = 0;
        for (int32_t j = 0; j < kDimension; ++j) {
          dot += queries_[q * kDimension + j] * embeddings_[e * kDimension + j];
        }
        scored[e] = {-dot, ids_[e]};
      }
      std::partial_sort(scored.begin(), scored.begin() + kK, scored.end());
      for (int64_t i = 0; i < kK; ++i) {
        exact.push_back(scored[i].second);
      }
    }
    return exact;
  }

  // The fraction of 'expected' ids that 'actual' finds for the same query.
  static double recall(
      const std::vector<int64_t>& expected,
      const std::vector<int64_t>& actual) {
    int64_t found = 0;
    for (int64_t q = 0; q < kNumQueries; ++q) {
      const std::unordered_set<int64_t> wanted(
          expected.begin() + q * kK, expected.begin() + (q + 1) * kK);
      for (int64_t i = 0; i < kK; ++i) {
        found += wanted.count(actual[q * kK + i]);
      }
    }
    return static_cast<double>(found) / (kNumQueries * kK);
  }

  std::vector<float> embeddings_;
  std::vector<float> queries_;
  std::vector<int64_t> ids_;
};

TEST_F(GpuLocalVectorIndexTest, searchMatchesCpu) {
  const auto exact = exactIds();
  for (const auto type :
       {VectorIndexType::kFlat,
        VectorIndexType::kIvfFlat,
        VectorIndexType::kIvfSq,
        VectorIndexType::kHnswSq}) {
    SCOPED_TRACE(VectorIndexTypeName::toName(type));
    const auto bytes = build(type);
    // Every list, so that the IVF searches are exhaustive on both devices.
    const auto options =
        type == VectorIndexType::kIvfFlat || type == VectorIndexType::kIvfSq
        ? LocalVectorIndexOptions{.nprobe = 16'384}
        : LocalVectorIndexOptions{};
    const LocalVectorIndex cpu(bytes, VectorIndexMetric::kCosine, options);
    useGpu();
    const LocalVectorIndex gpu(bytes, VectorIndexMetric::kCosine, options);
    setLocalVectorIndexAccelerator(nullptr);
    EXPECT_FALSE(cpu.accelerated());
    EXPECT_TRUE(gpu.accelerated());

    const auto cpuResults = search(cpu);
    const auto gpuResults = search(gpu);
    switch (type) {
      case VectorIndexType::kFlat:
      case VectorIndexType::kIvfFlat:
        EXPECT_EQ(exact, gpuResults.ids);
        for (size_t i = 0; i < cpuResults.scores.size(); ++i) {
          EXPECT_NEAR(cpuResults.scores[i], gpuResults.scores[i], 1e-4);
        }
        break;
      case VectorIndexType::kIvfSq:
        EXPECT_GE(recall(cpuResults.ids, gpuResults.ids), 0.95);
        break;
      case VectorIndexType::kHnswSq:
        // CAGRA searches the reconstructed embeddings at full precision.
        EXPECT_GE(recall(exact, gpuResults.ids), 0.9);
        break;
      case VectorIndexType::kCagra:
        break;
    }
  }
}

TEST_F(GpuLocalVectorIndexTest, padding) {
  const std::vector<float> embeddings{1, 0, 0, 1, -1, 0};
  const std::vector<int64_t> ids{10, 20, 30};
  useGpu();
  for (const auto type :
       {VectorIndexType::kFlat,
        VectorIndexType::kIvfFlat,
        VectorIndexType::kIvfSq,
        VectorIndexType::kHnswSq}) {
    SCOPED_TRACE(VectorIndexTypeName::toName(type));
    std::string bytes;
    LocalVectorIndex::build(
        type,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings, 2),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    const LocalVectorIndex index(bytes, VectorIndexMetric::kCosine, {});
    // Three embeddings are too few for a CAGRA graph.
    EXPECT_EQ(type != VectorIndexType::kHnswSq, index.accelerated());

    const std::vector<float> query{0.8, 0.6};
    std::vector<int64_t> resultIds(4);
    std::vector<float> scores(4);
    index.search(Embeddings(query, 2), /*k=*/4, resultIds, scores);
    EXPECT_THAT(resultIds, testing::ElementsAre(10, 20, 30, -1));
    EXPECT_NEAR(0.8, scores[0], 0.01);
    EXPECT_NEAR(0.6, scores[1], 0.01);
    EXPECT_NEAR(-0.8, scores[2], 0.05);
  }
}

TEST_F(GpuLocalVectorIndexTest, cagraStoredFormat) {
  useGpu({.build = true});
  const auto bytes = build(VectorIndexType::kCagra);
  setLocalVectorIndexAccelerator(nullptr);

  // The bytes are an IndexHNSWCagra, so the CPU searches them with no GPU.
  const LocalVectorIndex cpu(bytes, VectorIndexMetric::kCosine, {});
  EXPECT_EQ(VectorIndexType::kCagra, cpu.type());
  EXPECT_EQ(kNumEmbeddings, cpu.size());
  EXPECT_FALSE(cpu.accelerated());
  const auto exact = exactIds();
  EXPECT_GE(recall(exact, search(cpu).ids), 0.9);

  // The same bytes are cloned to the GPU once, and hnsw.efSearch is the
  // CAGRA intermediate result count.
  useGpu();
  const LocalVectorIndex gpu(
      bytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.efSearch = 128});
  EXPECT_TRUE(gpu.accelerated());
  EXPECT_GE(recall(exact, search(gpu).ids), 0.9);
}

TEST_F(GpuLocalVectorIndexTest, cagraRequiresGpuBuild) {
  EXPECT_THROW(build(VectorIndexType::kCagra), facebook::velox::VeloxUserError);
}

TEST_F(GpuLocalVectorIndexTest, minSearchSize) {
  const auto bytes = build(VectorIndexType::kFlat);
  useGpu({.minSearchSize = kNumEmbeddings + 1});
  EXPECT_FALSE(
      LocalVectorIndex(bytes, VectorIndexMetric::kCosine, {}).accelerated());
  useGpu({.minSearchSize = kNumEmbeddings});
  EXPECT_TRUE(
      LocalVectorIndex(bytes, VectorIndexMetric::kCosine, {}).accelerated());
}

TEST_F(GpuLocalVectorIndexTest, kAboveIndexSize) {
  const auto bytes = build(VectorIndexType::kFlat);
  useGpu();
  const LocalVectorIndex index(bytes, VectorIndexMetric::kCosine, {});
  ASSERT_TRUE(index.accelerated());
  const auto results = search(index, /*k=*/kNumEmbeddings + 1);
  EXPECT_EQ(-1, results.ids.back());
  EXPECT_NE(-1, results.ids[kNumEmbeddings - 1]);
}

TEST_F(GpuLocalVectorIndexTest, gpuBuild) {
  const auto exact = exactIds();
  for (const auto type : {VectorIndexType::kIvfFlat, VectorIndexType::kIvfSq}) {
    SCOPED_TRACE(VectorIndexTypeName::toName(type));
    useGpu({.build = true});
    const auto gpuBytes = build(type);
    setLocalVectorIndexAccelerator(nullptr);
    const auto cpuBytes = build(type);

    // The CPU reads the GPU build's bytes as its own.
    const LocalVectorIndex gpuBuilt(
        gpuBytes,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{.nprobe = 16'384});
    const LocalVectorIndex cpuBuilt(
        cpuBytes,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{.nprobe = 16'384});
    EXPECT_FALSE(gpuBuilt.accelerated());
    EXPECT_EQ(type, gpuBuilt.type());
    EXPECT_EQ(kNumEmbeddings, gpuBuilt.size());
    EXPECT_EQ(kDimension, gpuBuilt.dimension());
    EXPECT_EQ(cpuBytes.size(), gpuBytes.size());

    const auto gpuResults = search(gpuBuilt);
    if (type == VectorIndexType::kIvfFlat) {
      EXPECT_EQ(exact, gpuResults.ids);
    } else {
      EXPECT_GE(recall(search(cpuBuilt).ids, gpuResults.ids), 0.95);
    }

    // At the default nprobe, the GPU-trained lists find about what the
    // CPU-trained lists do.
    const LocalVectorIndex gpuDefault(gpuBytes, VectorIndexMetric::kCosine, {});
    const LocalVectorIndex cpuDefault(cpuBytes, VectorIndexMetric::kCosine, {});
    EXPECT_NEAR(
        recall(exact, search(cpuDefault).ids),
        recall(exact, search(gpuDefault).ids),
        0.1);
  }
}

// Reads a stored index and checks that spherical k-means left unit centroids.
// FAISS does not serialize ClusteringParameters, so cp.spherical reads back
// as its default; the centroid norms are what the bytes record.
void expectSphericalCentroids(std::string_view bytes) {
  faiss::ZeroCopyIOReader reader(
      reinterpret_cast<uint8_t*>(const_cast<char*>(bytes.data())),
      bytes.size());
  std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));
  auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
  if (ivf == nullptr || ivf->quantizer == nullptr) {
    EXPECT_NE(ivf, nullptr);
    return;
  }
  std::vector<float> centroid(ivf->d);
  EXPECT_EQ(ivf->nlist, ivf->quantizer->ntotal);
  for (faiss::idx_t i = 0; i < ivf->quantizer->ntotal; ++i) {
    ivf->quantizer->reconstruct(i, centroid.data());
    float norm = 0;
    for (float value : centroid) {
      norm += value * value;
    }
    EXPECT_NEAR(1.f, std::sqrt(norm), 1e-3) << i;
  }
}

// Dequantized embeddings of an id-mapped index, in id-map order.
std::unordered_map<int64_t, int64_t> reconstructedRows(
    std::string_view bytes,
    std::vector<float>& vectors) {
  faiss::ZeroCopyIOReader reader(
      reinterpret_cast<uint8_t*>(const_cast<char*>(bytes.data())),
      bytes.size());
  std::unique_ptr<faiss::Index> index(faiss::read_index(&reader));
  auto* idMap = dynamic_cast<faiss::IndexIDMap*>(index.get());
  if (idMap == nullptr || idMap->index == nullptr) {
    EXPECT_NE(idMap, nullptr);
    return {};
  }
  vectors.resize(idMap->ntotal * idMap->d);
  idMap->index->reconstruct_n(0, idMap->ntotal, vectors.data());
  std::unordered_map<int64_t, int64_t> rowOfId;
  for (faiss::idx_t i = 0; i < idMap->ntotal; ++i) {
    rowOfId.emplace(idMap->id_map[i], i);
  }
  EXPECT_EQ(static_cast<size_t>(idMap->ntotal), rowOfId.size());
  return rowOfId;
}

TEST_F(GpuLocalVectorIndexTest, buildOff) {
  useGpu({.build = false});
  const auto withGpu = build(VectorIndexType::kIvfFlat);
  setLocalVectorIndexAccelerator(nullptr);
  // Without a GPU build, the bytes are the CPU build's.
  EXPECT_EQ(build(VectorIndexType::kIvfFlat), withGpu);
}

// The score of a returned id is the cosine of that embedding, larger first,
// and a short result is padded with -1 only after its last neighbor.
void expectScoreContract(
    const std::vector<float>& queries,
    const std::vector<float>& embeddings,
    const std::unordered_map<int64_t, int64_t>& rowOfId,
    const std::vector<int64_t>& ids,
    const std::vector<float>& scores,
    int64_t numQueries,
    int64_t k,
    float tolerance) {
  ASSERT_EQ(ids.size(), static_cast<size_t>(numQueries * k));
  ASSERT_EQ(scores.size(), ids.size());
  const auto dimension =
      static_cast<int64_t>(embeddings.size() / rowOfId.size());
  for (int64_t query = 0; query < numQueries; ++query) {
    bool padded = false;
    float previous = std::numeric_limits<float>::infinity();
    std::unordered_set<int64_t> seen;
    for (int64_t neighbor = 0; neighbor < k; ++neighbor) {
      const auto slot = query * k + neighbor;
      const auto id = ids[slot];
      if (id == LocalVectorIndex::kPaddingId) {
        padded = true;
        continue;
      }
      EXPECT_FALSE(padded) << query << " " << neighbor;
      EXPECT_TRUE(seen.insert(id).second) << id;
      const auto row = rowOfId.at(id);
      float dot = 0;
      for (int64_t axis = 0; axis < dimension; ++axis) {
        dot += queries[query * dimension + axis] *
            embeddings[row * dimension + axis];
      }
      EXPECT_NEAR(dot, scores[slot], tolerance) << query << " " << id;
      EXPECT_LE(scores[slot], previous + tolerance);
      previous = scores[slot];
    }
  }
}

size_t idMismatches(
    const std::vector<int64_t>& left,
    const std::vector<int64_t>& right) {
  size_t mismatches = 0;
  for (size_t i = 0; i < left.size(); ++i) {
    mismatches += left[i] != right[i];
  }
  return mismatches;
}

// One query searched alone agrees with that query's rows in a batch search.
void expectBatchAgrees(
    const LocalVectorIndex& index,
    const std::vector<float>& queries,
    int64_t query,
    const std::vector<int64_t>& batchIds,
    const std::vector<float>& batchScores,
    int64_t k) {
  const auto dimension = static_cast<int32_t>(queries.size() / kNumQueries);
  std::vector<int64_t> ids(k);
  std::vector<float> scores(k);
  index.search(
      Embeddings(queries.data() + query * dimension, /*size=*/1, dimension),
      k,
      ids,
      scores);
  EXPECT_EQ(
      std::vector<int64_t>(
          batchIds.begin() + query * k, batchIds.begin() + (query + 1) * k),
      ids);
  for (int64_t neighbor = 0; neighbor < k; ++neighbor) {
    EXPECT_NEAR(batchScores[query * k + neighbor], scores[neighbor], 1e-4);
  }
}

TEST_F(GpuLocalVectorIndexTest, scoreIsCosineOfReturnedId) {
  std::unordered_map<int64_t, int64_t> rowOfId;
  for (int64_t i = 0; i < kNumEmbeddings; ++i) {
    rowOfId.emplace(ids_[i], i);
  }
  const auto exact = exactIds();

  const auto bytes = build(VectorIndexType::kFlat);
  useGpu();
  const LocalVectorIndex gpuFlat(bytes, VectorIndexMetric::kCosine, {});
  ASSERT_TRUE(gpuFlat.accelerated());
  const auto gpuFlatResults = search(gpuFlat);
  setLocalVectorIndexAccelerator(nullptr);
  const LocalVectorIndex cpuFlat(bytes, VectorIndexMetric::kCosine, {});
  const auto cpuFlatResults = search(cpuFlat);
  EXPECT_EQ(0, idMismatches(cpuFlatResults.ids, gpuFlatResults.ids));
  expectScoreContract(
      queries_,
      embeddings_,
      rowOfId,
      gpuFlatResults.ids,
      gpuFlatResults.scores,
      kNumQueries,
      kK,
      1e-4);
  expectBatchAgrees(
      gpuFlat, queries_, 0, gpuFlatResults.ids, gpuFlatResults.scores, kK);
  expectBatchAgrees(
      gpuFlat,
      queries_,
      kNumQueries - 1,
      gpuFlatResults.ids,
      gpuFlatResults.scores,
      kK);

  const auto ivfBytes = build(VectorIndexType::kIvfFlat);
  for (const int32_t nprobe : {1, 16'384}) {
    SCOPED_TRACE(nprobe);
    const LocalVectorIndexOptions options{.nprobe = nprobe};
    useGpu();
    const LocalVectorIndex gpu(ivfBytes, VectorIndexMetric::kCosine, options);
    ASSERT_TRUE(gpu.accelerated());
    const auto gpuResults = search(gpu);
    setLocalVectorIndexAccelerator(nullptr);
    const LocalVectorIndex cpu(ivfBytes, VectorIndexMetric::kCosine, options);
    const auto cpuResults = search(cpu);
    EXPECT_EQ(0, idMismatches(cpuResults.ids, gpuResults.ids));
    for (size_t i = 0; i < cpuResults.scores.size(); ++i) {
      EXPECT_NEAR(cpuResults.scores[i], gpuResults.scores[i], 1e-4);
    }
    expectScoreContract(
        queries_,
        embeddings_,
        rowOfId,
        gpuResults.ids,
        gpuResults.scores,
        kNumQueries,
        kK,
        1e-4);
    if (nprobe == 1) {
      EXPECT_NE(exact, gpuResults.ids);
      expectBatchAgrees(
          gpu, queries_, 0, gpuResults.ids, gpuResults.scores, kK);
    } else {
      EXPECT_EQ(exact, gpuResults.ids);
    }
  }

  // HNSW-SQ is rebuilt as CAGRA over the dequantized embeddings, so its score
  // is the cosine of the reconstructed vector.
  const auto hnswBytes = build(VectorIndexType::kHnswSq);
  std::vector<float> reconstructed;
  const auto reconstructedRow = reconstructedRows(hnswBytes, reconstructed);
  useGpu();
  const LocalVectorIndex hnsw(hnswBytes, VectorIndexMetric::kCosine, {});
  ASSERT_TRUE(hnsw.accelerated());
  const auto hnswResults = search(hnsw);
  EXPECT_GE(recall(exact, hnswResults.ids), 0.9);
  expectScoreContract(
      queries_,
      reconstructed,
      reconstructedRow,
      hnswResults.ids,
      hnswResults.scores,
      kNumQueries,
      kK,
      1e-3);

  useGpu({.build = true});
  const auto cagraBytes = build(VectorIndexType::kCagra);
  setLocalVectorIndexAccelerator(nullptr);
  const LocalVectorIndex cagraCpu(cagraBytes, VectorIndexMetric::kCosine, {});
  EXPECT_FALSE(cagraCpu.accelerated());
  const auto cagraCpuResults = search(cagraCpu);
  expectScoreContract(
      queries_,
      embeddings_,
      rowOfId,
      cagraCpuResults.ids,
      cagraCpuResults.scores,
      kNumQueries,
      kK,
      1e-3);
  EXPECT_GE(recall(exact, cagraCpuResults.ids), 0.9);

  useGpu();
  const LocalVectorIndex cagraGpu(
      cagraBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.efSearch = 128});
  ASSERT_TRUE(cagraGpu.accelerated());
  const auto cagraGpuResults = search(cagraGpu);
  expectScoreContract(
      queries_,
      embeddings_,
      rowOfId,
      cagraGpuResults.ids,
      cagraGpuResults.scores,
      kNumQueries,
      kK,
      1e-3);
  EXPECT_GE(recall(exact, cagraGpuResults.ids), 0.9);
}

TEST_F(GpuLocalVectorIndexTest, gpuBuiltIndexSearchesOnGpu) {
  const auto exact = exactIds();
  std::unordered_map<int64_t, int64_t> rowOfId;
  for (int64_t i = 0; i < kNumEmbeddings; ++i) {
    rowOfId.emplace(ids_[i], i);
  }

  useGpu({.build = true});
  const auto ivfBytes = build(VectorIndexType::kIvfFlat);
  expectSphericalCentroids(ivfBytes);
  useGpu();
  const LocalVectorIndex ivf(
      ivfBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.nprobe = 16'384});
  ASSERT_TRUE(ivf.accelerated());
  const auto ivfResults = search(ivf);
  EXPECT_EQ(exact, ivfResults.ids);
  expectScoreContract(
      queries_,
      embeddings_,
      rowOfId,
      ivfResults.ids,
      ivfResults.scores,
      kNumQueries,
      kK,
      1e-4);

  useGpu({.build = true});
  const auto sqBytes = build(VectorIndexType::kIvfSq);
  expectSphericalCentroids(sqBytes);
  setLocalVectorIndexAccelerator(nullptr);
  const auto cpuSq = search(LocalVectorIndex(
      sqBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.nprobe = 16'384}));
  useGpu();
  const LocalVectorIndex sq(
      sqBytes,
      VectorIndexMetric::kCosine,
      LocalVectorIndexOptions{.nprobe = 16'384});
  ASSERT_TRUE(sq.accelerated());
  const auto gpuSq = search(sq);
  EXPECT_GE(recall(cpuSq.ids, gpuSq.ids), 0.95);
  float maxGap = 0;
  size_t compared = 0;
  for (size_t i = 0; i < cpuSq.ids.size(); ++i) {
    if (cpuSq.ids[i] == gpuSq.ids[i] &&
        cpuSq.ids[i] != LocalVectorIndex::kPaddingId) {
      maxGap = std::max(maxGap, std::fabs(cpuSq.scores[i] - gpuSq.scores[i]));
      ++compared;
    }
  }
  EXPECT_GT(compared, cpuSq.ids.size() / 2);
  EXPECT_LT(maxGap, 0.05f);
}

// Flat and IVF agree with the CPU on raw inner products, including tied ids.
TEST_F(GpuLocalVectorIndexTest, rawInnerProductAndTies) {
  const std::vector<float> embeddings{2, 0, 2, 0, 0, 3};
  const std::vector<int64_t> ids{5, 6, 7};
  const std::vector<float> query{2, 1};
  useGpu();
  for (const auto type : {VectorIndexType::kFlat, VectorIndexType::kIvfFlat}) {
    SCOPED_TRACE(VectorIndexTypeName::toName(type));
    std::string bytes;
    LocalVectorIndex::build(
        type,
        VectorIndexMetric::kCosine,
        LocalVectorIndexOptions{},
        Embeddings(embeddings, 2),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    const LocalVectorIndex gpu(bytes, VectorIndexMetric::kCosine, {});
    ASSERT_TRUE(gpu.accelerated());
    std::vector<int64_t> gpuIds(3);
    std::vector<float> gpuScores(3);
    gpu.search(Embeddings(query, 2), /*k=*/3, gpuIds, gpuScores);

    setLocalVectorIndexAccelerator(nullptr);
    const LocalVectorIndex cpu(bytes, VectorIndexMetric::kCosine, {});
    std::vector<int64_t> cpuIds(3);
    std::vector<float> cpuScores(3);
    cpu.search(Embeddings(query, 2), /*k=*/3, cpuIds, cpuScores);
    useGpu();

    EXPECT_THAT(gpuIds, testing::UnorderedElementsAre(5, 6, 7));
    EXPECT_THAT(cpuIds, testing::UnorderedElementsAre(5, 6, 7));
    // Dots with (2, 0), (2, 0) and (0, 3). The two equal vectors both stay.
    EXPECT_NEAR(4, gpuScores[0], 1e-3);
    EXPECT_NEAR(4, gpuScores[1], 1e-3);
    EXPECT_NEAR(3, gpuScores[2], 1e-3);
    EXPECT_THAT(
        std::vector<int64_t>(gpuIds.begin(), gpuIds.begin() + 2),
        testing::UnorderedElementsAre(5, 6));
  }
}

} // namespace
} // namespace facebook::velox::common::vectorindex

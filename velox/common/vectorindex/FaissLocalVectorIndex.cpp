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

#include "velox/common/vectorindex/FaissLocalVectorIndex.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/zerocopy_io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/vectorindex/CallbackWriter.h"
#include "velox/common/vectorindex/SingleThreadedFaiss.h"

namespace facebook::velox::common::vectorindex {

namespace {

constexpr int64_t kMinDefaultNlist{8};
constexpr int64_t kMaxDefaultNlist{16'384};
// The fewest embeddings per list FAISS recommends for training.
constexpr int64_t kMinEmbeddingsPerList{39};
constexpr int32_t kDefaultM{32};
constexpr int32_t kDefaultEfConstruction{200};
constexpr int32_t kDefaultEfSearch{64};

int32_t resolveNlist(std::optional<int32_t> nlist, int64_t numEmbeddings) {
  const int64_t requested = nlist.has_value()
      ? *nlist
      : std::clamp<int64_t>(
            4 * std::sqrt(static_cast<double>(numEmbeddings)),
            kMinDefaultNlist,
            kMaxDefaultNlist);
  return std::max<int64_t>(
      1, std::min(requested, numEmbeddings / kMinEmbeddingsPerList));
}

std::unique_ptr<faiss::Index> makeIvfIndex(
    VectorIndexKind kind,
    faiss::MetricType metric,
    bool spherical,
    const LocalVectorIndexOptions& options,
    int64_t numEmbeddings,
    int32_t dimension) {
  const auto nlist = resolveNlist(options.nlist, numEmbeddings);
  auto quantizer = std::make_unique<faiss::IndexFlat>(dimension, metric);
  std::unique_ptr<faiss::IndexIVF> index;
  if (kind == VectorIndexKind::kIvfFlat) {
    index = std::make_unique<faiss::IndexIVFFlat>(
        quantizer.get(), dimension, nlist, metric);
  } else {
    index = std::make_unique<faiss::IndexIVFScalarQuantizer>(
        quantizer.get(),
        dimension,
        nlist,
        faiss::ScalarQuantizer::QT_8bit,
        metric);
  }
  index->own_fields = true;
  std::ignore = quantizer.release();
  index->cp.spherical = spherical;
  // resolveNlist() keeps kMinEmbeddingsPerList per list except for groups too
  // small for two lists, which FAISS would warn about on every build.
  index->cp.min_points_per_centroid = 1;
  index->nprobe = std::min<size_t>(options.nprobe.value_or(nlist), nlist);
  return index;
}

// Wraps 'index' in an IndexIDMap, which keeps the ids for index types that do
// not keep them.
std::unique_ptr<faiss::Index> withIds(std::unique_ptr<faiss::Index> index) {
  auto idMap = std::make_unique<faiss::IndexIDMap>(index.get());
  idMap->own_fields = true;
  std::ignore = index.release();
  return idMap;
}

std::unique_ptr<faiss::Index> makeIndex(
    VectorIndexKind kind,
    faiss::MetricType metric,
    bool spherical,
    const LocalVectorIndexOptions& options,
    int64_t numEmbeddings,
    int32_t dimension) {
  switch (kind) {
    case VectorIndexKind::kFlat:
      return withIds(std::make_unique<faiss::IndexFlat>(dimension, metric));
    case VectorIndexKind::kIvfFlat:
    case VectorIndexKind::kIvfSq:
      return makeIvfIndex(
          kind, metric, spherical, options, numEmbeddings, dimension);
    case VectorIndexKind::kHnswSq: {
      auto graph = std::make_unique<faiss::IndexHNSWSQ>(
          dimension,
          faiss::ScalarQuantizer::QT_8bit,
          options.m.value_or(kDefaultM),
          metric);
      graph->hnsw.efConstruction =
          options.efConstruction.value_or(kDefaultEfConstruction);
      graph->hnsw.efSearch = options.efSearch.value_or(kDefaultEfSearch);
      return withIds(std::move(graph));
    }
  }
  VELOX_UNREACHABLE();
}

// Returns the kind of a local index in the layout build() writes (see
// makeIndex), and the index that holds its search settings. Throws a user
// error for any other FAISS index.
std::pair<VectorIndexKind, faiss::Index*> classify(faiss::Index* index) {
  if (auto* idMap = dynamic_cast<faiss::IndexIDMap*>(index)) {
    if (dynamic_cast<faiss::IndexHNSWSQ*>(idMap->index)) {
      return {VectorIndexKind::kHnswSq, idMap->index};
    }
    if (dynamic_cast<faiss::IndexFlat*>(idMap->index)) {
      return {VectorIndexKind::kFlat, idMap->index};
    }
  } else if (dynamic_cast<faiss::IndexIVFScalarQuantizer*>(index)) {
    return {VectorIndexKind::kIvfSq, index};
  } else if (dynamic_cast<faiss::IndexIVFFlat*>(index)) {
    return {VectorIndexKind::kIvfFlat, index};
  }
  VELOX_USER_FAIL("Invalid local vector index: unsupported FAISS index");
}

// Returns true if any embedding of 'index', in the layout build() writes, has
// LocalVectorIndex::kPaddingId.
bool hasMissingId(const faiss::Index* index) {
  if (const auto* idMap = dynamic_cast<const faiss::IndexIDMap*>(index)) {
    return std::find(
               idMap->id_map.begin(),
               idMap->id_map.end(),
               LocalVectorIndex::kPaddingId) != idMap->id_map.end();
  }
  const auto* ivf = dynamic_cast<const faiss::IndexIVF*>(index);
  VELOX_CHECK_NOT_NULL(ivf);
  const auto* lists = ivf->invlists;
  for (size_t list = 0; list < lists->nlist; ++list) {
    const faiss::InvertedLists::ScopedIds ids(lists, list);
    const auto* begin = ids.ids;
    const auto* end = begin + lists->list_size(list);
    if (std::find(begin, end, LocalVectorIndex::kPaddingId) != end) {
      return true;
    }
  }
  return false;
}

} // namespace

void FaissLocalVectorIndex::build(
    VectorIndexKind kind,
    VectorIndexMetric metric,
    const LocalVectorIndexOptions& options,
    const Embeddings& embeddings,
    std::span<const int64_t> ids,
    folly::FunctionRef<void(std::string_view)> write) {
  VELOX_CHECK_EQ(ids.size(), embeddings.size());
  VELOX_USER_CHECK(
      std::find(ids.begin(), ids.end(), kPaddingId) == ids.end(),
      "Local vector index embedding id must not be -1");
  SingleThreadedFaiss singleThreaded;
  switch (metric) {
    case VectorIndexMetric::kCosine: {
      auto index = makeIndex(
          kind,
          faiss::METRIC_INNER_PRODUCT,
          /*spherical=*/true,
          options,
          embeddings.size(),
          embeddings.dimension());
      index->train(embeddings.size(), embeddings.data());
      index->add_with_ids(embeddings.size(), embeddings.data(), ids.data());
      CallbackWriter writer(write);
      faiss::write_index(index.get(), &writer);
      return;
    }
  }
  VELOX_UNREACHABLE();
}

namespace {

// Clusters 'numPoints' points of 'dimension' floats at 'points', normalized
// to length 1, into 'numClusters' clusters, at most 'numPoints', with
// spherical k-means. Returns the centroids one after another.
std::vector<float> sphericalKMeans(
    int32_t dimension,
    int64_t numClusters,
    const float* points,
    int64_t numPoints) {
  faiss::ClusteringParameters parameters;
  parameters.spherical = true;
  // The caller chooses the training points, so FAISS neither samples them nor
  // warns that there are few per centroid.
  parameters.min_points_per_centroid = 1;
  parameters.max_points_per_centroid = std::numeric_limits<int>::max();
  faiss::Clustering clustering(dimension, numClusters, parameters);
  faiss::IndexFlatIP index(dimension);
  clustering.train(numPoints, points, index);
  return std::move(clustering.centroids);
}

// Trains 'numClusters' centroids, fewer than 'embeddings', in two levels (see
// FaissLocalVectorIndex::trainCentroids).
std::vector<float> twoLevelSphericalKMeans(
    int64_t numClusters,
    const Embeddings& embeddings) {
  const auto dimension = embeddings.dimension();
  const auto numPoints = embeddings.size();
  const auto coarse = sphericalKMeans(
      dimension,
      static_cast<int64_t>(std::ceil(std::sqrt(numClusters))),
      embeddings.data(),
      numPoints);
  const int64_t numCoarse = coarse.size() / dimension;

  std::vector<faiss::idx_t> assignment(numPoints);
  {
    std::vector<float> scores(numPoints);
    faiss::IndexFlatIP index(dimension);
    index.add(numCoarse, coarse.data());
    index.search(
        numPoints, embeddings.data(), 1, scores.data(), assignment.data());
  }

  // Groups the embeddings by coarse cluster, in their order within each.
  std::vector<int64_t> offsets(numCoarse + 1, 0);
  for (auto cluster : assignment) {
    ++offsets[cluster + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<float> grouped(numPoints * dimension);
  {
    auto next = offsets;
    for (int64_t i = 0; i < numPoints; ++i) {
      std::copy_n(
          embeddings.data() + i * dimension,
          dimension,
          grouped.data() + next[assignment[i]]++ * dimension);
    }
  }

  // Coarse cluster i gets floor(c(i + 1)) - floor(c(i)) fine clusters, where
  // c(i) = numClusters * offsets[i] / numPoints: the shares add up to
  // 'numClusters', and none exceeds its cluster's embeddings because
  // 'numClusters' is less than 'numPoints'.
  const auto boundary = [&](int64_t i) {
    return numClusters * offsets[i] / numPoints;
  };
  // Threads take coarse clusters one at a time because their sizes differ.
  // An exception must not leave the parallel region, so the first one stops
  // the others and is rethrown after it.
  std::vector<std::vector<float>> fine(numCoarse);
  std::atomic<int64_t> nextCluster{0};
  std::exception_ptr error;
  std::mutex errorMutex;
#pragma omp parallel
  {
    for (auto i = nextCluster++; i < numCoarse; i = nextCluster++) {
      const auto share = boundary(i + 1) - boundary(i);
      if (share == 0) {
        continue;
      }
      try {
        fine[i] = sphericalKMeans(
            dimension,
            share,
            grouped.data() + offsets[i] * dimension,
            offsets[i + 1] - offsets[i]);
      } catch (...) {
        std::lock_guard<std::mutex> lock(errorMutex);
        if (!error) {
          error = std::current_exception();
        }
        nextCluster = numCoarse;
      }
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }

  std::vector<float> centroids;
  centroids.reserve(numClusters * dimension);
  for (const auto& clusterCentroids : fine) {
    centroids.insert(
        centroids.end(), clusterCentroids.begin(), clusterCentroids.end());
  }
  return centroids;
}

} // namespace

// static
std::vector<float> FaissLocalVectorIndex::trainCentroids(
    VectorIndexMetric metric,
    int64_t numClusters,
    const Embeddings& embeddings) {
  VELOX_CHECK_GT(numClusters, 0);
  if (numClusters >= embeddings.size()) {
    return {
        embeddings.data(),
        embeddings.data() + embeddings.size() * embeddings.dimension()};
  }
  // TODO: Cap the threads of concurrent trainings, which each take all of
  // FAISS's OpenMP threads and so oversubscribe the host.
  switch (metric) {
    case VectorIndexMetric::kCosine:
      if (numClusters > kMaxFlatClusters) {
        return twoLevelSphericalKMeans(numClusters, embeddings);
      }
      return sphericalKMeans(
          embeddings.dimension(),
          numClusters,
          embeddings.data(),
          embeddings.size());
  }
  VELOX_UNREACHABLE();
}

// static
int64_t FaissLocalVectorIndex::size(std::string_view bytes) {
  // Every layout build() writes starts with a four-character code, then the
  // index header: the dimension, an int32, and the number of embeddings, an
  // int64.
  static constexpr size_t kNumEmbeddingsOffset{8};
  VELOX_USER_CHECK_GE(
      bytes.size(),
      kNumEmbeddingsOffset + sizeof(int64_t),
      "Invalid local vector index: too short");
  const auto code = bytes.substr(0, 4);
  VELOX_USER_CHECK(
      code == "IxMp" || code == "IwFl" || code == "IwSq",
      "Invalid local vector index: unsupported FAISS index");
  int64_t numEmbeddings;
  std::memcpy(
      &numEmbeddings, bytes.data() + kNumEmbeddingsOffset, sizeof(int64_t));
  VELOX_USER_CHECK_GT(
      numEmbeddings, 0, "Local vector index must have at least one embedding");
  return numEmbeddings;
}

FaissLocalVectorIndex::FaissLocalVectorIndex(
    std::string_view bytes,
    VectorIndexMetric metric,
    const LocalVectorIndexOptions& searchOptions) {
  VELOX_CHECK(
      !searchOptions.nlist.has_value(),
      "Local vector index search options may not set nlist");
  VELOX_CHECK(
      !searchOptions.m.has_value(),
      "Local vector index search options may not set m");
  VELOX_CHECK(
      !searchOptions.efConstruction.has_value(),
      "Local vector index search options may not set efConstruction");
  // Some FAISS versions take a non-const pointer; the reader only reads it.
  faiss::ZeroCopyIOReader reader(
      reinterpret_cast<uint8_t*>(const_cast<char*>(bytes.data())),
      bytes.size());
  try {
    index_.reset(faiss::read_index(&reader));
  } catch (const faiss::FaissException& e) {
    VELOX_USER_FAIL("Invalid local vector index: {}", e.what());
  }
  switch (metric) {
    case VectorIndexMetric::kCosine:
      VELOX_USER_CHECK(
          index_->metric_type == faiss::METRIC_INNER_PRODUCT,
          "Local vector index does not match the metric: {}",
          metric);
      break;
  }
  VELOX_USER_CHECK_GT(
      index_->ntotal, 0, "Local vector index must have at least one embedding");

  const auto [kind, index] = classify(index_.get());
  kind_ = kind;
  VELOX_USER_CHECK(
      !hasMissingId(index_.get()),
      "Local vector index embedding id must not be -1");
  searchOptions.validate(kind_);
  if (searchOptions.nprobe.has_value()) {
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    ivf->nprobe = std::min<size_t>(*searchOptions.nprobe, ivf->nlist);
  }
  if (searchOptions.efSearch.has_value()) {
    dynamic_cast<faiss::IndexHNSW*>(index)->hnsw.efSearch =
        *searchOptions.efSearch;
  }
}

FaissLocalVectorIndex::~FaissLocalVectorIndex() = default;

VectorIndexKind FaissLocalVectorIndex::kind() const {
  return kind_;
}

int32_t FaissLocalVectorIndex::dimension() const {
  return index_->d;
}

int64_t FaissLocalVectorIndex::size() const {
  return index_->ntotal;
}

void FaissLocalVectorIndex::search(
    const Embeddings& queries,
    int64_t k,
    std::span<int64_t> ids,
    std::span<float> scores) const {
  VELOX_CHECK_GE(k, 1);
  VELOX_USER_CHECK_EQ(
      queries.dimension(),
      dimension(),
      "Query embedding length must match the vector index's");
  VELOX_CHECK_EQ(ids.size(), queries.size() * k);
  VELOX_CHECK_EQ(scores.size(), queries.size() * k);
  SingleThreadedFaiss singleThreaded;
  const auto* ivf = dynamic_cast<const faiss::IndexIVF*>(index_.get());
  if (ivf == nullptr) {
    index_->search(
        queries.size(), queries.data(), k, scores.data(), ids.data());
    return;
  }
  // IndexIVF::search adds its statistics to a process-wide global without
  // synchronization, so concurrent searches race on it.
  // IndexIVF::search_preassigned takes the caller's statistics instead.
  // TODO: Call IndexIVF::search once it can take the caller's statistics.
  const auto numProbes = static_cast<faiss::idx_t>(ivf->nprobe);
  std::vector<faiss::idx_t> lists(queries.size() * numProbes);
  std::vector<float> listScores(queries.size() * numProbes);
  ivf->quantizer->search(
      queries.size(),
      queries.data(),
      numProbes,
      listScores.data(),
      lists.data());
  faiss::IndexIVFStats stats;
  ivf->search_preassigned(
      queries.size(),
      queries.data(),
      k,
      lists.data(),
      listScores.data(),
      scores.data(),
      ids.data(),
      /*store_pairs=*/false,
      /*params=*/nullptr,
      &stats);
}

namespace {

class FaissReader : public LocalVectorIndexReader {
 public:
  FaissReader(VectorIndexMetric metric, LocalVectorIndexOptions searchOptions)
      : metric_{metric}, searchOptions_{std::move(searchOptions)} {}

  std::unique_ptr<LocalVectorIndex> open(
      std::string_view bytes) const override {
    return std::make_unique<FaissLocalVectorIndex>(
        bytes, metric_, searchOptions_);
  }

 private:
  const VectorIndexMetric metric_;
  const LocalVectorIndexOptions searchOptions_;
};

class FaissFormat : public LocalVectorIndexFormat {
 public:
  std::unique_ptr<LocalVectorIndexReader> makeReader(
      VectorIndexMetric metric,
      const LocalVectorIndex::SearchOptions& searchOptions) const override {
    return std::make_unique<FaissReader>(
        metric,
        LocalVectorIndexOptions::parseSearch(
            {searchOptions.begin(), searchOptions.end()}));
  }
};

} // namespace

// static
std::shared_ptr<LocalVectorIndexFormat> FaissLocalVectorIndex::format() {
  return std::make_shared<FaissFormat>();
}

} // namespace facebook::velox::common::vectorindex

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

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <tuple>

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
    VectorIndexType type,
    faiss::MetricType metric,
    bool spherical,
    const LocalVectorIndexOptions& options,
    int64_t numEmbeddings,
    int32_t dimension) {
  const auto nlist = resolveNlist(options.nlist, numEmbeddings);
  auto quantizer = std::make_unique<faiss::IndexFlat>(dimension, metric);
  std::unique_ptr<faiss::IndexIVF> index;
  if (type == VectorIndexType::kIvfFlat) {
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
  index->nprobe = std::min(options.nprobe.value_or((nlist + 9) / 10), nlist);
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
    VectorIndexType type,
    faiss::MetricType metric,
    bool spherical,
    const LocalVectorIndexOptions& options,
    int64_t numEmbeddings,
    int32_t dimension) {
  switch (type) {
    case VectorIndexType::kFlat:
      return withIds(std::make_unique<faiss::IndexFlat>(dimension, metric));
    case VectorIndexType::kIvfFlat:
    case VectorIndexType::kIvfSq:
      return makeIvfIndex(
          type, metric, spherical, options, numEmbeddings, dimension);
    case VectorIndexType::kHnswSq: {
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
    case VectorIndexType::kCagra: {
      // Stored as IndexHNSWCagra, wrapped so the embeddings keep their ids.
      // base_level_only keeps the CAGRA graph as the one level both the CPU
      // and the GPU search. The GPU build fills it.
      auto graph = std::make_unique<faiss::IndexHNSWCagra>(
          dimension, options.m.value_or(kDefaultM), metric);
      graph->base_level_only = true;
      graph->hnsw.efSearch = options.efSearch.value_or(kDefaultEfSearch);
      return withIds(std::move(graph));
    }
  }
  VELOX_UNREACHABLE();
}

// Returns the type of a local index in the layout build() writes (see
// makeIndex), and the index that holds its search settings. Throws a user
// error for any other FAISS index.
std::pair<VectorIndexType, faiss::Index*> classify(faiss::Index* index) {
  if (auto* idMap = dynamic_cast<faiss::IndexIDMap*>(index)) {
    if (dynamic_cast<faiss::IndexHNSWCagra*>(idMap->index)) {
      return {VectorIndexType::kCagra, idMap->index};
    }
    if (dynamic_cast<faiss::IndexHNSWSQ*>(idMap->index)) {
      return {VectorIndexType::kHnswSq, idMap->index};
    }
    if (dynamic_cast<faiss::IndexFlat*>(idMap->index)) {
      return {VectorIndexType::kFlat, idMap->index};
    }
  } else if (dynamic_cast<faiss::IndexHNSWCagra*>(index)) {
    return {VectorIndexType::kCagra, index};
  } else if (dynamic_cast<faiss::IndexIVFScalarQuantizer*>(index)) {
    return {VectorIndexType::kIvfSq, index};
  } else if (dynamic_cast<faiss::IndexIVFFlat*>(index)) {
    return {VectorIndexType::kIvfFlat, index};
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
  if (dynamic_cast<const faiss::IndexHNSW*>(index) != nullptr) {
    // A CAGRA graph without an id map numbers its embeddings from 0.
    return false;
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

std::mutex& acceleratorMutex() {
  static std::mutex mutex;
  return mutex;
}

std::shared_ptr<const LocalVectorIndexAccelerator>& acceleratorSlot() {
  static std::shared_ptr<const LocalVectorIndexAccelerator> accelerator;
  return accelerator;
}

} // namespace

void setLocalVectorIndexAccelerator(
    std::shared_ptr<const LocalVectorIndexAccelerator> accelerator) {
  const std::lock_guard<std::mutex> lock(acceleratorMutex());
  acceleratorSlot() = std::move(accelerator);
}

std::shared_ptr<const LocalVectorIndexAccelerator>
localVectorIndexAccelerator() {
  const std::lock_guard<std::mutex> lock(acceleratorMutex());
  return acceleratorSlot();
}

void LocalVectorIndex::build(
    VectorIndexType type,
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
          type,
          faiss::METRIC_INNER_PRODUCT,
          /*spherical=*/true,
          options,
          embeddings.size(),
          embeddings.dimension());
      const auto accelerator = localVectorIndexAccelerator();
      if (accelerator == nullptr ||
          !accelerator->build(type, *index, embeddings, ids)) {
        // A CAGRA graph is built by NN-descent on the GPU and stored as an
        // IndexHNSWCagra. There is no CPU build of it.
        VELOX_USER_CHECK(
            type != VectorIndexType::kCagra,
            "A CAGRA local vector index is built on the GPU");
        index->train(embeddings.size(), embeddings.data());
        index->add_with_ids(embeddings.size(), embeddings.data(), ids.data());
      }
      CallbackWriter writer(write);
      faiss::write_index(index.get(), &writer);
      return;
    }
  }
  VELOX_UNREACHABLE();
}

LocalVectorIndex::LocalVectorIndex(
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
          VectorIndexMetricName::toName(metric));
      break;
  }
  VELOX_USER_CHECK_GT(
      index_->ntotal, 0, "Local vector index must have at least one embedding");

  const auto [type, index] = classify(index_.get());
  type_ = type;
  VELOX_USER_CHECK(
      !hasMissingId(index_.get()),
      "Local vector index embedding id must not be -1");
  searchOptions.validate(type_);
  if (searchOptions.nprobe.has_value()) {
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    ivf->nprobe = std::min<size_t>(*searchOptions.nprobe, ivf->nlist);
  }
  if (searchOptions.efSearch.has_value()) {
    dynamic_cast<faiss::IndexHNSW*>(index)->hnsw.efSearch =
        *searchOptions.efSearch;
  }
  if (const auto accelerator = localVectorIndexAccelerator()) {
    accelerated_ = accelerator->load(type_, *index_);
  }
}

LocalVectorIndex::~LocalVectorIndex() = default;

VectorIndexType LocalVectorIndex::type() const {
  return type_;
}

int32_t LocalVectorIndex::dimension() const {
  return index_->d;
}

int64_t LocalVectorIndex::size() const {
  return index_->ntotal;
}

void LocalVectorIndex::search(
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
  if (accelerated_ != nullptr &&
      accelerated_->search(queries, k, ids, scores)) {
    return;
  }
  SingleThreadedFaiss singleThreaded;
  index_->search(queries.size(), queries.data(), k, scores.data(), ids.data());
}

} // namespace facebook::velox::common::vectorindex

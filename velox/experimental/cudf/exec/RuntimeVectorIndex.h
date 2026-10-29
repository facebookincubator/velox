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
#pragma once

#include <cudf/column/column_view.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// One hit from a packed query. buildRow is -1 when the slot is empty.
struct VectorJoinHit {
  int32_t probeRow{-1};
  int64_t buildRow{-1};
  int64_t buildId{0};
  float score{0};
};

/// Per-shard vector index. The build column views must outlive the index.
/// IVF-PQ is trained from a bounded prefix and extended one batch at a time.
/// Shards that are too small for clustering are searched exactly.
class RuntimeVectorIndex {
 public:
  struct Batch {
    cudf::column_view vectors;
    cudf::column_view buildIds;
    int64_t rowStart{0};
  };

  struct Options {
    int32_t nLists{1};
    int32_t pqBits{8};
    int32_t pqDim{0};
    int32_t trainRows{1};
    /// True when a larger score is a closer match.
    bool largerIsBetter{false};
    bool cosine{false};
  };

  ~RuntimeVectorIndex();

  RuntimeVectorIndex(const RuntimeVectorIndex&) = delete;
  RuntimeVectorIndex& operator=(const RuntimeVectorIndex&) = delete;

  int64_t indexedRows() const;
  int32_t dimension() const;
  int32_t nLists() const;
  bool approximate() const;

  /// Searches packed, valid probe vectors. 'queryRows' maps each packed query
  /// back to its row in the probe batch. Results are grouped by query, with
  /// 'k' slots each, and already reranked by the exact score.
  std::vector<VectorJoinHit> search(
      const float* queries,
      const int32_t* queryRows,
      int64_t numQueries,
      int32_t k,
      int32_t candidateK,
      int32_t nProbes,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

 private:
  struct Impl;
  explicit RuntimeVectorIndex(std::unique_ptr<Impl> impl);

  friend std::unique_ptr<RuntimeVectorIndex> buildRuntimeVectorIndex(
      const std::vector<Batch>& batches,
      const Options& options,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  std::unique_ptr<Impl> impl_;
};

/// Packs the valid rows of an ARRAY(REAL) column into a dense matrix.
struct PackedVectors {
  int64_t count{0};
  int32_t dimension{0};
  std::unique_ptr<rmm::device_uvector<float>> values;
  std::unique_ptr<rmm::device_uvector<int32_t>> rowIds;
};

PackedVectors packQueryVectors(
    const cudf::column_view& vectors,
    int32_t expectedDimension,
    bool cosine,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

std::unique_ptr<RuntimeVectorIndex> buildRuntimeVectorIndex(
    const std::vector<RuntimeVectorIndex::Batch>& batches,
    const RuntimeVectorIndex::Options& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox

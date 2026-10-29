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

#pragma once

#include <cstdint>
#include <string_view>

#include <folly/Function.h>

#include "velox/common/vectorindex/VectorIndexMetric.h"

namespace facebook::velox::common::vectorindex {

/// The codec of a vector index: the centroids of its clusters, serialized as a
/// FAISS index. A cluster's id is its centroid's position. The codec is the
/// plain FAISS serialization, as faiss::write_index writes it, so
/// faiss::read_index rebuilds it with no container around it.
class VectorIndexCodec {
 public:
  /// Serializes 'numClusters' centroids of 'dimension' floats, stored one
  /// after another in cluster-id order, into a codec for 'metric'. Passes the
  /// codec's bytes to 'write' in order, a chunk at a time, so the caller can
  /// place them without an intermediate copy. Normalizes 'centroids' in place
  /// for kCosine.
  static void pack(
      float* centroids,
      int64_t numClusters,
      int32_t dimension,
      VectorIndexMetric metric,
      folly::FunctionRef<void(std::string_view)> write);
};

} // namespace facebook::velox::common::vectorindex

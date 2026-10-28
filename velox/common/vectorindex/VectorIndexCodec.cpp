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

#include "velox/common/vectorindex/VectorIndexCodec.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

namespace {

// Passes each chunk FAISS writes to a callback.
struct CallbackWriter : faiss::IOWriter {
  explicit CallbackWriter(folly::FunctionRef<void(std::string_view)> write)
      : write_{write} {}

  size_t operator()(const void* ptr, size_t size, size_t numItems) override {
    write_(std::string_view(static_cast<const char*>(ptr), size * numItems));
    return numItems;
  }

 private:
  folly::FunctionRef<void(std::string_view)> write_;
};

} // namespace

void VectorIndexCodec::pack(
    float* centroids,
    int64_t numClusters,
    int32_t dimension,
    VectorIndexMetric metric,
    folly::FunctionRef<void(std::string_view)> write) {
  VELOX_CHECK_GT(numClusters, 0);
  VELOX_CHECK_GT(dimension, 0);
  switch (metric) {
    case VectorIndexMetric::kCosine: {
      faiss::fvec_renorm_L2(dimension, numClusters, centroids);
      faiss::IndexFlatIP index(dimension);
      index.add(numClusters, centroids);
      CallbackWriter writer(write);
      faiss::write_index(&index, &writer);
      return;
    }
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::common::vectorindex

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/index/VectorIndexUtility.h"

#include <cmath>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {

faiss::MetricType toFaissMetric(VectorDistanceMetric metric) {
  switch (metric) {
    case VectorDistanceMetric::kL2:
      return faiss::METRIC_L2;
    case VectorDistanceMetric::kCosine:
    case VectorDistanceMetric::kDotProduct:
      return faiss::METRIC_INNER_PRODUCT;
    default:
      NIMBLE_UNREACHABLE(
          "Unknown distance metric: {}", static_cast<int>(metric));
  }
}

void normalizeVectors(
    VectorDistanceMetric metric,
    uint64_t numVectors,
    uint32_t dimensions,
    float* vectors) {
  NIMBLE_CHECK_NOT_NULL(vectors, "Vector data must not be null");
  if (metric != VectorDistanceMetric::kCosine) {
    return;
  }

  for (uint64_t i = 0; i < numVectors; ++i) {
    float* vector = vectors + i * dimensions;
    float squaredNorm{0};
    for (uint32_t j = 0; j < dimensions; ++j) {
      squaredNorm += vector[j] * vector[j];
    }
    const auto norm = std::sqrt(squaredNorm);
    if (norm > 0) {
      for (uint32_t j = 0; j < dimensions; ++j) {
        vector[j] /= norm;
      }
    }
  }
}

serialization::VectorDistanceMetric toSerializedMetric(
    VectorDistanceMetric metric) {
  switch (metric) {
    case VectorDistanceMetric::kL2:
      return serialization::VectorDistanceMetric_L2;
    case VectorDistanceMetric::kCosine:
      return serialization::VectorDistanceMetric_Cosine;
    case VectorDistanceMetric::kDotProduct:
      return serialization::VectorDistanceMetric_DotProduct;
    default:
      NIMBLE_UNREACHABLE(
          "Unknown distance metric: {}", static_cast<int>(metric));
  }
}

VectorDistanceMetric fromSerializedMetric(int8_t metric) {
  switch (metric) {
    case static_cast<int8_t>(serialization::VectorDistanceMetric_L2):
      return VectorDistanceMetric::kL2;
    case static_cast<int8_t>(serialization::VectorDistanceMetric_Cosine):
      return VectorDistanceMetric::kCosine;
    case static_cast<int8_t>(serialization::VectorDistanceMetric_DotProduct):
      return VectorDistanceMetric::kDotProduct;
    default:
      NIMBLE_CHECK_FILE(
          false,
          "Unknown FlatBuffer distance metric: {}",
          static_cast<int>(metric));
      return VectorDistanceMetric::kL2;
  }
}

serialization::VectorIndexType toSerializedIndexType(
    VectorIndexType indexType) {
  switch (indexType) {
    case VectorIndexType::kIvfFlat:
      return serialization::VectorIndexType_IVF_FLAT;
    case VectorIndexType::kIvfSq8:
      return serialization::VectorIndexType_IVF_SQ8;
    case VectorIndexType::kIvfPq:
      return serialization::VectorIndexType_IVF_PQ;
    case VectorIndexType::kIvfRaBitQ:
      return serialization::VectorIndexType_IVF_RABITQ;
    case VectorIndexType::kHnswSq8:
      return serialization::VectorIndexType_HNSW_SQ8;
    default:
      NIMBLE_UNREACHABLE(
          "Unknown vector index type: {}", static_cast<int>(indexType));
  }
}

VectorIndexType fromSerializedIndexType(int8_t indexType) {
  switch (indexType) {
    case static_cast<int8_t>(serialization::VectorIndexType_IVF_FLAT):
      return VectorIndexType::kIvfFlat;
    case static_cast<int8_t>(serialization::VectorIndexType_IVF_SQ8):
      return VectorIndexType::kIvfSq8;
    case static_cast<int8_t>(serialization::VectorIndexType_IVF_PQ):
      return VectorIndexType::kIvfPq;
    case static_cast<int8_t>(serialization::VectorIndexType_IVF_RABITQ):
      return VectorIndexType::kIvfRaBitQ;
    case static_cast<int8_t>(serialization::VectorIndexType_HNSW_SQ8):
      return VectorIndexType::kHnswSq8;
    default:
      NIMBLE_CHECK_FILE(
          false,
          "Unknown FlatBuffer index type: {}",
          static_cast<int>(indexType));
      return VectorIndexType::kIvfFlat;
  }
}

} // namespace facebook::nimble::index

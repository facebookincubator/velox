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
#pragma once

#include <cstdint>

#include <faiss/MetricType.h>

#include "velox/dwio/nimble/index/VectorIndexConfig.h"
#include "velox/dwio/nimble/tablet/VectorIndexGenerated.h"

namespace facebook::nimble::index {

/// Converts a Nimble vector distance metric to its FAISS representation.
faiss::MetricType toFaissMetric(VectorDistanceMetric metric);

/// Normalizes vectors in place when cosine similarity is configured.
void normalizeVectors(
    VectorDistanceMetric metric,
    uint64_t numVectors,
    uint32_t dimensions,
    float* vectors);

/// Converts a Nimble vector distance metric to its serialized representation.
serialization::VectorDistanceMetric toSerializedMetric(
    VectorDistanceMetric metric);

/// Validates and converts a serialized vector distance metric byte.
VectorDistanceMetric fromSerializedMetric(int8_t metric);

/// Converts a Nimble vector index type to its serialized representation.
serialization::VectorIndexType toSerializedIndexType(VectorIndexType indexType);

/// Validates and converts a serialized vector index type byte.
VectorIndexType fromSerializedIndexType(int8_t indexType);

} // namespace facebook::nimble::index

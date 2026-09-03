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
#include <memory>
#include <vector>

#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook {

// Generates `batchCount` fuzzed flat row batches of `size` rows conforming to
// `type`, seeded deterministically by `seed`.
std::vector<velox::RowVectorPtr> generateBatches(
    const std::shared_ptr<const velox::RowType>& type,
    size_t batchCount,
    size_t size,
    uint32_t seed,
    velox::memory::MemoryPool& pool);

struct ChunkSizeResults {
  uint32_t stripeCount;
  uint32_t minChunkCount;
  uint32_t maxChunkCount;
};

// Reads back every stream of every stripe from `reader` and asserts that chunk
// raw sizes stay within [minStreamChunkRawSize, maxStreamChunkRawSize] (modulo
// a tolerance and string-value edge cases), returning per-stream chunk counts.
ChunkSizeResults validateChunkSize(
    nimble::BatchReader& reader,
    uint64_t minStreamChunkRawSize,
    uint64_t maxStreamChunkRawSize);

} // namespace facebook

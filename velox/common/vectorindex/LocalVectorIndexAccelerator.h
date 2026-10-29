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
#include <memory>
#include <span>

#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/VectorIndexType.h"

namespace faiss {
struct Index;
} // namespace faiss

namespace facebook::velox::common::vectorindex {

/// Runs the work of local indexes (see LocalVectorIndex) on a device such as a
/// GPU. A local index keeps its bytes in the FAISS serialization either way,
/// so a plan reads and writes the same intermediate results with or without
/// an accelerator. LocalVectorIndex uses the accelerator set by
/// setLocalVectorIndexAccelerator(), if any, and the FAISS CPU index for the
/// work the accelerator declines.
class LocalVectorIndexAccelerator {
 public:
  /// A copy of a local index on the device.
  class Index {
   public:
    virtual ~Index() = default;

    /// Searches as LocalVectorIndex::search() does, with the same arguments
    /// and results. Returns false, having written nothing, to have the CPU
    /// index search instead.
    virtual bool search(
        const Embeddings& queries,
        int64_t k,
        std::span<int64_t> ids,
        std::span<float> scores) const = 0;
  };

  virtual ~LocalVectorIndexAccelerator() = default;

  /// Returns a device copy of 'index', a local index of 'type' read from its
  /// bytes with its search settings applied, or null to search 'index' on the
  /// CPU. The copy may refer to 'index', which outlives it.
  virtual std::unique_ptr<Index> load(
      VectorIndexType type,
      const faiss::Index& index) const = 0;

  /// Trains 'index', an empty local index of 'type' as
  /// LocalVectorIndex::build() makes it, on 'embeddings', then adds them with
  /// 'ids'. Returns false, leaving 'index' empty, to have build() train and
  /// fill it on the CPU. A device build need not give the same bytes as the
  /// CPU build, nor the same bytes on every run.
  virtual bool build(
      VectorIndexType type,
      faiss::Index& index,
      const Embeddings& embeddings,
      std::span<const int64_t> ids) const = 0;
};

/// Sets the accelerator for local indexes read or built from then on. Null
/// keeps all their work on the CPU, as it is by default. Local indexes already
/// read keep the accelerator they were read with.
void setLocalVectorIndexAccelerator(
    std::shared_ptr<const LocalVectorIndexAccelerator> accelerator);

/// The accelerator setLocalVectorIndexAccelerator() last set, or null.
std::shared_ptr<const LocalVectorIndexAccelerator>
localVectorIndexAccelerator();

} // namespace facebook::velox::common::vectorindex

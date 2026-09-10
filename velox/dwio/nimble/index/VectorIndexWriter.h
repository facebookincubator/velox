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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/VectorIndexConfig.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::index {

/// Builds a FAISS-based vector similarity search index during file writes.
///
/// Accumulates float vectors from each configured column during write() calls,
/// then trains and serializes the FAISS indexes during close(). Each FAISS blob
/// is stored in a metadata section referenced by one optional directory.
class VectorIndexWriter {
 public:
  /// Creates a VectorIndexWriter from one or more index configurations.
  static std::unique_ptr<VectorIndexWriter> create(
      std::span<const VectorIndexConfig> configs,
      const velox::RowTypePtr& inputType,
      velox::memory::MemoryPool* pool);

  virtual ~VectorIndexWriter();

  /// Extracts float vectors from the configured column and buffers them.
  virtual void write(const velox::VectorPtr& input);

  /// Writes each FAISS blob as a metadata section and their directory as an
  /// optional section.
  virtual void close(
      const CreateMetadataSectionFn& createMetadataFn,
      const WriteOptionalSectionFn& writeMetadataFn);

 private:
  // Collects all state required to build one configured index.
  struct Accumulator {
    // Defines how vectors are encoded and indexed.
    VectorIndexConfig config;

    // Locates the vector column in each top-level row batch.
    velox::column_index_t columnIndex{0};

    // Stores vectors contiguously in file-row order. The values for row i
    // occupy [i * dimensions, (i + 1) * dimensions) until close() builds the
    // index and releases this pool-tracked buffer.
    velox::BufferPtr vectors;

    // Tracks the number of top-level rows represented in vectors.
    uint64_t numVectors{0};
  };

  // Owns an index between FAISS serialization and metadata-section persistence.
  struct SerializedIndex {
    // Contains the complete pool-tracked FAISS index representation.
    velox::BufferPtr serializedData;

    // Limits the view to bytes emitted by FAISS rather than buffer capacity.
    size_t serializedSize{0};

    // Records the effective IVF partition count after data-size clamping.
    uint32_t numPartitions{0};
  };

  // Describes a persisted index while assembling the output directory.
  struct PersistedIndex {
    // Identifies the matching accumulator.
    size_t accumulatorIndex{0};

    // Records the effective IVF partition count after data-size clamping.
    uint32_t numPartitions{0};

    // Locates the serialized FAISS blob in the Nimble file.
    MetadataSection indexSection;
  };

  VectorIndexWriter(
      std::vector<Accumulator> accumulators,
      velox::memory::MemoryPool* pool);

  // Builds the FAISS index from accumulated vectors.
  SerializedIndex buildIndex(Accumulator& accumulator) const;

  // Grows an accumulator geometrically while enforcing its configured limit.
  void ensureVectorCapacity(Accumulator& accumulator, size_t numValues) const;

  // Serializes descriptors for every successfully built vector index.
  std::string serializeDirectory(std::span<const PersistedIndex> indexes) const;

  // Accounts for the potentially large vector accumulation buffer.
  velox::memory::MemoryPool* const pool_;

  // Holds one independent accumulator per configured vector index.
  std::vector<Accumulator> accumulators_;

  // Prevents writes and duplicate finalization after close().
  bool closed_{false};
};

/// Creates a non-null writer for one or more index configurations. The
/// implementation must consume or copy configs during the call and must not
/// retain the span storage.
///
/// Injected through WriterOptions rather than resolved from a global so that a
/// writer which never configures a vector index does not have to link an index
/// implementation, and the heavyweight similarity-search libraries it depends
/// on.
using VectorIndexWriterFactory =
    std::function<std::unique_ptr<VectorIndexWriter>(
        std::span<const VectorIndexConfig> configs,
        const velox::RowTypePtr& inputType,
        velox::memory::MemoryPool* pool)>;

} // namespace facebook::nimble::index

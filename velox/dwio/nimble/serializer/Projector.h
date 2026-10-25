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

#include <memory>
#include <string_view>
#include <vector>

#include "folly/io/Cursor.h"
#include "folly/io/IOBuf.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::nimble::serde {

using Subfield = velox::common::Subfield;

/// Projects columns and subfields from serialized Nimble data without decoding.
///
/// Copies only the byte ranges for selected columns/subfields, preserving
/// compression and avoiding decode-encode overhead.
///
/// Thread-safe: the projector is immutable after construction, so one instance
/// can serve concurrent project() calls.
///
/// Supported projections:
/// - Top-level columns: "a", "b", "c"
/// - Nested struct fields: "a.b.c", "struct_col.field1"
/// - FlatMap keys: "flatmap_col[\"key1\"]"
///
/// Not supported (requires re-encoding):
/// - Regular MAP key projection
/// - Array element slicing
///
/// Example usage:
/// @code
///   auto inputSchema = nimble::convertToNimbleType(veloxType);
///   std::vector<Subfield> subfields = {
///       Subfield("a"),
///       Subfield("b.c"),
///       Subfield("flatmap[\"key1\"]"),
///   };
///
///   Projector projector(inputSchema, subfields, pool, {});
///
///   // Project multiple buffers
///   for (const auto& buffer : buffers) {
///     auto projected = projector.project(buffer);
///   }
///
///   // Get output schema for deserialization
///   auto outputSchema = projector.projectedSchema();
/// @endcode
class Projector {
 public:
  struct Options {
    /// Output serialization format version. Defaults to kProjection (the
    /// Projector-specific version byte for the two-array sparse trailer).
    /// kLegacyCompact is read-only and rejected at construction.
    SerializationVersion projectVersion{SerializationVersion::kProjection};

    /// Encoding type for the indices array of the sparse stream-sizes
    /// trailer. Supported types: Trivial, Varint, Delta, FixedBitWidth.
    EncodingType streamIndicesEncodingType{EncodingType::FixedBitWidth};

    /// Encoding type for the sizes array of the sparse stream-sizes
    /// trailer. Supported types: Trivial, Varint, Delta, FixedBitWidth.
    EncodingType streamSizesEncodingType{EncodingType::FixedBitWidth};

    /// Optional velox type with up-to-date column names from the current
    /// table schema. After schema evolution (e.g., column renames), the
    /// inputSchema (nimble type from serialization time) may have outdated
    /// names. When set, inputSchema column names are updated to match
    /// projectType via positional matching.
    /// When null, projectSubfield names must match inputSchema exactly.
    velox::TypePtr projectType{nullptr};
  };

  /// @param inputSchema Nimble schema of serialized data.
  /// @param projectSubfields Columns to project (using projectType names if
  ///        set).
  /// @param pool Memory pool for encoding/decoding offsets in sparse format.
  ///        Must not be null.
  /// @param options Includes optional projectType for schema evolution.
  /// @throws If a subfield path cannot be resolved against the schema.
  Projector(
      std::shared_ptr<const Type> inputSchema,
      const std::vector<Subfield>& projectSubfields,
      velox::memory::MemoryPool* pool,
      Options options);

  /// Projects a single input buffer. The output IOBuf chain references the
  /// input memory via zero-copy cloning — the input data must remain valid
  /// while the output is in use.
  /// @param input Serialized Nimble buffer.
  /// @return Projected buffer as IOBuf (possibly chained).
  folly::IOBuf project(std::string_view input) const;

  /// Projects multiple input buffers. Each output references its corresponding
  /// input memory — all inputs must remain valid while outputs are in use.
  /// @param inputs Vector of serialized Nimble buffers.
  /// @return Vector of projected buffers.
  std::vector<folly::IOBuf> project(
      const std::vector<std::string_view>& inputs) const;

  /// Projects a single input IOBuf (may be chained). The output IOBuf chain
  /// shares underlying memory with the input via zero-copy cloning — the input
  /// must remain valid while the output is in use.
  folly::IOBuf project(const folly::IOBuf& input) const;

  /// Projects multiple input IOBufs. Each output shares memory with its
  /// corresponding input — all inputs must remain valid while outputs are in
  /// use.
  std::vector<folly::IOBuf> project(
      const std::vector<folly::IOBuf>& inputs) const;

  /// Returns the projected schema.
  /// The schema has compact stream indices starting from 0.
  std::shared_ptr<const Type> projectedSchema() const {
    return projection_.nimbleType;
  }

  /// Returns the input stream indices that will be copied.
  /// Only for testing.
  const std::vector<uint32_t>& testingInputStreamIndices() const {
    return projection_.streamOffsets;
  }

  /// Returns whether input stream indices are sorted (fast path eligible).
  /// Only for testing.
  bool testingInputStreamsSorted() const {
    return inputStreamsSorted_;
  }

 private:
  // Maps an input stream index to its output stream index.
  struct StreamMapping {
    uint32_t inputStreamIdx;
    size_t outputStreamIdx;
  };

  // Mutable state of one project() call, carried through the projection steps
  // on the calling thread's stack so the projector itself stays immutable.
  struct ProjectionContext;

  // Projects a contiguous (non-chained) IOBuf using raw pointer arithmetic.
  void projectContiguous(ProjectionContext& context) const;

  // Projects a chained IOBuf using folly::io::Cursor.
  void projectChained(ProjectionContext& context) const;

  // Writes the output header, decodes the input trailer, and initializes the
  // projected output state. Called once the input header is decoded.
  void prepareProjection(uint32_t rowCount, ProjectionContext& context) const;

  // The stream projection helpers append the selected stream bytes to
  // context.output in output stream order and record each copied stream with
  // recordOutputStream(). The selected input streams are
  // projection_.streamOffsets, indexed by output stream.

  // Projects selected streams from a contiguous IOBuf in unsorted order.
  // Walks the selected input streams in output order; for each,
  // binary-searches the sparse (streamIndices, streamSizes) trailer for the
  // corresponding bytes. Output IOBufs are emitted in output order with run
  // merging across contiguous-in-input selected streams.
  void projectStreamsContiguousUnsorted(ProjectionContext& context) const;

  // Projects selected streams from a contiguous IOBuf in sorted order.
  // Two-pointer merge over the sparse (streamIndices, streamSizes) trailer and
  // the already-sorted selected input streams.
  void projectStreamsContiguousSorted(ProjectionContext& context) const;

  // Projects selected streams from a chained IOBuf in sorted order.
  // Two-pointer merge; cursor advances stay aligned to the sparse stream
  // bytes only.
  void projectStreamsChainedSorted(ProjectionContext& context) const;

  // Projects selected streams from a chained IOBuf in unsorted order.
  // Same merge-walk shape as the contiguous unsorted variant; output buffers
  // are sorted by output stream index before chaining.
  void projectStreamsChainedUnsorted(ProjectionContext& context) const;

  // Records the size of a copied output stream. A stream with bytes also sets
  // the output barrier when the input requires one and either a Hybrid FlatMap
  // is projected or the stream is a Row/FlatMap null stream.
  void recordOutputStream(
      size_t outputStreamIdx,
      uint32_t streamSize,
      ProjectionContext& context) const;

  // Patches the output header flags, appends the trailer, and returns the
  // output chain.
  folly::IOBuf buildProjectedOutput(ProjectionContext& context) const;

  velox::memory::MemoryPool* const pool_;
  const Options options_;
  const std::shared_ptr<const Type> inputSchema_;
  const NimbleTypeProjection projection_;

  // True when projection_.streamOffsets is already sorted (no FlatMap key
  // reordering). Enables fast-path projection that avoids sorting and
  // reordering overhead.
  const bool inputStreamsSorted_;

  // Hybrid FlatMap groups are independent per-batch blocks even when a selected
  // group is absent from a particular batch and contributes no stream bytes.
  const bool hasProjectedHybridFlatMaps_;

  // Mapping from input stream index to output stream index, sorted by input
  // stream index. Empty when inputStreamsSorted_ is true.
  // Used by projectStreamsChainedUnsorted() for forward pass extraction.
  const std::vector<StreamMapping> sortedStreamMappings_;
};

} // namespace facebook::nimble::serde

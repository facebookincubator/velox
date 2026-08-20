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
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "folly/Range.h"
#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/velox/FieldReader.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/type/Subfield.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwio::common {
class TypeWithId;
} // namespace facebook::velox::dwio::common

namespace facebook::nimble {

namespace serde {
class StreamDataParser;
} // namespace serde

/// Deserializer converts serialized nimble streams into Velox vectors.
///
/// This class provides a lightweight deserialization interface for
/// materializing one or more serialized batches into a vector matching the
/// configured schema. It supports FlatMap reconstruction and omitted
/// null/in-map streams according to Deserializer::Options.
class Deserializer {
 public:
  using Options = DeserializerOptions;
  using Subfield = velox::common::Subfield;

  Deserializer(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool);

  Deserializer(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool,
      DeserializerOptions options);

  /// Builds a deserializer that reads selected subfields from a Row schema.
  /// Empty selectedSubfields reads all fields.
  /// Output Row types contain only fields reached by selectedSubfields, ordered
  /// lexicographically by field name at each projected Row level.
  Deserializer(
      std::shared_ptr<const Type> schema,
      const std::vector<Subfield>& selectedSubfields,
      velox::memory::MemoryPool* pool,
      DeserializerOptions options);

  ~Deserializer();

  Deserializer(Deserializer&&) = delete;
  Deserializer& operator=(Deserializer&&) = delete;
  Deserializer(const Deserializer&) = delete;
  Deserializer& operator=(const Deserializer&) = delete;

  /// Decodes a single serialized batch, appending the whole batch to
  /// `output`. For kTablet batches with a per-batch rowRange in the
  /// header, only rows in that range are decoded (via skip+next). All
  /// other versions decode the full batch.
  void deserialize(std::string_view data, velox::VectorPtr& output) const;

  /// Decodes a sequence of serialized batches and appends the concatenated
  /// rows to `output`. Per-batch rowRange handling matches the single-batch
  /// overload above.
  void deserialize(
      const std::vector<std::string_view>& data,
      velox::VectorPtr& output) const;

  /// Decodes a sequence of serialized batches and records the number of rows
  /// each input batch contributes to `output` in `outputRowCounts`.
  /// Per-batch rowRange handling matches the overload above.
  void deserialize(
      const std::vector<std::string_view>& data,
      velox::VectorPtr& output,
      std::vector<uint32_t>& outputRowCounts) const;

  /// Decodes each batch in `data`, appending only the rows in
  /// `rowRanges[i]` for `data[i]`. The output is the concatenation of the
  /// selected rows across batches, in order.
  ///
  /// `rowRanges[i]` is relative to the rows the batch already exposes, not
  /// to the batch's physical row numbering. A kTablet batch whose header
  /// carries a rowRange exposes only that window, so a caller range narrows
  /// it further rather than replacing it: header `[500, 600)` combined with
  /// caller `[0, 50)` selects physical rows `[500, 550)`. For every other
  /// version the batch exposes all `rowCount` rows, so `rowRanges[i]` is
  /// simply the physical range.
  ///
  /// Equivalent output to
  ///   for each i: deserialize(data[i]).slice(rowRanges[i].startRow,
  ///                                          rowRanges[i].numRows())
  /// concatenated, but rows outside each range are never materialized —
  /// they are skipped via the FieldReader's `skip()` primitive. Note this
  /// equivalence holds under the relative interpretation: the single-batch
  /// `deserialize` already applies the header window, so slicing its output
  /// is the same as narrowing within that window.
  ///
  /// Preconditions:
  ///   * `data` is non-empty and `data.size() == rowRanges.size()`.
  ///   * For each `i`: `rowRanges[i].startRow <= rowRanges[i].endRow` and
  ///     `rowRanges[i].endRow <= ` the number of rows `data[i]` exposes
  ///     (the header rowRange's `numRows()` for kTablet, the batch rowCount
  ///     otherwise).
  void deserialize(
      const std::vector<std::string_view>& data,
      const std::vector<nimble::RowRange>& rowRanges,
      velox::VectorPtr& output) const;

 private:
  // Invoked by constructors to build parser, reader factory, reader, and stream
  // decoders from the requested schema and field selection.
  void initialize(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>&
          schemaWithId,
      const std::function<bool(uint32_t)>& isSelected);

  // Builds projected output type, selected column paths, FlatMap key filters,
  // then initializes readers for the projected schema.
  void initializeColumnProjection(
      const velox::TypePtr& veloxType,
      const std::vector<Subfield>& selectedSubfields);

  // Creates deserializers for a type and its FlatMap inMap streams.
  void createDeserializersForType(const Type& type, uint32_t depth);

  // Projection keeps only stream offsets selected by the reader tree. When
  // projection is disabled, every stream is selected.
  bool shouldDecodeStream(offset_size streamOffset) const {
    return !hasColumnProjection_ ||
        (streamOffset < selectedStreamOffsetFlags_.size() &&
         selectedStreamOffsetFlags_[streamOffset]);
  }

  // Creates FieldReaderParams from DeserializerOptions, including decode
  // executor, parallel decode threshold, and flatmap-as-struct settings.
  FieldReaderParams createFieldReaderParams() const;

  void deserialize(
      folly::Range<const std::string_view*> data,
      velox::VectorPtr& output) const;

  // Shared body of all public `deserialize` overloads. Loops over `data`,
  // calls `appendBatch` per batch, then `decodeRun`. Pass an empty
  // `rowRanges` to decode whatever range each batch exposes (its header
  // rowRange, or the whole batch); pass a `data.size()`-sized `rowRanges`
  // to narrow within that range per batch (`rowRanges[i]` for `data[i]`).
  // When set, `outputRowCounts` receives the number of rows contributed by
  // each input batch.
  void deserializeImpl(
      folly::Range<const std::string_view*> data,
      folly::Range<const nimble::RowRange*> rowRanges,
      velox::VectorPtr& output,
      std::vector<uint32_t>* outputRowCounts) const;

  // Open run of non-barrier batches that can be decoded together.
  struct DecodeRun {
    uint32_t rows{0};
    uint32_t batches{0};
  };

  // Channel mapping for one projected Row child. The reader decodes selected
  // fields into their original source channels; projection wraps them into the
  // compact output Row using outputChannel.
  struct IdentityProjection {
    IdentityProjection(
        velox::column_index_t inputChannel,
        velox::column_index_t outputChannel)
        : inputChannel{inputChannel}, outputChannel{outputChannel} {}

    velox::column_index_t inputChannel;
    velox::column_index_t outputChannel;
  };

  // Maps projected Row children to decoded channels. Nested plans are present
  // only for recursively projected Row children.
  struct OutputProjection {
    std::vector<IdentityProjection> identityProjections;
    std::vector<OutputProjection> childProjections;
  };

  struct ProjectedField {
    ProjectedField* ensureChild(const std::string& name);

    // A path ending at this node selects its complete source type; otherwise,
    // only the descendants in children are retained.
    bool selectWholeField{false};
    folly::F14FastMap<std::string, std::unique_ptr<ProjectedField>> children;
  };

  // Recursively builds a projected field type and decoded-input channel
  // mapping.
  static velox::TypePtr buildProjectedType(
      const velox::TypePtr& source,
      const ProjectedField& selected,
      OutputProjection& projection);

  // Builds the projected type and its decoded-input channel mapping.
  static velox::RowTypePtr buildProjectedType(
      const velox::RowTypePtr& sourceType,
      const std::vector<Subfield>& selectedSubfields,
      OutputProjection& outputProjection);

  // Queues `batch`'s streams onto the pending decode run and records the
  // batch's effective rowRange in `runRanges_` (run-local coordinates).
  //
  // `rowRange` narrows within the range the batch already exposes (the
  // kTablet header rowRange, or the full batch for other versions) and is
  // relative to that range's start, so it cannot widen the window or
  // address rows outside it. When set, requires
  // `rowRange->startRow <= rowRange->endRow <= exposed numRows()`.
  //
  // If the batch requires a null barrier, `decodeRun` fires before and
  // after queuing so the barrier batch decodes standalone. A rowRange
  // applies there too: a barrier batch is a one-batch run, which is exactly
  // the unit the range narrows. Returns the number of rows this batch adds to
  // the output.
  uint32_t appendBatch(
      std::string_view batch,
      std::optional<nimble::RowRange> rowRange,
      DecodeRun& run,
      velox::VectorPtr& output) const;

  // Registers this batch's physical stream segments and synthesizes omitted
  // FlatMap in-map segments when older serializers leave them implicit.
  void appendStreamSegments(
      uint32_t rowCount,
      uint32_t startRow,
      bool requiresBarrier) const;

  // Appends a decoded run to the accumulated output vector.
  void appendToOutput(velox::VectorPtr&& decoded, velox::VectorPtr& output)
      const;

  // Wraps selected Row children in the projected Row type after physical
  // decoding. Returns decoded unchanged when projection is disabled.
  velox::VectorPtr projectOutput(velox::VectorPtr&& decoded) const;

  // Applies the projected Row type and source-to-output channel mapping.
  velox::VectorPtr projectOutput(
      velox::VectorPtr&& source,
      const velox::TypePtr& projectedType,
      const OutputProjection& projection) const;

  // Decodes the batches queued since the last `decodeRun`. Reads the
  // per-batch ranges from `runRanges_`, feeds a coalesced skip+next
  // sequence to `reader_`, and appends each decoded vector to `output`.
  // Leaves `run` empty, `runRanges_` empty, and the reader reset.
  void decodeRun(DecodeRun& run, velox::VectorPtr& output) const;

  // --- Const members (set at construction, never modified) ---
  const std::shared_ptr<const Type> schema_;
  velox::memory::MemoryPool* const pool_;
  const DeserializerOptions options_;

  // If column projection is enabled, only selected fields in the schema will be
  // deserialized.
  const bool hasColumnProjection_;

  // --- Non-const members (assigned in constructor body) ---
  // Output Row type used only by projected reads to wrap decoded children.
  velox::RowTypePtr outputType_;
  std::unique_ptr<OutputProjection> outputProjection_;
  // Requested keys for projected top-level FlatMap fields.
  folly::F14FastMap<std::string, FeatureSelection> flatMapFeatureSelector_;
  std::unique_ptr<FieldReaderFactory> rootFactory_;
  std::unique_ptr<FieldReader> reader_;
  mutable std::unique_ptr<serde::StreamDataParser> parser_;
  // Rows to decode from each queued batch, in run-local coordinates.
  // `runRanges_[i]` is the range for the i'th batch appended so far in the
  // current decode run. Fill via `appendBatch`, drain via `decodeRun`.
  // Empty outside of an active `deserialize*` call.
  mutable std::vector<nimble::RowRange> runRanges_;

  // --- Mutable members (modified during deserialization) ---
  mutable folly::F14FastMap<uint32_t, std::unique_ptr<Decoder>>
      deserializerMap_;

  // Flat vector indexed by stream offset for O(1) lookup in deserialize().
  // Non-owning pointers; ownership stays in deserializerMap_.
  mutable std::vector<Decoder*> deserializers_;

  // Stream offsets selected by the reader tree. Empty when column projection is
  // disabled.
  std::vector<bool> selectedStreamOffsetFlags_;

  // --- FlatMap omitted in-map stream reconstruction ---
  // Maps FlatMap in-map stream offsets to their child value types.
  folly::F14FastMap<uint32_t, const Type*> inMapChildTypes_;

  static constexpr uint32_t kInvalidInMapOffset =
      std::numeric_limits<uint32_t>::max();

  // Reverse lookup from a FlatMap child value-stream offset to its in-map
  // stream offset. Entries that are not FlatMap child value streams use
  // kInvalidInMapOffset.
  std::vector<uint32_t> valueOffsetToInMap_;

  // Per-batch stream presence, indexed by stream offset.
  mutable std::vector<bool> streamPresentFlags_;
  // Stream offsets present in the current batch. Used to find omitted FlatMap
  // in-map streams.
  mutable std::vector<uint32_t> presentStreamOffsets_;
};

} // namespace facebook::nimble

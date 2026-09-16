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
#include <optional>
#include <span>
#include <vector>

#include "folly/Function.h"
#include "folly/container/F14Map.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"

namespace facebook::nimble {

class SharedDictionaryAlphabet;

/// Loads a shared dictionary alphabet when its first encoded chunk is read.
using DictionaryAlphabetLoader =
    folly::Function<std::shared_ptr<const SharedDictionaryAlphabet>()>;

class ReaderBase {
 public:
  static std::shared_ptr<ReaderBase> create(
      std::unique_ptr<velox::dwio::common::BufferedInput> input,
      const velox::dwio::common::ReaderOptions& options);

  /// Creates a ReaderBase sharing a cached TabletReader and pre-loaded
  /// schemas. The tablet's metadata (footer, stripes, ClusterIndex) is shared
  /// across all ReaderBase instances using the same tablet. Each ReaderBase
  /// still owns its own BufferedInput for data IO.
  ///
  /// Takes the entry by shared_ptr and retains it: the returned ReaderBase
  /// aliases its tablet pointer onto the entry, so the entry cannot retire
  /// while this reader is still reading through it. There is one
  /// CachedTabletReader per file, shared by every consumer, so moving members
  /// out of it would empty it for everyone else.
  static std::shared_ptr<ReaderBase> create(
      std::unique_ptr<velox::dwio::common::BufferedInput> input,
      const std::shared_ptr<CachedTabletReader>& cachedTablet,
      const velox::dwio::common::ReaderOptions& options);

  velox::dwio::common::BufferedInput& input() {
    return *input_;
  }

  const TabletReader& tablet() const {
    return *tablet_;
  }

  velox::memory::MemoryPool* pool() const {
    return pool_;
  }

  const std::shared_ptr<velox::random::RandomSkipTracker>& randomSkip() const {
    return randomSkip_;
  }

  /// Returns the Nimble-native schema representation stored in the file.
  /// This is the internal schema format read from the file footer, containing
  /// Nimble-specific type information like stream descriptors, FlatMap
  /// structure, and encoding metadata. Used for low-level stream access and
  /// decoding.
  ///
  /// Column names in nimbleSchema() match those in fileSchema() since both are
  /// derived from the same file footer.
  const std::shared_ptr<const Type>& nimbleSchema() const {
    return nimbleSchema_;
  }

  /// Returns the Velox-compatible schema representation stored in the file.
  /// Converted from nimbleSchema() to Velox RowType. Column names come from
  /// the file footer and may differ from the table schema (from metastore) due
  /// to schema evolution (e.g., column renames).
  ///
  /// Note: When reading, column matching between file schema and table schema
  /// is done by column ID (position), not by name. The scanSpec_ uses table
  /// schema column names for query processing. Use Reader::updateColumnNames()
  /// to align file schema column names with table schema when needed.
  const velox::RowTypePtr& fileSchema() const {
    return fileSchema_;
  }

  /// Returns the file schema with node IDs for column projection.
  /// This is a TypeWithId wrapper around fileSchema() that assigns unique node
  /// IDs to each schema node (columns, nested fields). These IDs are used by
  /// column readers to map schema nodes to Nimble stream offsets during
  /// reading.
  ///
  /// When scanSpec_ is provided, the TypeWithId is created with scan spec
  /// context. The scanSpec_ uses table schema column names (from metastore),
  /// while fileSchema_ has file column names. Column matching is done by
  /// column ID (position), not by name.
  ///
  /// Lazily initialized on first access and cached for subsequent calls.
  const std::shared_ptr<const velox::dwio::common::TypeWithId>&
  fileSchemaWithId() const {
    if (!fileSchemaWithId_) {
      fileSchemaWithId_ = scanSpec_
          ? velox::dwio::common::TypeWithId::create(fileSchema_, *scanSpec_)
          : velox::dwio::common::TypeWithId::create(fileSchema_);
    }
    return fileSchemaWithId_;
  }

  /// File-level column statistics from the vectorized stats optional section.
  /// Empty if absent.
  const std::vector<std::unique_ptr<ColumnStatistics>>& fileColumnStats()
      const {
    return fileColumnStats_;
  }

 private:
  ReaderBase(
      std::unique_ptr<velox::dwio::common::BufferedInput> input,
      std::shared_ptr<TabletReader> tablet,
      const std::shared_ptr<velox::random::RandomSkipTracker>& randomSkip,
      const std::shared_ptr<velox::common::ScanSpec>& scanSpec,
      std::shared_ptr<const Type> nimbleSchema,
      velox::RowTypePtr fileSchema,
      velox::memory::MemoryPool* pool);

  const std::unique_ptr<velox::dwio::common::BufferedInput> input_;
  const std::shared_ptr<TabletReader> tablet_;
  velox::memory::MemoryPool* const pool_;
  const std::shared_ptr<velox::random::RandomSkipTracker> randomSkip_;
  const std::shared_ptr<velox::common::ScanSpec> scanSpec_;
  const std::shared_ptr<const Type> nimbleSchema_;
  const velox::RowTypePtr fileSchema_;
  // File-level column statistics deserialized from the vectorized stats
  // optional section at construction.
  const std::vector<std::unique_ptr<ColumnStatistics>> fileColumnStats_;
  mutable std::shared_ptr<const velox::dwio::common::TypeWithId>
      fileSchemaWithId_;
};

/// Physical file location of a stream within a stripe. Used to plan IO
/// without immediately enqueueing reads, enabling callers to collect regions
/// across multiple stripes for a single coalesced load.
struct StreamLocation {
  uint32_t streamId{};
  velox::common::Region region;
};

/// Cloned BufferedInput for lazy I/O columns, loaded on first lazy access.
/// All lazy columns in a stripe share a single LazyInput so their
/// streams are coalesced into one WS read instead of N separate reads.
///
/// NOTE: this object is not thread-safe.
class LazyInput {
 public:
  explicit LazyInput(std::unique_ptr<velox::dwio::common::BufferedInput> input)
      : input_(std::move(input)) {}

  /// Triggers WS I/O to load all enqueued lazy streams. Idempotent —
  /// only the first call fetches data; subsequent calls are no-ops.
  void load();

  /// Returns the underlying BufferedInput for stream routing.
  velox::dwio::common::BufferedInput* bufferedInput() const {
    return input_.get();
  }

 private:
  std::unique_ptr<velox::dwio::common::BufferedInput> input_;
  bool loaded_{false};
};

class StripeStreams {
 public:
  explicit StripeStreams(const std::shared_ptr<ReaderBase>& readerBase)
      : readerBase_(readerBase) {}

  void setStripe(int stripe) {
    stripe_ = stripe;
    lazyInput_.reset();
    dictionaryInputs_.clear();
    // Keep previous stripe's shared_ptrs (StripeGroup, ChunkStatsGroup)
    // alive while loading the new stripe. This prevents the weak-pointer
    // cache entries from expiring when consecutive stripes share the same
    // group index, avoiding redundant metadata re-reads and re-parses.
    auto prevHolder = std::move(stripeIdentifier_);
    stripeIdentifier_ = readerBase_->tablet().stripeIdentifier(stripe_);
  }

  bool hasStream(int streamId) const {
    return streamRegion(streamId).has_value();
  }

  /// Enqueue a stream for loading. When lazyColumnIo is true, routes to the
  /// lazy input clone; otherwise to the main (eager) input.
  std::unique_ptr<velox::dwio::common::SeekableInputStream> enqueue(
      int streamId,
      bool lazyColumnIo = false);

  /// Returns the physical file location of each requested stream in the
  /// current stripe. Streams that do not exist or have zero size return
  /// nullopt at the corresponding index. Does not enqueue IO.
  std::vector<std::optional<StreamLocation>> locateStreams(
      std::span<const uint32_t> streamIds) const;

  void load() {
    readerBase_->input().load(velox::dwio::common::LogType::STREAM_BUNDLE);
  }

  /// Create a lazy input clone for lazy column I/O. Ownership is held
  /// by StripeStreams; the clone is valid until the next setStripe() call.
  LazyInput* createLazyInput();

  /// Loads the lazy input clone, if one exists. Idempotent.
  void loadLazyInput() {
    if (lazyInput_) {
      lazyInput_->load();
    }
  }

  int32_t stripeIndex() const {
    return stripe_;
  }

  const index::ClusterIndex* clusterIndex() const {
    return readerBase_->tablet().clusterIndex();
  }

  std::shared_ptr<index::StreamIndex> streamIndex(int streamId) const;

  /// Returns a lazy alphabet loader for a value stream, or nullptr if it has
  /// no dictionary binding in the current stripe.
  DictionaryAlphabetLoader dictionaryAlphabetLoader(uint32_t valueStreamId);

 private:
  struct DictionaryInput {
    // Enqueued stripe dictionary stream held until its alphabet is first
    // requested.
    std::unique_ptr<velox::dwio::common::SeekableInputStream> input;
    // Serialized byte length of the dictionary stream.
    uint64_t size{};
  };

  std::optional<velox::common::Region> streamRegion(int streamId) const;

  const std::shared_ptr<ReaderBase> readerBase_;

  int stripe_{};
  // Owns the lazy input clone for the current stripe. Reset on
  // setStripe(). StructColumnReader holds a raw LazyInput* pointer,
  // guarded by the numReads_ version check.
  std::unique_ptr<LazyInput> lazyInput_;
  std::optional<StripeIdentifier> stripeIdentifier_;
  // Inputs awaiting alphabet creation, keyed by projected value stream ID.
  folly::F14FastMap<uint32_t, DictionaryInput> dictionaryInputs_;
};

} // namespace facebook::nimble

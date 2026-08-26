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

#include <cmath>
#include <set>

#include <folly/Executor.h>
#include <folly/coro/Task.h>

#include "folly/container/F14Map.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/velox/NimbleConfig.h"
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/StreamData.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatsUtils.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::nimble {

struct InputBufferGrowthStats {
  std::atomic<uint64_t> count{0};
  // realloc bytes would be interesting, but requires a bit more
  // trouble to get.
  std::atomic<uint64_t> itemCount{0};
};

// A pool of decoding contexts.  Decoding contexts are used to decode a vector
// and its associated selectivity vector.  The pool is used to avoid
// repeated allocations of decoding context.
class DecodingContextPool {
 public:
  class DecodingContext {
   public:
    explicit DecodingContext(
        DecodingContextPool& pool,
        std::unique_ptr<velox::DecodedVector> decodedVector,
        std::unique_ptr<velox::SelectivityVector> selectivityVector);

    ~DecodingContext();

    velox::DecodedVector& decode(
        const velox::VectorPtr& vector,
        const range_helper::OrderedRanges<velox::vector_size_t>& ranges);

   private:
    DecodingContextPool& pool_;
    std::unique_ptr<velox::DecodedVector> decodedVector_;
    std::unique_ptr<velox::SelectivityVector> selectivityVector_;
  };

  explicit DecodingContextPool(
      std::function<void(void)> vectorDecoderVisitor = []() {});

  DecodingContext reserveContext();
  size_t size() const;

 private:
  std::mutex mutex_;
  std::vector<std::pair<
      std::unique_ptr<velox::DecodedVector>,
      std::unique_ptr<velox::SelectivityVector>>>
      pool_;
  std::function<void(void)> vectorDecoderVisitor_;

  void addContext(
      std::unique_ptr<velox::DecodedVector> decodedVector,
      std::unique_ptr<velox::SelectivityVector> selectivityVector);
};

class MemoryPoolHolder {
 public:
  velox::memory::MemoryPool& operator*() {
    return *poolPtr_;
  }

  velox::memory::MemoryPool* get() const {
    return poolPtr_;
  }

  velox::memory::MemoryPool* operator->() const {
    return get();
  }

  template <typename Creator>
  static MemoryPoolHolder create(
      velox::memory::MemoryPool& pool,
      const Creator& creator) {
    MemoryPoolHolder holder;
    if (pool.isLeaf()) {
      holder.poolPtr_ = &pool;
    } else {
      holder.ownedPool_ = creator(pool);
      holder.poolPtr_ = holder.ownedPool_.get();
    }
    return holder;
  }

 private:
  MemoryPoolHolder() = default;

  std::shared_ptr<velox::memory::MemoryPool> ownedPool_;
  velox::memory::MemoryPool* poolPtr_{nullptr};
};

/// NOTE: this class is not thread-safe.
class FieldWriterContext {
 public:
  explicit FieldWriterContext(
      velox::memory::MemoryPool& memoryPool,
      std::unique_ptr<velox::memory::MemoryReclaimer> reclaimer = nullptr,
      std::function<void(void)> vectorDecoderVisitor = []() {})
      : bufferMemoryPool_{MemoryPoolHolder::create(
            memoryPool,
            [&](auto& pool) {
              return pool.addLeafChild(
                  "field_writer_buffer", true, std::move(reclaimer));
            })},
        inputBufferGrowthPolicy_{
            DefaultInputBufferGrowthPolicy::withDefaultRanges()},
        decodingContextPool_{std::move(vectorDecoderVisitor)} {
    resetStringBuffer();
  }

  virtual ~FieldWriterContext() = default;

  inline MemoryPoolHolder& bufferMemoryPool() {
    return bufferMemoryPool_;
  }

  inline const MemoryPoolHolder& bufferMemoryPool() const {
    return bufferMemoryPool_;
  }

  inline std::mutex& flatMapSchemaMutex() {
    return flatMapSchemaMutex_;
  }

  inline SchemaBuilder& schemaBuilder() {
    return schemaBuilder_;
  }

  inline const SchemaBuilder& schemaBuilder() const {
    return schemaBuilder_;
  }

  inline bool hasFlatMapNodeId(uint32_t nodeId) const {
    return flatMapNodes_.contains(nodeId);
  }

  /// Registers a flat map node by ID, optionally with predefined keys.
  /// Empty set means dynamic key discovery; non-empty set means predefined
  /// keys in sorted order.
  void addFlatMapNodeId(uint32_t nodeId, std::set<std::string> keys = {}) {
    NIMBLE_CHECK(
        flatMapNodes_.find(nodeId) == flatMapNodes_.end(),
        "Flat map column already set for node {}",
        nodeId);
    flatMapNodes_[nodeId] = std::move(keys);
  }

  /// Returns predefined keys for a flat map node, or nullptr if the
  /// node has no predefined keys or is not a flat map node.
  const std::set<std::string>* getFlatMapNodeKeys(uint32_t nodeId) const {
    auto it = flatMapNodes_.find(nodeId);
    if (it == flatMapNodes_.end() || it->second.empty()) {
      return nullptr;
    }
    return &it->second;
  }

  inline void reserveFlatMapNodes(size_t size) {
    flatMapNodes_.reserve(size);
  }

  /// Returns the set of flat map node IDs.
  inline folly::F14FastSet<uint32_t> flatMapNodeIds() const {
    folly::F14FastSet<uint32_t> ids;
    ids.reserve(flatMapNodes_.size());
    for (const auto& [nodeId, _] : flatMapNodes_) {
      ids.insert(nodeId);
    }
    return ids;
  }

  inline folly::F14FastSet<uint32_t>& dictionaryArrayNodeIds() {
    return dictionaryArrayNodeIds_;
  }

  inline const folly::F14FastSet<uint32_t>& dictionaryArrayNodeIds() const {
    return dictionaryArrayNodeIds_;
  }

  inline void clearAndReserveDictionaryArrayNodeIds(size_t size) {
    dictionaryArrayNodeIds_.clear();
    dictionaryArrayNodeIds_.reserve(size);
  }

  inline bool hasDictionaryArrayNodeId(uint32_t nodeId) const {
    return dictionaryArrayNodeIds_.contains(nodeId);
  }

  inline folly::F14FastSet<uint32_t>& deduplicatedMapNodeIds() {
    return deduplicatedMapNodeIds_;
  }

  inline bool hasDeduplicatedMapNodeId(uint32_t nodeId) const {
    return deduplicatedMapNodeIds_.contains(nodeId);
  }

  inline const folly::F14FastSet<uint32_t>& deduplicatedMapNodeIds() const {
    return deduplicatedMapNodeIds_;
  }

  inline void clearAndReserveDeduplicatedMapNodeIds(size_t size) {
    deduplicatedMapNodeIds_.clear();
    deduplicatedMapNodeIds_.reserve(size);
  }

  inline bool ignoreTopLevelNulls() const {
    return ignoreTopLevelNulls_;
  }

  inline void setIgnoreTopLevelNulls(bool value) {
    ignoreTopLevelNulls_ = value;
  }

  inline uint32_t maxFlatMapKeys() const {
    return maxFlatMapKeys_;
  }

  inline void setMaxFlatMapKeys(uint32_t value) {
    maxFlatMapKeys_ = value;
  }

  void setParallelEncoding(
      folly::Executor* executor,
      uint32_t maxEncodeParallelism = 8,
      uint32_t minStreamsPerEncodeUnit = 2) {
    NIMBLE_CHECK_NOT_NULL(executor, "encodeExecutor must not be null");
    encodeExecutor_ = executor;
    maxEncodeParallelism_ = maxEncodeParallelism;
    minStreamsPerEncodeUnit_ = minStreamsPerEncodeUnit;
  }

  folly::Executor* encodeExecutor() const {
    return encodeExecutor_;
  }

  uint32_t maxEncodeParallelism() const {
    return maxEncodeParallelism_;
  }

  uint32_t minStreamsPerEncodeUnit() const {
    return minStreamsPerEncodeUnit_;
  }

  inline std::unique_ptr<InputBufferGrowthPolicy>& inputBufferGrowthPolicy() {
    return inputBufferGrowthPolicy_;
  }

  inline const std::unique_ptr<InputBufferGrowthPolicy>&
  inputBufferGrowthPolicy() const {
    return inputBufferGrowthPolicy_;
  }

  inline InputBufferGrowthStats& inputBufferGrowthStats() {
    return inputBufferGrowthStats_;
  }

  inline const InputBufferGrowthStats& inputBufferGrowthStats() const {
    return inputBufferGrowthStats_;
  }

  inline void recordChunkSize(uint64_t encodedBytes) {
    chunkSizeStats_.addValue(encodedBytes);
  }

  inline const velox::RuntimeMetric& chunkSizeStats() const {
    return chunkSizeStats_;
  }

  inline std::vector<ColumnStatistics*> columnStats() {
    std::vector<ColumnStatistics*> statsViews;
    statsViews.reserve(fileStats_.size());
    for (auto& stat : fileStats_) {
      statsViews.push_back(stat.get());
    }
    return statsViews;
  }

  inline const std::vector<std::vector<std::unique_ptr<ColumnStatistics>>>&
  stripeStats() const {
    return stripeStats_;
  }

  void handleFlatmapFieldAddEvent(
      const TypeBuilder& typeBuilder,
      std::string_view flatmapKey,
      const TypeBuilder& flatmapTypeBuilder) {
    if (flatmapFieldAddedEventHandler_) {
      flatmapFieldAddedEventHandler_(
          typeBuilder, flatmapKey, flatmapTypeBuilder);
    }
  }

  inline void setFlatmapFieldAddedEventHandler(
      std::function<
          void(const TypeBuilder&, std::string_view, const TypeBuilder&)>
          handler) {
    flatmapFieldAddedEventHandler_ = std::move(handler);
  }

  inline void handleAddedType(TypeBuilder& typeBuilder, uint32_t nodeId) const {
    typeAddedHandler_(typeBuilder, nodeId);
  }

  inline void setTypeAddedHandler(
      std::function<void(TypeBuilder&, uint32_t)>&& handler) {
    typeAddedHandler_ = std::move(handler);
  }

  inline DecodingContextPool::DecodingContext decodingContext() {
    return decodingContextPool_.reserveContext();
  }

  inline Buffer& stringBuffer() {
    return *buffer_;
  }

  inline void resetStringBuffer() {
    buffer_ = std::make_unique<Buffer>(*bufferMemoryPool_);
  }

  inline const std::vector<std::pair<uint32_t, std::unique_ptr<StreamData>>>&
  streams() {
    return streams_;
  }

  inline NullsStreamData& createNullsStreamData(
      const StreamDescriptorBuilder& descriptor,
      uint32_t nodeId) {
    return static_cast<NullsStreamData&>(
        *streams_
             .emplace_back(
                 nodeId,
                 std::make_unique<NullsStreamData>(
                     *bufferMemoryPool_, descriptor, *inputBufferGrowthPolicy_))
             .second);
  }

  template <typename T>
  ContentStreamData<T>& createContentStreamData(
      const StreamDescriptorBuilder& descriptor,
      uint32_t nodeId) {
    return static_cast<ContentStreamData<T>&>(
        *streams_
             .emplace_back(
                 nodeId,
                 std::make_unique<ContentStreamData<T>>(
                     *bufferMemoryPool_, descriptor, *inputBufferGrowthPolicy_))
             .second);
  }

  template <typename T>
  NullableContentStreamData<T>& createNullableContentStreamData(
      const StreamDescriptorBuilder& descriptor,
      uint32_t nodeId) {
    return static_cast<NullableContentStreamData<T>&>(
        *streams_
             .emplace_back(
                 nodeId,
                 std::make_unique<NullableContentStreamData<T>>(
                     *bufferMemoryPool_, descriptor, *inputBufferGrowthPolicy_))
             .second);
  }

  nimble::NullableContentStringStreamData&
  createNullableContentStringStreamData(
      const nimble::StreamDescriptorBuilder& descriptor,
      uint32_t nodeId) {
    return static_cast<nimble::NullableContentStringStreamData&>(
        *streams_
             .emplace_back(
                 nodeId,
                 std::make_unique<nimble::NullableContentStringStreamData>(
                     *bufferMemoryPool_,
                     descriptor,
                     *inputBufferGrowthPolicy_,
                     *stringBufferGrowthPolicy_))
             .second);
  }

  inline bool disableSharedStringBuffers() const {
    return disableSharedStringBuffers_;
  }

  void createStatsCollector(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
    statsCollectors_.emplace_back(StatisticsCollector::create(type));
    for (const auto& child : type->getChildren()) {
      createStatsCollector(child);
    }
  }

  void wrapSharedStatsCollector(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
      bool shared) {
    shared = shared || hasFlatMapNodeId(type->id());
    if (shared) {
      statsCollectors_[type->id()] = SharedStatisticsCollector::wrap(
          std::move(statsCollectors_[type->id()]));
    }

    for (const auto& child : type->getChildren()) {
      wrapSharedStatsCollector(child, shared);
    }
  }

  void initStatsCollectors(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
    createStatsCollector(type);

    // NOTE: When supporting deduplicated stats, we set just the stats on
    // the deduplicated type writers only. Those nodes record logical stats as
    // is, but then record the children's logical nodes as deduplicated at
    // finalization.
    for (auto id : dictionaryArrayNodeIds()) {
      statsCollectors_[id] = DeduplicatedStatisticsCollector::wrap(
          std::move(statsCollectors_[id]));
    }

    for (auto id : deduplicatedMapNodeIds()) {
      statsCollectors_[id] = DeduplicatedStatisticsCollector::wrap(
          std::move(statsCollectors_[id]));
    }

    // Wrap shared stats collector for flatmap nodes last to fully protected
    // concurrent stats updates across flatmap value writers.
    wrapSharedStatsCollector(type, false);

    schemaWithId_ = type;
  }

  StatisticsCollector* getStatsCollector(uint32_t nodeId) const {
    if (statsCollectors_.empty()) {
      return nullptr;
    }
    NIMBLE_CHECK_LT(
        nodeId,
        statsCollectors_.size(),
        "Mismatched cardinality for stats and schema. Likely uninitialized stats collectors.");
    return statsCollectors_[nodeId].get();
  }

  // After potential parallel process of field writers and their stats,
  // sequentially roll up and fix up the stats according to the hierarchy.
  // 1. roll up logical and physical sizes
  // 2. wrap all ancestors of deduplicated types
  // 3. backfill both known and newly wrapped deduplicated stats
  /// Finalizes the current stripe's collectors, snapshots their statistics
  /// into stripeStats_, and resets the collectors for the next stripe.
  void finalizeStripeStatsCollectors() {
    NIMBLE_CHECK(!statsFinalized_);
    finalizeStatsCollector(schemaWithId_);
    statsFinalized_ = true;
    snapshotStripeStats();
    resetStatsCollectors();
  }

  /// Rebuilds file-level column statistics by merging every per-stripe
  /// snapshot in stripeStats_. Falls back to finalizing the live collectors
  /// directly when no stripe snapshots were taken (single-stripe file).
  void finalizeFileStatsFromStripes() {
    fileStats_.clear();
    if (stripeStats_.empty()) {
      if (!statsFinalized_) {
        finalizeStatsCollector(schemaWithId_);
        statsFinalized_ = true;
      }
      fileStats_.reserve(statsCollectors_.size());
      for (auto& collector : statsCollectors_) {
        auto* stats = collector->shared()
            ? collector->as<SharedStatisticsCollector>()->getBaseStatistics()
            : collector->getStatsView();
        fileStats_.push_back(stats->clone());
      }
      return;
    }
    fileStats_.reserve(stripeStats_.front().size());
    for (const auto& stat : stripeStats_.front()) {
      fileStats_.push_back(stat->clone());
    }
    for (size_t stripe = 1; stripe < stripeStats_.size(); ++stripe) {
      NIMBLE_CHECK_EQ(stripeStats_[stripe].size(), fileStats_.size());
      for (size_t column = 0; column < fileStats_.size(); ++column) {
        mergeColumnStats(*fileStats_[column], *stripeStats_[stripe][column]);
      }
    }
  }

 protected:
  MemoryPoolHolder bufferMemoryPool_;
  std::mutex flatMapSchemaMutex_;
  SchemaBuilder schemaBuilder_;

  folly::F14FastMap<uint32_t, std::set<std::string>> flatMapNodes_;
  folly::F14FastSet<uint32_t> dictionaryArrayNodeIds_;
  folly::F14FastSet<uint32_t> deduplicatedMapNodeIds_;
  bool ignoreTopLevelNulls_{false};
  bool disableSharedStringBuffers_{false};
  uint32_t maxFlatMapKeys_{kDefaultMaxFlatMapKeys};

  folly::Executor* encodeExecutor_{nullptr};
  uint32_t maxEncodeParallelism_{0};
  uint32_t minStreamsPerEncodeUnit_{1};

  std::unique_ptr<InputBufferGrowthPolicy> inputBufferGrowthPolicy_;
  std::unique_ptr<InputBufferGrowthPolicy> stringBufferGrowthPolicy_;
  InputBufferGrowthStats inputBufferGrowthStats_;
  velox::RuntimeMetric chunkSizeStats_;

  std::function<void(const TypeBuilder&, std::string_view, const TypeBuilder&)>
      flatmapFieldAddedEventHandler_;

  std::function<void(TypeBuilder&, uint32_t)> typeAddedHandler_{
      [](TypeBuilder&, uint32_t) {}};

 private:
  // Merges min/max from source into target for a scalar statistics subclass
  // (Integral/FloatingPoint/String). Keeps the widest range across stripes.
  template <typename StatsT>
  static void mergeMinMax(
      ColumnStatistics& target,
      const ColumnStatistics& source) {
    auto* targetStats = target.as<StatsT>();
    const auto* sourceStats = source.as<StatsT>();
    if (sourceStats->getMin().has_value() &&
        (!targetStats->getMin().has_value() ||
         *sourceStats->getMin() < *targetStats->getMin())) {
      targetStats->min_ = sourceStats->getMin();
    }
    if (sourceStats->getMax().has_value() &&
        (!targetStats->getMax().has_value() ||
         *sourceStats->getMax() > *targetStats->getMax())) {
      targetStats->max_ = sourceStats->getMax();
    }
  }

  // NaN-aware min selection: returns true when source's min should replace
  // target's. A present value always beats a missing one, and a real value
  // beats NaN so NaN never wins as the minimum; otherwise the smaller value
  // wins. NaN is handled explicitly because IEEE comparisons against NaN are
  // always false.
  static bool shouldUseSourceFloatingPointMin(
      const FloatingPointStatistics& target,
      const FloatingPointStatistics& source) {
    if (!source.getMin().has_value()) {
      return false;
    }
    if (!target.getMin().has_value()) {
      return true;
    }
    if (std::isnan(*target.getMin())) {
      return false;
    }
    if (std::isnan(*source.getMin())) {
      return true;
    }
    return *source.getMin() < *target.getMin();
  }

  // Mirror of shouldUseSourceFloatingPointMin for the maximum: a real value
  // beats NaN so NaN never wins as the maximum; otherwise the larger value
  // wins.
  static bool shouldUseSourceFloatingPointMax(
      const FloatingPointStatistics& target,
      const FloatingPointStatistics& source) {
    if (!source.getMax().has_value()) {
      return false;
    }
    if (!target.getMax().has_value()) {
      return true;
    }
    if (std::isnan(*target.getMax())) {
      return false;
    }
    if (std::isnan(*source.getMax())) {
      return true;
    }
    return *source.getMax() > *target.getMax();
  }

  static void mergeFloatingPointMinMax(
      ColumnStatistics& target,
      const ColumnStatistics& source) {
    auto* targetStats = target.as<FloatingPointStatistics>();
    const auto* sourceStats = source.as<FloatingPointStatistics>();
    if (shouldUseSourceFloatingPointMin(*targetStats, *sourceStats)) {
      targetStats->min_ = sourceStats->getMin();
    }
    if (shouldUseSourceFloatingPointMax(*targetStats, *sourceStats)) {
      targetStats->max_ = sourceStats->getMax();
    }
  }

  static void mergeColumnStats(
      ColumnStatistics& target,
      const ColumnStatistics& source) {
    NIMBLE_CHECK(
        target.getType() == source.getType(),
        "Merging stats with mismatched types: {} vs {}.",
        static_cast<uint8_t>(target.getType()),
        static_cast<uint8_t>(source.getType()));
    // Deduplicated stats delegate their base metrics (value/null count, sizes)
    // to the wrapped base statistics, so merge into that base rather than
    // target's own (unused) member fields, and accumulate the dedup-specific
    // counters separately.
    if (target.getType() == StatType::DEDUPLICATED) {
      auto* targetStats = target.as<DeduplicatedColumnStatistics>();
      const auto* sourceStats = source.as<DeduplicatedColumnStatistics>();
      targetStats->dedupedCount_ += sourceStats->getDedupedCount();
      targetStats->dedupedLogicalSize_ += sourceStats->getDedupedLogicalSize();
      auto* targetBase = targetStats->getBaseStatistics();
      const auto* sourceBase = sourceStats->getBaseStatistics();
      NIMBLE_CHECK(
          (targetBase == nullptr) == (sourceBase == nullptr),
          "Mismatched deduplicated stats structure during merge.");
      if (targetBase != nullptr && sourceBase != nullptr) {
        mergeColumnStats(*targetBase, *sourceBase);
      }
      return;
    }

    target.valueCount_ += source.getValueCount();
    target.nullCount_ += source.getNullCount();
    target.logicalSize_ += source.getLogicalSize();
    target.physicalSize_ += source.getPhysicalSize();
    switch (target.getType()) {
      case StatType::INTEGRAL:
        mergeMinMax<IntegralStatistics>(target, source);
        break;
      case StatType::FLOATING_POINT:
        mergeFloatingPointMinMax(target, source);
        break;
      case StatType::STRING:
        mergeMinMax<StringStatistics>(target, source);
        break;
      case StatType::DEFAULT:
        break;
      case StatType::DEDUPLICATED:
        NIMBLE_UNREACHABLE("Deduplicated stats are handled above.");
    }
  }

  // Clones the current per-column statistics into a new per-stripe snapshot
  // appended to stripeStats_. Uses the deduplicated base statistics for shared
  // collectors so each snapshot mirrors the file-level column layout.
  void snapshotStripeStats() {
    auto& stripe = stripeStats_.emplace_back();
    stripe.reserve(statsCollectors_.size());
    for (auto& collector : statsCollectors_) {
      auto* stats = collector->shared()
          ? collector->as<SharedStatisticsCollector>()->getBaseStatistics()
          : collector->getStatsView();
      stripe.push_back(stats->clone());
    }
  }

  void resetStatsCollectors() {
    // finalizeStatsCollector() wraps ancestors of deduplicated nodes into a
    // DeduplicatedStatisticsCollector in place. That wrapping must be undone
    // between stripes; otherwise the next stripe's finalization would take the
    // "already deduplicated" branch and skip rolling child logical/physical
    // sizes up into the ancestor, corrupting the file-level stats
    // reconstruction. Genuine deduplicated nodes (configured at init via
    // dictionary array / deduplicated map ids) must remain wrapped.
    for (uint32_t id = 0; id < statsCollectors_.size(); ++id) {
      auto& collector = statsCollectors_[id];
      if (collector->getType() == StatType::DEDUPLICATED &&
          !hasDictionaryArrayNodeId(id) && !hasDeduplicatedMapNodeId(id)) {
        collector = collector->asChecked<DeduplicatedStatisticsCollector>()
                        ->releaseBaseCollector();
      }
    }
    for (auto& collector : statsCollectors_) {
      collector->reset();
    }
    statsFinalized_ = false;
  }

  void finalizeStatsCollector(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
    auto statsCollector = statsCollectors_[type->id()].get();

    // Known deduplicated nodes record logical stats as is, but then record the
    // children's logical sizes as deduplicated at finalization, to avoid double
    // counting.
    if (statsCollector->getType() == StatType::DEDUPLICATED) {
      // Handle the case where the collector might be wrapped in a
      // SharedStatisticsCollector.
      if (statsCollector->shared()) {
        auto sharedCollector = statsCollector->as<SharedStatisticsCollector>();
        for (const auto& child : type->getChildren()) {
          finalizeStatsCollector(child);
          auto childStatsCollector = getStatsCollector(child->id());
          sharedCollector->updateBaseCollector([&](StatisticsCollector*
                                                       baseCollector) {
            auto dedupedStatsCollector =
                baseCollector->as<DeduplicatedStatisticsCollector>();
            NIMBLE_DCHECK_NOT_NULL(
                dedupedStatsCollector,
                "Expected DeduplicatedStatisticsCollector for DEDUPLICATED type");
            // Only add children's logical sizes for flatmaps, which don't
            // include children's sizes in their own collectStatistics.
            // Sliding window maps and array with offsets use
            // getRawSizeFromVector which already includes children's sizes.
            // TODO(huamengjiang): fix the behavior of deduplicated stats for
            // flatmaps when properly supporting them.
            if (hasFlatMapNodeId(type->id())) {
              dedupedStatsCollector->addLogicalSize(
                  childStatsCollector->getLogicalSize());
            }
            dedupedStatsCollector->recordDeduplicatedStats(
                childStatsCollector->getValueCount(),
                childStatsCollector->getLogicalSize());
            dedupedStatsCollector->addPhysicalSize(
                childStatsCollector->getPhysicalSize());
          });
        }
      } else {
        auto dedupedStatsCollector =
            statsCollector->as<DeduplicatedStatisticsCollector>();
        NIMBLE_DCHECK_NOT_NULL(
            dedupedStatsCollector,
            "Expected DeduplicatedStatisticsCollector for DEDUPLICATED type");
        for (const auto& child : type->getChildren()) {
          finalizeStatsCollector(child);
          auto childStatsCollector = getStatsCollector(child->id());
          dedupedStatsCollector->recordDeduplicatedStats(
              childStatsCollector->getValueCount(),
              childStatsCollector->getLogicalSize());
          dedupedStatsCollector->addPhysicalSize(
              childStatsCollector->getPhysicalSize());
        }
      }

      return;
    }

    // Roll up logical and physical sizes. Find out if any child has
    // deduplicated stats.
    bool isDeduplicated = false;
    for (const auto& child : type->getChildren()) {
      finalizeStatsCollector(child);
      auto childStatsCollector = getStatsCollector(child->id());
      isDeduplicated |=
          (childStatsCollector->getType() == StatType::DEDUPLICATED);

      statsCollector->addLogicalSize(childStatsCollector->getLogicalSize());
      statsCollector->addPhysicalSize(childStatsCollector->getPhysicalSize());
    }

    // Backfill deduplicated ancestors.
    if (isDeduplicated &&
        statsCollectors_[type->id()]->getType() != StatType::DEDUPLICATED) {
      statsCollectors_[type->id()] = DeduplicatedStatisticsCollector::wrap(
          std::move(statsCollectors_[type->id()]));

      auto dedupedStatsCollector =
          statsCollectors_[type->id()]->as<DeduplicatedStatisticsCollector>();
      for (const auto& child : type->getChildren()) {
        auto childStatsCollector = getStatsCollector(child->id());
        if (childStatsCollector->getType() == StatType::DEDUPLICATED) {
          if (childStatsCollector->shared()) {
            auto dedupedStats =
                childStatsCollector->as<SharedStatisticsCollector>()
                    ->getBaseStatistics()
                    ->as<DeduplicatedColumnStatistics>();
            dedupedStatsCollector->recordDeduplicatedStats(
                dedupedStats->getDedupedCount(),
                dedupedStats->getDedupedLogicalSize());
          } else {
            auto dedupedStats =
                childStatsCollector->as<DeduplicatedStatisticsCollector>()
                    ->getStatsView()
                    ->as<DeduplicatedColumnStatistics>();
            dedupedStatsCollector->recordDeduplicatedStats(
                dedupedStats->getDedupedCount(),
                dedupedStats->getDedupedLogicalSize());
          }
        } else {
          dedupedStatsCollector->recordDeduplicatedStats(
              childStatsCollector->getValueCount(),
              childStatsCollector->getLogicalSize());
        }
      }
    }
  }

  std::unique_ptr<Buffer> buffer_;
  DecodingContextPool decodingContextPool_;
  std::vector<std::pair<uint32_t, std::unique_ptr<StreamData>>> streams_;
  std::shared_ptr<const velox::dwio::common::TypeWithId> schemaWithId_;
  std::vector<std::unique_ptr<StatisticsCollector>> statsCollectors_;
  std::vector<std::vector<std::unique_ptr<ColumnStatistics>>> stripeStats_;
  std::vector<std::unique_ptr<ColumnStatistics>> fileStats_;
  bool statsFinalized_{false};
};

using OrderedRanges = range_helper::OrderedRanges<velox::vector_size_t>;

class FieldWriter {
 public:
  FieldWriter(
      FieldWriterContext& context,
      std::shared_ptr<TypeBuilder> typeBuilder)
      : context_{context}, typeBuilder_{std::move(typeBuilder)} {}

  virtual ~FieldWriter() = default;

  // Writes the vector to internal buffers.
  virtual void write(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges) = 0;

  virtual folly::coro::Task<void> co_write(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges) {
    write(vector, ranges);
    co_return;
  }

  // Collects stats and clears interanl state and any accumulated data in
  // internal buffers.
  virtual void reset() = 0;

  // Called when all writes are done, allowing field writers to finalize
  // internal state.
  virtual void close() {}

  const std::shared_ptr<TypeBuilder>& typeBuilder() {
    return typeBuilder_;
  }

  static std::unique_ptr<FieldWriter> create(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type);

  static std::unique_ptr<FieldWriter> create(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
      std::function<void(TypeBuilder&, uint32_t)> typeAddedHandler);

 protected:
  // Computes the number of parallel write tasks based on max parallelism and
  // minimum streams per task. The result is clamped to [1, numChildren].
  uint32_t computeParallelWriteTaskCount(uint32_t numChildren) const;

  // Returns true if child fields should be written in parallel.
  bool parallelWriteEnabled(uint32_t numChildren) const;

  FieldWriterContext& context_;
  std::shared_ptr<TypeBuilder> typeBuilder_;
};

} // namespace facebook::nimble

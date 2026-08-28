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

#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/MetricsLogger.h"
#include "velox/dwio/nimble/common/NimbleConfig.h"
#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/tablet/StripeGroup.h"
#include "velox/dwio/nimble/writer/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/writer/BufferPolicy.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"

#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>
#include "velox/common/base/SpillConfig.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

namespace detail {
std::unordered_map<std::string, std::string> defaultMetadata();
} // namespace detail

/// Options used by Velox writer that affect the output file format.
/// NOTE: The object could be large when encodingOverrides are supplied. It's
/// strongly advised to move instead of copying it.
struct WriterOptions {
  /// Builds Encoding::Options from WriterOptions fields.
  Encoding::Options buildEncodingOptions() const {
    return {
        .useVarintRowCount = experimentalCompactRowCountEncoding,
        .blockBitPackingBlockSize = blockBitPackingBlockSize,
        .fixedBitWidthUseExactBits = fixedBitWidthUseExactBits,
        .allowNestedAlpSelection = allowNestedAlpSelection,
        .sharedDictionaryAlphabet = {},
        .fsstCompressionTargetRatio = fsstCompressionTargetRatio};
  }

  /// Property bag for storing user metadata in the file.
  std::unordered_map<std::string, std::string> metadata =
      detail::defaultMetadata();

  /// Shared dictionary encoding settings.
  /// EXPERIMENTAL: Shared dictionary encoding is not production-ready. Do not
  /// enable for production tables without consulting the Nimble team (oncall:
  /// dwios).
  SharedDictionaryEncodingConfig experimentalSharedDictionaryEncoding{};

  /// Enable column statistics collection. When false, the writer skips
  /// collecting per-column statistics, reducing write CPU cost.
  bool enableStatsCollection{true};

  /// Enable vectorized stats for applicable schema shapes.
  bool enableVectorizedStats{true};

  /// When true, chunk-level position index is built for all streams,
  /// enabling O(1) chunk-level seeking within stripes. Independent of
  /// the cluster index (clusterIndexConfig). When clusterIndexConfig is set,
  /// chunk index is always enabled regardless of this flag.
  /// EXPERIMENTAL: Not production-ready. Do not enable for production tables
  /// without consulting the Nimble team (oncall: dwios).
  // TODO: keeps the chunkIndex name for now; rename to the chunkStats naming
  // once per-chunk null/min/max stats are fully rolled out.
  bool enableChunkIndex{false};

  /// Skip writing chunk stats for a stripe group if the average number
  /// of chunks per stream is below this threshold. 0 disables chunk stats
  /// skipping.
  float chunkStatsMinAvgChunks{2};

  /// NOTE: !!! This is under experimentation and please do not turn on in
  /// production use case !!!
  /// Selects how per-stripe-group stream offsets/sizes are serialized:
  /// - kRaw: flat stripe-major uint32 arrays (default; all readers).
  /// - kStreamMajor: one encoding per stream.
  StripeGroup::EncodingLayout experimentalStripeGroupEncodingLayout{
      StripeGroup::EncodingLayout::kRaw};

  /// Candidate encodings (with read-cost weights) the writer considers when
  /// encoding per-stripe-group stream offsets/sizes (kStreamMajor). Only
  /// encodings with O(1) stateless point access (via EncodingView) are valid.
  /// Empty (default) lets the writer pick its own candidate set (Constant,
  /// Trivial, FixedBitWidth).
  std::vector<std::pair<EncodingType, float>>
      experimentalStripeGroupEncodingLayoutReadFactors{};

  /// If set, the cluster index on the specified columns will be built during
  /// writing. It is the primary index over key-ordered data and stores
  /// per-chunk boundary keys per stripe-group partition for binary-search
  /// pruning.
  /// EXPERIMENTAL: Cluster index is not production-ready. Do not enable for
  /// production tables without consulting the Nimble team (oncall: dwios).
  std::shared_ptr<const index::IndexConfig> clusterIndexConfig{};

  /// Whether to omit cluster index key columns from normal data storage. The
  /// key columns must still be present in write input batches.
  /// EXPERIMENTAL: Cluster index is not production-ready. Do not enable for
  /// production tables without consulting the Nimble team (oncall: dwios).
  bool experimentalOmitClusterIndexKeyColumnStorage{false};

  /// Enables compact varint row-count encoding for encoded data streams. The
  /// value is persisted in file properties so readers can select the matching
  /// decoding behavior.
  /// EXPERIMENTAL: Compact row-count encoding is not production-ready. Do not
  /// enable for production tables without consulting the Nimble team (oncall:
  /// dwios).
  bool experimentalCompactRowCountEncoding{false};

  /// Dense index configurations, grouped by factory name. Each factory creates
  /// one writer that may produce multiple logical indexes.
  /// EXPERIMENTAL: Dense indexes are not production-ready. Do not enable for
  /// production tables without consulting the Nimble team (oncall: dwios).
  std::vector<std::shared_ptr<const index::IndexConfig>> denseIndexConfigs{};

  /// Columns that should be encoded as flat maps. Maps column name to a set
  /// of predefined key strings. When the set is empty, the column is
  /// treated as a flat map with dynamic key discovery. When non-empty, keys
  /// are predefined in sorted order to ensure all writers produce
  /// identical schemas regardless of data arrival order. Unknown keys not in
  /// the set will cause an error during writing.
  folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns{};

  /// Maximum number of distinct flat-map keys allowed per file; 0 means
  /// unlimited. Writing fails when exceeded, bounding the per-key native memory
  /// the flat-map writer holds.
  uint32_t maxFlatMapKeys{kDefaultMaxFlatMapKeys};

  /// Per-type string attributes (e.g. Iceberg "iceberg.id") keyed by pre-order
  /// schema node id (matching `TypeWithId::id()`: root = 0, then depth-first --
  /// ROW children in field order, ARRAY element, MAP key then value). Each
  /// value is the attribute bag forwarded verbatim to
  /// `TypeBuilder::setAttributes(...)` on the matching node and preserves
  /// insertion order end-to-end through schema serialization.
  ///
  /// Node ids that do not correspond to a node in the input schema are ignored,
  /// so callers may submit a superset and let schema evolution drop entries
  /// that no longer apply.
  ///
  /// Empty map (default) is a no-op: every existing NIMBLE writer produces
  /// byte-identical files.
  folly::F14FastMap<uint32_t, std::vector<std::pair<std::string, std::string>>>
      schemaAttributes{};

  /// When true, the writer skips encoding flat map in-map boolean streams that
  /// are all-true (every row has the key) or all-false (no row has the key).
  /// The reader infers the in-map state from value stream presence: all-true
  /// keys have value streams, all-false keys do not.
  ///
  /// NOTE: readers that do not infer omitted in-map streams require this to
  /// remain false so constant in-map streams are physically present.
  bool skipConstantFlatMapInMapStreams{false};

  /// Columns that should be encoded as dictionary arrays
  /// NOTE: For each column, ALL the arrays inside this column will be encoded
  /// using dictionary arrays. In the future we'll have finer control on
  /// individual arrays within a column.
  folly::F14FastSet<std::string> dictionaryArrayColumns{};

  /// Columns that should be encoded as dictionary map
  /// NOTE: For each column, ALL the maps inside this column will be encoded
  /// using dictionary maps.
  folly::F14FastSet<std::string> deduplicatedMapColumns{};

  /// The metric logger would come populated with access descriptor information,
  /// application generated query id or specific sampling configs.
  std::shared_ptr<MetricsLogger> metricsLogger{};

  /// Optional feature reordering config.
  /// The key for this config is a (top-level) flat map column ordinal and
  /// the value is an ordered collection of feature ids. When provided, the
  /// writer will make sure that flat map features are grouped together and
  /// ordered based on this config.
  std::optional<std::vector<std::tuple<size_t, std::vector<int64_t>>>>
      featureReordering{};

  /// Optional captured encoding layout tree.
  /// Encoding layout tree is overlayed on the writer tree and the captured
  /// encodings are attempted to be used first, before resolving to perform an
  /// encoding selection.
  /// Captured encodings can be used to speed up writes (as no encoding
  /// selection is needed at runtime) and cal also provide better selected
  /// encodings, based on history data.
  std::optional<EncodingLayoutTree> encodingLayoutTree{};

  /// Compression settings to be used when encoding and compressing data streams
  CompressionOptions compressionOptions{};

  /// Per-chunk compression of encoded data streams (layered on top of
  /// compressionOptions).
  /// EXPERIMENTAL / benchmark-only; only Uncompressed, Zstd and Lz4 are
  /// supported.
  /// NOTE: !!! Do NOT enable in production !!!
  CompressionParams chunkCompression{.type = CompressionType::Uncompressed};

  /// Block size for BlockBitPacking encoding.
  /// EXPERIMENTAL: BlockBitPacking encoding is not production-ready. Do not
  /// enable for production tables without consulting the Nimble team
  ///(oncall: dwios).
  uint16_t blockBitPackingBlockSize = kBlockBitPackingBlockSize;

  /// When true, FOR-family payloads use the exact required bit width. When
  /// false, FixedBitWidth and PFOR round to byte or bucket boundaries.
  bool fixedBitWidthUseExactBits{false};

  /// FSST is kept only when its final encoded size is at most this fraction of
  /// the original string bytes.
  double fsstCompressionTargetRatio{0.6};

  /// EXPERIMENTATION: Allows ALP to participate in nested floating-point
  /// encoding selection. False by default; do not enable for production until
  /// ALP is production-ready.
  bool allowNestedAlpSelection{false};

  /// Maximum number of scratch vector buffers retained by each per-encode-task
  /// scratch pool. 0 disables scratch vector buffer caching. Disabled by
  /// default; benchmark callers can set a non-zero value to opt in. Memory is
  /// bounded by cached buffer count, not bytes, and cached buffers are dropped
  /// under memory arbitration.
  uint32_t maxCachedEncodingScratchBuffers{0};

  /// Maximum number of scratch buffers retained by each nested encoding buffer
  /// pool. 0 disables nested encoding buffer caching. Disabled by default;
  /// callers can set a non-zero value to opt in.
  uint32_t maxCachedNestedEncodingBuffers{0};

  /// In low-memory mode, the writer is trying to perform smaller (and more
  /// precise) buffer allocations. This means that overall, the writer will
  /// consume less memory, but will come with an additional cost, of more
  /// reallocations and extra data copy.
  /// TODO: This options should be removed and integrated into the
  /// inputGrowthPolicyFactory option (e.g. allow the caller to set an
  /// ExactGrowthPolicy, as defined here:
  /// velox/dwio/nimble/writer/BufferGrowthPolicy.h)
  bool lowMemoryMode{false};

  /// If present, metadata sections above this threshold size will be
  /// compressed.
  std::optional<uint32_t> metadataCompressionThreshold{};

  /// When flushing data streams into chunks, streams with raw data size smaller
  /// than this threshold will not be flushed.
  /// Note: this threshold is ignored when it is time to flush a stripe.
  uint64_t minStreamChunkRawSize{512 << 10};

  /// When flushing data streams into chunks, streams with raw data size larger
  /// than this threshold will be broken down into multiple smaller chunks. Each
  /// chunk will be at most this size.
  uint64_t maxStreamChunkRawSize{20 << 20};

  /// Used in place of maxStreamChunkRawSize for tables with large schemas.
  uint64_t wideSchemaMaxStreamChunkRawSize{2 << 20};

  /// When the number of schema nodes exceeds this threshold we use
  /// wideSchemaMaxStreamChunkRawSize in place of maxStreamChunkRawSize.
  size_t largeSchemaThreshold{500};

  /// Number of streams to try chunking between memory pressure evaluations.
  /// Note: this is ignored when it is time to flush a stripe.
  size_t chunkedStreamBatchSize{1024};

  /// The factory function that produces the root encoding selection policy.
  /// Encoding selection policy is the way to balance the tradeoffs of
  /// different performance factors (at both read and write times). Heuristics
  /// based, ML based or specialized policies can be specified.
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      [encodingFactory = ManualEncodingSelectionPolicyFactory{}](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };

  /// Provides policy that controls stripe sizes and memory footprint.
  std::function<std::unique_ptr<FlushPolicy>()> flushPolicyFactory = []() {
    // Buffering 256MB data before encoding stripes.
    return std::make_unique<StripeRawSizeFlushPolicy>(256 << 20);
  };

  /// Optional content-driven cutting policy. When set, Writer routes
  // every write() through the BufferPolicy (bufferInput → drain writeBuffer)
  // and emits one stripe per emitted BufferRange, bypassing shouldFlush.
  // When unset (the default), the writer takes the legacy path: write the
  // whole batch, then consult flushPolicyFactory's shouldFlush.
  // See BufferPolicy.h for the interface + lifecycle.
  std::function<std::unique_ptr<BufferPolicy>()> bufferPolicyFactory{};

  // When the writer needs to buffer data, and internal buffers don't have
  /// enough capacity, the writer is using this policy to claculate the the new
  /// capacity for the vuffers.
  std::function<std::unique_ptr<InputBufferGrowthPolicy>()>
      inputGrowthPolicyFactory =
          []() -> std::unique_ptr<InputBufferGrowthPolicy> {
    return DefaultInputBufferGrowthPolicy::withDefaultRanges();
  };

  /// When per-stream string buffers are enabled (disableSharedStringBuffers),
  /// this policy controls how the string buffer vectors grow.
  std::function<std::unique_ptr<InputBufferGrowthPolicy>()>
      stringBufferGrowthPolicyFactory =
          []() -> std::unique_ptr<InputBufferGrowthPolicy> {
    return DefaultInputBufferGrowthPolicy::withStringBufferRanges();
  };

  /// Input-buffer growth policy for these options: ExactGrowthPolicy under
  /// lowMemoryMode, else the amortized factory.
  std::unique_ptr<InputBufferGrowthPolicy> makeInputBufferGrowthPolicy() const {
    return lowMemoryMode ? std::make_unique<ExactGrowthPolicy>()
                         : inputGrowthPolicyFactory();
  }

  /// String-buffer counterpart of makeInputBufferGrowthPolicy().
  std::unique_ptr<InputBufferGrowthPolicy> makeStringBufferGrowthPolicy()
      const {
    return lowMemoryMode ? std::make_unique<ExactGrowthPolicy>()
                         : stringBufferGrowthPolicyFactory();
  }

  std::function<std::unique_ptr<velox::memory::MemoryReclaimer>()>
      reclaimerFactory = []() { return nullptr; };

  const velox::common::SpillConfig* spillConfig{nullptr};

  /// Sink-level IO counters that the writer accumulates written bytes and
  /// write wall time into, as deltas, on every write, flush and close. Not
  /// owned; must outlive the writer. Left null (the default) by callers that do
  /// not participate in Velox operator-level IO accounting.
  velox::io::IoStatistics* ioStatistics{nullptr};

  /// If provided, internal encoding operations will happen in parallel using
  /// the specified executor.
  ///
  /// The KeepAlive wrapper ensures that the executor object will be kept alive
  /// (allocated), and that the pool will be open for receiving new tasks. A
  /// shared_ptr would only guarantee that the object is still allocated, but
  /// not necessarily open for new task (e.g. it could have been .join()'ed
  /// through a different reference). Because of that, many libraries only
  /// provide KeepAlive references to executors, not shared_ptr, so taking a
  /// KeepAlive also makes it more convenient to clients.
  ///
  /// As a result, if a KeepAlive is still being held, clients trying to
  /// destruct the last reference of a shared_ptr to that executor will block
  /// until all KeepAlive references are destructed.
  folly::Executor::KeepAlive<> encodingExecutor{};

  /// When maxEncodeParallelism > 0 and encodingExecutor is set,
  /// FieldWriter::write() operations will be parallelized using coroutines
  /// scheduled on encodingExecutor.
  uint32_t maxEncodeParallelism{0};
  uint32_t minStreamsPerEncodeUnit{1};

  bool enableChunking{true};

  /// This callback will be visited on access to getDecodedVector in order to
  /// monitor usage of decoded vectors vs. data that is passed-through in the
  /// writer. Default function is no-op since its used for tests only.
  std::function<void(void)> vectorDecoderVisitor{[]() {}};

  /// Whether writer should ignore the top level nulls in the input.
  bool ignoreTopLevelNulls{false};

  bool enableStreamDeduplication{true};

  /// When true, string fields use per-field buffers instead of a shared buffer.
  /// This enables incremental memory reclamation during chunking.
  bool disableSharedStringBuffers{false};

  /// When true, enables consistency check between fileRawSize (accumulated via
  /// RawSizeUtils) and the root column statistics during file close.
  /// This is used to validate that column statistics accurately track raw
  /// sizes, with the goal of eventually replacing RawSizeUtils accumulation
  /// with column statistics for non-deduplicated columns.
  bool enableStatsConsistencyCheck{true};

  // Cache the encoding layout from the first encoding of each stream and
  // replay it on subsequent chunks/stripes, skipping the full encoding
  // selection cascade.
  bool enableEncodingSelectionCache{false};
};

} // namespace facebook::nimble

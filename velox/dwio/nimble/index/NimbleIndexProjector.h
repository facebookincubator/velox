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

#include <folly/io/IOBuf.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "velox/common/caching/FileHandle.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/tablet/DataInput.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/type/Subfield.h"

namespace facebook::nimble {

class TabletReaderCache;
namespace serde {
class StreamSlicer;
}

using Subfield = velox::common::Subfield;

/// NimbleIndexProjector takes a batch of index lookup requests (point lookups
/// or range scans with already-encoded keys) and column projections, uses the
/// Nimble cluster index to locate relevant stripes and row ranges, then reads
/// and serializes the projected columns for transport.
///
/// The output uses kTablet serialization format with a fixed-shape
/// per-slice header in front of the stripe body+trailer:
///   NODE 1 (per-slice header):
///     [version:1B=3][rowCount:varint]
///     [startRow:varint][endRow:varint]
///     [resumeKeyLength:varint]   0 = no key; N>0 = key of length N-1
///     [resumeKey bytes]          present when resumeKeyLength > 0
///   NODE 2 (shared stripe body + trailer):
///     [stream_data_0...][encodingType:1B][stream_sizes][trailer_size:u32]
///
/// The stripe slice IOBuf contains a per-slice header node followed by a shared
/// body+trailer chain. Multiple requests that hit the same stripe with
/// different row ranges share the body+trailer bytes via refcounted IOBuf
/// clones; only the header node is unique per slice.
///
/// Usage:
///   auto result = projector.project(request, options);
///   for (size_t i = 0; i < result.responses.size(); ++i) {
///     const auto& response = result.responses[i];
///     for (const auto& slice : response.slices) {
///       // `slice` is a self-describing kTablet IOBuf chain with
///       // the request's row range and (on the last slice of a
///       // cut-short response) the resume key embedded in the header.
///     }
///   }
/// NOTE: NimbleIndexProjector is not thread-safe. Each thread must use its
/// own instance.
class NimbleIndexProjector {
 public:
  /// Destroys the projector and its cached stream slicer.
  ~NimbleIndexProjector();

  // TODO: projectedSubfields currently must match file schema column names.
  // Add table-to-file column name mapping for schema evolution support.
  static std::unique_ptr<NimbleIndexProjector> create(
      TabletReaderCache& tabletReaderCache,
      const velox::FileHandle& fileHandle,
      const std::vector<Subfield>& projectedSubfields,
      const velox::dwio::common::ReaderOptions& options);

  /// Creates a projector with projection metadata built by the caller.
  /// The projection must have been built from the schema of `tablet`.
  static std::unique_ptr<NimbleIndexProjector> create(
      std::shared_ptr<TabletReader> tablet,
      const velox::FileHandle& fileHandle,
      std::shared_ptr<const NimbleTypeProjection> projection,
      const velox::dwio::common::ReaderOptions& options);

  /// Options for controlling projection behavior.
  ///
  /// The three limits below (maxRows, maxBytes, maxRowsPerRequest) only bound
  /// how much data a single project() call returns; on their own they never
  /// decide whether a resume key is produced. Resume keys are governed solely
  /// by needResumeKey: when it is set, any request whose results were cut short
  /// by one of these limits carries a resume key; when it is unset, no resume
  /// keys are produced regardless of which limits fire.
  struct Options {
    /// Soft limit on total rows across all requests. 0 means no limit.
    /// When the running total exceeds this limit mid-stripe, the entire
    /// stripe is still included (stripe-boundary soft limit). Processing
    /// stops after that stripe completes.
    uint64_t maxRows{0};
    /// Soft limit on total serialized bytes across all requests. 0 means no
    /// limit. Like maxRows, operates at stripe granularity: at least one
    /// stripe is always included, then processing stops after the stripe
    /// that exceeds the budget.
    uint64_t maxBytes{0};
    /// Hard per-request row limit. 0 means no limit. Each request's row
    /// range is clipped so that it never returns more than this many rows.
    uint64_t maxRowsPerRequest{0};
    /// Maximum tolerated avoidable row overfetch per stripe before stream
    /// slicing is applied. 0.0 slices whenever any overfetch can be removed;
    /// 1.0 disables slicing.
    double maxOverfetchRowsRatio{1.0};
    /// When set, every request whose results were cut short by a limit
    /// (maxRows, maxBytes, or maxRowsPerRequest) is given a resume key so the
    /// caller can continue from where the request stopped (use the resume key
    /// as the new lowerKey with the original upperKey). When unset, no resume
    /// keys are produced even if a limit truncated the results.
    bool needResumeKey{false};
  };

  /// Request for a batch of index lookups.
  struct Request {
    /// Pre-encoded key bounds for each lookup.
    std::vector<velox::serializer::EncodedKeyBounds> keyBounds;
  };

  /// Response for a single request.
  struct Response {
    /// One self-describing kTablet IOBuf chain per (request × stripe)
    /// intersection. Each entry covers a contiguous row range within one
    /// stripe; the row range is embedded in the chain's header. Empty for
    /// miss. Slices for overlapping requests share the body+trailer bytes
    /// via refcounted SharedInfo on the second IOBuf node.
    ///
    /// If the response is truncated and has a resume key, it is embedded in
    /// the header of the last slice. Use `rocks::readResultResumeKey()` on
    /// the last slice to extract it.
    std::vector<folly::IOBuf> slices;

    /// When Options::needResumeKey is set and this request's results were cut
    /// short by a limit (maxRows, maxBytes, or maxRowsPerRequest), the encoded
    /// resume key for continuation. The caller constructs new key bounds using
    /// this as lowerKey with their original upperKey. nullopt if complete, a
    /// miss, or needResumeKey was not set.
    ///
    /// Also embedded in the last slice's per-slice header (when slices
    /// is non-empty) so consumers that hold an IOBuf can recover the key
    /// without keeping the Response struct around.
    std::optional<std::string> resumeKey;
  };

  /// Result of a project() call.
  struct Result {
    /// One entry per request, in request order.
    std::vector<Response> responses;
  };

  /// Projects the requested columns for the given batch of index lookups.
  /// Returns one Response per request, in order. Processes all relevant
  /// stripes internally.
  Result project(const Request& request, const Options& options);

  /// Returns the projected nimble schema. Preserves encoding-specific types
  /// (ArrayWithOffsets, SlidingWindowMap, FlatMap) from the file schema.
  /// Clients need this to build a Deserializer for the output data.
  const std::shared_ptr<const Type>& projectedNimbleType() const {
    return projection_->nimbleType;
  }

  /// Statistics captured during project().
  struct Stats {
    /// Number of stripes read from the tablet.
    uint32_t numReadStripes{0};
    /// Number of read stripes serialized through stream slicing.
    uint32_t numSlicedStripes{0};
    /// Total rows read from storage (entire stripe row counts). Includes
    /// rows outside the requested row ranges that are read because we
    /// fetch entire projected streams per stripe.
    uint64_t numReadRows{0};
    /// Total rows in the output result (only the requested row ranges).
    /// The difference numReadRows - numProjectedRows is over-fetched rows.
    uint64_t numProjectedRows{0};
    /// Total serialized output bytes.
    uint64_t numOutputBytes{0};

    /// Time spent looking up stripes and row ranges via the tablet index.
    velox::CpuWallTiming lookupTiming;
    /// Time spent building per-stripe stream and request-range plans.
    velox::CpuWallTiming prepareTiming;
    /// Time spent loading stripe stream data from tablet.
    velox::CpuWallTiming scanTiming;
    /// Time spent serializing projected streams into kTablet format.
    velox::CpuWallTiming projectionTiming;

    std::string toString() const;
  };

  /// Returns cumulative statistics across all project() calls.
  const Stats& stats() const {
    return stats_;
  }

 private:
  NimbleIndexProjector(
      std::shared_ptr<TabletReader> tablet,
      std::shared_ptr<velox::ReadFile> file,
      std::unique_ptr<DataInput> dataInput,
      std::shared_ptr<const NimbleTypeProjection> projection,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<velox::io::IoStatistics> ioStats);

  // A request index paired with its stripe-relative row range.
  struct StripeRange {
    uint32_t requestIndex{};
    // Stripe-relative row range, already intersected with stripe boundaries.
    RowRange rowRange;
  };

  // CSR (compressed sparse row) layout mapping stripes to request row ranges.
  // All StripeRange entries are stored in a single flat vector (`entries`),
  // with `offsets[i]` marking where stripe i's entries begin.
  struct StripeRanges {
    uint32_t startStripe{0};
    uint32_t numStripes{0};
    // Flat storage of all per-stripe row ranges, grouped by stripe.
    std::vector<StripeRange> ranges;
    // offsets[i] = start index in ranges for stripe i (relative to
    // startStripe). Size = numStripes + 1.
    std::vector<uint32_t> offsets;

    std::span<const StripeRange> getRanges(uint32_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, numStripes);
      return {
          ranges.data() + offsets[stripeIndex],
          offsets[stripeIndex + 1] - offsets[stripeIndex]};
    }

    void clear() {
      startStripe = 0;
      numStripes = 0;
      ranges.clear();
      offsets.clear();
    }
  };

  // Initializes per-project() state: stores request/options pointers and
  // resets running totals.
  void initRequest(const Request& request, const Options& options);

  // Clears all per-project() state set by initRequest(). Invoked on scope
  // exit so a subsequent project() call starts from a clean slate.
  void clearRequest();

  // Looks up all requests via the cluster index and maps them to stripes.
  // Populates ctx_.stripeRanges.
  void lookupStripes();

  // Output of prepareStripes(). Per-stripe vectors are indexed by stripe
  // offset in the compact plan and only contain stripes retained for
  // processing.
  struct ScanPlan {
    // Absolute stripe indices in processing order.
    std::vector<uint32_t> stripeIndices;
    // Total rows in each planned stripe.
    std::vector<uint32_t> numRows;
    // Whether each planned stripe needs the Row/FlatMap null-barrier path.
    std::vector<bool> requiresNullBarriers;
    // Number of projected streams present in each planned stripe.
    std::vector<uint32_t> numStreams;
    // Total logical bytes across all projected streams in each planned stripe.
    std::vector<uint64_t> projectedBytes;
    // File offsets for stripe stream payloads.
    std::vector<uint64_t> stripeFileOffsets;
    // Flat reusable storage for all per-stripe projected stream slots. Each
    // stripe occupies projection_->streamOffsets.size() consecutive entries.
    std::vector<StripeGroup::StreamLocation> projectedStreams;
    // Start offsets into stripeRanges for each planned stripe. The last entry
    // is the total range count.
    std::vector<size_t> stripeRangeOffsets;
    // Flat reusable storage for all request ranges retained by planned stripes.
    std::vector<StripeRange> stripeRanges;
    bool truncated{false};
  };

  // Returns this stripe's projected stream slots from the flat ScanPlan
  // storage.
  std::span<const StripeGroup::StreamLocation> stripeProjectedStreams(
      size_t stripeOffset) const;

  // Returns this stripe's request ranges from the flat ScanPlan storage.
  std::span<const StripeRange> plannedStripeRanges(size_t stripeOffset) const;

  // Appends one stripe to the plan and returns its projected byte count.
  uint64_t appendStripePlan(uint32_t stripeIndex, size_t rangeOffset);

  // Computes the stripe-relative body range based on request row ranges and
  // Options::maxOverfetchRowsRatio.
  RowRange stripeRowRangeToPack(size_t stripeOffset) const;

  // Records the resume key for a request that reached its maxRowsPerRequest cap
  // in the stripe at `stripeOffset` (index relative to
  // stripeRanges.startStripe, not an absolute stripe index). `readEndRow` is
  // the request's stripe-relative end row after clipping. When `partialRead` is
  // true the cap fell inside the stripe, so the key is the row-precise key at
  // `readEndRow`; otherwise the cap landed on the stripe boundary and the key
  // resumes from the next stripe still holding this request. The stripe's
  // absolute start row is derived from `stripeOffset`. Idempotent (no-op once a
  // key is recorded); callers gate on Options::needResumeKey.
  void setResumeKey(
      uint32_t requestIndex,
      uint32_t stripeOffset,
      uint32_t readEndRow,
      bool partialRead);

  // Applies row and byte limits to the looked-up stripe ranges, computes stream
  // metadata for selected stripes, and populates ctx_.plan.
  void prepareStripes();

  // Enqueues all projected streams from ctx_.plan into DataInput and issues a
  // single coalesced load() call.
  void loadStripes();

  // Serializes each stripe's loaded streams, builds per-request results,
  // and finalizes the result.
  Result processStripes();

  // Computes per-stripe pack ranges and returns the upper-bound byte estimate
  // used to allocate the sliced-output arena. Returns 0 when no stripe will be
  // sliced.
  uint64_t prepareStripePackRanges();

  // Estimates sliced stream bytes for a partial stripe pack.
  uint64_t estimateSlicedStripeBytes(
      size_t stripeOffset,
      const RowRange& packRange) const;

  // Returns the per-project() sliced-output arena.
  Buffer& sliceOutputBuffer();

  // Transfers sliced-output arena chunks to the shared owner referenced by the
  // packed stripe IOBufs.
  void finalizeSliceOutputBuffer();

  // Loaded projected stream views and optional source deduplication metadata.
  struct StripeStreamViews {
    void clear() {
      streams.clear();
      presentIndices.clear();
      canonicalIndices.clear();
    }

    // Indexed by projected stream index; absent streams have empty views.
    std::vector<std::string_view> streams;
    // Present stream indices in projected stream order. Populated only when
    // canonical streams are resolved.
    std::vector<size_t> presentIndices;
    // Canonical projected index for each present stream. Populated only when
    // canonical streams are resolved.
    std::vector<std::optional<size_t>> canonicalIndices;
  };

  // Collects loaded stream views for one stripe. Optionally resolves source
  // deduplication and populates the canonical stream metadata.
  StripeStreamViews& collectStripeStreamViews(
      size_t stripeOffset,
      bool resolveCanonicalStreams);

  // Serialized stripe body and metadata computed while packing.
  struct PackedStripe {
    folly::IOBuf body;
    RowRange rowRange;
    bool requiresNullBarrier{false};
    bool streamHasChunkHeader{false};
  };

  // Selects full or partial packing based on the requested stripe range.
  PackedStripe packStripe(size_t stripeOffset, const RowRange& packRange);

  // Packs a full stripe zero-copy while preserving source stream deduplication.
  PackedStripe packFullStripe(size_t stripeOffset);

  // Slices and packs a partial stripe without reusing source deduplication.
  PackedStripe packPartialStripe(
      size_t stripeOffset,
      const RowRange& packRange);

  // When Options::needResumeKey is set, attaches resume keys to requests whose
  // results were cut short: per-request keys from maxRowsPerRequest caps, plus
  // (on global maxRows/maxBytes truncation) keys for requests that still have
  // data in an unprocessed stripe. No-op when needResumeKey is unset.
  void setResumeKeys(Result& result);

  // Iterates planned stripes and ctx_.packedStripes to build per-request
  // output slices. Each slice clones the shared stripe body and prepends a
  // per-request header with the row range and (on the last slice of a
  // truncated response) the resume key.
  void buildResult(Result& result);

  inline uint32_t stripeRowCount(uint32_t stripe) const {
    return static_cast<uint32_t>(tablet_->stripeRowCount(stripe));
  }

  // Computes the stripe-relative row range by intersecting the file-level
  // rowRangeLimit with the stripe boundaries.
  RowRange stripeRowRange(uint32_t stripe, const RowRange& rowRangeLimit) const;

  const std::shared_ptr<velox::ReadFile> file_;
  const std::shared_ptr<TabletReader> tablet_;
  const std::shared_ptr<velox::io::IoStatistics> ioStats_;
  velox::memory::MemoryPool* const pool_;
  std::unique_ptr<DataInput> dataInput_;
  const ClusterIndex* const clusterIndex_;
  const uint32_t numStripes_{0};

  const std::shared_ptr<const NimbleTypeProjection> projection_;
  // Reused across stripes; its raw input format is fixed by the tablet.
  const std::unique_ptr<serde::StreamSlicer> streamSlicer_;

  // Per-project() call state. Set by initRequest(), populated through the
  // pipeline (lookupStripes → prepareStripes → loadStripes → processStripes),
  // and reset on return.
  struct ProjectionContext {
    const Request* request{nullptr};
    const Options* options{nullptr};
    uint32_t numRequests{0};
    // CSR mapping from stripes to request row ranges. Populated by
    // lookupStripes(), read-only during stripe processing.
    StripeRanges stripeRanges;
    // Populated by prepareStripes().
    ScanPlan plan;
    // Per-request flag: true if the request has ranges in any planned stripe.
    // Set by prepareStripes(), used by setResumeKeys().
    std::vector<bool> hasStripeRanges;
    // Per-request resume keys set when a request reaches maxRowsPerRequest and
    // Options::needResumeKey is enabled.
    std::vector<std::optional<std::string>> resumeKeys;
    // Flat array of enqueue indices, logically
    // [stripeOffset * numProjectedStreams + streamIndex]. nullopt for absent
    // streams. Populated by loadStripes().
    std::vector<std::optional<uint32_t>> dataInputIndices;
    // Handle keeping loaded data alive for zero-copy BufferRefs.
    DataInput::Handle dataHandle;
    // Serialized stripe bodies and metadata, one per planned stripe. Populated
    // by processStripes(), consumed during buildResult().
    std::vector<PackedStripe> packedStripes;
    // Per planned stripe row range selected for packing. Full-stripe ranges use
    // zero-copy packing; narrower ranges use StreamSlicer.
    std::vector<RowRange> stripePackRanges;
    // Reusable per-request vectors.
    std::vector<uint64_t> rowsPerRequest;
    std::vector<size_t> sliceCounts;
    std::vector<size_t> emittedSlices;
    // Shared arena for all sliced stream bytes produced by one project() call.
    // The arena is transferred into sliceOutputChunks after all sliced stripes
    // have been packed, letting result IOBufs share the same chunk owner.
    std::unique_ptr<Buffer> sliceOutputBuffer;
    std::shared_ptr<std::vector<velox::BufferPtr>> sliceOutputChunks;

    // Reusable per-stripe inputs for the kTablet trailer. packStripe()
    // rebuilds these vectors for each stripe and clears them on exit to avoid
    // repeated allocations across stripes.
    struct PackScratch {
      void clear() {
        streamIds.clear();
        streamSizeIndices.clear();
        uniqueStreamSizes.clear();
        canonicalStreamSizeIndices.clear();
      }

      void reserve(uint32_t numProjectedStreams) {
        streamIds.reserve(numProjectedStreams);
        streamSizeIndices.reserve(numProjectedStreams);
        uniqueStreamSizes.reserve(numProjectedStreams);
        canonicalStreamSizeIndices.reserve(numProjectedStreams);
      }

      // Present projected stream slots, in projected-stream order.
      std::vector<uint32_t> streamIds;
      // For each streamIds entry, the body-order index of the unique stream
      // bytes used by that slot.
      std::vector<uint32_t> streamSizeIndices;
      // Sizes for unique stream byte ranges, in body order.
      std::vector<uint32_t> uniqueStreamSizes;
      // Index into uniqueStreamSizes for each projected stream slot. Duplicate
      // slots reuse their canonical stream's index.
      std::vector<std::optional<uint32_t>> canonicalStreamSizeIndices;
    };
    PackScratch packScratch;
    // Reusable loaded-stream views filled while packing one stripe.
    StripeStreamViews streamViewsScratch;
  };
  ProjectionContext ctx_;

  Stats stats_;
};

} // namespace facebook::nimble

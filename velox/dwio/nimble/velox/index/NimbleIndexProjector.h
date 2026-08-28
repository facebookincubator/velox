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
#include <string>
#include <vector>

#include <folly/io/IOBuf.h>

#include "velox/common/caching/FileHandle.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/tablet/DataInput.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
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
    // Index into the caller's request vector.
    uint32_t requestIndex{};
    // Stripe-relative row range, already intersected with stripe boundaries.
    RowRange rowRange;
  };

  // One lookup result mapped to the physical tablet stripes it touches.
  struct ResolvedRequest {
    // Index into the caller's request vector.
    uint32_t requestIndex{};
    // First physical tablet stripe touched by rowRange.
    uint32_t startStripe{};
    // One past the last physical tablet stripe touched by rowRange.
    uint32_t endStripe{};
    // File-level row range returned by the cluster index.
    RowRange rowRange;
  };

  // One request range resolved to a physical tablet stripe before grouping.
  struct ResolvedStripe {
    // Physical tablet stripe containing range.
    uint32_t tabletStripeIndex{};
    // Request range intersected with tabletStripeIndex.
    StripeRange range;
  };

  // Initializes per-project() state: stores request/options pointers and
  // resets running totals.
  void initRequest(const Request& request, const Options& options);

  // Clears all per-project() state set by initRequest(). Invoked on scope
  // exit so a subsequent project() call starts from a clean slate.
  void clearRequest();

  // Looks up all requests via the cluster index and maps them to plan stripes.
  void lookupStripes();

  // Output of lookupStripes() and prepareStripes(). Per-stripe vectors are
  // indexed by plan stripe index: a dense slot over planned tablet stripes.
  class ScanPlan {
   public:
    // Returns the number of tablet stripes selected by index lookup.
    size_t numLookupStripes() const {
      return stripeIndices_.size();
    }

    // Returns the number of lookup stripes kept after applying read limits.
    size_t numLoadStripes() const {
      return numLoadStripes_;
    }

    // Returns the physical tablet stripe id for a dense plan stripe index.
    uint32_t tabletStripeIndexAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, stripeIndices_.size());
      return stripeIndices_[stripeIndex];
    }

    // Returns request ranges for one plan stripe. Ranges for unprepared stripes
    // are the original lookup result; ranges for prepared stripes are updated
    // in place, with pruned entries marked empty.
    std::span<const StripeRange> rangesAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, stripeIndices_.size());
      NIMBLE_CHECK_LT(
          stripeIndex + 1,
          stripeRangeOffsets_.size(),
          "Lookup stripe range offsets must include a final sentinel");
      const auto beginOffset = stripeRangeOffsets_[stripeIndex];
      const auto endOffset = stripeRangeOffsets_[stripeIndex + 1];
      NIMBLE_CHECK_LE(
          endOffset,
          stripeRanges_.size(),
          "Lookup stripe request range exceeds plan storage");
      return std::span{stripeRanges_}.subspan(
          beginOffset, endOffset - beginOffset);
    }

    // Returns mutable request ranges for in-place pruning.
    std::span<StripeRange> mutableRangesAt(size_t stripeIndex) {
      NIMBLE_CHECK_LT(stripeIndex, stripeIndices_.size());
      NIMBLE_CHECK_LT(
          stripeIndex + 1,
          stripeRangeOffsets_.size(),
          "Stripe range offsets must include a final sentinel");
      const auto beginOffset = stripeRangeOffsets_[stripeIndex];
      const auto endOffset = stripeRangeOffsets_[stripeIndex + 1];
      NIMBLE_CHECK_LE(
          endOffset,
          stripeRanges_.size(),
          "Stripe request range exceeds plan storage");
      return std::span{stripeRanges_}.subspan(
          beginOffset, endOffset - beginOffset);
    }

    // Reserves storage for the lookup result grouped by stripe.
    void reserve(size_t numStripes, size_t numRanges) {
      stripeIndices_.reserve(numStripes);
      stripeRanges_.reserve(numRanges);
      stripeRangeOffsets_.reserve(numStripes + 1);
    }

    // Appends a tablet stripe id to the planned-stripe mapping.
    void addStripe(uint32_t tabletStripeIndex) {
      stripeIndices_.push_back(tabletStripeIndex);
    }

    // Appends a request range to the current lookup stripe group.
    void addRange(StripeRange stripeRange) {
      stripeRanges_.push_back(stripeRange);
    }

    // Appends a lookup range group boundary.
    void addRangeOffset(uint32_t offset) {
      stripeRangeOffsets_.push_back(offset);
    }

    // Resets read-time state while preserving lookup-time stripe groups.
    void reset(size_t numProjectedStreams) {
      const auto numLookupStripes = this->numLookupStripes();
      numProjectedStreams_ = numProjectedStreams;
      numLoadStripes_ = 0;
      numStripeRows_.assign(numLookupStripes, 0);
      requiresNullBarriers_.assign(numLookupStripes, false);
      numStripeStreams_.assign(numLookupStripes, 0);
      projectedStripeBytes_.assign(numLookupStripes, 0);
      stripeFileOffsets_.assign(numLookupStripes, 0);
      projectedStreams_.clear();
      projectedStreams_.resize(numLookupStripes * numProjectedStreams);
      truncated_ = false;
    }

    // Marks the plan as stopped by a global row or byte limit.
    void setTruncated() {
      truncated_ = true;
    }

    // Returns whether a global row or byte limit stopped planning early.
    bool truncated() const {
      return truncated_;
    }

    // Initializes per-stripe stream metadata before stream locations are read.
    void initStripe(size_t stripeIndex, uint32_t numRows, uint64_t fileOffset) {
      NIMBLE_CHECK_LT(stripeIndex, numStripeRows_.size());
      numStripeRows_[stripeIndex] = numRows;
      requiresNullBarriers_[stripeIndex] = false;
      numStripeStreams_[stripeIndex] = 0;
      projectedStripeBytes_[stripeIndex] = 0;
      stripeFileOffsets_[stripeIndex] = fileOffset;
      numLoadStripes_ = stripeIndex + 1;
    }

    // Records one present projected stream in a prepared stripe.
    void addProjectedStream(
        size_t stripeIndex,
        uint64_t streamSize,
        bool requiresNullBarrier) {
      NIMBLE_CHECK_LT(stripeIndex, numStripeStreams_.size());
      ++numStripeStreams_[stripeIndex];
      projectedStripeBytes_[stripeIndex] += streamSize;
      requiresNullBarriers_[stripeIndex] =
          requiresNullBarriers_[stripeIndex] || requiresNullBarrier;
    }

    // Returns the total row count for one prepared stripe.
    uint32_t numRowsAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, numStripeRows_.size());
      return numStripeRows_[stripeIndex];
    }

    // Returns whether a prepared stripe needs the null-barrier path.
    bool requiresNullBarrierAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, requiresNullBarriers_.size());
      return requiresNullBarriers_[stripeIndex];
    }

    // Returns the number of present projected streams in one prepared stripe.
    uint32_t numStreamsAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, numStripeStreams_.size());
      return numStripeStreams_[stripeIndex];
    }

    // Returns the logical projected bytes in one prepared stripe.
    uint64_t projectedBytesAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, projectedStripeBytes_.size());
      return projectedStripeBytes_[stripeIndex];
    }

    // Returns the file offset for one prepared stripe's stream payload.
    uint64_t stripeFileOffsetAt(size_t stripeIndex) const {
      NIMBLE_CHECK_LT(stripeIndex, stripeFileOffsets_.size());
      return stripeFileOffsets_[stripeIndex];
    }

    // Returns mutable projected stream locations for one plan stripe.
    std::span<StripeGroup::StreamLocation> projectedStreamsAt(
        size_t stripeIndex) {
      const auto offset = stripeIndex * numProjectedStreams_;
      NIMBLE_CHECK_LE(
          offset + numProjectedStreams_,
          projectedStreams_.size(),
          "Stripe projected stream range exceeds plan storage");
      return std::span{projectedStreams_}.subspan(offset, numProjectedStreams_);
    }

    // Returns projected stream locations for one plan stripe.
    std::span<const StripeGroup::StreamLocation> projectedStreamsAt(
        size_t stripeIndex) const {
      const auto offset = stripeIndex * numProjectedStreams_;
      NIMBLE_CHECK_LE(
          offset + numProjectedStreams_,
          projectedStreams_.size(),
          "Stripe projected stream range exceeds plan storage");
      return std::span{projectedStreams_}.subspan(offset, numProjectedStreams_);
    }

    // Clears lookup-time and read-time state.
    void clear() {
      stripeIndices_.clear();
      stripeRanges_.clear();
      stripeRangeOffsets_.clear();
      numLoadStripes_ = 0;
      numStripeRows_.clear();
      requiresNullBarriers_.clear();
      numStripeStreams_.clear();
      projectedStripeBytes_.clear();
      stripeFileOffsets_.clear();
      projectedStreams_.clear();
      numProjectedStreams_ = 0;
      truncated_ = false;
    }

   private:
    // Tablet stripe ids in ascending planned-stripe order.
    std::vector<uint32_t> stripeIndices_;
    // Request ranges grouped by plan stripe. During preparation, ranges are
    // pruned in place and dropped entries are marked empty so offsets stay
    // stable for unprepared stripes.
    std::vector<StripeRange> stripeRanges_;
    // Start offsets into stripeRanges_ for each plan stripe. The last
    // entry is the total lookup range count.
    std::vector<uint32_t> stripeRangeOffsets_;
    // Lookup-stripe prefix prepared for loading. Earlier slots can have all
    // ranges pruned when later requests still need reading.
    size_t numLoadStripes_{0};
    // Total rows in each plan stripe.
    std::vector<uint32_t> numStripeRows_;
    // Whether each plan stripe needs the Row/FlatMap null-barrier path.
    std::vector<bool> requiresNullBarriers_;
    // Number of projected streams present in each plan stripe.
    std::vector<uint32_t> numStripeStreams_;
    // Total logical bytes across all projected streams in each plan stripe.
    std::vector<uint64_t> projectedStripeBytes_;
    // File offsets for stripe stream payloads.
    std::vector<uint64_t> stripeFileOffsets_;
    // Flat reusable storage for all per-stripe projected stream slots. Each
    // stripe occupies projection_->streamOffsets.size() consecutive entries.
    std::vector<StripeGroup::StreamLocation> projectedStreams_;
    // Number of projected stream slots reserved for each plan stripe.
    size_t numProjectedStreams_{0};
    // True when global row or byte limits stop planning before all lookup
    // stripes are prepared.
    bool truncated_{false};
  };

  // Prepares one stripe for reading and returns its projected byte count.
  uint64_t prepareStripe(size_t stripeIndex);

  // Computes the stripe-relative body range based on request row ranges and
  // Options::maxOverfetchRowsRatio.
  RowRange stripeRowRangeToPack(size_t stripeIndex) const;

  // Records the resume key for a request that reached its maxRowsPerRequest cap
  // in the stripe at `stripeIndex`. `readEndRow` is the request's
  // stripe-relative end row after clipping. When `partialRead` is true the cap
  // fell inside the stripe, so the key is the row-precise key at `readEndRow`;
  // otherwise the cap landed on the stripe boundary and the key resumes from
  // the next stripe still holding this request. Idempotent (no-op once a key is
  // recorded); callers gate on Options::needResumeKey.
  void setResumeKey(
      uint32_t requestIndex,
      size_t stripeIndex,
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
      size_t stripeIndex,
      const RowRange& packRange) const;

  // Returns the per-project() sliced-output arena.
  Buffer& sliceOutputBuffer();

  // Transfers sliced-output arena chunks to the shared owner referenced by the
  // packed stripe IOBufs.
  void finalizeSliceOutputBuffer();

  // Loaded projected stream views and optional source deduplication metadata.
  struct StripeStreamViews {
    // Indexed by projected stream index; absent streams have empty views.
    std::vector<std::string_view> streams;
    // Present stream indices in projected stream order. Populated only when
    // canonical streams are resolved.
    std::vector<size_t> presentIndices;
    // Canonical projected index for each present stream. Populated only when
    // canonical streams are resolved.
    std::vector<std::optional<size_t>> canonicalIndices;

    void clear() {
      streams.clear();
      presentIndices.clear();
      canonicalIndices.clear();
    }
  };

  // Collects loaded stream views for one stripe. Optionally resolves source
  // deduplication and populates the canonical stream metadata.
  void collectStripeStreamViews(
      size_t stripeIndex,
      bool resolveCanonicalStreams);

  // Serialized stripe body and metadata computed while packing.
  struct PackedStripe {
    folly::IOBuf body;
    RowRange rowRange;
    bool requiresNullBarrier{false};
    bool streamHasChunkHeader{false};
  };

  // Selects full or partial packing based on the requested stripe range.
  PackedStripe packStripe(size_t stripeIndex, const RowRange& packRange);

  // Packs a full stripe zero-copy while preserving source stream deduplication.
  PackedStripe packFullStripe(size_t stripeIndex);

  // Slices and packs a partial stripe without reusing source deduplication.
  PackedStripe packPartialStripe(size_t stripeIndex, const RowRange& packRange);

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
    // Populated by lookupStripes() and prepareStripes().
    ScanPlan plan;
    // Per-request flag set when the request has ranges in any planned stripe.
    // Set by prepareStripes(), used by setResumeKeys().
    std::vector<bool> requestHasRanges;
    // Per-request resume keys set when a request reaches maxRowsPerRequest and
    // Options::needResumeKey is enabled.
    std::vector<std::optional<std::string>> resumeKeys;
    // Flat array of enqueue indices, logically
    // [stripeIndex * numProjectedStreams + streamIndex]. nullopt for
    // absent streams. Populated by loadStripes().
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
    // Lookup results that touched rows, mapped to physical tablet stripes.
    std::vector<ResolvedRequest> resolvedRequests;
    // Lookup results expanded to one entry per touched physical tablet stripe.
    std::vector<ResolvedStripe> resolvedStripes;
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
    StripeStreamViews loadedStreamViews;
  };
  ProjectionContext ctx_;

  Stats stats_;
};

} // namespace facebook::nimble

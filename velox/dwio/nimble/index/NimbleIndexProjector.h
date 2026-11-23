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

#include <folly/coro/Task.h>
#include <folly/io/IOBuf.h>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "velox/buffer/BufferPool.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Checksum.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/common/StripeRange.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/StripeStreams.h"
#include "velox/dwio/nimble/tablet/DataInput.h"
#include "velox/dwio/nimble/tablet/FileProperties.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/type/Subfield.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble {

class BlockCache;
class StreamReader;
class TabletReaderCache;
namespace serde {
class StreamSlicer;
}

using Subfield = velox::common::Subfield;

/// Projects columns for a batch of range lookups with already-encoded keys,
/// using the Nimble cluster index to read the matching stripe rows. Single-key
/// point lookups are available through lookup(encodedKey). Results can be
/// serialized for transport, decoded into a Velox vector, or supplied as
/// stripe streams to iterators.
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
/// A request cut short before any of its rows were read gets a single
/// header-only slice instead: NODE 1 alone, with rowCount 0, an empty row
/// range, and the resume key.
///
/// Usage:
///   auto result = projector.projectStreams(request, options);
///   for (size_t i = 0; i < result.responses.size(); ++i) {
///     const auto& response = result.responses[i];
///     for (const auto& slice : response.slices) {
///       // `slice` is a self-describing kTablet IOBuf chain with
///       // the request's row range and (on the last slice of a
///       // cut-short response) the resume key embedded in the header.
///     }
///   }
///
/// Shared dictionary encoding is not supported. The output carries projected
/// stream bytes only and has nowhere to put the alphabet a shared dictionary
/// stream decodes against, so create() rejects any tablet that records shared
/// dictionaries.
///
/// NOTE: NimbleIndexProjector is not thread-safe. Each thread must use its
/// own instance.
/// The caller must keep the memory pool alive through reads and asynchronous
/// prefetch cleanup.
class NimbleIndexProjector {
 public:
  /// Destroys the projector and its cached stream slicer.
  ~NimbleIndexProjector();

  /// Creates a projector over the tablet reader entry held by
  /// `tabletReaderCache` for `fileHandle`.
  ///
  /// When `blockCache` is non-null, stripe data reads route through
  /// `CachedDataInput` against the caller-owned cache; otherwise reads use
  /// `DirectDataInput` with no caching. The cache must outlive the projector.
  ///
  /// When ReaderOptions::coLoadKeyStreams() is set and the file's cluster index
  /// stores key chunks in stripes (KeyStreamLayout::kStripe), the projector
  /// co-loads key streams: it reads the stripes that a projection can select
  /// rows of in one load, each with its key stream, which holds the stripe's
  /// keys, and resolves the bounds from those key streams. loadStripeStreams()
  /// reads the key stream of each stripe too. Stream checksums verify the key
  /// streams like the projected streams. A co-load saves the read that waits
  /// for the key chunks, but reads the stripe streams even when a key is
  /// missing. Under Options::maxRows, Options::maxBytes or
  /// Options::maxRowsPerRequest, a projection picks the stripes to read from
  /// stripe metadata before the lookup, which does not tell how many rows of
  /// its stripe a bound selects, so it can read stripes past where the limits
  /// stop the projection. It also reads the stripe right after where the limits
  /// or a request's maxRowsPerRequest stop the projection, for its key stream,
  /// which holds the first key that a request continuing there resumes at, and
  /// it reads nothing past that stripe. It takes resume keys from the key
  /// streams it read, and resolves only the bounds in those streams: a request
  /// that the limits keep the projection from reaching gets its lowerKey back
  /// as its resume key, even when it matches no row. Without a co-load, the
  /// projector reads the key chunks before the stripe streams. Row windows and
  /// derived lookup requests use the index lookup before loading stripes, so
  /// their constraints are applied before stripe planning.
  ///
  // TODO: projectedSubfields currently must match file schema column names.
  // Add table-to-file column name mapping for schema evolution support.
  static std::unique_ptr<NimbleIndexProjector> create(
      TabletReaderCache& tabletReaderCache,
      const velox::FileHandle& fileHandle,
      const std::vector<Subfield>& projectedSubfields,
      BlockCache* blockCache,
      const velox::dwio::common::ReaderOptions& options);

  /// Creates a projector with projection metadata built by the caller.
  /// The projection must have been built from the schema of `tablet`.
  /// When `blockCache` is non-null, stripe data reads route through
  /// `CachedDataInput` against the caller-owned cache; otherwise reads use
  /// `DirectDataInput` with no caching. The cache must outlive the projector.
  /// ReaderOptions::coLoadKeyStreams() selects the co-load as create() above
  /// describes.
  static std::unique_ptr<NimbleIndexProjector> create(
      std::shared_ptr<TabletReader> tablet,
      const velox::FileHandle& fileHandle,
      std::shared_ptr<const NimbleTypeProjection> projection,
      BlockCache* blockCache,
      const velox::dwio::common::ReaderOptions& options);

  /// Options for controlling projection behavior.
  ///
  /// The three limits below (maxRows, maxBytes, maxRowsPerRequest) only bound
  /// how much data a single projection returns; on their own they never
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
    /// 1.0 disables slicing. Hybrid FlatMap projections always retain the
    /// full stripe because their streams use a key-major layout.
    double maxOverfetchRowsRatio{1.0};
    /// When set, every request whose results were cut short by a limit
    /// (maxRows, maxBytes, or maxRowsPerRequest) is given a resume key so the
    /// caller can continue from where the request stopped (use the resume key
    /// as the new lowerKey with the original upperKey). When unset, no resume
    /// keys are produced even if a limit truncated the results.
    bool needResumeKey{false};
    /// Caps the summed StripeGroupDescriptor::metadataSize of the stripe groups
    /// that one metadata load returns while planning which stripes to read. A
    /// load always returns the group of the stripe being planned, even when
    /// that group alone exceeds the cap. It then adds the next groups that hold
    /// stripes the key lookups found. It stops before the first group that
    /// would take the total past the cap, so 0 loads only the stripe's own
    /// group. Cached groups count toward the cap because planning holds them.
    /// Planning keeps the groups of a load until it reaches a stripe outside
    /// them, so a projection loads each group it uses at most once. A load of
    /// several groups coalesces nearby sections and can read the rest in
    /// parallel when the tablet reader has an IO executor. Its extra groups can
    /// include ones that a row or byte limit keeps planning from reaching. A
    /// read error in any of them fails the projection. Pinned metadata keeps
    /// them in memory for the reader's lifetime.
    ///
    /// A co-load (see create()) loads stripe groups the same way while it
    /// picks its stripes before the key lookups, and keeps the stream
    /// locations of the stripes it reads. Planning then takes the stream
    /// locations from the co-load and loads no group.
    uint64_t maxStripeGroupMetadataBytesPerLoad{0};
    /// Makes co_projectRows() build the lookup structures of each encoding
    /// view when it binds a stripe's streams, which pays off when a projection
    /// reads many rows of each stripe. When unset, each read derives only what
    /// it needs, which suits point lookups. Both settings return the same rows.
    /// Changing it between calls rebuilds the projector's StreamReader. See
    /// Encoding::Options::cacheEncodingView.
    bool cacheEncodingView{false};
  };

  /// Serialized output for a single request.
  struct SerializedResponse {
    /// One self-describing kTablet IOBuf chain per (request × stripe)
    /// intersection. Each entry covers a contiguous row range within one
    /// stripe; the row range is embedded in the chain's header. Empty when
    /// the request returns no rows and has no resume key. Slices for
    /// overlapping requests share the body+trailer bytes via refcounted
    /// SharedInfo on the second IOBuf node.
    ///
    /// If the response is truncated and has a resume key, it is embedded in
    /// the header of the last slice. Use `rocks::readResultResumeKey()` on
    /// the last slice to extract it. A request cut short before any of its
    /// rows were read holds one header-only slice with 0 rows and the key.
    std::vector<folly::IOBuf> slices;

    /// When Options::needResumeKey is set and this request's results were cut
    /// short by a limit (maxRows, maxBytes, or maxRowsPerRequest), the encoded
    /// resume key for continuation. The caller constructs new key bounds using
    /// this as lowerKey with their original upperKey. nullopt if complete, a
    /// miss, or needResumeKey was not set. A co-load (see create()) does not
    /// look up a request that the limits keep the projection from reaching, so
    /// it returns the lowerKey of such a request even when the request is a
    /// miss.
    ///
    /// Also embedded in the last slice's per-slice header so consumers that
    /// hold an IOBuf can recover the key without keeping the
    /// SerializedResponse struct around.
    std::optional<std::string> resumeKey;
  };

  /// Result of a projectStreams() call.
  struct SerializedResult {
    /// One entry per request, in request order.
    std::vector<SerializedResponse> responses;
  };

  /// Locates one request's rows in RowProjectionResult::values.
  struct RowProjectionResponse {
    /// First row for this request in the shared output vector.
    velox::vector_size_t offset{0};
    /// Number of rows returned for this request.
    velox::vector_size_t size{0};
    /// Encoded continuation key when the request was truncated.
    std::optional<std::string> resumeKey;
  };

  /// Holds request-major rows decoded directly into a Velox vector.
  struct RowProjectionResult {
    /// Flat projected rows from every request, grouped in request order.
    velox::RowVectorPtr values;
    /// One range into `values` per request.
    std::vector<RowProjectionResponse> responses;
  };

  /// Projects the requested columns for the given batch of index lookups.
  /// Returns one SerializedResponse per lookup position, in order. Processes
  /// all relevant stripes internally. 'request' must be a range scan.
  SerializedResult projectStreams(
      const index::IndexLookup::LookupRequest& request,
      const Options& options);

  /// Projects rows directly into a Velox vector. Flat scalar projections use
  /// random-access EncodingView reads when supported. Nested projections use
  /// the general FieldReader path without serializing an intermediate kTablet
  /// payload. 'request' must be a range scan. The request and options must
  /// remain alive until the returned task completes.
  folly::coro::Task<RowProjectionResult> co_projectRows(
      const index::IndexLookup::LookupRequest& request,
      const Options& options);

  /// Resolves a file-level row range into stripe selections.
  /// Returns an empty vector for an empty range.
  std::vector<StripeSelection> lookup(const RowRange& rowRange);

  /// Resolves key bounds into stripe selections, in stripe order.
  /// Returns an empty vector for a miss. The caller owns scan progress.
  std::vector<StripeSelection> lookup(
      const velox::serializer::EncodedKeyBounds& bounds);

  /// Resolves key bounds as lookup(bounds) does, and sets 'keyCursor' to a
  /// cursor over the keys of the selected rows, or to null on a miss, as
  /// ClusterIndexBase::seek() returns them. Decodes the keys of the stripe of
  /// 'keyChunk' from it, in place, instead of reading its key chunk, and reads
  /// the key chunks of other stripes through the cluster index: a caller that
  /// loaded the stripe of the lower bound with its key stream, or holds it,
  /// passes that key chunk.
  std::vector<StripeSelection> lookup(
      const velox::serializer::EncodedKeyBounds& bounds,
      const std::optional<index::ClusterIndexBase::KeyChunk>& keyChunk,
      std::unique_ptr<index::KeyCursor>& keyCursor);

  /// Returns the stripes whose rows a lookup of 'bounds' may select, from the
  /// stripe of the lower bound to the stripe of the upper bound, as
  /// ClusterIndexBase::lookupStripes() returns them. Reads index metadata but
  /// no key chunk. Only a Stripe layout cluster index supports this.
  nimble::StripeRange lookupStripes(
      const velox::serializer::EncodedKeyBounds& bounds);

  /// Loads projected streams for contiguous full stripes in one I/O batch.
  /// Returns one entry per stripe, in stripe order, including stripes whose
  /// projected streams are all absent. The caller applies row selections.
  /// Requires a non-zero stripe count. When the projector co-loads key streams
  /// (see create()), the batch also reads the key stream of each stripe, which
  /// ProjectedStripe::keyChunk then holds.
  std::vector<facebook::nimble::ProjectedStripe> loadStripeStreams(
      uint32_t stripeOffset,
      uint32_t numStripes);

  /// Returns whether the projector co-loads key streams (see create()):
  /// ReaderOptions::coLoadKeyStreams() is set, and the cluster index stores key
  /// chunks in stripes.
  bool coLoadKeyStreams() const {
    return coLoadKeyStreams_;
  }

  /// Returns the projected nimble schema. Preserves encoding-specific types
  /// (ArrayWithOffsets, SlidingWindowMap, FlatMap) from the file schema.
  /// Clients need this to build a Deserializer for the output data.
  const std::shared_ptr<const Type>& projectedNimbleType() const {
    return projection_->nimbleType;
  }

  /// Returns the cluster index used to resolve projected key ranges.
  const ClusterIndexBase& clusterIndex() const {
    return *velox::checkedNotNull(clusterIndex_);
  }

  /// Returns the number of rows in the source tablet.
  uint64_t numRows() const {
    return tablet_->tabletRowCount();
  }

  /// Returns the file properties used to configure decoding.
  const FileProperties& properties() const {
    return tablet_->properties();
  }

  /// Statistics captured during projection.
  struct Stats {
    /// Number of stripes that the projection reads rows from.
    uint32_t numReadStripes{0};

    // TODO: Skip the co-loaded stripes of keys that a bloom filter rules out
    // of the file.
    /// Number of stripes that the projection loads: the stripes of a co-load,
    /// otherwise the stripes that it reads rows from. numLoadedStripes minus
    /// numReadStripes counts the stripes that a co-load reads and no row comes
    /// from, such as the stripe of a missing key, or a stripe past where the
    /// limits or a request's Options::maxRowsPerRequest stop the projection.
    uint32_t numLoadedStripes{0};
    /// Number of read stripes serialized through stream slicing.
    uint32_t numSlicedStripes{0};
    /// Total rows read from storage (entire stripe row counts). Includes
    /// rows outside the requested row ranges that are read because we
    /// fetch entire projected streams per stripe.
    uint64_t numReadRows{0};
    /// Total rows in the output result (only the requested row ranges).
    /// The difference numReadRows - numProjectedRows is over-fetched rows.
    uint64_t numProjectedRows{0};
    /// Total serialized output bytes, projected stream bytes, or retained
    /// vector bytes.
    uint64_t numOutputBytes{0};
    /// Projected stream bytes charged against Options::maxBytes: the running
    /// total that decides where a byte-limited projection stops.
    uint64_t numPlannedBytes{0};
    /// Number of projections that stopped with stripes left to read because
    /// numPlannedBytes reached Options::maxBytes.
    uint32_t numMaxBytesTruncations{0};
    /// Number of loads that read cluster index key streams in the same load as
    /// stripe streams.
    uint32_t numKeyStreamCoLoads{0};

    /// Time spent looking up stripes and row ranges via the tablet index.
    velox::CpuWallTiming lookupTiming;
    /// Time spent building per-stripe stream and request-range plans.
    velox::CpuWallTiming prepareTiming;
    /// Time spent loading stripe stream data from tablet.
    velox::CpuWallTiming scanTiming;
    /// Time spent serializing projected streams into kTablet format.
    velox::CpuWallTiming projectionTiming;

    /// Returns a readable summary of counters and timings.
    std::string toString() const;
  };

  /// Returns approximate cumulative statistics. TSan builds synchronize
  /// snapshots with counter and timing updates.
  Stats stats() const;

  /// Enables or disables the CPU and wall clock reads that fill the timings of
  /// stats(). Reading the thread CPU clock is a system call, so a point lookup
  /// spends about 3 us on these reads. Disabled timings keep their current
  /// values. Enabled by default.
  void setTimingEnabled(bool enabled);

  /// Turns the verification of stream checksums on or off for the next
  /// requests, as ReaderOptions::verifyStreamChecksums() does at creation. A
  /// file whose stream trailers carry no checksum stays unverified. Must not
  /// be called while a request loads stripes.
  void setVerifyStreamChecksums(bool verifyStreamChecksums);

 private:
  // Applies a synchronous counter or timing update under the TSan guard.
  template <typename Update>
  void updateStats(Update&& update);

  // Accumulates one timing sample for the selected statistic.
  void recordStatsTiming(
      velox::CpuWallTiming Stats::* field,
      const velox::CpuWallTiming& timing);

  NimbleIndexProjector(
      std::shared_ptr<TabletReader> tablet,
      std::shared_ptr<velox::ReadFile> file,
      std::unique_ptr<DataInput> dataInput,
      std::shared_ptr<const NimbleTypeProjection> projection,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<velox::io::IoStatistics> ioStats,
      bool verifyStreamChecksums,
      bool coLoadKeyStreams);

  // A request index paired with its stripe-relative row range.
  struct StripeRange {
    uint32_t requestIndex{};
    // Stripe-relative row range, already intersected with stripe boundaries.
    RowRange rowRange;
  };

  // CSR (compressed sparse row) layout mapping stripes to request row ranges.
  // All StripeRange entries are stored in a single flat vector (`ranges`),
  // with `offsets[i]` marking where stripe i's entries begin.
  struct StripeRanges {
    // Ascending tablet stripe indices that carry ranges. Everything else here
    // is indexed by POSITION IN THIS VECTOR -- a "resolved stripe index" --
    // not by tablet stripe index. Each request covers one contiguous stripe
    // run, but the requests together only pin down an outer envelope, and for
    // scattered lookups that envelope is far wider than the set of stripes
    // that actually carry ranges -- 148k stripes between the first and last
    // against ~250 that hold anything, for 128 random probes. Indexing by
    // tablet stripe index would make the tables here scale with the envelope
    // rather than with the work.
    std::vector<uint32_t> resolvedStripes;
    // Flat storage of all per-stripe row ranges, grouped by resolved stripe
    // index.
    std::vector<StripeRange> ranges;
    // offsets[i] = start index in ranges for resolvedStripes[i].
    // Size = resolvedStripes.size() + 1.
    std::vector<uint32_t> offsets;

    /// @param resolvedStripeIndex Index into resolvedStripes, NOT a tablet
    /// stripe index.
    std::span<const StripeRange> getRanges(uint32_t resolvedStripeIndex) const {
      NIMBLE_CHECK_LT(resolvedStripeIndex, resolvedStripes.size());
      return {
          ranges.data() + offsets[resolvedStripeIndex],
          offsets[resolvedStripeIndex + 1] - offsets[resolvedStripeIndex]};
    }

    void clear() {
      resolvedStripes.clear();
      ranges.clear();
      offsets.clear();
    }
  };

  // Initializes per-request state: stores request/options pointers and
  // resets running totals.
  void initRequest(
      const index::IndexLookup::LookupRequest& request,
      const Options& options);

  // Clears all per-request state set by initRequest(). Invoked on scope
  // exit so a subsequent projection starts from a clean slate.
  void clearRequest();

  // Looks up all requests and maps them to stripes. Populates
  // ctx_.stripeRanges. Looks them up via the cluster index without a co-load,
  // which reads the key chunks that it needs from the file, and at their
  // boundary stripes with lookupCoLoadedRows() after one.
  void lookupRows();

  // Looks up the rows of each request after a co-load with
  // coLoadedRequestRows(), and returns them as the cluster index returns a
  // lookup.
  index::IndexLookup::LookupResult lookupCoLoadedRows() const;

  // Returns the rows of request 'requestIndex' after a co-load, which can be
  // none. The first stripe of a request in ctx_.coLoadRequestStripes holds
  // its lower key, its last stripe its upper key, and each stripe between them
  // only rows of the request. So it seeks the lower key in the key chunk of the
  // first stripe and the upper key in that of the last, and takes the stripes
  // between in full. Seeks a key only in a stripe that the co-load loaded: a
  // request whose first stripe the co-load did not load selects no row, and
  // the rows of one whose last stripe it did not load end at the first row of
  // that stripe.
  RowRange coLoadedRequestRows(uint32_t requestIndex) const;

  // Returns the key chunk of 'stripe' that the co-load loaded, or null when it
  // did not load the stripe.
  const ClusterIndexBase::KeyChunk* coLoadedKeyChunk(uint32_t stripe) const;

  // Loads, before lookupRows(), the stripes that the lookup of the requests can
  // select rows of, each with its key stream, in one load, and keeps their
  // key chunks in ctx_.coLoadedKeyChunks for lookupRows() and the resume keys.
  // Finds the stripes of each request from index metadata, which it keeps in
  // ctx_.coLoadRequestStripes for lookupRows(), and trims them at stripe level
  // with selectStripesToCoLoad(). prepareStripes() trims the stripes again at
  // row level after the lookup. Records in ctx_.unloadedRequests the requests
  // whose first stripes it did not load, of which the lookup selects no row,
  // as planning cannot reach them. Of a request whose last stripe it did not
  // load, the lookup can select rows past the loaded stripes, which planning
  // stops before.
  void coLoadStripes();

  // Picks the stripes that coLoadStripes() loads, in file order, from
  // metadata alone, and records them in ctx_.coLoadedStripes with the
  // locations of their streams in ctx_.coLoadedStreams. Gets the stripe groups
  // that locate the streams from a StripeGroupLoader of its own, which holds
  // the groups of one load at a time. 'requestStripes' holds the stripes of
  // each request, as ClusterIndexBase::lookupStripes() returns them. Without
  // limits, picks all of them. With limits, picks the stripes up to where the
  // limits of the options can stop prepareStripes(), then the stripe right
  // after there, for its key stream, and stops. Under maxRowsPerRequest, a
  // request stops asking for stripes once it may have reached the cap, and
  // asks for the stripe right after, for its key stream. A request selects
  // every row of each of its stripes but the first and the last, and maybe
  // none of those two, so the stripes reach at least as far as
  // prepareStripes() plans after the lookup, and their key streams hold every
  // resume key. Also picks the stripes up to there that hold no projected
  // stream, so that prepareStripes() drops them without stripe-group metadata.
  void selectStripesToCoLoad(
      std::span<const nimble::StripeRange> requestStripes);

  // Records 'stripe' for the co-load to read, with 'streams', the locations of
  // its projected streams and then of its key stream.
  void addCoLoadedStripe(
      uint32_t stripe,
      std::span<const StripeGroup::StreamMetadata> streams);

  // Loads the streams that ctx_.coLoadedStreams locates for each of
  // ctx_.coLoadedStripes in one load, verifying their checksums, and returns
  // the key chunk of each. Records the enqueue indices of their streams in
  // ctx_.dataInputIndices.
  std::vector<ClusterIndexBase::KeyChunk> loadCoLoadedStreams();

  // Returns the key chunk of each of ctx_.coLoadedStripes from the key streams
  // that loadCoLoadedStreams() loaded, without their stream trailers. Each
  // holds the handle of the load, ctx_.dataHandle.
  std::vector<ClusterIndexBase::KeyChunk> coLoadedKeyChunks() const;

  // Adds each of 'stripes', which ascend, to the co-load: records it in
  // ctx_.coLoadedStripes for loadCoLoadedStreams() to read with its projected
  // streams and its key stream, as selectStripesToCoLoad() does for the
  // stripes that it picks for a projection. 'stripeIdentifiers' identifies
  // each of 'stripes'.
  void addStripesToCoLoad(
      std::span<const uint32_t> stripes,
      std::span<const StripeIdentifier> stripeIdentifiers);

  // Points ctx_.dataInputIndices of each planned stripe at the streams that
  // loadCoLoadedStreams() loaded for it, which must have loaded every planned
  // stripe, in place of loadStripeStreams(). Drops the key streams and the
  // streams of the loaded stripes that prepareStripes() trimmed, then clears
  // what the co-load recorded.
  void assignCoLoadedStreams();

  // Output of prepareStripes(). Per-stripe vectors are indexed by stripe
  // offset in the compact plan and only contain stripes retained for
  // processing.
  struct ScanPlan {
    // Absolute stripe indices in processing order.
    std::vector<uint32_t> stripeIndices;
    // Total rows in each planned stripe.
    std::vector<uint32_t> numRows;
    // Whether each planned stripe must be decoded independently.
    std::vector<bool> requiredBarriers;
    // Number of projected streams present in each planned stripe.
    std::vector<uint32_t> numStreams;
    // Total logical bytes across all projected streams in each planned stripe.
    std::vector<uint64_t> projectedBytes;
    // File offsets for stripe stream payloads.
    std::vector<uint64_t> stripeFileOffsets;
    // Physical bytes spanning all streams in each stripe.
    std::vector<uint64_t> stripeSizes;
    // Flat reusable storage for all per-stripe projected stream slots. Each
    // stripe occupies projection_->streamOffsets.size() consecutive entries.
    std::vector<StripeGroup::StreamMetadata> projectedStreams;
    // Start offsets into stripeRanges for each planned stripe. The last entry
    // is the total range count.
    std::vector<size_t> stripeRangeOffsets;
    // Flat reusable storage for all request ranges retained by planned stripes.
    std::vector<StripeRange> stripeRanges;
    bool truncated{false};
  };

  // Returns this stripe's projected stream slots from the flat ScanPlan
  // storage.
  std::span<const StripeGroup::StreamMetadata> stripeProjectedStreams(
      size_t stripeOffset) const;

  // Returns this stripe's request ranges from the flat ScanPlan storage.
  std::span<const StripeRange> plannedStripeRanges(size_t stripeOffset) const;

  // Projected stream totals for one stripe, computed by locateStripeStreams()
  // before the stripe is known to be worth keeping.
  struct StripeStreams {
    uint32_t numStreams{0};
    uint64_t projectedBytes{0};
    bool requiredBarrier{false};
  };

  // Locates this stripe's projected streams from its loaded metadata, staging
  // them at the tail of ScanPlan::projectedStreams. The stripe is not part of
  // the plan until appendStripePlan() commits it; a stripe dropped in between
  // leaves the staged slots to be overwritten by the next stripe, and
  // prepareStripes() trims any left over.
  StripeStreams locateStripeStreams(const StripeIdentifier& stripe);

  // Stages the projected streams of 'stripe' like locateStripeStreams(), from
  // the locations that the co-load recorded instead of stripe-group metadata.
  // Returns no stream for a stripe that the co-load found holds none. Throws
  // for a stripe that the co-load did not locate, or of which it read only the
  // key stream.
  StripeStreams locateCoLoadedStreams(uint32_t stripe);

  // Converts 'projectedStreams', the locations of the projected streams of a
  // stripe, into their StripeStreams: how many hold bytes, how many bytes, and
  // whether decoding needs a barrier.
  StripeStreams toStripeStreams(
      std::span<const StripeGroup::StreamMetadata> projectedStreams) const;

  // Appends one stripe to the plan using streams already staged for it by
  // locateStripeStreams().
  void appendStripePlan(
      uint32_t stripeIndex,
      size_t rangeOffset,
      const StripeStreams& streams);

  // Computes the stripe-relative body range based on request row ranges and
  // Options::maxOverfetchRowsRatio. Hybrid FlatMap projections always retain
  // the full stripe because their physical streams are key-major.
  RowRange stripeRowRangeToPack(size_t stripeOffset) const;

  // Records the resume key for a request that reached its maxRowsPerRequest cap
  // in the stripe at `resolvedStripeIndex` (an index into
  // stripeRanges.resolvedStripes, not a tablet stripe index). `readEndRow` is
  // the request's stripe-relative end row after clipping. When `partialRead` is
  // true the cap fell inside the stripe, so the key is the row-precise key at
  // `readEndRow`; otherwise the cap landed on the stripe boundary and the key
  // resumes from the next stripe still holding this request. The stripe's
  // absolute start row is derived from `resolvedStripeIndex`.
  // Idempotent (no-op once a key is recorded); callers gate on
  // Options::needResumeKey.
  void setResumeKey(
      uint32_t requestIndex,
      uint32_t resolvedStripeIndex,
      uint32_t readEndRow,
      bool partialRead);

  // Returns the encoded key at row 'rowInStripe' of tablet stripe 'stripe'.
  // Reads it from the key chunk of the stripe that the co-load loaded, which
  // must hold the stripe, after a co-load, and via the cluster index, which
  // reads it from the file, without one.
  std::string keyAtStripeRow(uint32_t stripe, uint32_t rowInStripe) const;

  // Applies row and byte limits to the looked-up stripe ranges, computes stream
  // metadata for selected stripes, and populates ctx_.plan. After a co-load,
  // copies the stream locations that the co-load recorded instead of loading
  // stripe-group metadata.
  void prepareStripes();

  // Enqueues all projected streams from ctx_.plan into DataInput and issues a
  // single coalesced load() call.
  void loadStripeStreams();

  // Starts a load group for the stripe at 'stripeOffset' in the file, which
  // spans 'stripeSize' bytes when DataInput is cached, and enqueues each of
  // 'streams' that holds bytes, recording its enqueue index at the same
  // position in 'dataInputIndices'.
  void enqueueStripeStreams(
      uint64_t stripeOffset,
      uint64_t stripeSize,
      std::span<const StripeGroup::StreamMetadata> streams,
      std::span<std::optional<uint32_t>> dataInputIndices);

  // Loads the enqueued streams in one load into ctx_.dataHandle, verifying
  // their checksums when stream checksums are on.
  void loadEnqueuedStreams();

  // Builds serialized responses from the loaded stripe streams.
  SerializedResult processStripeStreams();

  // Checks one loaded stream against the checksum in its stream trailer, which
  // `data` ends with, throwing when they differ. `enqueueIndex`, the index
  // DataInput returned from enqueue(), only locates the stream for the error.
  void verifyStreamChecksum(uint32_t enqueueIndex, std::string_view data) const;

  // Describes the stream that the current load enqueued at `enqueueIndex`, for
  // a checksum error.
  std::string loadedStreamDescription(uint32_t enqueueIndex) const;

  // Validates the projected row schema.
  void validateProjection() const;

  // Returns the stream slicer, which the first call creates.
  serde::StreamSlicer& streamSlicer();

  // Builds streamChecksum_ and the stream trailer members when
  // `verifyStreamChecksums` is set and the file's stream trailers carry a
  // checksum.
  void initStreamChecksum(bool verifyStreamChecksums);

  // Resolves the request's stripes and loads their projected streams. Both key
  // stream layouts look up the rows of the requests and plan their stripes the
  // same way. A co-load of key streams loads the stripes before the lookup
  // instead of after the plan.
  void loadStripes(
      const index::IndexLookup::LookupRequest& request,
      const Options& options);

  // Returns decoding options backed by this projector's scratch pool.
  Encoding::Options encodingOptions(bool cacheEncodingView);

  // Creates the schema-derived reader on first use by co_projectRows().
  void ensureStreamReader();

  // Computes per-stripe pack ranges and returns the upper-bound byte estimate
  // used to allocate the sliced-output arena. Returns 0 when no stripe will be
  // sliced.
  uint64_t prepareStripePackRanges();

  // Estimates sliced stream bytes for a partial stripe pack.
  uint64_t estimateSlicedStripeBytes(
      size_t stripeOffset,
      const RowRange& packRange) const;

  // Returns the per-request sliced-output arena.
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
    bool requiredBarrier{false};
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

  // After global truncation (maxRows or maxBytes), resolves resume keys for the
  // requests it cut short and stores them in ctx_.resumeKeys: a request
  // continuing past the last planned stripe resumes at the next stripe, and a
  // request planning never reached resumes from its lower key, as does each of
  // ctx_.unloadedRequests. No-op when Options::needResumeKey is unset or the
  // plan was not truncated. First checks that planning stopped before
  // ctx_.unloadedRequests, as planning can read no rows of those.
  void prepareResumeKeys();

  // Iterates planned stripes and ctx_.packedStripes to build per-request
  // output slices. Each slice clones the shared stripe body and prepends a
  // per-request header with the row range and (on the last slice of a
  // truncated response) the resume key. A request that planning never reached
  // has no data slice, so its resume key gets a header-only slice.
  void buildResult(SerializedResult& result);

  inline uint32_t stripeRowCount(uint32_t stripe) const {
    return static_cast<uint32_t>(tablet_->stripeRowCount(stripe));
  }

  // Computes the stripe-relative row range by intersecting the file-level
  // rowRangeLimit with the stripe boundaries.
  RowRange stripeRowRange(uint32_t stripe, const RowRange& rowRangeLimit) const;

  // Keeps the source file alive throughout projected stripe loads.
  const std::shared_ptr<velox::ReadFile> file_;
  const std::shared_ptr<TabletReader> tablet_;
  const std::shared_ptr<velox::io::IoStatistics> ioStats_;
  // Owned by the caller throughout reads and asynchronous cleanup.
  velox::memory::MemoryPool* const pool_;
  const ClusterIndexBase* const clusterIndex_;
  const uint32_t numStripes_{0};
  // Whether the projector co-loads key streams: the creator asked for it, and
  // the cluster index stores its key chunks in stripes, as key streams that a
  // load can read with its stripe streams.
  const bool coLoadKeyStreams_{false};

  const std::shared_ptr<const NimbleTypeProjection> projection_;
  // True when every emitted slice must preserve a native HFM batch boundary,
  // even if the selected group has no physical stream in a stripe.
  const bool hasProjectedHybridFlatMaps_{false};
  // The streams that a co-load locates in each stripe: the projected streams,
  // then the key stream of the cluster index, which holds the key chunk of the
  // stripe. Empty unless the cluster index keeps its key chunks in stripes.
  const std::vector<uint32_t> coLoadStreamIds_;

  std::unique_ptr<DataInput> dataInput_;
  // Verifies a stream read from storage against the checksum in its stream
  // trailer. Built only when ReaderOptions::verifyStreamChecksums() or
  // setVerifyStreamChecksums() turns verification on and the file's stream
  // trailers carry a checksum, so null means this projector
  // does not verify. Stateful, so it relies on this class being
  // single-threaded.
  std::unique_ptr<Checksum> streamChecksum_;
  // When verifying: the stream trailer's size, read with every stream, and the
  // checksum's offset within it. Zero otherwise, so unverified reads load
  // exactly the stream's bytes.
  uint32_t streamTrailerSize_{0};
  uint32_t streamTrailerChecksumOffset_{0};
  // Whether the stats timings read the CPU and wall clocks.
  bool timingEnabled_{true};
  // Scratch allocations shared by successive EncodingView decodes. The
  // projector is single-threaded, matching BufferPool's ownership contract.
  velox::BufferPool encodingBufferPool_{velox::BufferPool::kDefaultCapacity};

  // Decodes projected rows into Velox vectors with the encoding options of each
  // request. The serialized path does not need this reader, so it is created
  // on the first co_projectRows() call.
  std::unique_ptr<StreamReader> streamReader_;

  // Reused across stripes; its raw input format is fixed by the tablet. Only
  // the serialized path slices streams, so the slicer is created on the first
  // slice and a projector that decodes rows never builds it.
  std::unique_ptr<serde::StreamSlicer> streamSlicer_;

  // One range tagged with the tablet stripe it falls in, before grouping.
  // Sorting a vector of these on (tabletStripeIndex, range.requestIndex) is
  // what produces the CSR grouping; the second key keeps each stripe's ranges
  // in request order, which the grouped layout preserves.
  struct ResolvedStripe {
    uint32_t tabletStripeIndex{};
    StripeRange range;
  };

  // Per-request state. Set by initRequest(), populated through the shared
  // pipeline (lookupRows → prepareStripes → loadStripeStreams), consumed by
  // whichever terminal step the caller asked for, and reset on return.
  struct ProjectionContext {
    const index::IndexLookup::LookupRequest* request{nullptr};
    const Options* options{nullptr};
    uint32_t numRequests{0};
    // CSR mapping from stripes to request row ranges. Populated by
    // lookupRows(), read-only during stripe processing.
    StripeRanges stripeRanges;
    // Populated by prepareStripes().
    ScanPlan plan;
    // Per-request resume keys, set by setResumeKey() when a request reaches
    // maxRowsPerRequest and by prepareResumeKeys() after global truncation.
    // All nullopt when Options::needResumeKey is unset.
    std::vector<std::optional<std::string>> resumeKeys;
    // Flat array of enqueue indices, logically
    // [stripeOffset * numProjectedStreams + streamIndex]. nullopt for absent
    // streams. Populated by loadStripeStreams(). loadCoLoadedStreams()
    // populates it by stripe positions in coLoadedStripes instead, in the
    // layout of coLoadedStreams, with the key stream of each stripe after its
    // projected streams. assignCoLoadedStreams() then turns it into the offsets
    // of the planned stripes and drops the key streams.
    std::vector<std::optional<uint32_t>> dataInputIndices;
    // Stripes whose projected streams coLoadStripes() loaded before planning
    // them, ascending. assignCoLoadedStreams() clears it once it has pointed
    // dataInputIndices at the streams of the planned stripes.
    std::vector<uint32_t> coLoadedStripes;
    // Locations of the projected streams and then of the key stream of each
    // of coLoadedStripes, numProjectedStreams + 1 per stripe, which
    // selectStripesToCoLoad() records from stripe-group metadata.
    // loadCoLoadedStreams() loads the streams from them, and prepareStripes()
    // plans the stripes from them, so neither needs stripe-group metadata.
    // assignCoLoadedStreams() clears it.
    std::vector<StripeGroup::StreamMetadata> coLoadedStreams;
    // Key chunks of coLoadedStripes, which point into the loaded data. Kept
    // past the lookup for the resume keys. Empty without a co-load, so that
    // the lookup then reads the key chunks that it needs from the file, and
    // keyAtStripeRow() reads keys from it.
    std::vector<ClusterIndexBase::KeyChunk> coLoadedKeyChunks;
    // Stripes of each request that ClusterIndexBase::lookupStripes() returns,
    // which coLoadStripes() records for lookupCoLoadedRows().
    std::vector<nimble::StripeRange> coLoadRequestStripes;
    // Requests whose first stripes a co-load did not load, ascending, as they
    // are past where the limits stop planning, so the lookup selects no row of
    // them. prepareResumeKeys() gives each its lower key back.
    std::vector<uint32_t> unloadedRequests;
    // Handle keeping loaded data alive for zero-copy BufferRefs.
    DataInput::Handle dataHandle;
    // Serialized stripe bodies and metadata, one per planned stripe. Populated
    // by processStripeStreams(), consumed during buildResult().
    std::vector<PackedStripe> packedStripes;
    // Per planned stripe row range selected for packing. Full-stripe ranges use
    // zero-copy packing; narrower ranges use StreamSlicer.
    std::vector<RowRange> stripePackRanges;
    // Reusable per-request vectors.
    std::vector<uint64_t> rowsPerRequest;
    // Stripes resolved from the cluster index, collected before grouping into
    // StripeRanges and reused across projection calls to keep the build
    // alloc-free. Holds one entry per (request, stripe) pair, so a stripe
    // repeats once per request that lands in it.
    std::vector<ResolvedStripe> resolvedStripesScratch;
    std::vector<size_t> sliceCounts;
    std::vector<size_t> emittedSlices;
    // Shared arena for all sliced stream bytes produced by one projection.
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

  // Guards stats updates and snapshots in TSan builds.
  mutable std::mutex statsLock_{};
  // Updated by the thread performing projection work.
  Stats stats_;
};

} // namespace facebook::nimble

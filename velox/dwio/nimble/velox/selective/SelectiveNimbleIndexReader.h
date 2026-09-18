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

#include <folly/container/F14Map.h>
#include <memory>
#include <vector>

#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble {

/// SelectiveNimbleIndexReader implements the IndexReader interface for
/// Nimble files with cluster indexes. It handles:
/// - Encoding index bounds into Nimble-specific encoded keys
/// - Looking up stripes and row ranges using the tablet index
/// - Managing stripe iteration and data reading
/// - Returning results in request order via the next() iterator pattern
///
/// This class moves the control logic (stripe iteration, row range computation,
/// key encoding) from HiveIndexReader into the format-specific reader, allowing
/// HiveIndexReader to focus on index bounds creation and result assembly.
class SelectiveNimbleIndexReader : public velox::dwio::common::IndexReader {
 public:
  using Result = velox::connector::IndexSource::Result;

  SelectiveNimbleIndexReader(
      std::shared_ptr<ReaderBase> readerBase,
      const velox::dwio::common::RowReaderOptions& options);

  ~SelectiveNimbleIndexReader() override = default;

  /// Starts a new batch lookup with the given index bounds.
  /// Encodes the bounds, looks up matching stripes, and prepares for iteration.
  void startLookup(
      const velox::serializer::IndexBounds& indexBounds,
      const Options& options) override;

  /// Returns true if there are more results to fetch from the current lookup.
  bool hasNext() const override;

  /// Returns the next batch of results from the current lookup.
  /// Results are returned in request order - all results for request N are
  /// returned before any results for request N+1.
  std::unique_ptr<Result> next(velox::vector_size_t maxOutputRows) override;

  /// Returns accumulated runtime stats for this index reader.
  folly::F14FastMap<std::string, velox::RuntimeMetric> stats() const override;

 private:
  using RowRangeMap = folly::F14FastMap<RowRange, size_t, RowRangeHash>;

  // Represents a read segment - a contiguous row range to read from the file.
  // Multiple requests may reference the same segment (when they share row
  // ranges).
  struct ReadSegment {
    // Stripe-relative row range to read. Reset per stripe in
    // buildStripeReadSegments().
    RowRange rowRange;
    // Request indices that reference this segment.
    std::vector<velox::vector_size_t> requestIndices;

    std::string toString() const;
  };

  // Per-request state for tracking output assembly.
  struct RequestState {
    // Number of stripes still pending for this request.
    int32_t pendingStripes{0};

    // File-level row range from the batch lookup. Set once in
    // resolveRequestStripes().
    RowRange rowRange;

    // Stripe-relative row range for output assembly. Set per stripe in
    // prepareStripeReading(). May be truncated by maxRowsPerRequest.
    RowRange stripeRowRange;

    velox::vector_size_t outputRows{0};

    struct OutputReference {
      size_t chunkIndex{};
      RowRange rowRange;
    };
    std::vector<OutputReference> outputRefs;

    // Returns true if no pending stripes and no output rows accumulated.
    bool empty() const {
      VELOX_CHECK_EQ(outputRows == 0, outputRefs.empty());
      return pendingStripes == 0 && outputRows == 0;
    }

    // Increments the pending stripe count when a new stripe is mapped to this
    // request.
    void incPendingStripes();

    // Decrements the pending stripe count when a stripe finishes processing
    // for this request (either with data or empty).
    void decPendingStripes();
  };

  // Output chunk with reference counting.
  struct OutputChunk {
    velox::RowVectorPtr data;
    int32_t refCount{0};

    explicit OutputChunk(velox::RowVectorPtr _data) : data(std::move(_data)) {}

    void addRef() {
      ++refCount;
    }
    void dropRef();
  };

  uint32_t stripeRowOffset(uint32_t stripe) const {
    return static_cast<uint32_t>(readerBase_->tablet().stripeStartRow(stripe));
  }

  uint32_t stripeRowCount(uint32_t stripe) const {
    return static_cast<uint32_t>(readerBase_->tablet().stripeRowCount(stripe));
  }

  // Computes the stripe-relative row range by intersecting the file-level
  // rowRangeLimit with the stripe boundaries.
  RowRange stripeRowRange(uint32_t stripe, const RowRange& rowRangeLimit)
      const {
    const auto stripeStart = stripeRowOffset(stripe);
    const auto stripeEnd = stripeStart + stripeRowCount(stripe);
    const auto startRow = std::max(rowRangeLimit.startRow, stripeStart);
    const auto endRow = std::min(rowRangeLimit.endRow, stripeEnd);
    if (startRow >= endRow) {
      return RowRange{};
    }
    return RowRange(startRow - stripeStart, endRow - stripeStart);
  }

  // Encodes raw index bounds into Nimble-specific encoded key format.
  // @param indexBounds The index bounds to encode.
  // @return Vector of encoded key bounds ready for index lookup.
  std::vector<velox::serializer::EncodedKeyBounds> encodeIndexBounds(
      const velox::serializer::IndexBounds& indexBounds);

  // Maps each request to the stripes it needs to read from by performing
  // index lookups for all encoded key bounds.
  void resolveRequestStripes();

  // Loads the current stripe if not already loaded.
  // @return True if a stripe was loaded, false if no more stripes.
  bool loadStripe();

  // Prepares a stripe for reading by initializing streams and column readers.
  // @param stripeIndex The index of the stripe to prepare.
  void prepareStripeReading(uint32_t stripeIndex);

  // Builds read segments from request ranges based on filter presence.
  // Dispatches to mergeStripeRowRanges or splitStripeRowRanges.
  void buildStripeReadSegments(
      const std::vector<std::pair<velox::vector_size_t, RowRange>>&
          requestRanges);

  // Builds read segments by merging overlapping row ranges.
  // Used when no filters are present for efficient reading with seek.
  void mergeStripeRowRanges(
      const std::vector<std::pair<velox::vector_size_t, RowRange>>&
          requestRanges);

  // Builds read segments by splitting overlapping row ranges into
  // non-overlapping segments. Used when filters are present.
  void splitStripeRowRanges(
      const std::vector<std::pair<velox::vector_size_t, RowRange>>&
          requestRanges);

  // Loads a stripe and performs index lookup to determine row ranges.
  // @param stripeIndex The index of the stripe to load.
  void initStripeColumnReader(uint32_t stripeIndex);

  // Prepares the column reader for reading the current segment by seeking to
  // the segment's start row. Does nothing if readSegmentIndex_ is out of
  // range.
  void prepareStripeSegmentRead();

  // Reads a fragment of data from the current stripe's read segment.
  // @param output The output vector to populate.
  // @return Number of rows read.
  uint64_t readStripeFragment(velox::RowVectorPtr& output);

  // Advances to the next read segment within the current stripe.
  void advanceStripeReadSegment();

  // Advances to the next stripe in the iteration.
  void advanceStripe();

  // Adds output data for a segment to the output chunks and tracks references.
  // @param segment The segment that produced this output.
  // @param output The output data to add.
  void addStripeSegmentOutput(
      const ReadSegment& segment,
      velox::RowVectorPtr output);

  // Tracks output references for requests in a segment.
  // @param segment The segment containing request indices.
  // @param outputIndex Index of the output chunk.
  // @param outputRows Number of rows in the output.
  void trackStripeSegmentOutputRefs(
      const ReadSegment& segment,
      size_t outputIndex,
      velox::vector_size_t outputRows);

  // Updates pending stripe counts for requests after processing a segment.
  // @param segment The segment that was processed.
  void updatePendingStripes(const ReadSegment& segment);

  // Updates the count of ready output requests that can be returned.
  void updateReadyOutputRequests();

  // Checks if enough output is ready to produce results.
  // @param maxOutputRows Maximum rows to produce.
  // @return True if output can be produced.
  bool canProduceOutput(velox::vector_size_t maxOutputRows) const;

  // Produces the next batch of output results.
  // @return Result containing output rows and request indices.
  std::unique_ptr<Result> produceOutput();

  // Removes a request from all stripes starting from startStripeIdx. Called
  // when a request has reached its maxRowsPerRequest_ limit and the remaining
  // rows will be processed within the current stripe.
  // @param requestIdx The request index to remove from subsequent stripes.
  // @param startStripeIdx The starting stripe index to remove from.
  void removeRequestFromSubsequentStripes(
      velox::vector_size_t requestIdx,
      size_t startStripeIdx);

  // Resets all state for a new lookup operation.
  void reset();

  // Adapts the output type for column reader construction. For
  // flatmap-as-struct columns (where outputType has ROW but the file has MAP),
  // replaces ROW with the file's MAP type so the column reader knows the
  // physical format.
  static velox::RowTypePtr adaptOutputTypeForReader(
      const velox::RowTypePtr& outputType,
      const velox::RowTypePtr& fileType,
      const velox::common::ScanSpec& scanSpec);

  const std::shared_ptr<ReaderBase> readerBase_;
  const velox::dwio::common::RowReaderOptions options_;
  const std::unique_ptr<const EncodingFactory> encodingFactory_;
  const std::unique_ptr<RowSizeTracker> rowSizeTracker_;
  const bool hasFilters_;

  // The user-facing output type. For flatmap-as-struct columns, this has ROW
  // types instead of MAP. Used when creating output batch vectors so that
  // copy() and output assembly use the correct column types.
  const velox::RowTypePtr outputType_;

  // The file-facing output type. For flatmap-as-struct columns, ROW types
  // are replaced with the file's MAP types so the column reader knows the
  // physical format. Used for column reader construction.
  const velox::RowTypePtr fileOutputType_;

  const ClusterIndex* const clusterIndex_;

  StripeStreams streams_;
  int32_t numStripes_{0};

  std::unique_ptr<velox::serializer::KeyEncoder> keyEncoder_;

  std::unique_ptr<velox::dwio::common::SelectiveColumnReader> columnReader_;
  velox::dwio::common::SplitStats splitStats_{
      velox::dwio::common::FileFormat::NIMBLE};

  // Current lookup state.
  size_t numRequests_{0};
  Options requestOptions_;
  std::vector<velox::serializer::EncodedKeyBounds> encodedKeyBounds_;
  std::vector<uint32_t> stripes_;
  folly::F14FastMap<uint32_t, std::vector<velox::vector_size_t>>
      stripeToRequests_;
  size_t stripeIndex_{0};
  bool stripeLoaded_{false};
  std::vector<ReadSegment> readSegments_;
  size_t readSegmentIndex_{0};

  // Output state.
  std::vector<RequestState> requestStates_;
  std::vector<OutputChunk> outputSegments_;
  velox::vector_size_t nextOutputRequest_{0};
  velox::vector_size_t lastReadyOutputRequest_{-1};
  velox::vector_size_t readyOutputRows_{0};

  void addStat(std::string_view name, int64_t value) {
    stats_[std::string(name)].addValue(value);
  }

  // Tracks distinct stripes loaded for the numDistinctStripesLoaded metric.
  folly::F14FastSet<uint32_t> loadedStripes_;

  // Accumulated runtime stats. Timing entries are pre-initialized in the
  // constructor with their unit so addValue() preserves the unit. stats()
  // emits only entries with count > 0.
  folly::F14FastMap<std::string, velox::RuntimeMetric> stats_;
};

} // namespace facebook::nimble

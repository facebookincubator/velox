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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleIndexReader.h"

#include <utility>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble {

using namespace facebook::velox;
using RuntimeCounter = velox::RuntimeCounter;

namespace {

Encoding::Options encodingOptions(const TabletReader& tablet) {
  Encoding::Options options;
  options.useVarintRowCount = tablet.features().compactRowCountEncoding();
  return options;
}

// Converts nimble SortOrders to velox::core::SortOrders.
std::vector<velox::core::SortOrder> toVeloxSortOrders(
    const std::vector<SortOrder>& sortOrders,
    size_t numBoundColumns) {
  std::vector<velox::core::SortOrder> veloxSortOrders;
  veloxSortOrders.reserve(numBoundColumns);
  for (size_t i = 0; i < numBoundColumns; ++i) {
    veloxSortOrders.push_back(sortOrders[i].toVeloxSortOrder());
  }
  return veloxSortOrders;
}

} // namespace

velox::RowTypePtr SelectiveNimbleIndexReader::adaptOutputTypeForReader(
    const velox::RowTypePtr& outputType,
    const velox::RowTypePtr& fileType,
    const velox::common::ScanSpec& scanSpec) {
  bool needsAdaptation = false;
  for (size_t i = 0; i < outputType->size(); ++i) {
    const auto* childSpec = scanSpec.childByName(outputType->nameOf(i));
    if (childSpec && childSpec->isFlatMapAsStruct()) {
      needsAdaptation = true;
      break;
    }
  }
  if (!needsAdaptation) {
    return outputType;
  }

  auto columnNames = outputType->names();
  std::vector<velox::TypePtr> columnTypes;
  columnTypes.reserve(outputType->size());

  for (size_t i = 0; i < outputType->size(); ++i) {
    const auto& fieldName = outputType->nameOf(i);
    const auto* childSpec = scanSpec.childByName(fieldName);
    NIMBLE_CHECK_NOT_NULL(childSpec);
    // fileTypeIdx may be nullopt when the column exists in the output schema
    // but not in the file schema (schema evolution).
    auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);

    if (childSpec->isFlatMapAsStruct() && fileTypeIdx.has_value()) {
      // Flatmap-as-struct: use the file's MAP type so the column reader knows
      // the physical format. The ScanSpec's isFlatMapAsStruct flag tells the
      // reader to produce struct output.
      NIMBLE_CHECK(
          outputType->childAt(i)->isRow() &&
          fileType->childAt(*fileTypeIdx)->isMap());
      columnTypes.push_back(fileType->childAt(*fileTypeIdx));
    } else {
      columnTypes.push_back(outputType->childAt(i));
    }
  }
  return velox::ROW(std::move(columnNames), std::move(columnTypes));
}

void SelectiveNimbleIndexReader::RequestState::incPendingStripes() {
  ++pendingStripes;
}

void SelectiveNimbleIndexReader::RequestState::decPendingStripes() {
  NIMBLE_CHECK_GT(pendingStripes, 0);
  --pendingStripes;
}

std::string SelectiveNimbleIndexReader::ReadSegment::toString() const {
  std::string requestIndicesStr;
  for (size_t i = 0; i < requestIndices.size(); ++i) {
    if (i > 0) {
      requestIndicesStr += ", ";
    }
    requestIndicesStr += std::to_string(requestIndices[i]);
  }
  return fmt::format(
      "ReadSegment{{rowRange: {}, requestIndices: [{}]}}",
      rowRange.toString(),
      requestIndicesStr);
}

void SelectiveNimbleIndexReader::OutputChunk::dropRef() {
  NIMBLE_CHECK_GT(refCount, 0);
  if (--refCount == 0) {
    data.reset();
  }
}

SelectiveNimbleIndexReader::SelectiveNimbleIndexReader(
    std::shared_ptr<ReaderBase> readerBase,
    const dwio::common::RowReaderOptions& options)
    : readerBase_(std::move(readerBase)),
      options_(options),
      encodingFactory_(
          options.stringDecoderZeroCopy()
              ? std::make_unique<const EncodingFactory>(
                    encodingOptions(readerBase_->tablet()))
              : std::make_unique<const legacy::EncodingFactory>(
                    encodingOptions(readerBase_->tablet()))),
      rowSizeTracker_(
          std::make_unique<RowSizeTracker>(readerBase_->fileSchemaWithId())),
      hasFilters_(options.scanSpec()->hasFilter()),
      outputType_(
          options.requestedType() ? options.requestedType()
                                  : readerBase_->fileSchema()),
      fileOutputType_(adaptOutputTypeForReader(
          outputType_,
          readerBase_->fileSchema(),
          *options.scanSpec())),
      clusterIndex_{readerBase_->tablet().clusterIndex()},
      streams_{readerBase_},
      numStripes_{static_cast<int32_t>(readerBase_->tablet().stripeCount())},
      stats_{
          {std::string(dwio::common::IndexReader::kIndexStripeLoadWallNanos),
           velox::RuntimeMetric(velox::RuntimeCounter::Unit::kNanos)},
          {std::string(dwio::common::IndexReader::kIndexStripeLoadCpuNanos),
           velox::RuntimeMetric(velox::RuntimeCounter::Unit::kNanos)},
          {std::string(dwio::common::IndexReader::kIndexDataDecodeWallNanos),
           velox::RuntimeMetric(velox::RuntimeCounter::Unit::kNanos)},
          {std::string(dwio::common::IndexReader::kIndexDataDecodeCpuNanos),
           velox::RuntimeMetric(velox::RuntimeCounter::Unit::kNanos)},
      } {}

void SelectiveNimbleIndexReader::startLookup(
    const velox::serializer::IndexBounds& indexBounds,
    const Options& options) {
  reset();

  numRequests_ = indexBounds.numRows();
  // Encode the index bounds.
  encodedKeyBounds_ = encodeIndexBounds(indexBounds);
  NIMBLE_CHECK_EQ(
      encodedKeyBounds_.size(),
      numRequests_,
      "Encoded key bounds size mismatch");

  // Cache options for use during output tracking.
  requestOptions_ = options;

  // Build stripe to request mapping.
  resolveRequestStripes();

  // Initialize request output tracking.
  nextOutputRequest_ = 0;
  lastReadyOutputRequest_ = -1;
  readyOutputRows_ = 0;

  // Report lookup stats.
  addStat(dwio::common::IndexReader::kNumIndexLookupRequests, numRequests_);
  addStat(dwio::common::IndexReader::kNumIndexLookupStripes, stripes_.size());
}

bool SelectiveNimbleIndexReader::hasNext() const {
  return numRequests_ > 0 &&
      (stripeIndex_ < stripes_.size() || readyOutputRows_ > 0 ||
       nextOutputRequest_ <= lastReadyOutputRequest_);
}

std::unique_ptr<connector::IndexSource::Result>
SelectiveNimbleIndexReader::next(vector_size_t maxOutputRows) {
  NIMBLE_CHECK_GT(numRequests_, 0, "No lookup in progress");

  // Process stripes until we can produce output or finish.
  while (stripeIndex_ < stripes_.size() && !canProduceOutput(maxOutputRows)) {
    if (!loadStripe()) {
      continue;
    }
    NIMBLE_CHECK(stripeLoaded_);

    // Process segments within this stripe.
    while (readSegmentIndex_ < readSegments_.size() &&
           !canProduceOutput(maxOutputRows)) {
      auto& segment = readSegments_[readSegmentIndex_];
      RowVectorPtr segmentOutput;
      const uint64_t readRows = readStripeFragment(segmentOutput);
      NIMBLE_CHECK_EQ(readRows, segment.rowRange.numRows());
      addStripeSegmentOutput(segment, segmentOutput);
      advanceStripeReadSegment();
    }

    advanceStripe();
  }

  return produceOutput();
}

std::vector<velox::serializer::EncodedKeyBounds>
SelectiveNimbleIndexReader::encodeIndexBounds(
    const velox::serializer::IndexBounds& indexBounds) {
  NIMBLE_CHECK_NOT_NULL(clusterIndex_);
  if (keyEncoder_ == nullptr) {
    const auto& sortOrders = clusterIndex_->sortOrders();
    keyEncoder_ = velox::serializer::KeyEncoder::create(
        indexBounds.indexColumns,
        asRowType(indexBounds.type()),
        toVeloxSortOrders(sortOrders, indexBounds.indexColumns.size()),
        readerBase_->pool());
  }
  return keyEncoder_->encodeIndexBounds(indexBounds);
}

void SelectiveNimbleIndexReader::resolveRequestStripes() {
  NIMBLE_CHECK(stripes_.empty());
  NIMBLE_CHECK(stripeToRequests_.empty());
  requestStates_.resize(numRequests_);

  const auto& tablet = readerBase_->tablet();

  // Batch lookup all requests at once.
  const auto result = clusterIndex_->lookup(
      index::IndexLookup::LookupRequest::rangeScan(encodedKeyBounds_));

  uint64_t numMatchedRows = 0;
  for (size_t requestIdx = 0; requestIdx < numRequests_; ++requestIdx) {
    const auto locations = result[requestIdx];
    if (locations.empty()) {
      continue;
    }

    numMatchedRows += locations[0].numRows();
    requestStates_[requestIdx].rowRange = locations[0];

    const auto startStripe =
        static_cast<int32_t>(tablet.rowToStripe(locations[0].startRow));
    auto endStripe = std::min(
        numStripes_,
        static_cast<int32_t>(tablet.rowToStripe(locations[0].endRow - 1) + 1));
    NIMBLE_CHECK_LT(startStripe, endStripe);

    // Add stripes to the mapping.
    // For no-filter case, we can prune stripes based on exact row counts
    // since input rows == output rows. We compute the exact row count per
    // stripe by intersecting the file-level row range with the stripe
    // boundaries. For filter case, we cannot prune stripes since we don't
    // know output rows ahead of time. Instead, truncation based on actual
    // output rows happens during output tracking in
    // trackStripeSegmentOutputRefs().
    const auto& rowRange = requestStates_[requestIdx].rowRange;
    uint64_t accumulatedRows = 0;
    for (uint32_t stripe = startStripe; stripe < endStripe; ++stripe) {
      stripeToRequests_[stripe].push_back(requestIdx);
      requestStates_[requestIdx].incPendingStripes();

      if (!hasFilters_ && requestOptions_.maxRowsPerRequest > 0) {
        NIMBLE_CHECK_LT(accumulatedRows, requestOptions_.maxRowsPerRequest);
        accumulatedRows += stripeRowRange(stripe, rowRange).numRows();
        if (accumulatedRows >= requestOptions_.maxRowsPerRequest) {
          break;
        }
      }
    }
  }

  // Collect and sort unique stripes.
  stripes_.reserve(stripeToRequests_.size());
  uint64_t numScannedRows = 0;
  for (const auto& [stripe, _] : stripeToRequests_) {
    stripes_.push_back(stripe);
    numScannedRows += stripeRowCount(stripe);
  }
  std::sort(stripes_.begin(), stripes_.end());

  addStat(dwio::common::IndexReader::kNumIndexScannedRows, numScannedRows);
  addStat(dwio::common::IndexReader::kNumIndexMatchedRows, numMatchedRows);
}

bool SelectiveNimbleIndexReader::loadStripe() {
  if (stripeLoaded_) {
    return true;
  }
  // Skip stripes if all requests have been removed due to stripe pruning.
  while (stripeIndex_ < stripes_.size()) {
    const uint32_t stripeIndex = stripes_[stripeIndex_];
    auto it = stripeToRequests_.find(stripeIndex);
    NIMBLE_CHECK(it != stripeToRequests_.end());
    if (it->second.empty()) {
      ++stripeIndex_;
      continue;
    }
    prepareStripeReading(stripeIndex);
    NIMBLE_CHECK(!readSegments_.empty());
    stripeLoaded_ = true;
    return true;
  }
  return false;
}

void SelectiveNimbleIndexReader::prepareStripeReading(uint32_t stripeIndex) {
  readSegments_.clear();
  readSegmentIndex_ = 0;

  auto it = stripeToRequests_.find(stripeIndex);
  NIMBLE_CHECK(it != stripeToRequests_.end());
  NIMBLE_CHECK(!it->second.empty());

  auto& requestIndices = it->second;

  // Load the stripe and build column reader.
  initStripeColumnReader(stripeIndex);

  // Compute stripe-relative row ranges from cached file-level row ranges.
  std::vector<RowRange> stripeRowRanges;
  stripeRowRanges.reserve(requestIndices.size());
  for (vector_size_t requestIndex : requestIndices) {
    stripeRowRanges.emplace_back(
        stripeRowRange(stripeIndex, requestStates_[requestIndex].rowRange));
  }

  // Filter out empty ranges and store per-request row ranges.
  // For no-filter case, also truncate row ranges based on
  // requestOptions_.maxRowsPerRequest since input rows == output rows.
  std::vector<std::pair<vector_size_t, RowRange>> requestRanges;
  requestRanges.reserve(requestIndices.size());
  for (size_t i = 0; i < requestIndices.size(); ++i) {
    auto& state = requestStates_[requestIndices[i]];
    NIMBLE_CHECK_GT(state.pendingStripes, 0);
    state.stripeRowRange = stripeRowRanges[i];
    NIMBLE_CHECK(
        !state.stripeRowRange.empty(),
        "Stripe row range should not be empty for a mapped request");

    // For no-filter case, truncate row range if accumulated output would
    // exceed requestOptions_.maxRowsPerRequest. This avoids reading
    // unnecessary data.
    if (!hasFilters_ && requestOptions_.maxRowsPerRequest > 0) {
      NIMBLE_CHECK_GT(requestOptions_.maxRowsPerRequest, state.outputRows);
      const auto remainingRows =
          requestOptions_.maxRowsPerRequest - state.outputRows;
      if (state.stripeRowRange.numRows() >=
          static_cast<vector_size_t>(remainingRows)) {
        state.stripeRowRange.endRow =
            state.stripeRowRange.startRow + remainingRows;
      }
    }

    requestRanges.emplace_back(requestIndices[i], state.stripeRowRange);
  }
  buildStripeReadSegments(requestRanges);
}

void SelectiveNimbleIndexReader::buildStripeReadSegments(
    const std::vector<std::pair<vector_size_t, RowRange>>& requestRanges) {
  if (!hasFilters_) {
    mergeStripeRowRanges(requestRanges);
  } else {
    splitStripeRowRanges(requestRanges);
  }
  NIMBLE_CHECK(!readSegments_.empty());
  addStat(
      dwio::common::IndexReader::kNumIndexLookupReadSegments,
      readSegments_.size());

  readSegmentIndex_ = 0;
  prepareStripeSegmentRead();
}

void SelectiveNimbleIndexReader::mergeStripeRowRanges(
    const std::vector<std::pair<vector_size_t, RowRange>>& requestRanges) {
  NIMBLE_CHECK(!hasFilters_);
  // No filter case: merge row ranges for efficient reading with seek.
  // Each request will reference the appropriate portion of the merged read.

  // Sort ranges by start row.
  std::vector<RowRange> sortedRanges;
  sortedRanges.reserve(requestRanges.size());
  for (const auto& [_, range] : requestRanges) {
    sortedRanges.push_back(range);
  }
  std::sort(
      sortedRanges.begin(),
      sortedRanges.end(),
      [](const auto& a, const auto& b) { return a.startRow < b.startRow; });

  // Merge overlapping ranges.
  std::vector<RowRange> mergedRanges;
  RowRange current = sortedRanges[0];
  for (size_t i = 1; i < sortedRanges.size(); ++i) {
    const auto& range = sortedRanges[i];
    if (range.startRow <= current.endRow) {
      current.endRow = std::max(current.endRow, range.endRow);
    } else {
      mergedRanges.push_back(current);
      current = range;
    }
  }
  mergedRanges.push_back(current);

  // Create one read segment per merged range.
  // Each segment references all requests that overlap with it.
  for (const auto& mergedRange : mergedRanges) {
    ReadSegment segment;
    segment.rowRange = mergedRange;
    for (const auto& [requestIdx, range] : requestRanges) {
      // Check if merged range contains the request range.
      if (mergedRange.contains(range)) {
        segment.requestIndices.push_back(requestIdx);
      }
    }
    // Each merged range must have at least one contained request.
    NIMBLE_CHECK(
        !segment.requestIndices.empty(),
        "Merged range [{}, {}) has no contained requests",
        mergedRange.startRow,
        mergedRange.endRow);
    readSegments_.push_back(std::move(segment));
  }
}

void SelectiveNimbleIndexReader::splitStripeRowRanges(
    const std::vector<std::pair<vector_size_t, RowRange>>& requestRanges) {
  NIMBLE_CHECK(hasFilters_);
  // Filter case: split overlapping ranges into non-overlapping segments.
  // Example: [0,100) and [50,150) -> [0,50), [50,100), [100,150)
  // Each segment is read separately, and requests reference their segments.

  // Collect all unique boundary points.
  std::vector<uint32_t> boundaries;
  boundaries.reserve(requestRanges.size() * 2);
  for (const auto& [_, range] : requestRanges) {
    boundaries.push_back(range.startRow);
    boundaries.push_back(range.endRow);
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(
      std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  // Create segments from consecutive boundary points.
  for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
    RowRange segmentRange{boundaries[i], boundaries[i + 1]};
    NIMBLE_CHECK(!segmentRange.empty());

    // Find which requests contain this segment.
    std::vector<vector_size_t> segmentRequestIndices;
    for (const auto& [requestIdx, range] : requestRanges) {
      if (range.contains(segmentRange)) {
        segmentRequestIndices.push_back(requestIdx);
      }
    }

    // Skip segments in gaps between non-overlapping request ranges.
    if (segmentRequestIndices.empty()) {
      continue;
    }
    ReadSegment segment;
    segment.rowRange = segmentRange;
    segment.requestIndices = std::move(segmentRequestIndices);
    readSegments_.push_back(std::move(segment));
  }
}

void SelectiveNimbleIndexReader::initStripeColumnReader(uint32_t stripeIndex) {
  addStat(dwio::common::RowReader::kNumStripeLoads, 1);
  loadedStripes_.insert(stripeIndex);

  CpuWallTiming timing;
  {
    CpuWallTimer timer(timing);
    streams_.setStripe(stripeIndex);
    NimbleParams params(
        *readerBase_->pool(),
        columnReaderStatistics_,
        readerBase_->nimbleSchema(),
        streams_,
        options_.trackRowSize() ? rowSizeTracker_.get() : nullptr,
        *encodingFactory_,
        options_.stringDecoderZeroCopy(),
        options_.preserveFlatMapsInMemory());

    columnReader_ = buildColumnReader(
        fileOutputType_,
        readerBase_->fileSchemaWithId(),
        params,
        *options_.scanSpec(),
        true);
    rowSizeTracker_->finalizeProjection();
    columnReader_->setIsTopLevel();

    streams_.load();
  }
  addStat(
      dwio::common::IndexReader::kIndexStripeLoadWallNanos, timing.wallNanos);
  addStat(dwio::common::IndexReader::kIndexStripeLoadCpuNanos, timing.cpuNanos);
}

uint64_t SelectiveNimbleIndexReader::readStripeFragment(RowVectorPtr& output) {
  if (readSegmentIndex_ >= readSegments_.size()) {
    return 0;
  }

  const auto& segment = readSegments_[readSegmentIndex_];
  const auto rowsToRead = segment.rowRange.endRow - segment.rowRange.startRow;
  NIMBLE_CHECK_GT(rowsToRead, 0);

  VectorPtr readOutput =
      BaseVector::create(outputType_, 0, readerBase_->pool());
  CpuWallTiming timing;
  {
    CpuWallTimer timer(timing);
    columnReader_->next(rowsToRead, readOutput, /*mutation=*/nullptr);
  }
  addStat(
      dwio::common::IndexReader::kIndexDataDecodeWallNanos, timing.wallNanos);
  addStat(dwio::common::IndexReader::kIndexDataDecodeCpuNanos, timing.cpuNanos);
  if (readOutput->size() > 0) {
    readOutput->loadedVector();
  }
  output = checkedPointerCast<RowVector>(readOutput);
  return rowsToRead;
}

void SelectiveNimbleIndexReader::advanceStripeReadSegment() {
  ++readSegmentIndex_;
  prepareStripeSegmentRead();
}

void SelectiveNimbleIndexReader::prepareStripeSegmentRead() {
  if (readSegmentIndex_ < readSegments_.size()) {
    columnReader_->seekTo(
        readSegments_[readSegmentIndex_].rowRange.startRow,
        /*readsNullsOnly=*/false);
  }
}

void SelectiveNimbleIndexReader::advanceStripe() {
  if (readSegmentIndex_ >= readSegments_.size()) {
    ++stripeIndex_;
    stripeLoaded_ = false;
    readSegments_.clear();
    readSegmentIndex_ = 0;
    columnReader_.reset();
  }
}

bool SelectiveNimbleIndexReader::canProduceOutput(
    vector_size_t maxOutputRows) const {
  return readyOutputRows_ >= maxOutputRows;
}

void SelectiveNimbleIndexReader::addStripeSegmentOutput(
    const ReadSegment& segment,
    RowVectorPtr output) {
  NIMBLE_CHECK_NOT_NULL(output);
  const auto outputRows = output->size();
  if (outputRows == 0) {
    updatePendingStripes(segment);
  } else {
    outputSegments_.emplace_back(std::move(output));
    const size_t outputIndex = outputSegments_.size() - 1;
    trackStripeSegmentOutputRefs(segment, outputIndex, outputRows);
    NIMBLE_CHECK_GT(outputSegments_[outputIndex].refCount, 0);
  }
  updateReadyOutputRequests();
}

void SelectiveNimbleIndexReader::trackStripeSegmentOutputRefs(
    const ReadSegment& segment,
    size_t outputIndex,
    vector_size_t outputRows) {
  NIMBLE_CHECK_GT(outputRows, 0);
  const auto startRow = segment.rowRange.startRow;
  const auto readRows = segment.rowRange.numRows();

  if (!hasFilters_) {
    // No filter: outputRows == readRows. Each request extracts its portion
    // based on its row range. Row range truncation is already done in
    // prepareStripeReading.
    NIMBLE_CHECK_EQ(outputRows, readRows);
    const auto readEndRow = segment.rowRange.endRow;
    for (vector_size_t requestIdx : segment.requestIndices) {
      auto& state = requestStates_[requestIdx];
      const auto& stripeRange = state.stripeRowRange;
      const auto overlapStart = std::max(startRow, stripeRange.startRow);
      const auto overlapEnd = std::min(readEndRow, stripeRange.endRow);
      const auto overlapRows = overlapEnd - overlapStart;

      state.outputRefs.emplace_back(
          RequestState::OutputReference{
              outputIndex,
              RowRange{overlapStart - startRow, overlapEnd - startRow}});
      outputSegments_[outputIndex].addRef();
      state.outputRows += overlapRows;
      VELOX_DCHECK(
          requestOptions_.maxRowsPerRequest == 0 ||
              state.outputRows <= requestOptions_.maxRowsPerRequest,
          "Output rows {} exceeds maxRowsPerRequest {} for request {}",
          state.outputRows,
          requestOptions_.maxRowsPerRequest,
          requestIdx);
    }
  } else {
    // With filter: outputRows <= readRows. Each request in the segment
    // gets the filtered output. Apply truncation if
    // requestOptions_.maxRowsPerRequest is set.
    NIMBLE_CHECK_LE(outputRows, readRows);

    for (vector_size_t requestIdx : segment.requestIndices) {
      auto& state = requestStates_[requestIdx];

      // Calculate how many rows to add, respecting the limit.
      auto rowsToAdd = outputRows;
      if (requestOptions_.maxRowsPerRequest > 0) {
        if (requestOptions_.maxRowsPerRequest == state.outputRows) {
          continue;
        }
        const auto remainingRows =
            requestOptions_.maxRowsPerRequest - state.outputRows;
        if (outputRows >= static_cast<vector_size_t>(remainingRows)) {
          rowsToAdd = remainingRows;
          removeRequestFromSubsequentStripes(requestIdx, stripeIndex_ + 1);
        }
      }

      state.outputRefs.emplace_back(
          RequestState::OutputReference{
              outputIndex, RowRange{0, static_cast<uint32_t>(rowsToAdd)}});
      outputSegments_[outputIndex].addRef();
      state.outputRows += rowsToAdd;
    }
  }
  updatePendingStripes(segment);
}

void SelectiveNimbleIndexReader::updatePendingStripes(
    const ReadSegment& segment) {
  for (vector_size_t requestIdx : segment.requestIndices) {
    const auto& stripeRange = requestStates_[requestIdx].stripeRowRange;
    if (segment.rowRange.endRow >= stripeRange.endRow) {
      --requestStates_[requestIdx].pendingStripes;
    }
  }
}

void SelectiveNimbleIndexReader::updateReadyOutputRequests() {
  for (auto i = lastReadyOutputRequest_ + 1;
       i < static_cast<vector_size_t>(numRequests_);
       ++i) {
    if (requestStates_[i].pendingStripes != 0) {
      break;
    }
    lastReadyOutputRequest_ = i;
    readyOutputRows_ += requestStates_[i].outputRows;
    // Skip empty output request.
    if (requestStates_[i].empty() && i == nextOutputRequest_) {
      ++nextOutputRequest_;
    }
  }
}

std::unique_ptr<connector::IndexSource::Result>
SelectiveNimbleIndexReader::produceOutput() {
  if (readyOutputRows_ == 0) {
    NIMBLE_CHECK_EQ(stripeIndex_, stripes_.size());
    reset();
    return nullptr;
  }

  NIMBLE_CHECK_LE(nextOutputRequest_, lastReadyOutputRequest_);

  // Count total output rows from all ready requests.
  vector_size_t totalOutputRows = 0;
  for (auto i = nextOutputRequest_; i <= lastReadyOutputRequest_; ++i) {
    totalOutputRows += requestStates_[i].outputRows;
  }
  NIMBLE_CHECK_EQ(totalOutputRows, readyOutputRows_);

  // Allocate output buffers for all ready requests.
  RowVectorPtr output = BaseVector::create<RowVector>(
      outputType_, totalOutputRows, readerBase_->pool());
  auto inputHits = AlignedBuffer::allocate<vector_size_t>(
      totalOutputRows, readerBase_->pool());
  auto* rawInputHits = inputHits->asMutable<vector_size_t>();

  // Build output for all ready requests.
  vector_size_t outputOffset = 0;
  for (auto requestIdx = nextOutputRequest_;
       requestIdx <= lastReadyOutputRequest_;
       ++requestIdx) {
    const auto& state = requestStates_[requestIdx];
    if (state.empty()) {
      continue;
    }

    for (const auto& ref : state.outputRefs) {
      auto& chunk = outputSegments_[ref.chunkIndex];
      const auto numRows = ref.rowRange.endRow - ref.rowRange.startRow;
      output->copy(
          chunk.data.get(), outputOffset, ref.rowRange.startRow, numRows);

      // Fill inputHits with the request index for each output row.
      std::fill(
          rawInputHits + outputOffset,
          rawInputHits + outputOffset + numRows,
          requestIdx);

      outputOffset += numRows;
      chunk.dropRef();
    }
  }
  NIMBLE_CHECK_EQ(outputOffset, totalOutputRows);

  nextOutputRequest_ = lastReadyOutputRequest_ + 1;
  readyOutputRows_ = 0;
  return std::make_unique<connector::IndexSource::Result>(
      std::move(inputHits), std::move(output));
}

void SelectiveNimbleIndexReader::removeRequestFromSubsequentStripes(
    velox::vector_size_t requestIdx,
    size_t startStripeIdx) {
  auto& state = requestStates_[requestIdx];
  // We must have at least one pending stripe (the current stripe) that will
  // be processed. We only remove from subsequent stripes.
  NIMBLE_CHECK_GE(state.pendingStripes, 1);
  for (size_t i = startStripeIdx;
       i < stripes_.size() && state.pendingStripes > 1;
       ++i) {
    const auto stripe = stripes_[i];
    auto& requests = stripeToRequests_[stripe];
    auto it = std::find(requests.begin(), requests.end(), requestIdx);
    NIMBLE_CHECK(
        it != requests.end(),
        "Request {} must exist in stripe {} as stripes are contiguous",
        requestIdx,
        stripe);
    requests.erase(it);
    state.decPendingStripes();
  }
}

void SelectiveNimbleIndexReader::reset() {
  numRequests_ = 0;
  requestOptions_ = {};
  encodedKeyBounds_.clear();
  stripes_.clear();
  stripeToRequests_.clear();
  stripeIndex_ = 0;
  stripeLoaded_ = false;
  readSegments_.clear();
  readSegmentIndex_ = 0;
  requestStates_.clear();
  outputSegments_.clear();
  nextOutputRequest_ = 0;
  lastReadyOutputRequest_ = -1;
  readyOutputRows_ = 0;
  columnReader_.reset();
}

folly::F14FastMap<std::string, velox::RuntimeMetric>
SelectiveNimbleIndexReader::stats() const {
  folly::F14FastMap<std::string, velox::RuntimeMetric> result;
  // Only emit metrics that received at least one addValue() call.
  // Pre-initialized timing entries that were never written are skipped.
  for (const auto& [name, metric] : stats_) {
    if (metric.count > 0) {
      result.emplace(name, metric);
    }
  }
  if (!loadedStripes_.empty()) {
    result[std::string(
        dwio::common::IndexReader::kNumIndexDistinctStripesLoaded)] =
        RuntimeMetric(static_cast<int64_t>(loadedStripes_.size()));
  }
  return result;
}

} // namespace facebook::nimble

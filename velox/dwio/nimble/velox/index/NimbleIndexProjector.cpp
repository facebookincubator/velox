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

#include "velox/dwio/nimble/velox/index/NimbleIndexProjector.h"

#include <algorithm>
#include <cmath>

#include "folly/ScopeGuard.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/serializer/StreamSlicer.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"

namespace facebook::nimble {

using namespace facebook::velox; // NOLINT(google-build-using-namespace)

namespace {

void validateReaderOptions(const velox::dwio::common::ReaderOptions& options) {
  NIMBLE_CHECK_NOT_NULL(
      options.dataIoStats(),
      "NimbleIndexProjector requires ReaderOptions::dataIoStats to be set");
  NIMBLE_CHECK_NOT_NULL(
      options.metadataIoStats(),
      "NimbleIndexProjector requires ReaderOptions::metadataIoStats to be set");
  NIMBLE_CHECK_NOT_NULL(
      options.indexIoStats(),
      "NimbleIndexProjector requires ReaderOptions::indexIoStats to be set");
}

} // namespace

std::string NimbleIndexProjector::Stats::toString() const {
  return fmt::format(
      "Stats(numReadStripes={}, numSlicedStripes={}, "
      "slicedStripePct={:.2f}%, numReadRows={}, numProjectedRows={}, "
      "numOutputBytes={}, "
      "lookupTiming=[{}], prepareTiming=[{}], scanTiming=[{}], "
      "projectionTiming=[{}])",
      numReadStripes,
      numSlicedStripes,
      numReadStripes == 0
          ? 0.0
          : 100.0 * static_cast<double>(numSlicedStripes) / numReadStripes,
      numReadRows,
      numProjectedRows,
      velox::succinctBytes(numOutputBytes),
      lookupTiming.toString(),
      prepareTiming.toString(),
      scanTiming.toString(),
      projectionTiming.toString());
}

namespace {

// Adapter that lets serializer helpers append directly into an already
// allocated IOBuf. resize() only advances the IOBuf length within existing
// tailroom; it never reallocates.
class IOBufAppender {
 public:
  explicit IOBufAppender(folly::IOBuf& buffer) : buffer_{buffer} {}

  size_t size() const {
    return buffer_.length();
  }

  void resize(size_t size) {
    const auto currentSize = buffer_.length();
    NIMBLE_CHECK_GE(
        size, currentSize, "IOBufAppender only supports appending data");
    const auto appendSize = size - currentSize;
    NIMBLE_CHECK_LE(
        appendSize,
        buffer_.tailroom(),
        "Estimated trailer tailroom is too small: {} > {}",
        appendSize,
        buffer_.tailroom());
    buffer_.append(appendSize);
  }

  char* data() {
    return reinterpret_cast<char*>(buffer_.writableData());
  }

 private:
  folly::IOBuf& buffer_;
};

constexpr auto kTabletTrailerEncoding = EncodingType::FixedBitWidth;

size_t estimateTabletTrailerSize(
    size_t numPresentStreams,
    size_t numUniqueStreams) {
  return serde::detail::estimateTrailerSize(
      numPresentStreams,
      numUniqueStreams,
      kTabletTrailerEncoding,
      kTabletTrailerEncoding,
      kTabletTrailerEncoding);
}

template <typename Buffer>
void writeTabletTrailer(
    const std::vector<uint32_t>& streamIds,
    const std::vector<uint32_t>& streamSizeIndices,
    const std::vector<uint32_t>& uniqueStreamSizes,
    Buffer& buffer) {
  serde::detail::writeTrailer(
      streamIds,
      streamSizeIndices,
      uniqueStreamSizes,
      kTabletTrailerEncoding,
      kTabletTrailerEncoding,
      kTabletTrailerEncoding,
      buffer);
}

std::unique_ptr<DataInput> createDataInput(
    const velox::FileHandle& fileHandle,
    const velox::dwio::common::ReaderOptions& options) {
  DirectDataInput::Options dataInputOptions;
  dataInputOptions.pool = &options.memoryPool();
  dataInputOptions.ioStats = options.dataIoStats();
  dataInputOptions.maxCoalesceDistance = options.maxCoalesceDistance();
  dataInputOptions.maxCoalesceBytes = options.maxCoalesceBytes();
  return std::make_unique<DirectDataInput>(
      fileHandle.file.get(), dataInputOptions);
}

void freeDataHandle(void* /*buf*/, void* userData) {
  delete static_cast<DataInput::Handle*>(userData);
}

} // namespace

std::unique_ptr<NimbleIndexProjector> NimbleIndexProjector::create(
    TabletReaderCache& tabletReaderCache,
    const velox::FileHandle& fileHandle,
    const std::vector<Subfield>& projectedSubfields,
    const velox::dwio::common::ReaderOptions& options) {
  validateReaderOptions(options);
  NIMBLE_CHECK(
      !options.cacheData(),
      "NimbleIndexProjector does not support data caching");
  auto cached = tabletReaderCache.get(
      fileHandle.file, TabletReader::configureOptions(options));
  auto projection =
      std::make_shared<NimbleTypeProjection>(buildProjectedNimbleType(
          cached->nimbleSchema().get(), projectedSubfields));
  // Aliased onto `cached` so the projector owns the cache entry, not just the
  // tablet. The entry holds the IoStatistics the tablet writes its metadata and
  // index reads into, and retiring it hands those totals off and stops watching
  // them -- so an entry that retired while this projector was still reading
  // would leave the rest of its IO reported by nobody.
  auto tablet = std::shared_ptr<TabletReader>(cached, cached->tablet().get());
  return create(std::move(tablet), fileHandle, std::move(projection), options);
}

std::unique_ptr<NimbleIndexProjector> NimbleIndexProjector::create(
    std::shared_ptr<TabletReader> tablet,
    const velox::FileHandle& fileHandle,
    std::shared_ptr<const NimbleTypeProjection> projection,
    const velox::dwio::common::ReaderOptions& options) {
  validateReaderOptions(options);
  NIMBLE_CHECK(
      !options.cacheData(),
      "NimbleIndexProjector does not support data caching");
  NIMBLE_CHECK_NOT_NULL(tablet);
  NIMBLE_CHECK_NOT_NULL(projection);
  return std::unique_ptr<NimbleIndexProjector>(new NimbleIndexProjector(
      std::move(tablet),
      fileHandle.file,
      createDataInput(fileHandle, options),
      std::move(projection),
      &options.memoryPool(),
      options.dataIoStats()));
}

NimbleIndexProjector::NimbleIndexProjector(
    std::shared_ptr<TabletReader> tablet,
    std::shared_ptr<velox::ReadFile> file,
    std::unique_ptr<DataInput> dataInput,
    std::shared_ptr<const NimbleTypeProjection> projection,
    velox::memory::MemoryPool* pool,
    std::shared_ptr<velox::io::IoStatistics> ioStats)
    : file_{std::move(file)},
      tablet_{std::move(tablet)},
      ioStats_{std::move(ioStats)},
      pool_{pool},
      dataInput_{std::move(dataInput)},
      clusterIndex_{tablet_->clusterIndex()},
      numStripes_{tablet_->stripeCount()},
      projection_{std::move(projection)},
      streamSlicer_{std::make_unique<serde::StreamSlicer>(
          projection_->nimbleType,
          pool_,
          serde::StreamSlicer::Options{
              .streamVersion = SerializationVersion::kTablet,
              .streamHasChunkHeader = true,
              .streamsUseVarintRowCount =
                  tablet_->properties().compactRowCountEncoding(),
          })} {
  NIMBLE_CHECK_NOT_NULL(
      clusterIndex_, "NimbleIndexProjector requires a tablet with an index");
  NIMBLE_CHECK_GT(numStripes_, 0, "NimbleIndexProjector requires stripes");

  // Rejects the whole file rather than only the projected streams: no cheap
  // per-stream binding query exists for file and external scopes, and
  // resolving an alphabet just to test for one would decode it.
  if (tablet_->hasStripeDictionaries() ||
      tablet_->hasFileOrExternalDictionaries()) {
    NIMBLE_UNSUPPORTED(
        "NimbleIndexProjector does not support shared dictionary encoding");
  }

  NIMBLE_CHECK_EQ(
      projection_->streamOffsets.size(),
      projection_->rowOrFlatMapNullStreams.size(),
      "Projected stream offsets and Row/FlatMap null stream mask must align");
}

NimbleIndexProjector::~NimbleIndexProjector() = default;

NimbleIndexProjector::Result NimbleIndexProjector::project(
    const Request& request,
    const Options& options) {
  initRequest(request, options);
  SCOPE_EXIT {
    clearRequest();
  };

  lookupStripes();
  prepareStripes();
  loadStripes();
  return processStripes();
}

void NimbleIndexProjector::setResumeKey(
    uint32_t requestIndex,
    uint32_t resolvedStripeIndex,
    uint32_t readEndRow,
    bool partialRead) {
  if (ctx_.resumeKeys[requestIndex].has_value()) {
    return;
  }
  // More rows remain iff we clipped mid-stripe, or the request continues into
  // the next stripe. A request occupies one contiguous stripe span, so
  // "continues" can only mean the immediately-next stripe -- no scan needed.
  bool hasMore = partialRead;
  const auto& resolvedStripes = ctx_.stripeRanges.resolvedStripes;
  const uint32_t nextResolvedStripeIndex = resolvedStripeIndex + 1;
  // Only the immediately-next stripe can continue the request, so skip the
  // scan when the next resolved stripe is not adjacent. Requests cover one
  // contiguous stripe run, so a non-adjacent successor could never hold this
  // request anyway -- the adjacency test just avoids scanning its ranges to
  // find that out.
  if (!hasMore && nextResolvedStripeIndex < resolvedStripes.size() &&
      resolvedStripes[nextResolvedStripeIndex] ==
          resolvedStripes[resolvedStripeIndex] + 1) {
    const auto ranges = ctx_.stripeRanges.getRanges(nextResolvedStripeIndex);
    hasMore = std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
      return range.requestIndex == requestIndex;
    });
  }
  // The first un-returned row is stripeStartRow + readEndRow in both cases:
  // the clip point for a mid-stripe cut, or this stripe's end (== the next
  // stripe's start) for a boundary cap.
  if (hasMore) {
    const uint32_t stripeIndex = resolvedStripes[resolvedStripeIndex];
    const auto stripeStartRow =
        static_cast<uint32_t>(tablet_->stripeStartRow(stripeIndex));
    ctx_.resumeKeys[requestIndex] =
        clusterIndex_->keyAtRow(stripeStartRow + readEndRow);
  }
}

void NimbleIndexProjector::prepareStripes() {
  velox::CpuWallTimer timer(stats_.prepareTiming);
  uint64_t totalRows{0};
  uint64_t totalBytes{0};
  const auto maxRowsPerRequest = ctx_.options->maxRowsPerRequest;
  const auto needResumeKey = ctx_.options->needResumeKey;
  auto& rowsPerRequest = ctx_.rowsPerRequest;
  rowsPerRequest.assign(ctx_.numRequests, 0);
  ctx_.hasStripeRanges.assign(ctx_.numRequests, false);
  ctx_.resumeKeys.assign(ctx_.numRequests, std::nullopt);
  // Reserve against the number of stripes that carry ranges, not the min..max
  // span: the CSR is now indexed by position in resolvedStripes, so the span
  // is an over-estimate whenever the requested stripes are sparse.
  const auto numResolvedStripes =
      static_cast<uint32_t>(ctx_.stripeRanges.resolvedStripes.size());
  ctx_.plan.stripeIndices.reserve(numResolvedStripes);
  ctx_.plan.numRows.reserve(numResolvedStripes);
  ctx_.plan.requiresNullBarriers.reserve(numResolvedStripes);
  ctx_.plan.numStreams.reserve(numResolvedStripes);
  ctx_.plan.projectedBytes.reserve(numResolvedStripes);
  ctx_.plan.stripeFileOffsets.reserve(numResolvedStripes);
  ctx_.plan.stripeRangeOffsets.reserve(numResolvedStripes + 1);
  ctx_.plan.projectedStreams.reserve(
      static_cast<size_t>(numResolvedStripes) *
      projection_->streamOffsets.size());
  ctx_.plan.stripeRanges.reserve(ctx_.stripeRanges.ranges.size());

  for (uint32_t resolvedStripeIndex = 0;
       resolvedStripeIndex < numResolvedStripes;
       ++resolvedStripeIndex) {
    auto spanRanges = ctx_.stripeRanges.getRanges(resolvedStripeIndex);
    const uint32_t stripeIndex =
        ctx_.stripeRanges.resolvedStripes[resolvedStripeIndex];

    const auto rangeOffset = ctx_.plan.stripeRanges.size();
    uint32_t numStripeRanges{0};

    uint64_t stripeRows{0};
    for (const auto& range : spanRanges) {
      auto stripeRange = range;
      const auto requestIndex = stripeRange.requestIndex;
      const auto numRows =
          static_cast<uint64_t>(stripeRange.rowRange.numRows());
      if (maxRowsPerRequest == 0) {
        rowsPerRequest[requestIndex] += numRows;
        stripeRows += numRows;
        ctx_.hasStripeRanges[requestIndex] = true;
        ctx_.plan.stripeRanges.push_back(stripeRange);
        ++numStripeRanges;
        continue;
      }

      if (rowsPerRequest[requestIndex] >= maxRowsPerRequest) {
        continue;
      }
      const auto remaining = maxRowsPerRequest - rowsPerRequest[requestIndex];
      const auto rowsToRead = std::min(numRows, remaining);
      rowsPerRequest[requestIndex] += rowsToRead;
      stripeRows += rowsToRead;

      // The hard cap cut this request mid-stripe: clip its range to the rows we
      // keep. Clipping is independent of resume keys and always applies.
      const bool partialRead = rowsToRead < numRows;
      if (partialRead) {
        stripeRange.rowRange.endRow =
            stripeRange.rowRange.startRow + static_cast<uint32_t>(rowsToRead);
      }

      // Once the request reaches its cap (via a mid-stripe clip or by landing
      // exactly on this stripe's end), record where it resumes.
      if (needResumeKey && rowsPerRequest[requestIndex] >= maxRowsPerRequest) {
        setResumeKey(
            requestIndex,
            resolvedStripeIndex,
            stripeRange.rowRange.endRow,
            partialRead);
      }
      ctx_.hasStripeRanges[requestIndex] = true;
      ctx_.plan.stripeRanges.push_back(stripeRange);
      ++numStripeRanges;
    }

    if (numStripeRanges == 0) {
      continue;
    }

    totalRows += stripeRows;
    totalBytes += appendStripePlan(stripeIndex, rangeOffset);
    if ((ctx_.options->maxRows > 0 && totalRows >= ctx_.options->maxRows) ||
        (ctx_.options->maxBytes > 0 && totalBytes >= ctx_.options->maxBytes)) {
      ctx_.plan.truncated = true;
      break;
    }
  }
  ctx_.plan.stripeRangeOffsets.push_back(ctx_.plan.stripeRanges.size());
}

void NimbleIndexProjector::initRequest(
    const Request& request,
    const Options& options) {
  NIMBLE_CHECK_NULL(ctx_.request, "project() is not reentrant");
  NIMBLE_CHECK_NULL(ctx_.options, "project() is not reentrant");
  NIMBLE_CHECK_GT(request.keyBounds.size(), 0, "keyBounds must not be empty");
  NIMBLE_CHECK(
      std::isfinite(options.maxOverfetchRowsRatio) &&
          options.maxOverfetchRowsRatio >= 0.0 &&
          options.maxOverfetchRowsRatio <= 1.0,
      "maxOverfetchRowsRatio must be between 0.0 and 1.0. Got: {}",
      options.maxOverfetchRowsRatio);
  ctx_.request = &request;
  ctx_.options = &options;
  ctx_.numRequests = static_cast<uint32_t>(request.keyBounds.size());
}

void NimbleIndexProjector::clearRequest() {
  ctx_.request = nullptr;
  ctx_.options = nullptr;
  ctx_.numRequests = 0;
  ctx_.stripeRanges.clear();
  ctx_.plan.stripeIndices.clear();
  ctx_.plan.numRows.clear();
  ctx_.plan.requiresNullBarriers.clear();
  ctx_.plan.numStreams.clear();
  ctx_.plan.projectedBytes.clear();
  ctx_.plan.stripeFileOffsets.clear();
  ctx_.plan.projectedStreams.clear();
  ctx_.plan.stripeRangeOffsets.clear();
  ctx_.plan.stripeRanges.clear();
  ctx_.plan.truncated = false;
  ctx_.hasStripeRanges.clear();
  ctx_.resumeKeys.clear();
  ctx_.dataInputIndices.clear();
  ctx_.dataHandle.reset();
  ctx_.packedStripes.clear();
  ctx_.stripePackRanges.clear();
  ctx_.rowsPerRequest.clear();
  ctx_.resolvedStripesScratch.clear();
  ctx_.sliceCounts.clear();
  ctx_.emittedSlices.clear();
  ctx_.sliceOutputBuffer.reset();
  ctx_.sliceOutputChunks.reset();
  ctx_.packScratch.clear();
  ctx_.streamViewsScratch.clear();
  dataInput_->clear();
}

RowRange NimbleIndexProjector::stripeRowRange(
    uint32_t stripe,
    const RowRange& rowRangeLimit) const {
  const auto stripeStart =
      static_cast<uint32_t>(tablet_->stripeStartRow(stripe));
  const auto stripeEnd = stripeStart + stripeRowCount(stripe);
  const auto startRow = std::max(rowRangeLimit.startRow, stripeStart);
  const auto endRow = std::min(rowRangeLimit.endRow, stripeEnd);
  if (startRow >= endRow) {
    return RowRange{};
  }
  return RowRange(startRow - stripeStart, endRow - stripeStart);
}

void NimbleIndexProjector::lookupStripes() {
  velox::CpuWallTimer timer(stats_.lookupTiming);

  const auto result = clusterIndex_->lookup(
      index::IndexLookup::LookupRequest::rangeScan(ctx_.request->keyBounds));

  struct ResolvedRequest {
    uint32_t requestIndex;
    uint32_t startStripe;
    uint32_t endStripe;
    RowRange rowRange;
  };

  std::vector<ResolvedRequest> resolvedRequests;
  resolvedRequests.reserve(ctx_.numRequests);
  for (uint32_t requestIndex = 0; requestIndex < ctx_.numRequests;
       ++requestIndex) {
    const auto& ranges = result[requestIndex];
    if (ranges.empty()) {
      continue;
    }
    NIMBLE_CHECK_EQ(ranges.size(), 1, "Expected single row range per lookup");
    const auto& range = ranges[0];
    NIMBLE_CHECK(!range.empty());

    const uint32_t startStripe = tablet_->rowToStripe(range.startRow);
    const uint32_t endStripe = tablet_->rowToStripe(range.endRow - 1) + 1;
    NIMBLE_CHECK_LE(endStripe, numStripes_);
    NIMBLE_CHECK_LT(startStripe, endStripe);

    resolvedRequests.push_back({requestIndex, startStripe, endStripe, range});
  }

  if (resolvedRequests.empty()) {
    return;
  }

  // Collect one entry per (request, stripe) pair, then group. Everything here
  // is O(entries) -- a few hundred -- where a span-indexed CSR would zero-fill
  // and prefix-sum the whole min..max stripe span, which for scattered probes
  // is three orders of magnitude larger than the number of stripes with
  // ranges.
  auto& scratch = ctx_.resolvedStripesScratch;
  scratch.clear();
  for (const auto& request : resolvedRequests) {
    for (uint32_t stripe = request.startStripe; stripe < request.endStripe;
         ++stripe) {
      scratch.push_back(
          {stripe,
           StripeRange{
               request.requestIndex,
               stripeRowRange(stripe, request.rowRange)}});
    }
  }
  // Ordering on (stripe, request) is a total order, so an unstable sort still
  // leaves the ranges within a stripe in request order, matching the
  // write-cursor fill this replaces.
  std::sort(
      scratch.begin(), scratch.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.tabletStripeIndex != rhs.tabletStripeIndex) {
          return lhs.tabletStripeIndex < rhs.tabletStripeIndex;
        }
        return lhs.range.requestIndex < rhs.range.requestIndex;
      });

  auto& stripeRanges = ctx_.stripeRanges;
  stripeRanges.ranges.reserve(scratch.size());
  stripeRanges.offsets.push_back(0);
  for (size_t i = 0; i < scratch.size(); ++i) {
    if (i == 0) {
      stripeRanges.resolvedStripes.push_back(scratch[i].tabletStripeIndex);
    } else if (
        scratch[i].tabletStripeIndex != scratch[i - 1].tabletStripeIndex) {
      stripeRanges.offsets.push_back(static_cast<uint32_t>(i));
      stripeRanges.resolvedStripes.push_back(scratch[i].tabletStripeIndex);
    }
    stripeRanges.ranges.push_back(scratch[i].range);
  }
  stripeRanges.offsets.push_back(static_cast<uint32_t>(scratch.size()));
  NIMBLE_CHECK_EQ(
      stripeRanges.offsets.size(),
      stripeRanges.resolvedStripes.size() + 1,
      "Stripe range offsets must have one entry per resolved stripe plus one");
}

void NimbleIndexProjector::loadStripes() {
  const auto numPlannedStripes = ctx_.plan.stripeIndices.size();
  if (numPlannedStripes == 0) {
    return;
  }
  velox::CpuWallTimer timer(stats_.scanTiming);

  uint32_t totalStreams = 0;
  for (const auto numStreams : ctx_.plan.numStreams) {
    totalStreams += numStreams;
  }
  dataInput_->reserve(totalStreams);

  const auto numProjectedStreams = projection_->streamOffsets.size();
  ctx_.dataInputIndices.resize(numPlannedStripes * numProjectedStreams);
  for (size_t stripeOffset = 0; stripeOffset < numPlannedStripes;
       ++stripeOffset) {
    dataInput_->startGroup();
    const auto dataInputBase = stripeOffset * numProjectedStreams;
    const auto streams = stripeProjectedStreams(stripeOffset);
    const auto stripeFileOffset = ctx_.plan.stripeFileOffsets[stripeOffset];
    for (size_t streamIndex = 0; streamIndex < streams.size(); ++streamIndex) {
      const auto& stream = streams[streamIndex];
      if (stream.size == 0) {
        continue;
      }
      ctx_.dataInputIndices[dataInputBase + streamIndex] = dataInput_->enqueue(
          velox::common::Region{stripeFileOffset + stream.offset, stream.size});
    }
  }
  ctx_.dataHandle = dataInput_->load();
}

NimbleIndexProjector::Result NimbleIndexProjector::processStripes() {
  velox::CpuWallTimer timer(stats_.projectionTiming);
  Result result;
  result.responses.resize(ctx_.numRequests);
  ctx_.packedStripes.resize(ctx_.plan.stripeIndices.size());
  const auto estimatedSliceOutputBytes = prepareStripePackRanges();
  const bool hasSlicedStripes = estimatedSliceOutputBytes > 0;
  if (hasSlicedStripes) {
    ctx_.sliceOutputBuffer =
        std::make_unique<Buffer>(*pool_, estimatedSliceOutputBytes);
    ctx_.sliceOutputChunks = std::make_shared<std::vector<velox::BufferPtr>>();
  }

  for (size_t i = 0; i < ctx_.plan.stripeIndices.size(); ++i) {
    ++stats_.numReadStripes;
    ctx_.packedStripes[i] = packStripe(i, ctx_.stripePackRanges[i]);
  }
  if (hasSlicedStripes) {
    finalizeSliceOutputBuffer();
  }
  setResumeKeys(result);
  buildResult(result);
  return result;
}

uint64_t NimbleIndexProjector::prepareStripePackRanges() {
  uint64_t estimatedSlicedBytes{0};
  ctx_.stripePackRanges.resize(ctx_.plan.stripeIndices.size());
  for (size_t stripeOffset{0}; stripeOffset < ctx_.plan.stripeIndices.size();
       ++stripeOffset) {
    const RowRange stripeRange{0, ctx_.plan.numRows[stripeOffset]};
    const auto packRange = stripeRowRangeToPack(stripeOffset);
    ctx_.stripePackRanges[stripeOffset] = packRange;
    if (packRange != stripeRange) {
      // Full-stripe packing reuses loaded stream views; only sliced stripes
      // need caller-owned output buffer capacity.
      estimatedSlicedBytes +=
          estimateSlicedStripeBytes(stripeOffset, packRange);
    }
  }
  return estimatedSlicedBytes;
}

uint64_t NimbleIndexProjector::estimateSlicedStripeBytes(
    size_t stripeOffset,
    const RowRange& packRange) const {
  NIMBLE_CHECK_LT(
      stripeOffset,
      ctx_.plan.stripeIndices.size(),
      "Stripe offset is out of range");
  const auto numRows = ctx_.plan.numRows[stripeOffset];
  NIMBLE_CHECK_GT(numRows, 0, "Stripe must contain rows");
  NIMBLE_CHECK_LE(
      packRange.numRows(), numRows, "Pack range cannot exceed stripe rows");
  const auto scaledBytes = static_cast<uint64_t>(std::ceil(
      static_cast<double>(ctx_.plan.projectedBytes[stripeOffset]) *
      packRange.numRows() / numRows));
  // Sliced stream payload bytes scale with rows, but per-stream metadata and
  // output-buffer allocation granularity need extra room.
  const auto estimatedStreamOverheadBytes =
      static_cast<uint64_t>(ctx_.plan.numStreams[stripeOffset]) * 32 + 4096;
  return std::max(
             scaledBytes + scaledBytes / 4,
             ctx_.plan.projectedBytes[stripeOffset] / 2) +
      estimatedStreamOverheadBytes;
}

Buffer& NimbleIndexProjector::sliceOutputBuffer() {
  NIMBLE_CHECK_NOT_NULL(
      ctx_.sliceOutputBuffer, "Sliced output buffer must be initialized");
  return *ctx_.sliceOutputBuffer;
}

void NimbleIndexProjector::finalizeSliceOutputBuffer() {
  NIMBLE_CHECK_NOT_NULL(
      ctx_.sliceOutputBuffer, "Sliced output buffer must be initialized");
  NIMBLE_CHECK_NOT_NULL(
      ctx_.sliceOutputChunks, "Sliced output chunk owner must be initialized");
  // Partial-stripe IOBuf nodes already hold shared_ptr copies to this vector.
  // Populate it now so those nodes keep the transferred Buffer chunks alive,
  // then drop the context's reference.
  *ctx_.sliceOutputChunks = ctx_.sliceOutputBuffer->transferBuffers();
  ctx_.sliceOutputBuffer.reset();
  ctx_.sliceOutputChunks.reset();
}

uint64_t NimbleIndexProjector::appendStripePlan(
    uint32_t stripeIndex,
    size_t rangeOffset) {
  auto& plan = ctx_.plan;
  const auto stripeOffset = plan.stripeIndices.size();
  const auto numProjectedStreams = projection_->streamOffsets.size();
  plan.stripeIndices.push_back(stripeIndex);
  plan.numRows.push_back(stripeRowCount(stripeIndex));
  plan.requiresNullBarriers.push_back(false);
  plan.numStreams.push_back(0);
  plan.projectedBytes.push_back(0);
  plan.stripeFileOffsets.push_back(tablet_->stripeOffset(stripeIndex));
  plan.stripeRangeOffsets.push_back(rangeOffset);
  plan.projectedStreams.resize((stripeOffset + 1) * numProjectedStreams);

  const auto stripeId = tablet_->stripeIdentifier(stripeIndex);
  auto projectedStreams = std::span{plan.projectedStreams}.subspan(
      stripeOffset * numProjectedStreams, numProjectedStreams);
  tablet_->streamLocations(
      stripeId, projection_->streamOffsets, projectedStreams);

  uint64_t projectedBytes{0};
  for (size_t i = 0; i < projectedStreams.size(); ++i) {
    const auto& stream = projectedStreams[i];
    if (stream.size == 0) {
      continue;
    }
    ++plan.numStreams.back();
    projectedBytes += stream.size;
    if (projection_->rowOrFlatMapNullStreams[i]) {
      // A present Row/FlatMap null stream means the slice may carry nulls.
      plan.requiresNullBarriers.back() = true;
    }
  }
  plan.projectedBytes.back() = projectedBytes;
  return projectedBytes;
}

std::span<const StripeGroup::StreamLocation>
NimbleIndexProjector::stripeProjectedStreams(size_t stripeOffset) const {
  const auto numProjectedStreams = projection_->streamOffsets.size();
  const auto projectedStreamsOffset = stripeOffset * numProjectedStreams;
  NIMBLE_CHECK_LE(
      projectedStreamsOffset + numProjectedStreams,
      ctx_.plan.projectedStreams.size(),
      "Stripe projected stream range exceeds plan storage");
  return std::span{ctx_.plan.projectedStreams}.subspan(
      projectedStreamsOffset, numProjectedStreams);
}

std::span<const NimbleIndexProjector::StripeRange>
NimbleIndexProjector::plannedStripeRanges(size_t stripeOffset) const {
  NIMBLE_CHECK_LT(
      stripeOffset,
      ctx_.plan.stripeIndices.size(),
      "Stripe offset out of range");
  NIMBLE_CHECK_LT(
      stripeOffset + 1,
      ctx_.plan.stripeRangeOffsets.size(),
      "Stripe range offsets must include a final sentinel");
  const auto beginOffset = ctx_.plan.stripeRangeOffsets[stripeOffset];
  const auto endOffset = ctx_.plan.stripeRangeOffsets[stripeOffset + 1];
  NIMBLE_CHECK_LE(
      endOffset,
      ctx_.plan.stripeRanges.size(),
      "Stripe request range exceeds plan storage");
  return std::span{ctx_.plan.stripeRanges}.subspan(
      beginOffset, endOffset - beginOffset);
}

RowRange NimbleIndexProjector::stripeRowRangeToPack(size_t stripeOffset) const {
  const RowRange stripeRange{0, ctx_.plan.numRows[stripeOffset]};
  if (ctx_.options->maxOverfetchRowsRatio >= 1.0) {
    return stripeRange;
  }

  uint32_t startRow = ctx_.plan.numRows[stripeOffset];
  uint32_t endRow = 0;
  for (const auto& range : plannedStripeRanges(stripeOffset)) {
    startRow = std::min(startRow, range.rowRange.startRow);
    endRow = std::max(endRow, range.rowRange.endRow);
  }
  NIMBLE_CHECK_LT(
      startRow, endRow, "Planned stripe must have non-empty ranges");
  const RowRange requestedRange{startRow, endRow};
  if (requestedRange == stripeRange) {
    return stripeRange;
  }

  const auto overfetchRows =
      ctx_.plan.numRows[stripeOffset] - requestedRange.numRows();
  const auto overfetchRowsRatio =
      static_cast<double>(overfetchRows) / ctx_.plan.numRows[stripeOffset];
  return overfetchRowsRatio > ctx_.options->maxOverfetchRowsRatio
      ? requestedRange
      : stripeRange;
}

NimbleIndexProjector::PackedStripe NimbleIndexProjector::packStripe(
    size_t stripeOffset,
    const RowRange& packRange) {
  const auto numProjectedStreams = projection_->streamOffsets.size();
  auto& packScratch = ctx_.packScratch;
  SCOPE_EXIT {
    packScratch.clear();
  };
  packScratch.reserve(numProjectedStreams);
  const RowRange stripeRange{0, ctx_.plan.numRows[stripeOffset]};
  if (packRange == stripeRange) {
    return packFullStripe(stripeOffset);
  }
  ++stats_.numSlicedStripes;
  return packPartialStripe(stripeOffset, packRange);
}

NimbleIndexProjector::PackedStripe NimbleIndexProjector::packFullStripe(
    size_t stripeOffset) {
  const auto numProjectedStreams = projection_->streamOffsets.size();
  auto& packScratch = ctx_.packScratch;
  const auto& loadedStreams =
      collectStripeStreamViews(stripeOffset, /*resolveCanonicalStreams=*/true);
  const auto numRows = ctx_.plan.numRows[stripeOffset];
  const RowRange stripeRange{0, numRows};

  // Each unique stream is wrapped zero-copy into the output chain. Streams that
  // are physically contiguous in the DataInput read buffer are merged into a
  // single IOBuf node — fewer nodes means cheaper appendToChain here and
  // cheaper cloneAsValue per request in buildResult. Every node takes ownership
  // of its own refcounted DataInput::Handle on the read buffer, so the whole
  // chain is managed (folly::IOBuf::isManaged()) and self-contained: each node
  // keeps the loaded data alive independently, so no node dangles if it is
  // separated from its siblings.
  std::unique_ptr<folly::IOBuf> chain;
  const char* runData = nullptr;
  uint64_t runLength = 0;
  const auto flushRun = [&]() {
    if (runData == nullptr) {
      return;
    }
    // TODO: each run allocates a fresh DataInput::Handle (a heap shared_ptr
    // copy pinning the read buffer). If this allocation shows up in CPU
    // profiling, use a slab cache or reuse a single Handle/buffer across the
    // chain's nodes.
    auto node = folly::IOBuf::takeOwnership(
        const_cast<char*>(runData),
        runLength,
        freeDataHandle,
        new DataInput::Handle(ctx_.dataHandle));
    if (chain == nullptr) {
      chain = std::move(node);
    } else {
      chain->appendToChain(std::move(node));
    }
    runData = nullptr;
    runLength = 0;
  };

  uint32_t bodyOffset{0};
  auto& canonicalStreamSizeIndices = packScratch.canonicalStreamSizeIndices;
  canonicalStreamSizeIndices.assign(numProjectedStreams, std::nullopt);
  for (const auto projectedIndex : loadedStreams.presentIndices) {
    const auto streamData = loadedStreams.streams[projectedIndex];
    NIMBLE_CHECK_GT(streamData.size(), 0, "Projected stream must not be empty");
    packScratch.streamIds.emplace_back(static_cast<uint32_t>(projectedIndex));

    const auto canonicalProjectedIndex =
        *loadedStreams.canonicalIndices[projectedIndex];
    if (canonicalProjectedIndex != projectedIndex) {
      NIMBLE_CHECK(
          canonicalStreamSizeIndices[canonicalProjectedIndex].has_value(),
          "Duplicate stream must refer to an earlier stream in the stripe");
      packScratch.streamSizeIndices.emplace_back(
          *canonicalStreamSizeIndices[canonicalProjectedIndex]);
      canonicalStreamSizeIndices[projectedIndex] =
          canonicalStreamSizeIndices[canonicalProjectedIndex];
      continue;
    }
    const auto streamSize = static_cast<uint32_t>(streamData.size());
    const auto streamSizeIndex =
        static_cast<uint32_t>(packScratch.uniqueStreamSizes.size());
    canonicalStreamSizeIndices[projectedIndex] = streamSizeIndex;
    packScratch.streamSizeIndices.emplace_back(streamSizeIndex);
    packScratch.uniqueStreamSizes.emplace_back(streamSize);
    bodyOffset += streamSize;

    if (runData != nullptr && streamData.data() == runData + runLength) {
      // Contiguous in the read buffer: extend the current run rather than
      // adding another IOBuf node.
      runLength += streamData.size();
    } else {
      flushRun();
      runData = streamData.data();
      runLength = streamData.size();
    }
  }
  flushRun();

  const auto estimatedTrailerSize = estimateTabletTrailerSize(
      packScratch.streamIds.size(), packScratch.uniqueStreamSizes.size());
  // The trailer is a small standalone node with an estimated upper-bound
  // capacity. createCombined() keeps the IOBuf object and byte storage in one
  // allocation, avoiding the extra backing-buffer allocation that create()
  // would use.
  auto trailer = folly::IOBuf::createCombined(estimatedTrailerSize);
  IOBufAppender trailerBuf{*trailer};
  writeTabletTrailer(
      packScratch.streamIds,
      packScratch.streamSizeIndices,
      packScratch.uniqueStreamSizes,
      trailerBuf);

  NIMBLE_CHECK_NOT_NULL(chain);
  const auto trailerSize = trailer->length();
  chain->appendToChain(std::move(trailer));

  // projectedBytes counts logical projected stream bytes, including duplicate
  // slots. bodyOffset is the physical body size after deduplication.
  const size_t bodySize = bodyOffset + trailerSize;
  stats_.numReadRows += numRows;
  stats_.numOutputBytes += bodySize;
  return {
      .body = std::move(*chain),
      .rowRange = stripeRange,
      .requiresNullBarrier = ctx_.plan.requiresNullBarriers[stripeOffset],
      .streamHasChunkHeader = true,
  };
}

NimbleIndexProjector::StripeStreamViews&
NimbleIndexProjector::collectStripeStreamViews(
    size_t stripeOffset,
    bool resolveCanonicalStreams) {
  const auto projectedStreams = stripeProjectedStreams(stripeOffset);
  const auto numProjectedStreams = projection_->streamOffsets.size();
  auto& loadedStreams = ctx_.streamViewsScratch;
  loadedStreams.clear();
  loadedStreams.streams.assign(numProjectedStreams, {});
  if (resolveCanonicalStreams) {
    loadedStreams.presentIndices.reserve(ctx_.plan.numStreams[stripeOffset]);
    loadedStreams.canonicalIndices.resize(numProjectedStreams);
  }

  uint32_t streamEnqueueBase{0};
  const auto dataInputBase = stripeOffset * numProjectedStreams;
  for (size_t i = 0; i < numProjectedStreams; ++i) {
    if (projectedStreams[i].size == 0) {
      continue;
    }
    NIMBLE_CHECK(
        ctx_.dataInputIndices[dataInputBase + i].has_value(),
        "Present projected stream must have an enqueued data input index");
    const auto enqueueIndex = *ctx_.dataInputIndices[dataInputBase + i];
    if (loadedStreams.presentIndices.empty()) {
      streamEnqueueBase = enqueueIndex;
    }
    const auto& streamLocation = projectedStreams[i];
    const auto& bufferRef = dataInput_->bufferRef(enqueueIndex);
    NIMBLE_CHECK_EQ(
        bufferRef.length,
        streamLocation.size,
        "Loaded stream length must match projected stream length");
    loadedStreams.streams[i] =
        std::string_view(bufferRef.data, bufferRef.length);
    if (!resolveCanonicalStreams) {
      continue;
    }
    loadedStreams.presentIndices.emplace_back(i);

    size_t canonicalProjectedIndex = i;
    if (bufferRef.canonicalIndex != enqueueIndex) {
      NIMBLE_CHECK_GE(
          bufferRef.canonicalIndex,
          streamEnqueueBase,
          "Duplicate stream must refer to the current stripe");
      const auto canonicalIndexOffset =
          bufferRef.canonicalIndex - streamEnqueueBase;
      NIMBLE_CHECK_LT(
          canonicalIndexOffset,
          loadedStreams.presentIndices.size(),
          "Duplicate stream must refer to an earlier stream in the stripe");
      canonicalProjectedIndex =
          loadedStreams.presentIndices[canonicalIndexOffset];
    }
    loadedStreams.canonicalIndices[i] = canonicalProjectedIndex;

    if (canonicalProjectedIndex != i) {
      loadedStreams.streams[i] = loadedStreams.streams[canonicalProjectedIndex];
      continue;
    }
  }
  return loadedStreams;
}

NimbleIndexProjector::PackedStripe NimbleIndexProjector::packPartialStripe(
    size_t stripeOffset,
    const RowRange& packRange) {
  NIMBLE_CHECK_GT(
      packRange.numRows(), 0, "Partial stripe range must not be empty");
  const auto& loadedStreams =
      collectStripeStreamViews(stripeOffset, /*resolveCanonicalStreams=*/false);
  auto sliced = streamSlicer_->slice(
      loadedStreams.streams,
      packRange.startRow,
      packRange.numRows(),
      &sliceOutputBuffer());

  size_t bodySize{0};
  auto& packScratch = ctx_.packScratch;
  for (uint32_t projectedIndex{0}; projectedIndex < sliced.streams.size();
       ++projectedIndex) {
    const auto stream = sliced.streams[projectedIndex];
    if (stream.empty()) {
      continue;
    }
    packScratch.streamIds.emplace_back(projectedIndex);
    packScratch.streamSizeIndices.emplace_back(
        static_cast<uint32_t>(packScratch.uniqueStreamSizes.size()));
    packScratch.uniqueStreamSizes.emplace_back(
        static_cast<uint32_t>(stream.size()));
    bodySize += stream.size();
  }
  NIMBLE_CHECK_GT(
      bodySize, 0, "Non-empty partial stripe must produce a sliced body");

  auto slicedBody = serde::StreamSlicer::takeOwnershipAsIOBuf(
      sliced.streams, std::shared_ptr<const void>{ctx_.sliceOutputChunks});
  auto chain = std::make_unique<folly::IOBuf>(std::move(slicedBody));
  NIMBLE_CHECK_EQ(
      bodySize,
      chain->computeChainDataLength(),
      "Sliced stream sizes must match the sliced body");

  const auto estimatedTrailerSize = estimateTabletTrailerSize(
      packScratch.streamIds.size(), packScratch.uniqueStreamSizes.size());
  auto trailer = folly::IOBuf::createCombined(estimatedTrailerSize);
  IOBufAppender trailerBuffer{*trailer};
  writeTabletTrailer(
      packScratch.streamIds,
      packScratch.streamSizeIndices,
      packScratch.uniqueStreamSizes,
      trailerBuffer);
  bodySize += trailer->length();
  chain->appendToChain(std::move(trailer));

  stats_.numReadRows += ctx_.plan.numRows[stripeOffset];
  stats_.numOutputBytes += bodySize;
  return {
      .body = std::move(*chain),
      .rowRange = packRange,
      .requiresNullBarrier = sliced.requiresNullBarrier,
      .streamHasChunkHeader = false,
  };
}

void NimbleIndexProjector::setResumeKeys(Result& result) {
  if (!ctx_.options->needResumeKey) {
    return;
  }
  for (size_t i = 0; i < result.responses.size(); ++i) {
    if (ctx_.resumeKeys[i].has_value()) {
      result.responses[i].resumeKey = ctx_.resumeKeys[i];
    }
  }
  if (!ctx_.plan.truncated) {
    return;
  }
  const auto lastStripeOffset = ctx_.plan.stripeIndices.size() - 1;
  const auto stripeIndex = ctx_.plan.stripeIndices[lastStripeOffset];
  const auto stripeRanges = plannedStripeRanges(lastStripeOffset);

  // No next stripe in the range map — all mapped requests end at this stripe.
  // resolvedStripes is ascending, so the successor is found by binary search;
  // a miss means the next stripe carries no ranges and nothing continues.
  const auto nextStripe = stripeIndex + 1;
  const auto& resolvedStripes = ctx_.stripeRanges.resolvedStripes;
  const auto nextStripeIt = std::lower_bound(
      resolvedStripes.begin(), resolvedStripes.end(), nextStripe);
  if (nextStripeIt == resolvedStripes.end() || *nextStripeIt != nextStripe) {
    return;
  }

  // No next stripe in the file — all requests are fully satisfied.
  const auto stripeStartRow =
      static_cast<uint32_t>(tablet_->stripeStartRow(stripeIndex));
  const auto nextStripeStartRow = stripeStartRow + stripeRowCount(stripeIndex);
  if (nextStripeStartRow >= tablet_->tabletRowCount()) {
    return;
  }

  auto resumeKey = clusterIndex_->keyAtRow(nextStripeStartRow);
  const auto nextRanges = ctx_.stripeRanges.getRanges(
      static_cast<uint32_t>(nextStripeIt - resolvedStripes.begin()));

  for (const auto& request : stripeRanges) {
    auto& response = result.responses[request.requestIndex];
    if (!response.resumeKey.has_value() &&
        std::any_of(
            nextRanges.begin(), nextRanges.end(), [&](const auto& range) {
              return range.requestIndex == request.requestIndex;
            })) {
      response.resumeKey = resumeKey;
    }
  }

  // For requests that were never started (no stripe ranges in any plan),
  // set resume key to their original lower key so the caller can retry.
  for (size_t i = 0; i < result.responses.size(); ++i) {
    auto& response = result.responses[i];
    if (!ctx_.hasStripeRanges[i] && !response.resumeKey.has_value()) {
      const auto& keyBounds = ctx_.request->keyBounds[i];
      NIMBLE_CHECK(
          keyBounds.lowerKey.has_value(),
          "Request {} has no lowerKey: unbounded lower requests start from "
          "stripe 0 and should have been processed before truncation",
          i);
      response.resumeKey = *keyBounds.lowerKey;
    }
  }
}

namespace {

/// Builds a kTablet IOBuf chain: [header] -> [shared body+trailer].
folly::IOBuf assembleStripeSlice(
    uint32_t numRows,
    bool requiresNullBarrier,
    bool streamEncodingUsesVarintRowCount,
    bool streamHasChunkHeader,
    RowRange rowRange,
    folly::IOBuf body,
    std::optional<std::string> resumeKey = std::nullopt) {
  serde::TabletChunkHeader header{
      .rowCount = numRows,
      .requiresNullBarrier = requiresNullBarrier,
      .streamEncodingUsesVarintRowCount = streamEncodingUsesVarintRowCount,
      .streamHasChunkHeader = streamHasChunkHeader,
      .rowRange = rowRange,
      .resumeKey = std::move(resumeKey),
  };
  auto serializedHeader =
      std::make_unique<folly::IOBuf>(serde::createTabletChunkHeader(header));
  serializedHeader->appendToChain(
      std::make_unique<folly::IOBuf>(std::move(body)));
  return std::move(*serializedHeader);
}

} // namespace

void NimbleIndexProjector::buildResult(Result& result) {
  // Build per-response slice counts for reserve.
  auto& sliceCounts = ctx_.sliceCounts;
  sliceCounts.assign(ctx_.numRequests, 0);
  for (size_t i = 0; i < ctx_.plan.stripeIndices.size(); ++i) {
    for (const auto& range : plannedStripeRanges(i)) {
      ++sliceCounts[range.requestIndex];
    }
  }
  for (size_t i = 0; i < ctx_.numRequests; ++i) {
    result.responses[i].slices.reserve(sliceCounts[i]);
  }

  // Track per-response how many slices have been emitted so we know when
  // we're at the last one (for embedding the resume key).
  auto& emittedSlices = ctx_.emittedSlices;
  emittedSlices.assign(ctx_.numRequests, 0);

  for (size_t i = 0; i < ctx_.plan.stripeIndices.size(); ++i) {
    auto& packedStripe = ctx_.packedStripes[i];

    for (const auto& range : plannedStripeRanges(i)) {
      NIMBLE_CHECK(!range.rowRange.empty());
      NIMBLE_CHECK(
          packedStripe.rowRange.contains(range.rowRange),
          "Packed stripe range {} must contain request range {}",
          packedStripe.rowRange.toString(),
          range.rowRange.toString());
      const RowRange packedRelativeRange{
          range.rowRange.startRow - packedStripe.rowRange.startRow,
          range.rowRange.endRow - packedStripe.rowRange.startRow};
      stats_.numProjectedRows += range.rowRange.numRows();
      ++emittedSlices[range.requestIndex];
      const bool isLastSlice =
          emittedSlices[range.requestIndex] == sliceCounts[range.requestIndex];
      auto& response = result.responses[range.requestIndex];
      response.slices.emplace_back(assembleStripeSlice(
          packedStripe.rowRange.numRows(),
          packedStripe.requiresNullBarrier,
          tablet_->properties().compactRowCountEncoding(),
          packedStripe.streamHasChunkHeader,
          packedRelativeRange,
          packedStripe.body.cloneAsValue(),
          isLastSlice ? response.resumeKey : std::nullopt));
    }
  }
}

} // namespace facebook::nimble

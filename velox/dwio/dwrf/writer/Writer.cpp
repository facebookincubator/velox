/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/dwio/dwrf/writer/Writer.h"

#include <folly/ScopeGuard.h>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/utils/ProtoUtils.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/dwrf/writer/LayoutPlanner.h"

namespace facebook::velox::dwrf {

EncodingIter::EncodingIter(
    const proto::StripeFooter& footer,
    const std::vector<proto::StripeEncryptionGroup>& encryptionGroups,
    int32_t encryptionGroupIndex,
    google::protobuf::RepeatedPtrField<proto::ColumnEncoding>::const_iterator
        current,
    google::protobuf::RepeatedPtrField<proto::ColumnEncoding>::const_iterator
        currentEnd)
    : footer_{footer},
      encryptionGroups_{encryptionGroups},
      encryptionGroupIndex_{encryptionGroupIndex},
      current_{std::move(current)},
      currentEnd_{std::move(currentEnd)} {
  // Move to the end or a valid position.
  if (current_ == currentEnd_) {
    next();
  }
}

/* static */ EncodingIter EncodingIter::begin(
    const proto::StripeFooter& footer,
    const std::vector<proto::StripeEncryptionGroup>& encryptionGroups) {
  return EncodingIter{
      footer,
      encryptionGroups,
      -1,
      footer.encoding().begin(),
      footer.encoding().end()};
}

/* static */ EncodingIter EncodingIter::end(
    const proto::StripeFooter& footer,
    const std::vector<proto::StripeEncryptionGroup>& encryptionGroups) {
  if (encryptionGroups.empty()) {
    const auto& footerEncodings = footer.encoding();
    return EncodingIter{
        footer,
        encryptionGroups,
        -1,
        footerEncodings.end(),
        footerEncodings.end()};
  }
  const auto lastEncryptionGroupIndex =
      folly::to<int32_t>(encryptionGroups.size()) - 1;
  const auto& lastEncryptionGroupEncodings = encryptionGroups.back().encoding();
  return EncodingIter{
      footer,
      encryptionGroups,
      lastEncryptionGroupIndex,
      lastEncryptionGroupEncodings.end(),
      lastEncryptionGroupEncodings.end()};
}

void EncodingIter::next() {
  // The check is needed for the initial position
  // if footer is empty but encryption groups are not.
  if (current_ != currentEnd_) {
    ++current_;
  }
  // Get to the absolute end or a valid position.
  while (current_ == currentEnd_) {
    if (encryptionGroupIndex_ == encryptionGroups_.size() - 1) {
      return;
    }
    const auto& encryptionGroup = encryptionGroups_.at(++encryptionGroupIndex_);
    current_ = encryptionGroup.encoding().begin();
    currentEnd_ = encryptionGroup.encoding().end();
  }
  return;
}

EncodingIter& EncodingIter::operator++() {
  next();
  return *this;
}

EncodingIter EncodingIter::operator++(int) {
  auto current = *this;
  next();
  return current;
}

bool EncodingIter::operator==(const EncodingIter& other) const {
  return current_ == other.current_;
}

bool EncodingIter::operator!=(const EncodingIter& other) const {
  return current_ != other.current_;
}

EncodingIter::reference EncodingIter::operator*() const {
  return *current_;
}

EncodingIter::pointer EncodingIter::operator->() const {
  return &(*current_);
}

EncodingManager::EncodingManager(
    const encryption::EncryptionHandler& encryptionHandler)
    : encryptionHandler_{encryptionHandler} {
  initEncryptionGroups();
}

proto::ColumnEncoding& EncodingManager::addEncodingToFooter(uint32_t nodeId) {
  if (encryptionHandler_.isEncrypted(nodeId)) {
    auto index = encryptionHandler_.getEncryptionGroupIndex(nodeId);
    return *encryptionGroups_.at(index).add_encoding();
  } else {
    return *footer_.add_encoding();
  }
}

proto::Stream* EncodingManager::addStreamToFooter(
    uint32_t nodeId,
    uint32_t& currentIndex) {
  if (encryptionHandler_.isEncrypted(nodeId)) {
    currentIndex = encryptionHandler_.getEncryptionGroupIndex(nodeId);
    return encryptionGroups_.at(currentIndex).add_streams();
  } else {
    currentIndex = std::numeric_limits<uint32_t>::max();
    return footer_.add_streams();
  }
}

std::string* EncodingManager::addEncryptionGroupToFooter() {
  return footer_.add_encryptiongroups();
}

proto::StripeEncryptionGroup EncodingManager::getEncryptionGroup(uint32_t i) {
  return encryptionGroups_.at(i);
}

const proto::StripeFooter& EncodingManager::getFooter() const {
  return footer_;
}

EncodingIter EncodingManager::begin() const {
  return EncodingIter::begin(footer_, encryptionGroups_);
}

EncodingIter EncodingManager::end() const {
  return EncodingIter::end(footer_, encryptionGroups_);
}

void EncodingManager::initEncryptionGroups() {
  // initialize encryption groups
  if (encryptionHandler_.isEncrypted()) {
    auto count = encryptionHandler_.getEncryptionGroupCount();
    // We use uint32_t::max to represent non-encrypted when adding streams, so
    // make sure number of encryption groups is smaller than that
    DWIO_ENSURE_LT(count, std::numeric_limits<uint32_t>::max());
    encryptionGroups_.resize(count);
  }
}

namespace {
// We currently use previous stripe raw size as the proxy for the expected
// stripe raw size. For the first stripe, we are more conservative about
// flush overhead memory unless we know otherwise, e.g. perhaps from
// encoding DB work.
// This can be fitted linearly just like flush overhead or perhaps
// figured out from the schema.
// This can be simplified with Slice::estimateMemory().
size_t estimateNextWriteSize(const WriterContext& context, size_t numRows) {
  // This is 0 for first slice. We are assuming reasonable input for now.
  return folly::to<size_t>(ceil(context.getAverageRowSize() * numRows));
}
} // namespace

void Writer::write(const VectorPtr& slice) {
  auto& context = getContext();
  size_t offset = 0;
  // Calculate length increment based on linear projection of micro batch size.
  // Total length is capped later.
  const auto& sliceMemoryEstimate = slice->retainedSize();
  const auto& sliceRowCount = slice->size();
  const size_t lengthIncrement = std::max<size_t>(
      1UL,
      sliceMemoryEstimate > 0 ? folly::to<size_t>(std::floor(
                                    1.0 * context.rawDataSizePerBatch /
                                    sliceMemoryEstimate * sliceRowCount))
                              : folly::to<size_t>(sliceRowCount));
  if (UNLIKELY(
          sliceMemoryEstimate == 0 ||
          sliceMemoryEstimate > context.rawDataSizePerBatch)) {
    LOG(WARNING) << fmt::format(
        "Unpopulated or huge vector memory estimate! Micro batch size {} rows. "
        "Slice memory estimate {} bytes. Batching threshold {} bytes.",
        lengthIncrement,
        sliceMemoryEstimate,
        context.rawDataSizePerBatch);
  }
  while (offset < sliceRowCount) {
    size_t length = lengthIncrement;
    if (context.isIndexEnabled) {
      length =
          std::min<size_t>(length, context.indexStride - context.indexRowCount);
    }

    length = std::min(length, sliceRowCount - offset);
    VELOX_CHECK_GT(length, 0);
    bool flushDecision = shouldFlush(context, length);
    if (flushDecision) {
      // Try abandoning inefficiency dictionary encodings early
      // and see if we can delay the flush.
      if (writer_->tryAbandonDictionaries(false)) {
        flushDecision = shouldFlush(context, length);
      }
      if (flushDecision) {
        flush();
      }
    }

    auto rawSize =
        writer_->write(slice, common::Ranges::of(offset, offset + length));
    offset += length;
    getContext().incRawSize(rawSize);

    if (context.isIndexEnabled &&
        context.indexRowCount >= context.indexStride) {
      createRowIndexEntry();
    }
  }
}

uint64_t Writer::flushTimeMemoryUsageEstimate(
    const WriterContext& context,
    size_t nextWriteSize) const {
  return context.getTotalMemoryUsage() +
      context.getEstimatedStripeSize(nextWriteSize) +
      context.getEstimatedFlushOverhead(context.stripeRawSize + nextWriteSize);
}

// Flush if we cannot take one additional slice/stride based on current stripe
// raw size.
bool Writer::overMemoryBudget(const WriterContext& context, size_t writeLength)
    const {
  size_t nextWriteSize = estimateNextWriteSize(context, writeLength);
  return flushTimeMemoryUsageEstimate(context, nextWriteSize) >
      context.getMemoryBudget();
}

dwio::common::StripeProgress getStripeProgress(const WriterContext& context) {
  return dwio::common::StripeProgress{
      .stripeIndex = context.stripeIndex,
      .stripeRowCount = context.stripeRowCount,
      .totalMemoryUsage = context.getTotalMemoryUsage(),
      .stripeSizeEstimate = std::max(
          context.getEstimatedStripeSize(context.stripeRawSize),
          // The stripe size estimate is only more accurate from the second
          // stripe onward because it uses past stripe states in heuristics.
          // We need to additionally bound it with output stream size based
          // estimate for the first stripe.
          context.stripeIndex == 0 ? context.getEstimatedOutputStreamSize()
                                   : 0)};
}

// Writer will flush to make more memory if the incoming stride would make
// it exceed memory budget with the default flush policy. Other policies
// can intentionally throw and expect the application to retry.
//
// The current approach is to assume that the customer passes in slices of
// similar sizes, perhaps even bounded by a configurable amount. We then
// compute the soft_cap = hard_budget - expected_increment_per_slice, and
// compare that against a dynamically determined flush_overhead +
// current_total_usage and try to flush preemptively after writing each
// slice/stride to bring the current memory usage below the soft_cap again.
//
// Using less memory than the soft_cap ensures being able to
// write a new slice/stride, unless the slice/stride is drastically bigger
// than the previous ones.
bool Writer::shouldFlush(const WriterContext& context, size_t nextWriteLength) {
  // TODO: ideally, the heurstics to keep under the memory budget thing
  // shouldn't be a first class concept for writer and should be wrapped in
  // flush policy or some other abstraction for pluggability of the additional
  // logic.

  // If we are hitting memory budget before satisfying flush criteria, try
  // entering low memory mode to work with less memory-intensive encodings.
  bool overBudget = overMemoryBudget(context, nextWriteLength);
  bool stripeProgressDecision =
      flushPolicy_->shouldFlush(getStripeProgress(context));
  auto dwrfFlushDecision = flushPolicy_->shouldFlushDictionary(
      stripeProgressDecision, overBudget, context);

  if (UNLIKELY(dwrfFlushDecision == FlushDecision::ABANDON_DICTIONARY)) {
    enterLowMemoryMode();
    // Recalculate memory usage due to encoding switch.
    // We can still be over budget either due to not having enough budget to
    // switch encoding or switching encoding not reducing memory footprint
    // enough.
    overBudget = overMemoryBudget(context, nextWriteLength);
    stripeProgressDecision =
        flushPolicy_->shouldFlush(getStripeProgress(context));
  }

  const bool decision = overBudget || stripeProgressDecision ||
      dwrfFlushDecision == FlushDecision::FLUSH_DICTIONARY;
  if (decision) {
    VLOG(1) << fmt::format(
        "overMemoryBudget: {}, dictionaryMemUsage: {}, outputStreamSize: {}, generalMemUsage: {}, estimatedStripeSize: {}",
        overBudget,
        context.getMemoryUsage(MemoryUsageCategory::DICTIONARY)
            .getCurrentBytes(),
        context.getEstimatedOutputStreamSize(),
        context.getMemoryUsage(MemoryUsageCategory::GENERAL).getCurrentBytes(),
        context.getEstimatedStripeSize(context.stripeRawSize));
  }
  return decision;
}

void Writer::setLowMemoryMode() {
  getContext().setLowMemoryMode();
}

// Low memory allows for the writer to write the same data with a lower
// memory budget.
// Currently this method is only called locally to switch encoding if
// we couldn't meet flush criteria without exceeding memory budget.
// NOTE: switching encoding is not a good mitigation for immediate memory
// pressure because the switch consumes even more memory than a flush.
void Writer::enterLowMemoryMode() {
  auto& context = getContext();
  // Until we have capability to abandon dictionary after the first
  // stripe, do nothing and rely solely on flush to comply with budget.
  if (UNLIKELY(context.checkLowMemoryMode() && context.stripeIndex == 0)) {
    // Idempotent call to switch to less memory intensive encodings.
    writer_->tryAbandonDictionaries(true);
  }
}

void Writer::flushStripe(bool close) {
  auto& context = getContext();
  int64_t preFlushStreamMemoryUsage =
      context.getMemoryUsage(MemoryUsageCategory::OUTPUT_STREAM)
          .getCurrentBytes();
  if (context.stripeRowCount == 0) {
    return;
  }

  dwio::common::MetricsLog::StripeFlushMetrics metrics;
  metrics.writerVersion =
      writerVersionToString(context.getConfig(Config::WRITER_VERSION));
  metrics.outputStreamMemoryEstimate = context.getEstimatedOutputStreamSize();
  metrics.stripeSizeEstimate =
      context.getEstimatedStripeSize(context.stripeRawSize);

  if (context.isIndexEnabled && context.indexRowCount > 0) {
    createRowIndexEntry();
  }

  auto preFlushMem = context.getTotalMemoryUsage();

  auto& handler = context.getEncryptionHandler();
  EncodingManager encodingManager{handler};

  writer_->flush([&](uint32_t nodeId) -> proto::ColumnEncoding& {
    return encodingManager.addEncodingToFooter(nodeId);
  });

  // Collects the memory increment from flushing data to output streams.
  auto flushOverhead =
      context.getMemoryUsage(MemoryUsageCategory::OUTPUT_STREAM)
          .getCurrentBytes() -
      preFlushStreamMemoryUsage;
  context.recordFlushOverhead(flushOverhead);
  metrics.flushOverhead = flushOverhead;

  auto postFlushMem = context.getTotalMemoryUsage();

  auto& sink = getSink();
  auto stripeOffset = sink.size();

  uint32_t lastIndex = 0;
  uint64_t offset = 0;
  auto addStream = [&](const DwrfStreamIdentifier& stream, const auto& out) {
    proto::Stream* s;
    uint32_t currentIndex;
    auto nodeId = stream.encodingKey().node;
    s = encodingManager.addStreamToFooter(nodeId, currentIndex);

    // set offset only when needed, ie. when offset of current stream cannot be
    // calculated based on offset and length of previous stream. In that case,
    // it must be that current stream and previous stream doesn't belong to same
    // encryption group or neither are encrypted. So the logic is simplified to
    // check if group index are the same for current and previous stream
    if (offset > 0 && lastIndex != currentIndex) {
      s->set_offset(offset);
    }
    lastIndex = currentIndex;

    // Jolly/Presto readers can't read streams bigger than 2GB.
    validateStreamSize(stream, out.size());

    s->set_kind(static_cast<proto::Stream_Kind>(stream.kind()));
    s->set_node(nodeId);
    s->set_column(stream.column());
    s->set_sequence(stream.encodingKey().sequence);
    s->set_length(out.size());
    s->set_usevints(context.getConfig(Config::USE_VINTS));
    offset += out.size();

    context.recordPhysicalSize(stream, out.size());
  };

  // TODO: T45025996 Discard all empty streams at flush time.
  // deals with streams
  uint64_t indexLength = 0;
  sink.setMode(WriterSink::Mode::Index);
  auto planner = layoutPlannerFactory_(getStreamList(context), encodingManager);
  planner->plan();
  planner->iterateIndexStreams([&](auto& streamId, auto& content) {
    DWIO_ENSURE(
        isIndexStream(streamId.kind()),
        "unexpected stream kind ",
        streamId.kind());
    indexLength += content.size();
    addStream(streamId, content);
    sink.addBuffers(content);
  });

  uint64_t dataLength = 0;
  sink.setMode(WriterSink::Mode::Data);
  planner->iterateDataStreams([&](auto& streamId, auto& content) {
    DWIO_ENSURE(
        !isIndexStream(streamId.kind()),
        "unexpected stream kind ",
        streamId.kind());
    dataLength += content.size();
    addStream(streamId, content);
    sink.addBuffers(content);
  });
  DWIO_ENSURE_GT(dataLength, 0);
  metrics.stripeSize = dataLength;

  if (handler.isEncrypted()) {
    // fill encryption metadata
    for (uint32_t i = 0; i < handler.getEncryptionGroupCount(); ++i) {
      auto group = encodingManager.addEncryptionGroupToFooter();
      writeProtoAsString(
          *group,
          encodingManager.getEncryptionGroup(i),
          std::addressof(handler.getEncryptionProviderByIndex(i)));
    }
  }

  // flush footer
  uint64_t footerOffset = sink.size();
  DWIO_ENSURE_EQ(footerOffset, stripeOffset + dataLength + indexLength);

  sink.setMode(WriterSink::Mode::Footer);
  writeProto(encodingManager.getFooter());
  sink.setMode(WriterSink::Mode::None);

  auto& stripe = addStripeInfo();
  stripe.set_offset(stripeOffset);
  stripe.set_indexlength(indexLength);
  stripe.set_datalength(dataLength);
  stripe.set_footerlength(sink.size() - footerOffset);

  // set encryption key metadata
  if (handler.isEncrypted() && context.stripeIndex == 0) {
    for (uint32_t i = 0; i < handler.getEncryptionGroupCount(); ++i) {
      *stripe.add_keymetadata() =
          handler.getEncryptionProviderByIndex(i).getKey();
    }
  }

  context.recordAverageRowSize();
  context.recordCompressionRatio(dataLength);

  auto totalMemoryUsage = context.getTotalMemoryUsage();
  metrics.limit = totalMemoryUsage;
  metrics.availableMemory = context.getMemoryBudget() - totalMemoryUsage;

  auto& dictionaryDataMemoryUsage =
      context.getMemoryUsage(MemoryUsageCategory::DICTIONARY);
  metrics.dictionaryMemory = dictionaryDataMemoryUsage.getCurrentBytes();
  // TODO: what does this try to capture?
  metrics.maxDictSize = dictionaryDataMemoryUsage.getMaxBytes();

  metrics.stripeIndex = context.stripeIndex;
  metrics.rawStripeSize = context.stripeRawSize;
  metrics.rowsInStripe = context.stripeRowCount;
  metrics.compressionRatio = context.getCompressionRatio();
  metrics.flushOverheadRatio = context.getFlushOverheadRatio();
  metrics.averageRowSize = context.getAverageRowSize();
  metrics.groupSize = 0;
  metrics.close = close;

  LOG(INFO) << fmt::format(
      "Stripe {}: Flush overhead = {}, data length = {}, pre flush mem = {}, post flush mem = {}. Closing = {}",
      metrics.stripeIndex,
      metrics.flushOverhead,
      metrics.stripeSize,
      preFlushMem,
      postFlushMem,
      metrics.close);
  // Add flush overhead and other ratio logging.
  context.metricLogger->logStripeFlush(metrics);

  // prepare for next stripe
  context.nextStripe();
  writer_->reset();
}

void Writer::flushInternal(bool close) {
  auto& context = getContext();
  auto& footer = getFooter();
  auto& sink = getSink();
  {
    CpuWallTimer timer{context.flushTiming};
    flushStripe(close);

    // if this is the last stripe, add footer
    if (close) {
      auto& handler = context.getEncryptionHandler();
      std::vector<std::vector<proto::FileStatistics>> stats;
      proto::Encryption* encryption = nullptr;

      // initialize encryption related metadata only when there is data written
      if (handler.isEncrypted() && footer.stripes_size() > 0) {
        auto count = handler.getEncryptionGroupCount();
        stats.resize(count);
        encryption = footer.mutable_encryption();
        encryption->set_keyprovider(
            encryption::toProto(handler.getKeyProviderType()));
        for (uint32_t i = 0; i < count; ++i) {
          encryption->add_encryptiongroups();
        }
      }

      std::optional<uint32_t> lastRoot;
      std::unordered_map<proto::ColumnStatistics*, proto::ColumnStatistics*>
          statsMap;
      writer_->writeFileStats([&](uint32_t nodeId) -> proto::ColumnStatistics& {
        auto entry = footer.add_statistics();
        if (!encryption || !handler.isEncrypted(nodeId)) {
          return *entry;
        }

        auto root = handler.getEncryptionRoot(nodeId);
        auto groupIndex = handler.getEncryptionGroupIndex(nodeId);
        auto& group = stats.at(groupIndex);
        if (!lastRoot || root != lastRoot.value()) {
          // this is a new root, add to the footer, and use a new slot
          group.emplace_back();
          encryption->mutable_encryptiongroups(groupIndex)->add_nodes(root);
        }
        lastRoot = root;
        auto encryptedStats = group.back().add_statistics();
        statsMap[entry] = encryptedStats;
        return *encryptedStats;
      });

#define COPY_STAT(from, to, stat) \
  if (from->has_##stat()) {       \
    to->set_##stat(from->stat()); \
  }

      // fill basic stats
      for (auto& pair : statsMap) {
        COPY_STAT(pair.second, pair.first, numberofvalues);
        COPY_STAT(pair.second, pair.first, hasnull);
        COPY_STAT(pair.second, pair.first, rawsize);
        COPY_STAT(pair.second, pair.first, size);
      }

#undef COPY_STAT

      // set metadata for each encryption group
      if (encryption) {
        for (uint32_t i = 0; i < handler.getEncryptionGroupCount(); ++i) {
          auto group = encryption->mutable_encryptiongroups(i);
          // set stats. No need to set key metadata since it just reused the
          // same key of the first stripe
          for (auto& s : stats.at(i)) {
            writeProtoAsString(
                *group->add_statistics(),
                s,
                std::addressof(handler.getEncryptionProviderByIndex(i)));
          }
        }
      }

      writeFooter(*schema_->type);
    }

    // flush to sink
    sink.flush();
  }

  if (close) {
    context.metricLogger->logFileClose(
        dwio::common::MetricsLog::FileCloseMetrics{
            .writerVersion = writerVersionToString(
                context.getConfig(Config::WRITER_VERSION)),
            .footerLength = footer.contentlength(),
            .fileSize = sink.size(),
            .cacheSize = sink.getCacheSize(),
            .numCacheBlocks = sink.getCacheOffsets().size() - 1,
            .cacheMode = static_cast<int32_t>(sink.getCacheMode()),
            .numOfStripes = context.stripeIndex,
            .rowCount = context.stripeRowCount,
            .rawDataSize = context.stripeRawSize,
            .numOfStreams = context.getStreamCount(),
            .totalMemory = context.getTotalMemoryUsage(),
            .dictionaryMemory =
                context.getMemoryUsage(MemoryUsageCategory::DICTIONARY)
                    .getCurrentBytes(),
            .generalMemory =
                context.getMemoryUsage(MemoryUsageCategory::GENERAL)
                    .getCurrentBytes()});
  }
}

void Writer::flush() {
  flushInternal(false);
}

void Writer::close() {
  auto exitGuard = folly::makeGuard([this]() { flushPolicy_->onClose(); });
  flushInternal(true);
  WriterBase::close();
}

} // namespace facebook::velox::dwrf

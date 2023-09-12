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

dwio::common::StripeProgress getStripeProgress(const WriterContext& context) {
  return dwio::common::StripeProgress{
      .stripeIndex = context.stripeIndex(),
      .stripeRowCount = context.stripeRowCount(),
      .totalMemoryUsage = context.getTotalMemoryUsage(),
      .stripeSizeEstimate = std::max(
          context.getEstimatedStripeSize(context.stripeRawSize()),
          // The stripe size estimate is only more accurate from the second
          // stripe onward because it uses past stripe states in heuristics.
          // We need to additionally bound it with output stream size based
          // estimate for the first stripe.
          context.stripeIndex() == 0 ? context.getEstimatedOutputStreamSize()
                                     : 0)};
}
} // namespace

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool)
    : writerBase_(std::make_unique<WriterBase>(std::move(sink))),
      schema_{dwio::common::TypeWithId::create(options.schema)} {
  VELOX_CHECK(
      !pool->isLeaf(),
      "Memory pool {} for DWRF writer can't be leaf",
      pool->name());
  auto handler =
      (options.encryptionSpec ? encryption::EncryptionHandler::create(
                                    schema_,
                                    *options.encryptionSpec,
                                    options.encrypterFactory.get())
                              : nullptr);
  writerBase_->initContext(options.config, std::move(pool), std::move(handler));
  auto& context = writerBase_->getContext();
  context.buildPhysicalSizeAggregators(*schema_);
  if (options.flushPolicyFactory == nullptr) {
    flushPolicy_ = std::make_unique<DefaultFlushPolicy>(
        context.stripeSizeFlushThreshold(),
        context.dictionarySizeFlushThreshold());
  } else {
    flushPolicy_ = options.flushPolicyFactory();
  }

  if (options.layoutPlannerFactory != nullptr) {
    layoutPlanner_ = options.layoutPlannerFactory(*schema_);
  } else {
    layoutPlanner_ = std::make_unique<LayoutPlanner>(*schema_);
  }

  if (options.columnWriterFactory == nullptr) {
    writer_ = BaseColumnWriter::create(writerBase_->getContext(), *schema_);
  } else {
    writer_ = options.columnWriterFactory(writerBase_->getContext(), *schema_);
  }
}

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options)
    : Writer{
          std::move(sink),
          options,
          options.memoryPool->addAggregateChild(fmt::format(
              "writer_node_{}",
              folly::to<std::string>(folly::Random::rand64())))} {}

void Writer::write(const VectorPtr& input) {
  auto& context = writerBase_->getContext();
  // Calculate length increment based on linear projection of micro batch size.
  // Total length is capped later.
  const auto& estimatedInputMemoryBytes = input->estimateFlatSize();
  const auto inputRowCount = input->size();
  const size_t writeBatchSize = std::max<size_t>(
      1UL,
      estimatedInputMemoryBytes > 0
          ? folly::to<size_t>(std::floor(
                1.0 * context.rawDataSizePerBatch() /
                estimatedInputMemoryBytes * inputRowCount))
          : folly::to<size_t>(inputRowCount));
  if (FOLLY_UNLIKELY(
          estimatedInputMemoryBytes == 0 ||
          estimatedInputMemoryBytes > context.rawDataSizePerBatch())) {
    VLOG(1) << fmt::format(
        "Unpopulated or huge vector memory estimate! Micro write batch size {} rows. "
        "Input vector memory estimate {} bytes. Batching threshold {} bytes.",
        writeBatchSize,
        estimatedInputMemoryBytes,
        context.rawDataSizePerBatch());
  }
  size_t rowOffset = 0;
  while (rowOffset < inputRowCount) {
    size_t numRowsToWrite = writeBatchSize;
    if (context.indexEnabled()) {
      // Do not write cross an index row block.
      numRowsToWrite = std::min<size_t>(
          numRowsToWrite, context.indexStride() - context.indexRowCount());
    }

    numRowsToWrite = std::min(numRowsToWrite, inputRowCount - rowOffset);
    VELOX_CHECK_GT(numRowsToWrite, 0);
    bool doFlush = shouldFlush(context, numRowsToWrite);
    if (doFlush) {
      // Try abandoning inefficiency dictionary encodings early and see if we
      // can delay the flush.
      if (writer_->tryAbandonDictionaries(false)) {
        doFlush = shouldFlush(context, numRowsToWrite);
      }
      if (doFlush) {
        flush();
      }
    }

    const auto rawSize = writer_->write(
        input, common::Ranges::of(rowOffset, rowOffset + numRowsToWrite));
    rowOffset += numRowsToWrite;
    context.incRawSize(rawSize);

    if (context.indexEnabled() &&
        context.indexRowCount() >= context.indexStride()) {
      createRowIndexEntry();
    }
  }
}

uint64_t Writer::flushTimeMemoryUsageEstimate(
    const WriterContext& context,
    size_t nextWriteSize) const {
  return context.getTotalMemoryUsage() +
      context.getEstimatedStripeSize(nextWriteSize) +
      context.getEstimatedFlushOverhead(
          context.stripeRawSize() + nextWriteSize);
}

bool Writer::overMemoryBudget(const WriterContext& context, size_t numRows)
    const {
  // Flush if we cannot take one additional slice/stride based on current stripe
  // raw size.
  const size_t nextWriteSize = estimateNextWriteSize(context, numRows);
  return flushTimeMemoryUsageEstimate(context, nextWriteSize) >
      context.getMemoryBudget();
}

bool Writer::shouldFlush(const WriterContext& context, size_t nextWriteRows) {
  // TODO: ideally, the heurstics to keep under the memory budget thing
  // shouldn't be a first class concept for writer and should be wrapped in
  // flush policy or some other abstraction for pluggability of the additional
  // logic.

  // If we are hitting memory budget before satisfying flush criteria, try
  // entering low memory mode to work with less memory-intensive encodings.
  bool overBudget = overMemoryBudget(context, nextWriteRows);
  bool stripeProgressDecision =
      flushPolicy_->shouldFlush(getStripeProgress(context));
  auto dictionaryFlushDecision = flushPolicy_->shouldFlushDictionary(
      stripeProgressDecision, overBudget, context);

  if (FOLLY_UNLIKELY(
          dictionaryFlushDecision == FlushDecision::ABANDON_DICTIONARY)) {
    enterLowMemoryMode();
    // Recalculate memory usage due to encoding switch.
    // We can still be over budget either due to not having enough budget to
    // switch encoding or switching encoding not reducing memory footprint
    // enough.
    overBudget = overMemoryBudget(context, nextWriteRows);
    stripeProgressDecision =
        flushPolicy_->shouldFlush(getStripeProgress(context));
  }

  const bool shouldFlush = overBudget || stripeProgressDecision ||
      dictionaryFlushDecision == FlushDecision::FLUSH_DICTIONARY;
  if (shouldFlush) {
    VLOG(1) << fmt::format(
        "overMemoryBudget: {}, dictionaryMemUsage: {}, outputStreamSize: {}, generalMemUsage: {}, estimatedStripeSize: {}",
        overBudget,
        context.getMemoryUsage(MemoryUsageCategory::DICTIONARY),
        context.getEstimatedOutputStreamSize(),
        context.getMemoryUsage(MemoryUsageCategory::GENERAL),
        context.getEstimatedStripeSize(context.stripeRawSize()));
  }
  return shouldFlush;
}

void Writer::setLowMemoryMode() {
  writerBase_->getContext().setLowMemoryMode();
}

void Writer::enterLowMemoryMode() {
  auto& context = writerBase_->getContext();
  // Until we have capability to abandon dictionary after the first
  // stripe, do nothing and rely solely on flush to comply with budget.
  if (FOLLY_UNLIKELY(
          context.checkLowMemoryMode() && context.stripeIndex() == 0)) {
    // Idempotent call to switch to less memory intensive encodings.
    writer_->tryAbandonDictionaries(true);
  }
}

void Writer::flushStripe(bool close) {
  auto& context = writerBase_->getContext();
  const int64_t preFlushStreamMemoryUsage =
      context.getMemoryUsage(MemoryUsageCategory::OUTPUT_STREAM);
  if (context.stripeRowCount() == 0) {
    return;
  }

  dwio::common::MetricsLog::StripeFlushMetrics metrics;
  metrics.writerVersion =
      writerVersionToString(context.getConfig(Config::WRITER_VERSION));
  metrics.outputStreamMemoryEstimate = context.getEstimatedOutputStreamSize();
  metrics.stripeSizeEstimate =
      context.getEstimatedStripeSize(context.stripeRawSize());

  if (context.indexEnabled() && context.indexRowCount() > 0) {
    createRowIndexEntry();
  }

  const auto preFlushMem = context.getTotalMemoryUsage();

  const auto& handler = context.getEncryptionHandler();
  EncodingManager encodingManager{handler};

  writer_->flush([&](uint32_t nodeId) -> proto::ColumnEncoding& {
    return encodingManager.addEncodingToFooter(nodeId);
  });

  // Collects the memory increment from flushing data to output streams.
  const auto flushOverhead =
      context.getMemoryUsage(MemoryUsageCategory::OUTPUT_STREAM) -
      preFlushStreamMemoryUsage;
  context.recordFlushOverhead(flushOverhead);
  metrics.flushOverhead = flushOverhead;

  const auto postFlushMem = context.getTotalMemoryUsage();

  auto& sink = writerBase_->getSink();
  auto stripeOffset = sink.size();

  uint32_t lastIndex = 0;
  uint64_t offset = 0;
  const auto addStream = [&](const DwrfStreamIdentifier& stream,
                             const auto& out) {
    uint32_t currentIndex;
    const auto nodeId = stream.encodingKey().node();
    proto::Stream* s = encodingManager.addStreamToFooter(nodeId, currentIndex);

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
    writerBase_->validateStreamSize(stream, out.size());

    s->set_kind(static_cast<proto::Stream_Kind>(stream.kind()));
    s->set_node(nodeId);
    s->set_column(stream.column());
    s->set_sequence(stream.encodingKey().sequence());
    s->set_length(out.size());
    s->set_usevints(context.getConfig(Config::USE_VINTS));
    offset += out.size();

    context.recordPhysicalSize(stream, out.size());
  };

  // TODO: T45025996 Discard all empty streams at flush time.
  // deals with streams
  uint64_t indexLength = 0;
  sink.setMode(WriterSink::Mode::Index);
  auto result = layoutPlanner_->plan(encodingManager, getStreamList(context));
  result.iterateIndexStreams([&](auto& streamId, auto& content) {
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
  result.iterateDataStreams([&](auto& streamId, auto& content) {
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
      writerBase_->writeProtoAsString(
          *group,
          encodingManager.getEncryptionGroup(i),
          std::addressof(handler.getEncryptionProviderByIndex(i)));
    }
  }

  // flush footer
  const uint64_t footerOffset = sink.size();
  DWIO_ENSURE_EQ(footerOffset, stripeOffset + dataLength + indexLength);

  sink.setMode(WriterSink::Mode::Footer);
  writerBase_->writeProto(encodingManager.getFooter());
  sink.setMode(WriterSink::Mode::None);

  auto& stripe = writerBase_->addStripeInfo();
  stripe.set_offset(stripeOffset);
  stripe.set_indexlength(indexLength);
  stripe.set_datalength(dataLength);
  stripe.set_footerlength(sink.size() - footerOffset);

  // set encryption key metadata
  if (handler.isEncrypted() && context.stripeIndex() == 0) {
    for (uint32_t i = 0; i < handler.getEncryptionGroupCount(); ++i) {
      *stripe.add_keymetadata() =
          handler.getEncryptionProviderByIndex(i).getKey();
    }
  }

  context.recordAverageRowSize();
  context.recordCompressionRatio(dataLength);

  const auto totalMemoryUsage = context.getTotalMemoryUsage();
  metrics.limit = totalMemoryUsage;
  metrics.availableMemory = context.getMemoryBudget() - totalMemoryUsage;

  auto& dictionaryPool = context.getMemoryPool(MemoryUsageCategory::DICTIONARY);
  metrics.dictionaryMemory = dictionaryPool.currentBytes();
  // TODO: what does this try to capture?
  metrics.maxDictSize = dictionaryPool.stats().peakBytes;

  metrics.stripeIndex = context.stripeIndex();
  metrics.rawStripeSize = context.stripeRawSize();
  metrics.rowsInStripe = context.stripeRowCount();
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
  context.metricLogger()->logStripeFlush(metrics);

  // prepare for next stripe
  context.nextStripe();
  writer_->reset();
}

void Writer::flushInternal(bool close) {
  auto& context = writerBase_->getContext();
  auto& footer = writerBase_->getFooter();
  auto& sink = writerBase_->getSink();
  {
    CpuWallTimer timer{context.flushTiming()};
    flushStripe(close);

    // if this is the last stripe, add footer
    if (close) {
      const auto& handler = context.getEncryptionHandler();
      std::vector<std::vector<proto::FileStatistics>> stats;
      proto::Encryption* encryption = nullptr;

      // initialize encryption related metadata only when there is data written
      if (handler.isEncrypted() && footer.stripes_size() > 0) {
        const auto count = handler.getEncryptionGroupCount();
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
            writerBase_->writeProtoAsString(
                *group->add_statistics(),
                s,
                std::addressof(handler.getEncryptionProviderByIndex(i)));
          }
        }
      }

      writerBase_->writeFooter(*schema_->type());
    }

    // flush to sink
    sink.flush();
  }

  if (close) {
    context.metricLogger()->logFileClose(
        dwio::common::MetricsLog::FileCloseMetrics{
            .writerVersion = writerVersionToString(
                context.getConfig(Config::WRITER_VERSION)),
            .footerLength = footer.contentlength(),
            .fileSize = sink.size(),
            .cacheSize = sink.getCacheSize(),
            .numCacheBlocks = sink.getCacheOffsets().size() - 1,
            .cacheMode = static_cast<int32_t>(sink.getCacheMode()),
            .numOfStripes = context.stripeIndex(),
            .rowCount = context.stripeRowCount(),
            .rawDataSize = context.stripeRawSize(),
            .numOfStreams = context.getStreamCount(),
            .totalMemory = context.getTotalMemoryUsage(),
            .dictionaryMemory =
                context.getMemoryUsage(MemoryUsageCategory::DICTIONARY),
            .generalMemory =
                context.getMemoryUsage(MemoryUsageCategory::GENERAL)});
  }
}

void Writer::flush() {
  flushInternal(false);
}

void Writer::close() {
  auto exitGuard = folly::makeGuard([this]() { flushPolicy_->onClose(); });
  flushInternal(true);
  writerBase_->close();
}

void Writer::abort() {
  // NOTE: we need to reset column writer as all its dependent objects (e.g.
  // writer context) will be reset by writer base abort.
  writer_.reset();
  writerBase_->abort();
}

dwrf::WriterOptions getDwrfOptions(const dwio::common::WriterOptions& options) {
  std::map<std::string, std::string> configs;
  if (options.compressionKind.has_value()) {
    configs.emplace(
        Config::COMPRESSION.configKey(),
        std::to_string(options.compressionKind.value()));
  }
  dwrf::WriterOptions dwrfOptions;
  dwrfOptions.config = Config::fromMap(configs);
  dwrfOptions.schema = options.schema;
  dwrfOptions.memoryPool = options.memoryPool;
  return dwrfOptions;
}

std::unique_ptr<dwio::common::Writer> DwrfWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options) {
  auto dwrfOptions = getDwrfOptions(options);
  return std::make_unique<Writer>(std::move(sink), dwrfOptions);
}

void registerDwrfWriterFactory() {
  dwio::common::registerWriterFactory(std::make_shared<DwrfWriterFactory>());
}

void unregisterDwrfWriterFactory() {
  dwio::common::unregisterWriterFactory(dwio::common::FileFormat::DWRF);
}

} // namespace facebook::velox::dwrf

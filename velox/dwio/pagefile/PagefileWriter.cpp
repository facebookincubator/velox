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

#include "velox/dwio/pagefile/PagefileWriter.h"

#include <folly/ScopeGuard.h>

#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/utils/ProtoUtils.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/dwrf/writer/LayoutPlanner.h"
#include "velox/exec/MemoryReclaimer.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::pagefile {

namespace {

#define NON_RECLAIMABLE_SECTION_CHECK() \
  VELOX_CHECK(nonReclaimableSection_ == nullptr || *nonReclaimableSection_);
} // namespace

PagefileWriter::PagefileWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool) :
      sink_(std::move(sink)),
      schema_{dwio::common::TypeWithId::create(options.schema)},
      spillConfig_{options.spillConfig},
      nonReclaimableSection_(options.nonReclaimableSection),
      pool_(pool) {
  VELOX_CHECK(
      spillConfig_ == nullptr || nonReclaimableSection_ != nullptr,
      "nonReclaimableSection_ must be set if writer memory reclaim is enabled");

  setMemoryReclaimers(pool);
  setState(State::kRunning);
}

void PagefileWriter::setMemoryReclaimers(
    const std::shared_ptr<memory::MemoryPool>& pool) {
  VELOX_CHECK(
      !pool->isLeaf(),
      "The root memory pool for pagefile writer can't be leaf: {}",
      pool->name());
  VELOX_CHECK_NULL(pool->reclaimer());

  if ((pool->parent() == nullptr) || (pool->parent()->reclaimer() == nullptr)) {
    return;
  }

  pool->setReclaimer(MemoryReclaimer::create(this));
}

void PagefileWriter::write(const VectorPtr& input) {
  checkRunning();
  NON_RECLAIMABLE_SECTION_CHECK();

  //auto& context = writerBase_->getContext();
  // Calculate length increment based on linear projection of micro batch size.
  // Total length is capped later.
  const auto& estimatedInputMemoryBytes = input->estimateFlatSize();
  const auto inputRowCount = input->size();
  /*const size_t writeBatchSize = std::max<size_t>(
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
  }*/

  size_t rowOffset = 0;
  size_t numRowsToWrite = inputRowCount;
  while (rowOffset < inputRowCount) {

    numRowsToWrite = std::min(numRowsToWrite, inputRowCount - rowOffset);
    VELOX_CHECK_GT(numRowsToWrite, 0);

    ensureWriteFits(
        estimatedInputMemoryBytes * numRowsToWrite / inputRowCount,
        numRowsToWrite);

    TestValue::adjust("facebook::velox::dwrf::Writer::write", this);

    flush();

    //const auto rawSize = writer_->write(
    //    input, common::Ranges::of(rowOffset, rowOffset + numRowsToWrite));
    rowOffset += numRowsToWrite;
    //context.incRawSize(rawSize);
  }
}

bool PagefileWriter::canReclaim() const {
  return spillConfig_ != nullptr;
}

void PagefileWriter::ensureWriteFits(size_t appendBytes, size_t appendRows) {
  if (!canReclaim()) {
    return;
  }

  const uint64_t totalMemoryUsage = pool_->usedBytes();
  if (totalMemoryUsage == 0) {
    return;
  }

  // Allows the memory arbitrator to reclaim memory from this file writer if the
  // memory reservation below has triggered memory arbitration.
  memory::ReclaimableSectionGuard reclaimGuard(nonReclaimableSection_);

  // TODO(Aditi) : Compute the estimate of the bytes needed to write using
  // max(bytes_needed_by_input, bytes_needed_for_output);
  //const size_t estimatedAppendMemoryBytes =
  //    std::max(appendBytes, context.estimateNextWriteSize(appendRows));
  const size_t estimatedAppendMemoryBytes = appendBytes;
  const double estimatedMemoryGrowthRatio =
      (double)estimatedAppendMemoryBytes / totalMemoryUsage;
  if (!maybeReserveMemory(estimatedMemoryGrowthRatio)) {
    return;
  }
}

bool PagefileWriter::maybeReserveMemory(
    double estimatedMemoryGrowthRatio) {
  VELOX_CHECK(!*nonReclaimableSection_);
  VELOX_CHECK(canReclaim());
  auto& pool = *pool_;
  const uint64_t availableReservation = pool.availableReservation();
  const uint64_t usedReservationBytes = pool.usedBytes();
  const uint64_t minReservationBytes =
      usedReservationBytes * spillConfig_->minSpillableReservationPct / 100;
  const uint64_t estimatedIncrementBytes =
      usedReservationBytes * estimatedMemoryGrowthRatio;
  if ((availableReservation > minReservationBytes) &&
      (availableReservation > 2 * estimatedIncrementBytes)) {
    return true;
  }

  const uint64_t bytesToReserve = std::max(
      estimatedIncrementBytes * 2,
      usedReservationBytes * spillConfig_->spillableReservationGrowthPct / 100);
  return pool.maybeReserve(bytesToReserve);
}

void PagefileWriter::releaseMemory() {
  if (!canReclaim()) {
    return;
  }
  pool_->release();
}

void PagefileWriter::flushInternal(bool close) {
  TestValue::adjust("facebook::velox::pagefile::PagefileWriter::flushInternal", this);
  auto exitGuard = folly::makeGuard([this]() { releaseMemory(); });

  //auto& context = writerBase_->getContext();
  //auto& footer = writerBase_->getFooter();
  auto& sink = *sink_;
  {
    CpuWallTimer timer{context.flushTiming()};
    flushStripe(close);


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


      writerBase_->writeFooter(*schema_->type());
    }

  // flush to sink
  sink.flush();

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

void PagefileWriter::flush() {
  checkRunning();
  flushInternal(false);
}

void PagefileWriter::close() {
  checkRunning();
  auto exitGuard = folly::makeGuard([this]() {
    flushPolicy_->onClose();
    setState(State::kClosed);
  });
  flushInternal(true);
  writerBase_->close();
}

void PagefileWriter::abort() {
  checkRunning();
  auto exitGuard = folly::makeGuard([this]() { setState(State::kAborted); });
  // NOTE: we need to reset column writer as all its dependent objects (e.g.
  // writer context) will be reset by writer base abort.
  writer_.reset();
  writerBase_->abort();
}

std::unique_ptr<memory::MemoryReclaimer> PagefileWriter::MemoryReclaimer::create(
    PagefileWriter* writer) {
  return std::unique_ptr<memory::MemoryReclaimer>(
      new PagefileWriter::MemoryReclaimer(writer));
}

bool PagefileWriter::MemoryReclaimer::reclaimableBytes(
    const memory::MemoryPool& /*unused*/,
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = 0;
  if (!writer_->canReclaim()) {
    return false;
  }
  // What to reclaim though ?
  //const uint64_t memoryUsage = writer_->getContext().getTotalMemoryUsage();
  //if (memoryUsage < writer_->spillConfig_->writerFlushThresholdSize) {
  //  return false;
  //}
  //reclaimableBytes = memoryUsage;
  return true;
}

uint64_t PagefileWriter::MemoryReclaimer::reclaim(
    memory::MemoryPool* pool,
    uint64_t targetBytes,
    uint64_t /*unused*/,
    memory::MemoryReclaimer::Stats& stats) {
  if (!writer_->canReclaim()) {
    return 0;
  }

  if (*writer_->nonReclaimableSection_) {
    RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
    LOG(WARNING)
        << "Can't reclaim from pagefile writer which is under non-reclaimable section: "
        << pool->name();
    ++stats.numNonReclaimableAttempts;
    return 0;
  }
  if (!writer_->isRunning()) {
    LOG(WARNING) << "Can't reclaim from a not running pagefile writer: "
                 << pool->name() << ", state: " << writer_->state();
    ++stats.numNonReclaimableAttempts;
    return 0;
  }
  //const uint64_t memoryUsage = writer_->getContext().getTotalMemoryUsage();
  //if (memoryUsage < writer_->spillConfig_->writerFlushThresholdSize) {
  //  RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
  //  LOG(WARNING)
  //      << "Can't reclaim memory from dwrf writer pool " << pool->name()
  //      << " which doesn't have sufficient memory to flush, writer memory usage: "
  //      << succinctBytes(memoryUsage) << ", writer flush memory threshold: "
  //      << succinctBytes(writer_->spillConfig_->writerFlushThresholdSize);
  //  ++stats.numNonReclaimableAttempts;
  //  return 0;
  //}

  auto reclaimBytes = memory::MemoryReclaimer::run(
      [&]() {
        int64_t reclaimedBytes{0};
        memory::ScopedReclaimedBytesRecorder recorder(pool, &reclaimedBytes);
        writer_->flushInternal(false);
        return reclaimedBytes;
      },
      stats);
  return reclaimBytes;
}

std::unique_ptr<dwio::common::Writer> PagefileWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options) {
  return std::make_unique<PagefileWriter>(
      std::move(sink), options,
      options.memoryPool->addAggregateChild(fmt::format(
          "{}.pagefile.{}",
          options.memoryPool->name(),
          folly::to<std::string>(folly::Random::rand64()))));
}

void registerPagefileWriterFactory() {
  dwio::common::registerWriterFactory(std::make_shared<PagefileWriterFactory>());
}

void unregisterPagefileWriterFactory() {
  dwio::common::unregisterWriterFactory(dwio::common::FileFormat::PAGEFILE);
}

} // namespace facebook::velox::pagefile

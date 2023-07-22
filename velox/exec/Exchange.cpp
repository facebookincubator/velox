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
#include "velox/exec/Exchange.h"
#include <velox/common/base/Exceptions.h>
#include <velox/common/memory/Memory.h>
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {

SerializedPage::SerializedPage(
    std::unique_ptr<folly::IOBuf> iobuf,
    std::function<void(folly::IOBuf&)> onDestructionCb)
    : iobuf_(std::move(iobuf)),
      iobufBytes_(chainBytes(*iobuf_.get())),
      onDestructionCb_(onDestructionCb) {
  VELOX_CHECK_NOT_NULL(iobuf_);
  for (auto& buf : *iobuf_) {
    int32_t bufSize = buf.size();
    ranges_.push_back(ByteRange{
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(buf.data())),
        bufSize,
        0});
  }
}

SerializedPage::~SerializedPage() {
  if (onDestructionCb_) {
    onDestructionCb_(*iobuf_.get());
  }
}

void SerializedPage::prepareStreamForDeserialize(ByteStream* input) {
  input->resetInput(std::move(ranges_));
}

std::shared_ptr<ExchangeSource> ExchangeSource::create(
    const std::string& taskId,
    int destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* pool) {
  for (auto& factory : factories()) {
    auto result = factory(taskId, destination, queue, pool);
    if (result) {
      return result;
    }
  }
  VELOX_FAIL("No ExchangeSource factory matches {}", taskId);
}

// static
std::vector<ExchangeSource::Factory>& ExchangeSource::factories() {
  static std::vector<Factory> factories;
  return factories;
}

namespace {
class LocalExchangeSource : public ExchangeSource {
 public:
  LocalExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool)
      : ExchangeSource(taskId, destination, queue, pool) {}

  bool shouldRequestLocked() override {
    if (atEnd_) {
      return false;
    }
    return !requestPending_.exchange(true);
  }

  void request() override {
    request(kMaxBytes);
  }

  void request(uint64_t bytes) override {
    auto buffers = PartitionedOutputBufferManager::getInstance().lock();
    VELOX_CHECK_NOT_NULL(buffers, "invalid PartitionedOutputBufferManager");
    VELOX_CHECK(requestPending_);
    auto requestedSequence = sequence_;
    auto self = shared_from_this();
    buffers->getData(
        taskId_,
        destination_,
        bytes,
        sequence_,
        // Since this lambda may outlive 'this', we need to capture a
        // shared_ptr to the current object (self).
        [self, requestedSequence, buffers, this](
            std::vector<std::unique_ptr<folly::IOBuf>> data, int64_t sequence) {
          if (requestedSequence > sequence) {
            VLOG(2) << "Receives earlier sequence than requested: task "
                    << taskId_ << ", destination " << destination_
                    << ", requested " << sequence << ", received "
                    << requestedSequence;
            int64_t nExtra = requestedSequence - sequence;
            VELOX_CHECK(nExtra < data.size());
            data.erase(data.begin(), data.begin() + nExtra);
            sequence = requestedSequence;
          }
          std::vector<std::unique_ptr<SerializedPage>> pages;
          bool atEnd = false;
          for (auto& inputPage : data) {
            if (!inputPage) {
              atEnd = true;
              // Keep looping, there could be extra end markers.
              continue;
            }
            inputPage->unshare();
            pages.push_back(
                std::make_unique<SerializedPage>(std::move(inputPage)));
            inputPage = nullptr;
          }
          numPages_ += pages.size();
          int64_t ackSequence = kNoReply;
          {
            std::vector<ContinuePromise> promises;
            {
              std::lock_guard<std::mutex> l(queue_->mutex());
              uint64_t bytes = 0;
              if (!queue_->enableFlowControl()) {
                requestPending_ = false;
              }
              for (auto& page : pages) {
                bytes += page->size();
                queue_->enqueueLocked(std::move(page), promises);
              }
              queue_->recordReplyLocked(bytes);
              if (atEnd) {
                requestPending_ = false;
                queue_->enqueueLocked(nullptr, promises);
                atEnd_ = true;
              }
              if (pages.size() > 0) {
                ackSequence = sequence_ = sequence + pages.size();
              }
            }
            for (auto& promise : promises) {
              promise.setValue();
            }
          }

          if (atEnd_) {
            buffers->deleteResults(taskId_, destination_);
          }

          // Outside of queue mutex. Sends a delete results, ack or new
          // request to self and requests more sources if there is space in the
          // queue.
          if (auto bytes = queue_->requestIfDue(self)) {
            self->request(bytes);
            return;
          }

          if (!atEnd_ && ackSequence != kNoReply) {
            buffers->acknowledge(taskId_, destination_, ackSequence);
          }
        });
  }

  void close() override {
    auto buffers = PartitionedOutputBufferManager::getInstance().lock();
    buffers->deleteResults(taskId_, destination_);
  }

  void deleteResults() {
    auto buffers = PartitionedOutputBufferManager::getInstance().lock();
    buffers->deleteResults(taskId_, destination_);
  }

  folly::F14FastMap<std::string, int64_t> stats() const override {
    return {{"localExchangeSource.numPages", numPages_}};
  }

 private:
  static constexpr uint64_t kMaxBytes = 32 * 1024 * 1024; // 32 MB

  // Records the total number of pages fetched from sources.
  int64_t numPages_{0};
};

std::unique_ptr<ExchangeSource> createLocalExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* pool) {
  if (strncmp(taskId.c_str(), "local://", 8) == 0) {
    return std::make_unique<LocalExchangeSource>(
        taskId, destination, std::move(queue), pool);
  }
  return nullptr;
}

} // namespace

uint64_t ExchangeQueue::expectedSerializedPageSize() const {
  // Initially, max size / number of sources. Later, average of received sizes.
  constexpr int32_t kMinExpectedSize =
      PartitionedOutputBuffer::kMinDestinationSize;
  if (!numSources_) {
    return kMinExpectedSize;
  }
  // If not all sources have arrived, assume 100 sources. Otherwise,
  // use numSources_
  auto numSourcesGuess =
      noMoreSources_ ? numSources_ : std::max(numSources_, 100);
  // Initially, expected page size is maxBytes_ / receivedPages_.
  // As we get more data, we can make the guess as receivedBytes_ /
  // receivedPages_. This formula ensures the above.
  return std::max<int64_t>(
      kMinExpectedSize,
      (maxBytes_ + receivedBytes_) / (receivedPages_ + numSourcesGuess));
}

int64_t ExchangeQueue::requestIfDue(
    const std::shared_ptr<ExchangeSource>& replySource) {
  auto client = client_.lock();
  if (client) {
    return client->requestIfDue(replySource);
  }
  return false;
}

bool ExchangeQueue::enableFlowControl() const {
  if (auto client = client_.lock()) {
    return client->enableFlowControl();
  }
  return false;
}

void ExchangeClient::addRemoteTaskId(const std::string& taskId) {
  std::shared_ptr<ExchangeSource> toRequest;
  std::shared_ptr<ExchangeSource> toClose;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    bool duplicate = !taskIds_.insert(taskId).second;
    if (duplicate) {
      // Do not add sources twice. Presto protocol may add duplicate sources
      // and the task updates have no guarantees of arriving in order.
      return;
    }

    std::shared_ptr<ExchangeSource> source;
    try {
      source = ExchangeSource::create(taskId, destination_, queue_, pool_);
    } catch (const VeloxException& e) {
      throw;
    } catch (const std::exception& e) {
      // Task ID can be very long. Truncate to 256 characters.
      VELOX_FAIL(
          "Failed to create ExchangeSource: {}. Task ID: {}.",
          e.what(),
          taskId.substr(0, 126));
    }

    if (closed_) {
      toClose = std::move(source);
    } else {
      sources_.push_back(source);
      queue_->addSourceLocked();
      toRequest = source;
    }
  }

  // Outside of lock.
  if (toClose) {
    return toClose->close();
  }
  if (!toRequest) {
    return;
  }
  std::shared_ptr<ExchangeSource> empty;
  if (requestIfDue(empty)) {
    return;
  }
  if (enableFlowControl()) {
    return;
  }

  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (!toRequest->shouldRequestLocked()) {
      toRequest = nullptr;
    }
  }
  if (toRequest) {
    toRequest->request();
  }
}

void ExchangeClient::noMoreRemoteTasks() {
  queue_->noMoreSources();
}

void ExchangeClient::close() {
  std::vector<std::shared_ptr<ExchangeSource>> sources;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return;
    }
    closed_ = true;
    sources = std::move(sources_);
  }

  // Outside of mutex.
  for (auto& source : sources) {
    source->close();
  }
  queue_->close();
}

folly::F14FastMap<std::string, RuntimeMetric> ExchangeClient::stats() const {
  folly::F14FastMap<std::string, RuntimeMetric> stats;
  std::lock_guard<std::mutex> l(queue_->mutex());
  for (const auto& source : sources_) {
    for (const auto& [name, value] : source->stats()) {
      stats[name].addValue(value);
    }
  }
  stats["peakBytes"] =
      RuntimeMetric(queue_->peakBytes(), RuntimeCounter::Unit::kBytes);
  stats["nothingRequestable"] = RuntimeMetric(numNothingRequestable_);
  return stats;
}

std::unique_ptr<SerializedPage> ExchangeClient::next(
    bool* atEnd,
    ContinueFuture* future) {
  std::unique_ptr<SerializedPage> page;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    *atEnd = false;
    page = queue_->dequeueLocked(atEnd, future);
    if (*atEnd) {
      return page;
    }
    if (page && queue_->totalBytes() > queue_->minBytes()) {
      return page;
    }
  }

  std::shared_ptr<ExchangeSource> empty;
  if (requestIfDue(empty)) {
    return page;
  }
  if (enableFlowControl()) {
    return page;
  }

  std::vector<std::shared_ptr<ExchangeSource>> toRequest;
  // There is space for more data, send requests to sources with no pending
  // request.
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    for (auto& source : sources_) {
      if (source->shouldRequestLocked()) {
        toRequest.push_back(source);
      }
    }
  }

  // Outside of lock
  for (auto& source : toRequest) {
    source->request();
  }
  return page;
}

int64_t ExchangeClient::requestIfDue(
    const std::shared_ptr<ExchangeSource>& replySource) {
  if (!enableFlowControl_) {
    return 0;
  }
  // Requests data from as many next sources as will fit based on average
  // response size and space in the queue. If 'replySource' is given and not at
  // end, sends a data request to it if it fits in the new  requests batch and
  // an ack if it had data but was not rerequested.

  std::vector<std::shared_ptr<ExchangeSource>> toRequest;
  int64_t requestSize = 0;
  int64_t requestedBytes = 0;
  bool isDirectRerequest = false;
  bool replySourcePending = false;
  if (replySource && !replySource->isAtEnd()) {
    VELOX_CHECK(replySource->isPending());
    replySourcePending = true;
  }
  bool requestMore = false;
  {
    bool fullRequestBatch = false;
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return 0;
    }
    if (queue_->numPending() == (replySourcePending ? 1 : 0)) {
      // If there are no sources pending or if replySource is the only one
      // pending and jus got a reply, there can be no pending bytes expected for
      // the queue.
      queue_->clearExpectedBytes();
    }
    int64_t space =
        queue_->maxBytes() - queue_->totalBytes() - queue_->expectedBytes();
    int64_t unit = queue_->expectedSerializedPageSize();
    int64_t numToRequest = std::max<int64_t>(0, space / unit);
    int32_t numRequestable =
        sources_.size() - queue_->numCompleted() - queue_->numPending();
    int32_t numRequestableCheck = 0;
    for (const auto& source : sources_) {
      numRequestableCheck += source->isRequestable();
    }
    VELOX_CHECK_EQ(numRequestable, numRequestableCheck);
    // Note that space can be negative. Make requestSize no less than minimum
    // reply.
    requestSize = numRequestable
        ? bits::roundUp(std::max<int64_t>(1, space / numRequestable), unit)
        : unit;
    // No new requests if there is no space and there is something already
    // received or expected.
    fullRequestBatch = numToRequest == 0 &&
        (queue_->totalBytes() > 0 || queue_->expectedBytes() > 0);
    for (auto i = 0; !fullRequestBatch && i < sources_.size(); ++i) {
      if (++nextSourceIndex_ >= sources_.size()) {
        nextSourceIndex_ = 0;
      }
      auto& source = sources_[nextSourceIndex_];
      // 'replySource' is pending, so will not be added to 'toRequest.
      if (source->shouldRequestLocked()) {
        requestedBytes += requestSize;
        toRequest.push_back(source);
        if (toRequest.size() >= numToRequest || requestedBytes > space) {
          // If we expect a full buffer from sources other than the one that
          // replied, we ack the reply instead of direct rerequest.
          fullRequestBatch = true;
          break;
        }
      }
    }
    if (replySource && !replySource->isAtEnd()) {
      // After requesting others, check if replySource 1. must send an ack, 2.
      // must send a rerequest, 3. becomes not pending.
      if (!fullRequestBatch) {
        // There is space after requesting other sources. Send a new request in
        // the place of ack. Saves a message.
        isDirectRerequest = true;
        requestedBytes += requestSize;
        requestMore = true;
      } else {
        // replySource will send an ack, and becomes not pending.
        --queue_->numPending();
        replySource->clearPendingLocked();
      }
    }
    if (!toRequest.empty()) {
      // If one source is already pending, substract it from the new request
      // count.
      queue_->recordRequestLocked(
          toRequest.size() - isDirectRerequest, requestedBytes);
    } else {
      ++numNothingRequestable_;
    }
  }

  // Outside of lock
  for (auto& source : toRequest) {
    source->request(requestSize);
  }
  return requestMore ? requestSize : 0;
}

ExchangeClient::~ExchangeClient() {
  close();
}

std::string ExchangeClient::toString() const {
  std::stringstream out;
  for (auto& source : sources_) {
    out << source->toString() << std::endl;
  }
  return out.str();
}

std::string ExchangeClient::toJsonString() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["closed"] = closed_;
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  for (auto& source : sources_) {
    clientsObj[std::to_string(index++)] = source->toJsonString();
  }
  return folly::toPrettyJson(obj);
}

bool Exchange::getSplits(ContinueFuture* future) {
  if (operatorCtx_->driverCtx()->driverId != 0) {
    // When there are multiple pipelines, a single operator, the one from
    // pipeline 0, is responsible for feeding splits into shared ExchangeClient.
    return false;
  }
  if (noMoreSplits_) {
    return false;
  }
  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId_, split, *future);
    if (reason == BlockingReason::kNotBlocked) {
      if (split.hasConnectorSplit()) {
        auto remoteSplit = std::dynamic_pointer_cast<RemoteConnectorSplit>(
            split.connectorSplit);
        VELOX_CHECK(remoteSplit, "Wrong type of split");
        exchangeClient_->addRemoteTaskId(remoteSplit->taskId);
        ++stats_.wlock()->numSplits;
      } else {
        exchangeClient_->noMoreRemoteTasks();
        noMoreSplits_ = true;
        if (atEnd_) {
          operatorCtx_->task()->multipleSplitsFinished(
              stats_.rlock()->numSplits);
          recordStats();
        }
        return false;
      }
    } else {
      return true;
    }
  }
}

BlockingReason Exchange::isBlocked(ContinueFuture* future) {
  if (currentPage_ || atEnd_) {
    return BlockingReason::kNotBlocked;
  }

  // Start fetching data right away. Do not wait for all
  // splits to be available.

  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  ContinueFuture dataFuture;
  currentPage_ = exchangeClient_->next(&atEnd_, &dataFuture);
  if (currentPage_ || atEnd_) {
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(numSplits);
      recordStats();
    }
    return BlockingReason::kNotBlocked;
  }

  // We have a dataFuture and we may also have a splitFuture_.

  if (splitFuture_.valid()) {
    // Block until data becomes available or more splits arrive.
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return BlockingReason::kWaitForSplit;
  }

  // Block until data becomes available.
  *future = std::move(dataFuture);
  return BlockingReason::kWaitForProducer;
}

bool Exchange::isFinished() {
  return atEnd_;
}

RowVectorPtr Exchange::getOutput() {
  if (!currentPage_) {
    return nullptr;
  }

  uint64_t rawInputBytes{0};
  if (!inputStream_) {
    inputStream_ = std::make_unique<ByteStream>();
    rawInputBytes += currentPage_->size();
    currentPage_->prepareStreamForDeserialize(inputStream_.get());
  }

  getSerde()->deserialize(
      inputStream_.get(), operatorCtx_->pool(), outputType_, &result_);

  {
    auto lockedStats = stats_.wlock();
    lockedStats->rawInputBytes += rawInputBytes;
    lockedStats->addInputVector(result_->estimateFlatSize(), result_->size());
  }

  if (inputStream_->atEnd()) {
    currentPage_ = nullptr;
    inputStream_ = nullptr;
  }

  return result_;
}

void Exchange::recordStats() {
  auto lockedStats = stats_.wlock();
  for (const auto& [name, value] : exchangeClient_->stats()) {
    auto it = lockedStats->runtimeStats.find(name);
    if (it == lockedStats->runtimeStats.end()) {
      lockedStats->runtimeStats[name] = value;
    } else {
      lockedStats->runtimeStats[name].merge(value);
    }
  }
}

VectorSerde* Exchange::getSerde() {
  return getVectorSerde();
}

VELOX_REGISTER_EXCHANGE_SOURCE_METHOD_DEFINITION(
    ExchangeSource,
    createLocalExchangeSource);

} // namespace facebook::velox::exec

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
#pragma once

#include <velox/common/memory/Memory.h>
#include <velox/common/memory/MemoryAllocator.h>
#include <memory>
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

// Corresponds to Presto SerializedPage, i.e. a container for
// serialize vectors in Presto wire format.
class SerializedPage {
 public:
  static constexpr int kSerializedPageOwner = -11;

  // Construct from IOBuf chain.
  explicit SerializedPage(
      std::unique_ptr<folly::IOBuf> iobuf,
      std::function<void(folly::IOBuf&)> onDestructionCb = nullptr);

  ~SerializedPage();

  // Returns the size of the serialized data in bytes.
  uint64_t size() const {
    return iobufBytes_;
  }

  // Makes 'input' ready for deserializing 'this' with
  // VectorStreamGroup::read().
  void prepareStreamForDeserialize(ByteStream* input);

  std::unique_ptr<folly::IOBuf> getIOBuf() const {
    return iobuf_->clone();
  }

 private:
  static int64_t chainBytes(folly::IOBuf& iobuf) {
    int64_t size = 0;
    for (auto& range : iobuf) {
      size += range.size();
    }
    return size;
  }

  // Buffers containing the serialized data. The memory is owned by 'iobuf_'.
  std::vector<ByteRange> ranges_;

  // IOBuf holding the data in 'ranges_.
  std::unique_ptr<folly::IOBuf> iobuf_;

  // Number of payload bytes in 'iobuf_'.
  const int64_t iobufBytes_;

  // Callback that will be called on destruction of the SerializedPage,
  // primarily used to free externally allocated memory backing folly::IOBuf
  // from caller. Caller is responsible to pass in proper cleanup logic to
  // prevent any memory leak.
  std::function<void(folly::IOBuf&)> onDestructionCb_;
};

class ExchangeSource;
class ExchangeClient;

// Queue of results retrieved from source. Owned by shared_ptr by
// Exchange and client threads and registered callbacks waiting
// for input.
class ExchangeQueue {
 public:
  explicit ExchangeQueue(
      int64_t minBytes,
      int64_t maxBytes,
      std::weak_ptr<ExchangeClient> client)
      : minBytes_(minBytes), maxBytes_(maxBytes), client_(std::move(client)) {}

  ~ExchangeQueue() {
    clearAllPromises();
  }

  std::mutex& mutex() {
    return mutex_;
  }

  bool empty() const {
    return queue_.empty();
  }

  int32_t numCompleted() const {
    return numCompleted_;
  }

  int32_t& numPending() {
    return numPending_;
  }

  void enqueueLocked(
      std::unique_ptr<SerializedPage>&& page,
      std::vector<ContinuePromise>& promises) {
    if (page == nullptr) {
      ++numCompleted_;
      --numPending_;
      auto completedPromises = checkCompleteLocked();
      promises.reserve(promises.size() + completedPromises.size());
      for (auto& promise : completedPromises) {
        promises.push_back(std::move(promise));
      }
      return;
    }
    auto pageSize = page->size();
    if (pageSize > peakPageSize_) {
      peakPageSize_ = pageSize;
    }
    totalBytes_ += page->size();
    receivedBytes_ += page->size();
    if (totalBytes_ > peakBytes_) {
      peakBytes_ = totalBytes_;
    }
    ++receivedPages_;
    queue_.push_back(std::move(page));
    if (!promises_.empty()) {
      // Resume one of the waiting drivers.
      promises.push_back(std::move(promises_.back()));
      promises_.pop_back();
    }
  }

  // If data is permanently not available, e.g. the source cannot be
  // contacted, this registers an error message and causes the reading
  // Exchanges to throw with the message.
  void setError(const std::string& error) {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (!error_.empty()) {
        return;
      }
      error_ = error;
      atEnd_ = true;
      // NOTE: clear the serialized page queue as we won't consume from an
      // errored queue.
      queue_.clear();
      promises = clearAllPromisesLocked();
    }
    clearPromises(promises);
  }

  std::unique_ptr<SerializedPage> dequeueLocked(
      bool* atEnd,
      ContinueFuture* future) {
    VELOX_CHECK(future);
    if (!error_.empty()) {
      *atEnd = true;
      throw std::runtime_error(error_);
    }
    if (queue_.empty()) {
      if (atEnd_) {
        *atEnd = true;
      } else {
        promises_.emplace_back("ExchangeQueue::dequeue");
        *future = promises_.back().getSemiFuture();
        *atEnd = false;
      }
      return nullptr;
    }
    auto page = std::move(queue_.front());
    queue_.pop_front();
    *atEnd = false;
    totalBytes_ -= page->size();
    return page;
  }

  // Returns a guess for a typical reply unit size. An expected reply size is
  // the requested bytes rounded up to the next multiple of the estimate. Actual
  // reply sizes have no hard limit.
  uint64_t expectedSerializedPageSize() const;

  uint64_t requestableSpace() const;

  // Marks that 'numRequests', each for 'bytes' have been started.
  void recordRequestLocked(int32_t numRequests, uint64_t bytes) {
    numPending_ += numRequests;
    expectedBytes_ += bytes;
  }

  /// Records that a pending request arrived with 'bytes' of payload. Decreases
  /// expected reply bytes.
  void recordReplyLocked(int64_t bytes) {
    if (numPending_ == 1) {
      expectedBytes_ = 0;
      return;
    }
    expectedBytes_ = std::max<int64_t>(
        0,
        expectedBytes_ -
            std::max<int64_t>(bytes, expectedBytes_ / numPending_));
  }

  // Returns the total bytes held by SerializedPages in 'this'.
  uint64_t totalBytes() const {
    return totalBytes_;
  }

  uint64_t peakBytes() const {
    return peakBytes_;
  }

  // Returns the target size for totalBytes(). An exchange client
  // should not fetch more data until the queue totalBytes() is below
  // minBytes().
  uint64_t minBytes() const {
    return minBytes_;
  }

  uint64_t maxBytes() const {
    return maxBytes_;
  }

  int64_t expectedBytes() const {
    return expectedBytes_;
  }

  void clearExpectedBytes() {
    expectedBytes_ = 0;
  }

  void addSourceLocked() {
    VELOX_CHECK(!noMoreSources_, "addSource called after noMoreSources");
    numSources_++;
  }

  void noMoreSources() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      noMoreSources_ = true;
      promises = checkCompleteLocked();
    }
    clearPromises(promises);
  }

  void close() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      promises = closeLocked();
    }
    clearPromises(promises);
  }

  /// After a reply has arrived, finds other sources to request. See the same
  /// method in ExchangeClient for the meaning of the arguments.
  void requestIfDue(
      const std::shared_ptr<ExchangeSource>& replySource,
      int64_t replySequence);

 private:
  std::vector<ContinuePromise> closeLocked() {
    queue_.clear();
    return clearAllPromisesLocked();
  }

  std::vector<ContinuePromise> checkCompleteLocked() {
    if (noMoreSources_ && numCompleted_ == numSources_) {
      atEnd_ = true;
      return clearAllPromisesLocked();
    }
    return {};
  }

  void clearAllPromises() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      promises = clearAllPromisesLocked();
    }
    clearPromises(promises);
  }

  std::vector<ContinuePromise> clearAllPromisesLocked() {
    return std::move(promises_);
  }

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  int numCompleted_ = 0;
  int numSources_ = 0;
  bool noMoreSources_ = false;
  bool atEnd_ = false;
  std::mutex mutex_;
  std::deque<std::unique_ptr<SerializedPage>> queue_;
  std::vector<ContinuePromise> promises_;
  // When set, all promises will be realized and the next dequeue will
  // throw an exception with this message.
  std::string error_;
  // Total size of SerializedPages in queue.
  uint64_t totalBytes_{0};

  // If 'totalBytes_' < 'minBytes_', an exchange should request more data from
  // producers.
  uint64_t minBytes_;

  // Maximum size of queue. Set request sizes and cap outstanding
  // requests so as not to overrun this when data arrives.
  uint64_t maxBytes_;

  std::weak_ptr<ExchangeClient> client_;

  // Number of sources with a pending request.
  int32_t numPending_{0};

  // Volume expected from pending requests.
  int64_t expectedBytes_{0};

  // Number of SerializedPages received.
  int64_t receivedPages_{0};

  // Total size of SerializedPages received. Used to calculate an average
  // expected size.
  int64_t receivedBytes_{0};
  int64_t peakBytes_{0};
  int64_t peakPageSize_{0};
};

class ExchangeSource : public std::enable_shared_from_this<ExchangeSource> {
 public:
  using Factory = std::function<std::shared_ptr<ExchangeSource>(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool)>;

  // Sequence number given to requestIfDue to mark request expired without
  // producing data.
  static constexpr int64_t kNoReply = ~0L;

  ExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool)
      : taskId_(taskId),
        destination_(destination),
        queue_(std::move(queue)),
        pool_(pool->shared_from_this()) {}

  virtual ~ExchangeSource() = default;

  static std::shared_ptr<ExchangeSource> create(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool);

  // Returns true if there is no request to the source pending or if
  // this should be retried. If true, the caller is expected to call
  // request(). This is expected to be called while holding lock over
  // queue_.mutex(). This sets the status of 'this' to be pending. The
  // caller is thus expected to call request() without holding a lock over
  // queue_.mutex(). This pattern prevents multiple exchange consumer
  // threads from issuing the same request.
  virtual bool shouldRequestLocked() = 0;

  void clearPendingLocked() {
    requestPending_ = false;
  }

  bool isPending() const {
    return requestPending_;
  }

  bool isRequestable() const {
    return !requestPending_ && !atEnd_;
  }

  bool isAtEnd() const {
    return atEnd_;
  }

  // Requests the producer to generate more data. Call only if
  // shouldRequest() was true. The object handles its own lifetime by
  // acquiring a shared_from_this() pointer if needed.  'mexBytes'
  // specifies a cap on the data to receive. The cap can be exceeded
  // if the producer has a single indivisible item that is larger than
  // the cap.
  virtual void request(uint64_t maxBytes) = 0;

  /// Informs the source that responses with sequence number < 'sequence' may be
  /// deleted.
  virtual void acknowledge(int64_t sequence) = 0;

  virtual void deleteResults() = 0;

  // Returns the size of the next request reply. This may be known if a previous
  // receive was aborted for e.g. no space in buffer.
  virtual std::optional<int64_t> nextReplySize() {
    return std::nullopt;
  }

  // Close the exchange source. May be called before all data
  // has been received and processed. This can happen in case
  // of an error or an operator like Limit aborting the query
  // once it received enough data.
  virtual void close() = 0;

  // Returns runtime statistics.
  virtual folly::F14FastMap<std::string, int64_t> stats() const = 0;

  virtual std::string toString() {
    std::stringstream out;
    out << "[ExchangeSource " << taskId_ << ":" << destination_
        << (requestPending_ ? " pending " : "") << (atEnd_ ? " at end" : "");
    return out.str();
  }

  static void registerFactory();

  static bool registerFactory(Factory factory) {
    factories().push_back(factory);
    return true;
  }

  static std::vector<Factory>& factories();

  // ID of the task producing data
  const std::string taskId_;
  // Destination number of 'this' on producer
  const int destination_;
  int64_t sequence_ = 0;
  std::shared_ptr<ExchangeQueue> queue_;
  std::atomic<bool> requestPending_{false};
  bool atEnd_ = false;

 protected:
  // Holds a shared reference on the memory pool as it might be still possible
  // to be accessed by external components after the query task is destroyed.
  // For instance, in Prestissimo, there might be a pending http request issued
  // by PrestoExchangeSource to fetch data from the remote task. When the http
  // response returns back, the task might have already terminated and deleted
  // so we need to hold an additional shared reference on the memory pool to
  // keeps it alive.
  const std::shared_ptr<memory::MemoryPool> pool_;
};

struct RemoteConnectorSplit : public connector::ConnectorSplit {
  const std::string taskId;

  explicit RemoteConnectorSplit(const std::string& _taskId)
      : ConnectorSplit(""), taskId(_taskId) {}

  std::string toString() const override {
    return fmt::format("Remote: {}", taskId);
  }
};

// Handle for a set of producers. This may be shared by multiple Exchanges, one
// per consumer thread.
class ExchangeClient {
 public:
  static constexpr int32_t kDefaultBufferSize = 32 << 20; // 32 MB.

  ExchangeClient(int destination, memory::MemoryPool* pool, int64_t bufferSize)
      : destination_(destination), pool_(pool) {
    VELOX_CHECK_NOT_NULL(pool_);
    VELOX_CHECK(
        destination >= 0,
        "Exchange client destination must be greater than zero, got {}",
        destination);
  }

  static std::shared_ptr<ExchangeClient> create(
      int destination,
      memory::MemoryPool* pool,
      int64_t bufferSize = kDefaultBufferSize) {
    auto client =
        std::make_shared<ExchangeClient>(destination, pool, bufferSize);
    client->queue_ =
        std::make_shared<ExchangeQueue>(bufferSize * 0.8, bufferSize, client);
    return client;
  }

  ~ExchangeClient();

  memory::MemoryPool* pool() const {
    return pool_;
  }

  // Creates an exchange source and starts fetching data from the specified
  // upstream task. If 'close' has been called already, creates an exchange
  // source and immediately closes it to notify the upstream task that data is
  // no longer needed. Repeated calls with the same 'taskId' are ignored.
  void addRemoteTaskId(const std::string& taskId);

  void noMoreRemoteTasks();

  // Closes exchange sources.
  void close();

  // Returns runtime statistics aggregates across all of the exchange sources.
  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

  std::shared_ptr<ExchangeQueue> queue() const {
    return queue_;
  }

  std::unique_ptr<SerializedPage> next(bool* atEnd, ContinueFuture* future);

  // Sends as many requests as are expected to fit in 'queue_' to
  // sources with no pending request. If called after receiving a
  // reply on the callback thread of the receive notify, 'replySource'
  // is sent either an acknowledge with 'replySequence' or a request
  // for more data, starting at 'replySequence'. In the case of
  // acknowledge, replySource is no longer pending, in the case of
  // re-request it stays pending. Whether an acknowledge or a request
  // for more is sent depends on whether there is space for a request
  // to 'replySource' after planning all the other possible
  // requests. The re-request can save a network send.  If
  // 'replySequence is kEmptyReply, then 'replySource' is marked as no
  // longer pending. if 'replySource' is nullptr, we are not calling after a
  // reply and just schedule as many sources as can be expected to fit in
  // 'queue_->maxBytes()'.
  void requestIfDue(
      const std::shared_ptr<ExchangeSource>& replySource,
      int64_t replySequence);

  std::string toString();

 private:
  const int destination_;
  memory::MemoryPool* const pool_;
  std::shared_ptr<ExchangeQueue> queue_;
  std::unordered_set<std::string> taskIds_;
  std::vector<std::shared_ptr<ExchangeSource>> sources_;
  // Next source to request. Clock hand over 'sources_'.
  int32_t nextSourceIndex_{0};

  bool closed_{false};
  int64_t numEmptyRequests_{0};
  int64_t numDirectRerequest_{0};
  int64_t numNothingRequestable_{0};
};

class Exchange : public SourceOperator {
 public:
  Exchange(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::ExchangeNode>& exchangeNode,
      std::shared_ptr<ExchangeClient> exchangeClient,
      const std::string& operatorType = "Exchange")
      : SourceOperator(
            ctx,
            exchangeNode->outputType(),
            operatorId,
            exchangeNode->id(),
            operatorType),
        planNodeId_(exchangeNode->id()),
        exchangeClient_(std::move(exchangeClient)) {}

  ~Exchange() override {
    close();
  }

  RowVectorPtr getOutput() override;

  void close() override {
    SourceOperator::close();
    currentPage_ = nullptr;
    result_ = nullptr;
    if (exchangeClient_) {
      exchangeClient_->close();
    }
    exchangeClient_ = nullptr;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  virtual VectorSerde* getSerde();

 private:
  /// Fetches splits from the task until there are no more splits or task
  /// returns a future that will be complete when more splits arrive. Adds
  /// splits to exchangeClient_. Returns true if received a future from the task
  /// and sets the 'future' parameter. Returns false if fetched all splits or if
  /// this operator is not the first operator in the pipeline and therefore is
  /// not responsible for fetching splits and adding them to the
  /// exchangeClient_.
  bool getSplits(ContinueFuture* future);

  void recordStats();

  const core::PlanNodeId planNodeId_;
  bool noMoreSplits_ = false;

  /// A future received from Task::getSplitOrFuture(). It will be complete when
  /// there are more splits available or no-more-splits signal has arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  RowVectorPtr result_;
  std::shared_ptr<ExchangeClient> exchangeClient_;
  std::unique_ptr<SerializedPage> currentPage_;
  std::unique_ptr<ByteStream> inputStream_;
  bool atEnd_{false};
};

} // namespace facebook::velox::exec

#define VELOX_REGISTER_EXCHANGE_SOURCE_METHOD_DEFINITION(class, function) \
  void class ::registerFactory() {                                        \
    facebook::velox::exec::ExchangeSource::registerFactory((function));   \
  }

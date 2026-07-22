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

#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.h"

#include <boost/asio/buffer.hpp>
#include <boost/beast/http.hpp>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/io/async/AsyncSocketException.h>
#include <folly/io/async/AsyncTransport.h>
#include <folly/io/async/EventBase.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::filesystems {
namespace {

namespace http = boost::beast::http;
using ErrorCode = boost::system::error_code;

enum class Failure : uint8_t { kNone, kSocket, kParser, kEarlyEof, kOverflow };
enum class WriteState : uint8_t { kPending, kSuccess, kFailure };

http::verb toBeastMethod(HttpMethod method) {
  switch (method) {
    case HttpMethod::kGet:
      return http::verb::get;
    case HttpMethod::kHead:
      return http::verb::head;
    case HttpMethod::kPost:
      return http::verb::post;
  }
  throw std::invalid_argument("unsupported ABFS HTTP method");
}

class FiberNotification {
 public:
  void post() noexcept {
    if (!pending_) {
      pending_ = true;
      baton_.post();
    }
  }

  bool wait(std::chrono::milliseconds timeout) {
    if (!pending_ && !baton_.timed_wait(timeout)) {
      return false;
    }
    pending_ = false;
    baton_.reset();
    return true;
  }

 private:
  folly::fibers::Baton baton_;
  bool pending_{false};
};

} // namespace

class FollyHttpConnection::TransportHolder {
 public:
  explicit TransportHolder(folly::AsyncTransportWrapper::UniquePtr transport)
      : transport_(std::move(transport)) {
    if (transport_ == nullptr) {
      throw std::invalid_argument("FollyHttpConnection requires a transport");
    }
  }

  folly::AsyncTransportWrapper* get() const noexcept {
    return transport_.get();
  }

  void close() noexcept {
    try {
      if (transport_ != nullptr) {
        transport_->closeNow();
      }
    } catch (...) {
    }
  }

 private:
  folly::AsyncTransportWrapper::UniquePtr transport_;
};

class ReadCallback final : public folly::AsyncTransport::ReadCallback {
 public:
  explicit ReadCallback(FollyHttpConnection::TransactionState* state)
      : state_(state) {}

  void getReadBuffer(void** buffer, size_t* length) noexcept override;
  void readDataAvailable(size_t length) noexcept override;
  void readEOF() noexcept override;
  void readErr(const folly::AsyncSocketException& exception) noexcept override;

 private:
  FollyHttpConnection::TransactionState* state_;
};

class WriteCallback final : public folly::AsyncTransport::WriteCallback {
 public:
  explicit WriteCallback(FollyHttpConnection::TransactionState* state)
      : state_(state) {}

  void writeSuccess() noexcept override;
  void writeErr(
      size_t bytesWritten,
      const folly::AsyncSocketException& exception) noexcept override;

 private:
  FollyHttpConnection::TransactionState* state_;
};

class FollyHttpConnection::TransactionState
    : public std::enable_shared_from_this<
          FollyHttpConnection::TransactionState> {
 public:
  TransactionState(
      std::shared_ptr<TransportHolder> transport,
      const HttpLimits& limits,
      const HttpTimeouts& timeouts,
      HttpTransactionRelease release)
      : limits_(limits),
        timeouts_(timeouts),
        transport_(std::move(transport)),
        release_(std::move(release)),
        ingress_(limits.maxIngressBytes),
        readCallback_(this),
        writeCallback_(this) {}

  ~TransactionState() {
    clearReadCallback();
    if (!released_) {
      finish(HttpTransactionOutcome::kAbandoned);
    }
  }

  void validate() const {
    if (limits_.maxIngressBytes == 0) {
      throw std::invalid_argument("ABFS HTTP ingress bound must be non-zero");
    }
    if (timeouts_.write.count() <= 0 ||
        timeouts_.firstByteAndHeaders.count() <= 0 ||
        timeouts_.bodyIdle.count() <= 0 || timeouts_.total.count() <= 0) {
      throw std::invalid_argument("ABFS HTTP timeouts must be positive");
    }
  }

  void writeSuccess() noexcept {
    completeWrite(WriteState::kSuccess);
  }

  void writeFailure() noexcept {
    completeWrite(WriteState::kFailure);
  }

  void completeWrite(WriteState state) noexcept {
    if (writeState_ != WriteState::kPending) {
      return;
    }
    writeState_ = state;
    writeTerminal_ = true;
    notification_.post();
  }

  void readData(size_t length) noexcept {
    try {
      if (failure_ != Failure::kNone ||
          length > ingress_.size() - ingressSize_) {
        failure_ = Failure::kOverflow;
        transport_->close();
        notification_.post();
        return;
      }
      ingressSize_ += length;
      notification_.post();
    } catch (...) {
      failure_ = Failure::kSocket;
      transport_->close();
      notification_.post();
    }
  }

  void readEof() noexcept {
    eof_ = true;
    notification_.post();
  }

  void readError() noexcept {
    try {
      failure_ = Failure::kSocket;
      transport_->close();
      notification_.post();
    } catch (...) {
      failure_ = Failure::kSocket;
    }
  }

  void installReadCallback() {
    transport()->setReadCB(&readCallback_);
    readCallbackInstalled_ = true;
  }

  void clearReadCallback() noexcept {
    try {
      if (readCallbackInstalled_) {
        transport()->setReadCB(nullptr);
        readCallbackInstalled_ = false;
      }
    } catch (...) {
    }
  }

  folly::AsyncTransportWrapper* transport() const noexcept {
    return transport_->get();
  }

  bool wait(std::chrono::milliseconds timeout) {
    return notification_.wait(timeout);
  }

  std::chrono::milliseconds remainingTotal() const {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - transactionStart_);
    if (elapsed >= timeouts_.total) {
      return std::chrono::milliseconds(0);
    }
    return timeouts_.total - elapsed;
  }

  void close() noexcept {
    clearReadCallback();
    transport_->close();
  }

  void finish(HttpTransactionOutcome outcome) noexcept {
    if (released_) {
      return;
    }
    releaseOutcome_ = outcome;
    released_ = true;
    clearReadCallback();
    if (outcome != HttpTransactionOutcome::kReusable) {
      transport_->close();
    }
    try {
      if (release_) {
        release_(outcome);
      }
    } catch (...) {
    }
  }

  [[noreturn]] void fail(Failure failure) {
    failure_ = failure;
    close();
    finish(HttpTransactionOutcome::kFailed);
    switch (failure) {
      case Failure::kEarlyEof:
        throw std::runtime_error(
            "ABFS HTTP response ended before Content-Length");
      case Failure::kOverflow:
        throw std::runtime_error(
            "ABFS HTTP ingress exceeded its configured bound");
      case Failure::kParser:
        throw std::runtime_error("ABFS HTTP response parsing failed");
      case Failure::kSocket:
        throw std::runtime_error("ABFS HTTP transport failed");
      case Failure::kNone:
        throw std::runtime_error("ABFS HTTP transaction failed");
    }
    throw std::runtime_error("ABFS HTTP transaction failed");
  }

  void prepareParser(HttpResponseBodyMode mode, bool isHeadRequest) {
    parser_ = std::make_unique<http::response_parser<http::buffer_body>>();
    parser_->header_limit(limits_.maxHeaderBytes);
    parser_->body_limit(std::numeric_limits<uint64_t>::max());
    parser_->skip(mode == HttpResponseBodyMode::kSkip || isHeadRequest);
  }

  void parseHeaders() {
    size_t consumedTotal{0};
    while (ingressSize_ != consumedTotal && !parser_->is_header_done()) {
      ErrorCode error;
      const auto consumed = parser_->put(
          boost::asio::buffer(
              ingress_.data() + consumedTotal, ingressSize_ - consumedTotal),
          error);
      consumedTotal += consumed;
      if (error == http::error::need_more ||
          error == http::error::need_buffer) {
        break;
      }
      if (error) {
        fail(Failure::kParser);
      }
      if (consumed == 0) {
        break;
      }
    }
    if (consumedTotal != 0) {
      std::memmove(
          ingress_.data(),
          ingress_.data() + consumedTotal,
          ingressSize_ - consumedTotal);
      ingressSize_ -= consumedTotal;
    }
  }

  void validateStatusLine() const {
    const auto& response = parser_->get();
    const auto statusLineBytes = 5 +
        std::to_string(response.version() / 10).size() + 1 +
        std::to_string(response.version() % 10).size() + 1 +
        std::to_string(response.result_int()).size() + 1 +
        response.reason().size() + 2;
    if (statusLineBytes > limits_.maxStatusLineBytes) {
      throw std::runtime_error("ABFS HTTP status line exceeded its limit");
    }
  }

  HttpResponseHead responseHead() const {
    const auto& response = parser_->get();
    validateStatusLine();
    HttpResponseHead head;
    head.version = {
        static_cast<uint16_t>(response.version() / 10),
        static_cast<uint16_t>(response.version() % 10),
    };
    head.statusCode = response.result_int();
    head.reason = std::string(response.reason());
    head.informationalResponseCount = informationalResponseCount_;
    for (const auto& field : response) {
      head.headers.emplace_back(
          std::string(field.name_string()), std::string(field.value()));
    }
    if (const auto contentLength = parser_->content_length()) {
      head.contentLength = *contentLength;
    }
    head.reusable = response.keep_alive();
    return head;
  }

  bool complete() const noexcept {
    return parser_->is_done();
  }

  int responseStatus() const noexcept {
    return parser_->get().result_int();
  }

  size_t
  readBody(uint8_t* buffer, size_t size, std::chrono::milliseconds timeout) {
    while (true) {
      if (ingressSize_ != 0) {
        auto& body = parser_->get().body();
        body.data = buffer;
        body.size = size;
        ErrorCode error;
        const auto consumed = parser_->put(
            boost::asio::buffer(ingress_.data(), ingressSize_), error);
        const auto produced = size - body.size;
        if (consumed != 0) {
          std::memmove(
              ingress_.data(),
              ingress_.data() + consumed,
              ingressSize_ - consumed);
          ingressSize_ -= consumed;
        }
        if (error && error != http::error::need_buffer &&
            error != http::error::need_more) {
          fail(Failure::kParser);
        }
        if (parser_->is_done()) {
          if (ingressSize_ != 0) {
            fail(Failure::kParser);
          }
          finish(
              parser_->keep_alive() ? HttpTransactionOutcome::kReusable
                                    : HttpTransactionOutcome::kClosed);
        }
        if (produced != 0 || parser_->is_done()) {
          return produced;
        }
        if (consumed != 0 && ingressSize_ != 0) {
          continue;
        }
      }
      if (complete()) {
        if (ingressSize_ != 0) {
          fail(Failure::kParser);
        }
        finish(
            parser_->keep_alive() ? HttpTransactionOutcome::kReusable
                                  : HttpTransactionOutcome::kClosed);
        return 0;
      }
      if (failure_ != Failure::kNone) {
        fail(failure_);
      }
      if (eof_) {
        ErrorCode error;
        parser_->put_eof(error);
        if (error || !parser_->is_done()) {
          fail(Failure::kEarlyEof);
        }
        finish(
            parser_->keep_alive() ? HttpTransactionOutcome::kReusable
                                  : HttpTransactionOutcome::kClosed);
        return 0;
      }
      if (!wait(timeout)) {
        close();
        finish(HttpTransactionOutcome::kTimedOut);
        throw std::runtime_error("ABFS HTTP body read timed out");
      }
    }
  }

  HttpLimits limits_;
  HttpTimeouts timeouts_;
  std::shared_ptr<TransportHolder> transport_;
  HttpTransactionRelease release_;
  std::vector<uint8_t> ingress_;
  size_t ingressSize_{0};
  std::unique_ptr<http::response_parser<http::buffer_body>> parser_;
  ReadCallback readCallback_;
  WriteCallback writeCallback_;
  std::string serializedRequest_;
  FiberNotification notification_;
  Failure failure_{Failure::kNone};
  WriteState writeState_{WriteState::kPending};
  bool readCallbackInstalled_{false};
  bool writeTerminal_{false};
  bool eof_{false};
  bool released_{false};
  size_t informationalResponseCount_{0};
  HttpTransactionOutcome releaseOutcome_{HttpTransactionOutcome::kAbandoned};
  std::chrono::steady_clock::time_point transactionStart_{
      std::chrono::steady_clock::now()};
};

class BodyTransaction final : public HttpBodyTransaction {
 public:
  BodyTransaction(
      std::shared_ptr<FollyHttpConnection::TransactionState> state,
      std::chrono::milliseconds timeout)
      : state_(std::move(state)), timeout_(timeout) {}

  ~BodyTransaction() override {
    if (state_ != nullptr && !state_->released_) {
      state_->finish(HttpTransactionOutcome::kAbandoned);
    }
  }

  size_t read(uint8_t* buffer, size_t size, std::chrono::milliseconds timeout)
      override {
    if (state_->released_) {
      if (state_->releaseOutcome_ == HttpTransactionOutcome::kReusable ||
          state_->releaseOutcome_ == HttpTransactionOutcome::kClosed) {
        return 0;
      }
      throw std::runtime_error(
          "ABFS HTTP body transaction is no longer active");
    }
    if (size == 0) {
      if (state_->complete()) {
        state_->finish(
            state_->parser_->keep_alive() ? HttpTransactionOutcome::kReusable
                                          : HttpTransactionOutcome::kClosed);
      }
      return 0;
    }
    if (buffer == nullptr) {
      throw std::invalid_argument("ABFS HTTP body read requires a buffer");
    }
    const auto waitTimeout =
        std::min(std::min(timeout, timeout_), state_->remainingTotal());
    if (waitTimeout.count() <= 0) {
      state_->close();
      state_->finish(HttpTransactionOutcome::kTimedOut);
      throw std::runtime_error("ABFS HTTP body read timed out");
    }
    return state_->readBody(buffer, size, waitTimeout);
  }

  bool complete() const noexcept override {
    return state_->complete();
  }

  void abandon() noexcept override {
    if (state_ != nullptr) {
      state_->finish(HttpTransactionOutcome::kAbandoned);
    }
  }

 private:
  std::shared_ptr<FollyHttpConnection::TransactionState> state_;
  std::chrono::milliseconds timeout_;
};

void ReadCallback::getReadBuffer(void** buffer, size_t* length) noexcept {
  static uint8_t fallbackByte{0};
  try {
    if (state_->failure_ == Failure::kNone &&
        state_->ingressSize_ < state_->ingress_.size()) {
      *buffer = state_->ingress_.data() + state_->ingressSize_;
      *length = state_->ingress_.size() - state_->ingressSize_;
      return;
    }
    state_->failure_ = Failure::kOverflow;
    state_->transport_->close();
    *buffer = &fallbackByte;
    *length = 1;
  } catch (...) {
    state_->failure_ = Failure::kOverflow;
    *buffer = &fallbackByte;
    *length = 1;
  }
}

void ReadCallback::readDataAvailable(size_t length) noexcept {
  state_->readData(length);
}

void ReadCallback::readEOF() noexcept {
  state_->readEof();
}

void ReadCallback::readErr(const folly::AsyncSocketException&) noexcept {
  state_->readError();
}

void WriteCallback::writeSuccess() noexcept {
  state_->writeSuccess();
}

void WriteCallback::writeErr(
    size_t,
    const folly::AsyncSocketException&) noexcept {
  state_->writeFailure();
}

FollyHttpConnection::FollyHttpConnection(
    folly::AsyncTransportWrapper::UniquePtr transport)
    : transport_(std::make_shared<TransportHolder>(std::move(transport))) {}

FollyHttpConnection::~FollyHttpConnection() {
  if (activeTransaction_ != nullptr) {
    activeTransaction_->finish(HttpTransactionOutcome::kAbandoned);
  }
  transport_->close();
}

folly::EventBase* FollyHttpConnection::eventBase() const noexcept {
  return transport_->get() == nullptr ? nullptr
                                      : transport_->get()->getEventBase();
}

bool FollyHttpConnection::usable() const noexcept {
  return transport_->get() != nullptr && transport_->get()->good() &&
      (activeTransaction_ == nullptr || activeTransaction_->released_);
}

HttpResponseTransaction FollyHttpConnection::send(
    const HttpRequest& request,
    const HttpLimits& limits,
    const HttpTimeouts& timeouts,
    HttpTransactionRelease release) {
  auto* eventBase = transport_->get()->getEventBase();
  if (eventBase == nullptr || !eventBase->isInEventBaseThread()) {
    throw std::logic_error(
        "ABFS HTTP send must run on the transport EventBase");
  }
  if (!folly::fibers::onFiber()) {
    throw std::logic_error("ABFS HTTP send must run inside a fiber");
  }
  if (activeTransaction_ != nullptr) {
    if (!activeTransaction_->released_ || !activeTransaction_->writeTerminal_) {
      throw std::logic_error(
          "ABFS HTTP connection already has an active response");
    }
    activeTransaction_.reset();
  }
  activeTransaction_.reset();

  auto state = std::make_shared<TransactionState>(
      transport_, limits, timeouts, std::move(release));
  activeTransaction_ = state;
  try {
    state->validate();
  } catch (...) {
    state->finish(HttpTransactionOutcome::kFailed);
    throw;
  }
  if (request.body.size() > limits.maxRequestBodyBytes) {
    state->finish(HttpTransactionOutcome::kFailed);
    throw std::invalid_argument("ABFS HTTP request body exceeded its limit");
  }
  http::request<http::string_body> beastRequest;
  beastRequest.method(toBeastMethod(request.method));
  beastRequest.target(request.target);
  beastRequest.version(11);
  for (const auto& [name, value] : request.headers) {
    beastRequest.insert(name, value);
  }
  beastRequest.body() = request.body;
  const auto hasLength =
      beastRequest.find(http::field::content_length) != beastRequest.end();
  const auto hasTransferEncoding =
      beastRequest.find(http::field::transfer_encoding) != beastRequest.end();
  if (!hasLength && !hasTransferEncoding && !request.body.empty()) {
    beastRequest.prepare_payload();
  }

  http::request_serializer<http::string_body> serializer(beastRequest);
  try {
    ErrorCode serializationError;
    while (!serializer.is_done()) {
      serializer.next(
          serializationError,
          [&](ErrorCode& visitorError, const auto& buffers) {
            const auto bytes = boost::asio::buffer_size(buffers);
            const auto offset = state->serializedRequest_.size();
            if (bytes > state->serializedRequest_.max_size() - offset) {
              visitorError =
                  make_error_code(boost::system::errc::value_too_large);
              return;
            }
            state->serializedRequest_.resize(offset + bytes);
            boost::asio::buffer_copy(
                boost::asio::buffer(
                    state->serializedRequest_.data() + offset, bytes),
                buffers);
            serializer.consume(bytes);
          });
      if (serializationError) {
        throw std::runtime_error(serializationError.message());
      }
    }
    state->writeState_ = WriteState::kPending;
    state->transport()->write(
        &state->writeCallback_,
        state->serializedRequest_.data(),
        state->serializedRequest_.size());
  } catch (...) {
    state->close();
    state->finish(HttpTransactionOutcome::kFailed);
    throw;
  }
  const auto writeTimeout = std::min(timeouts.write, state->remainingTotal());
  if (!state->wait(writeTimeout) ||
      state->writeState_ != WriteState::kSuccess) {
    state->close();
    const auto outcome = state->writeState_ == WriteState::kPending
        ? HttpTransactionOutcome::kTimedOut
        : HttpTransactionOutcome::kFailed;
    state->finish(outcome);
    throw std::runtime_error(
        outcome == HttpTransactionOutcome::kTimedOut
            ? "ABFS HTTP request write timed out"
            : "ABFS HTTP request write failed");
  }

  state->prepareParser(
      request.responseBodyMode, request.method == HttpMethod::kHead);
  state->installReadCallback();
  const auto headerStart = std::chrono::steady_clock::now();
  while (true) {
    while (!state->parser_->is_header_done()) {
      state->parseHeaders();
      if (state->parser_->is_header_done()) {
        try {
          state->validateStatusLine();
        } catch (...) {
          state->close();
          state->finish(HttpTransactionOutcome::kFailed);
          throw;
        }
        break;
      }
      if (state->failure_ != Failure::kNone) {
        state->fail(state->failure_);
      }
      if (state->eof_) {
        state->fail(Failure::kEarlyEof);
      }
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - headerStart);
      const auto phaseRemaining = elapsed >= timeouts.firstByteAndHeaders
          ? std::chrono::milliseconds(0)
          : timeouts.firstByteAndHeaders - elapsed;
      const auto waitTimeout =
          std::min(phaseRemaining, state->remainingTotal());
      if (waitTimeout.count() <= 0 || !state->wait(waitTimeout)) {
        state->close();
        state->finish(HttpTransactionOutcome::kTimedOut);
        throw std::runtime_error("ABFS HTTP response headers timed out");
      }
    }
    const auto status = state->responseStatus();
    if (status >= 100 && status < 200) {
      if (status == 101 ||
          ++state->informationalResponseCount_ >
              limits.maxInformationalResponses) {
        state->fail(Failure::kParser);
      }
      state->prepareParser(
          request.responseBodyMode, request.method == HttpMethod::kHead);
      continue;
    }
    break;
  }

  HttpResponseTransaction transaction;
  try {
    transaction.head = state->responseHead();
    if (state->parser_->is_done()) {
      if (state->ingressSize_ != 0) {
        state->fail(Failure::kParser);
      }
      state->finish(
          state->parser_->keep_alive() ? HttpTransactionOutcome::kReusable
                                       : HttpTransactionOutcome::kClosed);
    }
  } catch (...) {
    state->close();
    state->finish(HttpTransactionOutcome::kFailed);
    throw;
  }
  activeTransaction_ = state;
  transaction.body =
      std::make_unique<BodyTransaction>(state, timeouts.bodyIdle);
  return transaction;
}

} // namespace facebook::velox::filesystems

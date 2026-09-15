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

#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h"

#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.h"

#include <azure/core/http/http.hpp>
#include <azure/core/url.hpp>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/io/async/AsyncTimeout.h>
#include <folly/io/async/EventBase.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::filesystems {
namespace {

constexpr char kTransportErrorPrefix[] = "ABFS Azure transport failure: ";
constexpr size_t kRequestBodyChunkBytes{16 * 1'024};
constexpr size_t kResponseBodyChunkBytes{16 * 1'024};

Azure::Core::Http::TransportException transportException(
    const std::exception& exception) {
  return Azure::Core::Http::TransportException(
      std::string(kTransportErrorPrefix) + exception.what());
}

bool equalInsensitive(std::string left, std::string right) {
  if (left.size() != right.size()) {
    return false;
  }
  std::transform(
      left.begin(), left.end(), left.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
      });
  std::transform(
      right.begin(), right.end(), right.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
      });
  return left == right;
}

uint16_t effectivePort(const Azure::Core::Url& url) {
  if (url.GetPort() != 0) {
    return url.GetPort();
  }
  return equalInsensitive(url.GetScheme(), "https") ? 443 : 80;
}

std::string expectedScheme(AsyncChannelSecurity security) {
  return security == AsyncChannelSecurity::kTls ? "https" : "http";
}

bool hasHeader(const HttpHeaders& headers, std::string_view name) {
  return std::any_of(headers.begin(), headers.end(), [&](const auto& header) {
    return equalInsensitive(header.first, std::string{name});
  });
}

std::string hostHeader(const Azure::Core::Url& url) {
  auto host = url.GetHost();
  const auto port = url.GetPort();
  const auto defaultPort =
      equalInsensitive(url.GetScheme(), "https") ? 443 : 80;
  if (port != 0 && port != defaultPort) {
    host += ":" + std::to_string(port);
  }
  return host;
}

struct ConnectionLease {
  std::shared_ptr<FollyHttpConnection> connection;
  std::shared_ptr<class FollyHttpConnectionPool> pool;
  bool released{false};

  void release(HttpTransactionOutcome outcome) noexcept;
};

class FollyHttpConnectionPool final
    : public std::enable_shared_from_this<FollyHttpConnectionPool> {
 public:
  FollyHttpConnectionPool(
      AsyncChannelFactoryPtr factory,
      AsyncChannelEndpoint endpoint,
      HttpLimits limits,
      HttpTimeouts timeouts,
      size_t maxConnections)
      : factory_(std::move(factory)),
        endpoint_(std::move(endpoint)),
        limits_(limits),
        timeouts_(timeouts),
        maxConnections_(maxConnections) {
    if (maxConnections_ == 0) {
      throw std::invalid_argument(
          "ABFS HTTP connection pool limit must be positive");
    }
    if (timeouts_.connectionAcquire.count() <= 0 ||
        timeouts_.connectionIdle.count() <= 0) {
      throw std::invalid_argument(
          "ABFS HTTP connection pool timeouts must be positive");
    }
  }

  ~FollyHttpConnectionPool() {
    if (eventBase_ == nullptr) {
      return;
    }
    eventBase_->runImmediatelyOrRunInEventBaseThreadAndWait([this] {
      if (idleTimeout_ != nullptr) {
        idleTimeout_->cancelTimeout();
      }
      idle_.clear();
      idleTimeout_.reset();
    });
  }

  std::shared_ptr<FollyHttpConnection> acquire(
      std::chrono::milliseconds timeout) {
    assertExecutionContext();
    if (!folly::fibers::onFiber()) {
      throw std::logic_error("ABFS HTTP pool acquire must run inside a fiber");
    }
    evictExpiredIdleConnections();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      while (!idle_.empty()) {
        auto connection = std::move(idle_.front().connection);
        idle_.pop_front();
        scheduleIdleTimeout();
        if (!connection->usable()) {
          --totalConnections_;
          continue;
        }
        ++leasedConnections_;
        peakLeasedConnections_ =
            std::max(peakLeasedConnections_, leasedConnections_);
        return connection;
      }
      if (totalConnections_ < maxConnections_) {
        ++totalConnections_;
        try {
          auto connection = std::make_shared<FollyHttpConnection>(
              factory_->connect(endpoint_));
          bindEventBase(connection->eventBase());
          ++leasedConnections_;
          peakLeasedConnections_ =
              std::max(peakLeasedConnections_, leasedConnections_);
          return connection;
        } catch (...) {
          --totalConnections_;
          throw;
        }
      }

      auto waiter = std::make_shared<Waiter>();
      waiters_.push_back(waiter);
      ++waitingFibers_;
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now >= deadline
          ? std::chrono::milliseconds(0)
          : std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now);
      const auto waited = waiter->baton.timed_wait(remaining);
      if (!waited && waiter->state == WaiterState::kPending) {
        const auto iterator =
            std::find(waiters_.begin(), waiters_.end(), waiter);
        if (iterator != waiters_.end()) {
          waiters_.erase(iterator);
          --waitingFibers_;
          waiter->complete(WaiterState::kTimedOut);
        }
      }
      if (waiter->state == WaiterState::kConnection) {
        auto connection = std::move(waiter->connection);
        ++leasedConnections_;
        peakLeasedConnections_ =
            std::max(peakLeasedConnections_, leasedConnections_);
        return connection;
      }
      if (waiter->state == WaiterState::kRetry) {
        continue;
      }
      if (waiter->state == WaiterState::kTimedOut) {
        throw std::runtime_error("ABFS HTTP connection pool acquire timed out");
      }
    }
  }

  void release(
      std::shared_ptr<FollyHttpConnection> connection,
      HttpTransactionOutcome outcome) noexcept {
    try {
      assertExecutionContext();
      if (leasedConnections_ == 0) {
        return;
      }
      --leasedConnections_;
      const auto reusable = outcome == HttpTransactionOutcome::kReusable &&
          connection != nullptr && connection->usable();
      if (reusable) {
        if (!waiters_.empty()) {
          auto waiter = std::move(waiters_.front());
          waiters_.pop_front();
          --waitingFibers_;
          waiter->complete(WaiterState::kConnection, std::move(connection));
        } else {
          idle_.push_back(
              {std::move(connection),
               std::chrono::steady_clock::now() + timeouts_.connectionIdle});
          scheduleIdleTimeout();
        }
        return;
      }

      if (totalConnections_ != 0) {
        --totalConnections_;
      }
      connection.reset();
      if (!waiters_.empty()) {
        auto waiter = std::move(waiters_.front());
        waiters_.pop_front();
        --waitingFibers_;
        waiter->complete(WaiterState::kRetry);
      }
    } catch (...) {
    }
  }

  FollyHttpTransport::PoolMetrics metrics() const noexcept {
    return {
        maxConnections_,
        totalConnections_,
        leasedConnections_,
        idle_.size(),
        waitingFibers_,
        peakLeasedConnections_,
        idleConnectionEvictions_,
    };
  }

 private:
  struct IdleConnection {
    std::shared_ptr<FollyHttpConnection> connection;
    std::chrono::steady_clock::time_point deadline;
  };

  class IdleTimeout final : public folly::AsyncTimeout {
   public:
    IdleTimeout(folly::EventBase* eventBase, FollyHttpConnectionPool* pool)
        : folly::AsyncTimeout(eventBase), pool_(pool) {}

    void timeoutExpired() noexcept override {
      pool_->evictExpiredIdleConnections();
    }

   private:
    FollyHttpConnectionPool* pool_;
  };

  enum class WaiterState { kPending, kConnection, kRetry, kTimedOut };

  struct Waiter {
    folly::fibers::Baton baton;
    WaiterState state{WaiterState::kPending};
    std::shared_ptr<FollyHttpConnection> connection;

    void complete(
        WaiterState terminalState,
        std::shared_ptr<FollyHttpConnection> completedConnection = nullptr) {
      if (state != WaiterState::kPending) {
        throw std::logic_error("ABFS HTTP pool waiter completed twice");
      }
      state = terminalState;
      connection = std::move(completedConnection);
      baton.post();
    }
  };

  void bindEventBase(folly::EventBase* eventBase) {
    if (eventBase == nullptr) {
      throw std::logic_error("ABFS HTTP connection has no EventBase");
    }
    if (eventBase_ == nullptr) {
      eventBase_ = eventBase;
      idleTimeout_ = std::make_unique<IdleTimeout>(eventBase_, this);
    }
    assertExecutionContext();
  }

  void evictExpiredIdleConnections() noexcept {
    const auto now = std::chrono::steady_clock::now();
    while (!idle_.empty() && idle_.front().deadline <= now) {
      auto connection = std::move(idle_.front().connection);
      idle_.pop_front();
      if (totalConnections_ != 0) {
        --totalConnections_;
      }
      ++idleConnectionEvictions_;
      connection.reset();
    }
    scheduleIdleTimeout();
  }

  void scheduleIdleTimeout() noexcept {
    if (idleTimeout_ == nullptr) {
      return;
    }
    idleTimeout_->cancelTimeout();
    if (idle_.empty()) {
      return;
    }
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
        idle_.front().deadline - std::chrono::steady_clock::now());
    idleTimeout_->scheduleTimeout(
        std::max(remaining, std::chrono::milliseconds(1)));
  }

  void assertExecutionContext() const {
    if (eventBase_ != nullptr && !eventBase_->isInEventBaseThread()) {
      throw std::logic_error("ABFS HTTP pool used from the wrong EventBase");
    }
  }

  AsyncChannelFactoryPtr factory_;
  AsyncChannelEndpoint endpoint_;
  HttpLimits limits_;
  HttpTimeouts timeouts_;
  size_t maxConnections_{0};
  folly::EventBase* eventBase_{nullptr};
  std::deque<IdleConnection> idle_;
  std::deque<std::shared_ptr<Waiter>> waiters_;
  std::unique_ptr<IdleTimeout> idleTimeout_;
  size_t totalConnections_{0};
  size_t leasedConnections_{0};
  size_t waitingFibers_{0};
  size_t peakLeasedConnections_{0};
  size_t idleConnectionEvictions_{0};
};

void ConnectionLease::release(HttpTransactionOutcome outcome) noexcept {
  if (released) {
    return;
  }
  released = true;
  if (pool != nullptr) {
    pool->release(std::move(connection), outcome);
  } else {
    connection.reset();
  }
}

HttpMethod requestMethod(const Azure::Core::Http::Request& request) {
  const auto method = request.GetMethod().ToString();
  if (method == "GET") {
    return HttpMethod::kGet;
  }
  if (method == "HEAD") {
    return HttpMethod::kHead;
  }
  if (method == "POST") {
    return HttpMethod::kPost;
  }
  throw std::invalid_argument("ABFS Azure HTTP method is unsupported");
}

std::optional<int64_t> responseLength(const HttpResponseHead& head) {
  if (!head.contentLength.has_value() ||
      *head.contentLength > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(*head.contentLength);
}

} // namespace

FollyHttpTransport::FollyHttpTransport(
    AsyncChannelFactoryPtr factory,
    AsyncChannelEndpoint endpoint,
    HttpLimits limits,
    HttpTimeouts timeouts,
    size_t maxConnectionsPerEndpoint)
    : factory_(std::move(factory)),
      endpoint_(std::move(endpoint)),
      limits_(limits),
      timeouts_(timeouts),
      maxConnectionsPerEndpoint_(maxConnectionsPerEndpoint),
      pool_(
          std::make_shared<FollyHttpConnectionPool>(
              factory_,
              endpoint_,
              limits_,
              timeouts_,
              maxConnectionsPerEndpoint_)) {
  if (factory_ == nullptr) {
    throw std::invalid_argument(
        "ABFS Azure transport requires a channel factory");
  }
  if (endpoint_.serverName.empty()) {
    throw std::invalid_argument(
        "ABFS Azure transport requires an endpoint host");
  }
  if (endpoint_.connectAddress.getPort() == 0) {
    throw std::invalid_argument(
        "ABFS Azure transport requires an endpoint port");
  }
}

FollyHttpTransport::PoolMetrics FollyHttpTransport::poolMetrics() const {
  return std::static_pointer_cast<FollyHttpConnectionPool>(pool_)->metrics();
}

std::unique_ptr<Azure::Core::Http::RawResponse> FollyHttpTransport::Send(
    Azure::Core::Http::Request& request,
    const Azure::Core::Context& context) {
  const auto deadline = std::chrono::steady_clock::now() + timeouts_.total;
  context.ThrowIfCancelled();
  try {
    if (!folly::fibers::onFiber()) {
      throw std::logic_error(
          "ABFS Azure transport send must run inside a fiber");
    }
    const auto& url = request.GetUrl();
    const auto scheme = url.GetScheme();
    if (!equalInsensitive(scheme, expectedScheme(endpoint_.security)) ||
        !equalInsensitive(url.GetHost(), endpoint_.serverName) ||
        effectivePort(url) != endpoint_.connectAddress.getPort() ||
        url.GetRelativeUrl().empty()) {
      throw std::invalid_argument(
          "ABFS Azure request URL does not match endpoint");
    }

    const auto method = requestMethod(request);
    HttpRequest httpRequest;
    httpRequest.method = method;
    httpRequest.target = url.GetRelativeUrl();
    if (httpRequest.target.empty() || httpRequest.target.front() != '/') {
      httpRequest.target.insert(httpRequest.target.begin(), '/');
    }
    httpRequest.responseBodyMode = method == HttpMethod::kHead
        ? HttpResponseBodyMode::kSkip
        : HttpResponseBodyMode::kParse;
    for (const auto& header : request.GetHeaders()) {
      httpRequest.headers.emplace_back(header.first, header.second);
    }
    if (!hasHeader(httpRequest.headers, "Host")) {
      httpRequest.headers.emplace_back("Host", hostHeader(url));
    }

    if (auto bodyStream = request.GetBodyStream()) {
      std::array<uint8_t, kRequestBodyChunkBytes> buffer{};
      while (true) {
        context.ThrowIfCancelled();
        const auto bytes =
            bodyStream->Read(buffer.data(), buffer.size(), context);
        if (bytes == 0) {
          break;
        }
        if (bytes > buffer.size() ||
            httpRequest.body.size() > limits_.maxRequestBodyBytes ||
            bytes > limits_.maxRequestBodyBytes - httpRequest.body.size()) {
          throw std::length_error("ABFS Azure request body exceeded its limit");
        }
        httpRequest.body.append(
            reinterpret_cast<const char*>(buffer.data()), bytes);
      }
    }

    const auto remainingTotal = [&] {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return std::chrono::milliseconds(0);
      }
      return std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - now);
    };
    if (remainingTotal().count() <= 0) {
      throw std::runtime_error("ABFS HTTP transaction timed out");
    }
    auto pool = std::static_pointer_cast<FollyHttpConnectionPool>(pool_);
    auto connection =
        pool->acquire(std::min(timeouts_.connectionAcquire, remainingTotal()));
    auto lease = std::make_shared<ConnectionLease>();
    lease->connection = connection;
    lease->pool = pool;
    auto transactionTimeouts = timeouts_;
    transactionTimeouts.total = remainingTotal();
    if (transactionTimeouts.total.count() <= 0) {
      lease->release(HttpTransactionOutcome::kTimedOut);
      throw std::runtime_error("ABFS HTTP transaction timed out");
    }
    std::weak_ptr<ConnectionLease> weakLease = lease;
    auto transaction = connection->send(
        httpRequest,
        limits_,
        transactionTimeouts,
        [weakLease](HttpTransactionOutcome outcome) {
          if (const auto owner = weakLease.lock()) {
            owner->release(outcome);
          }
        });

    auto response = std::make_unique<Azure::Core::Http::RawResponse>(
        transaction.head.version.major,
        transaction.head.version.minor,
        static_cast<Azure::Core::Http::HttpStatusCode>(
            transaction.head.statusCode),
        transaction.head.reason);
    // RawResponse stores headers by name, so duplicate field multiplicity ends
    // here.
    for (const auto& header : transaction.head.headers) {
      response->SetHeader(header.first, header.second);
    }

    if (request.ShouldBufferResponse() || method == HttpMethod::kHead) {
      std::vector<uint8_t> body;
      std::array<uint8_t, kResponseBodyChunkBytes> buffer{};
      while (!transaction.body->complete()) {
        context.ThrowIfCancelled();
        const auto bytes = transaction.body->read(
            buffer.data(), buffer.size(), timeouts_.bodyIdle);
        if (bytes == 0) {
          if (!transaction.body->complete()) {
            throw std::runtime_error(
                "ABFS Azure response body ended before completion");
          }
          break;
        }
        if (body.size() > limits_.maxBufferedResponseBodyBytes ||
            bytes > limits_.maxBufferedResponseBodyBytes - body.size()) {
          throw std::length_error(
              "ABFS Azure buffered response body exceeded its limit");
        }
        body.insert(body.end(), buffer.begin(), buffer.begin() + bytes);
      }
      response->SetBody(std::move(body));
    } else {
      response->SetBodyStream(
          std::make_unique<FollyResponseBodyStream>(
              std::move(transaction.body),
              responseLength(transaction.head),
              timeouts_.bodyIdle,
              lease));
    }
    return response;
  } catch (const Azure::Core::Http::TransportException&) {
    throw;
  } catch (const std::exception& exception) {
    throw transportException(exception);
  } catch (...) {
    throw Azure::Core::Http::TransportException(
        std::string(kTransportErrorPrefix) + "unknown exception");
  }
}

} // namespace facebook::velox::filesystems

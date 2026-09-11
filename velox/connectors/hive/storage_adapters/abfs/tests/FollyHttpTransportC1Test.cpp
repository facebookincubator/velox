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

#include "velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h"

#include <azure/core/http/http.hpp>
#include <azure/storage/blobs.hpp>

#include <folly/SocketAddress.h>
#include <folly/fibers/FiberManager.h>
#include <folly/fibers/FiberManagerMap.h>
#include <folly/futures/Promise.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>

namespace facebook::velox::filesystems {
namespace {

constexpr size_t kRequestCount{64};
constexpr size_t kConnectionLimit{32};
constexpr size_t kBodyBytes{4 * 1'024 * 1'024};
constexpr size_t kFiberStackBytes{256 * 1'024};
constexpr size_t kIngressBytes{64 * 1'024};
constexpr size_t kPatternBytes{251 * 64};
constexpr auto kWaitTimeout = std::chrono::seconds(30);
static_assert(kPatternBytes % 251 == 0);

uint8_t bodyByte(size_t offset) {
  return static_cast<uint8_t>(offset % 251);
}

uint64_t computeExpectedChecksum() {
  uint64_t checksum{0};
  for (size_t offset = 0; offset < kBodyBytes; ++offset) {
    checksum = checksum * 1'000'003 + bodyByte(offset);
  }
  return checksum;
}

const uint64_t kExpectedChecksum = computeExpectedChecksum();

class C1Server {
 public:
  struct Metrics {
    size_t acceptedConnections{0};
    size_t peakActive{0};
    size_t completed{0};
    size_t headersSent{0};
    bool allRangesValid{true};
    bool allGetsValid{true};
  };

  C1Server() {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create C1 listener");
    }
    int reuse{1};
    setsockopt(listenSocket_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, static_cast<int>(kConnectionLimit)) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind C1 listener");
    }
    setNonBlocking(listenSocket_);
    socklen_t length = sizeof(address);
    if (getsockname(
            listenSocket_, reinterpret_cast<sockaddr*>(&address), &length) <
        0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect C1 listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~C1Server() {
    stop();
  }

  C1Server(const C1Server&) = delete;
  C1Server& operator=(const C1Server&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  bool waitForPeak(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, timeout, [this] {
      return peakActive_ >= kConnectionLimit || failed_;
    });
  }

  void releaseHeaders() {
    {
      std::lock_guard lock(mutex_);
      releaseHeaders_ = true;
    }
    condition_.notify_all();
  }

  bool waitForCompletion(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, timeout, [this] {
      return completed_ == kRequestCount || failed_;
    });
  }

  Metrics metrics() const {
    std::lock_guard lock(mutex_);
    return {
        acceptedConnections_,
        peakActive_,
        completed_,
        headersSent_,
        allRangesValid_,
        allGetsValid_,
    };
  }

  bool failed() const {
    std::lock_guard lock(mutex_);
    return failed_;
  }

  void stop() noexcept {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
      releaseHeaders_ = true;
    }
    condition_.notify_all();
    if (listenSocket_ >= 0) {
      shutdown(listenSocket_, SHUT_RDWR);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    if (listenSocket_ >= 0) {
      close(listenSocket_);
      listenSocket_ = -1;
    }
  }

 private:
  struct Connection {
    int socket{-1};
    std::string request;
    bool requestComplete{false};
    bool responseStarted{false};
    size_t bodySent{0};
    size_t headerSent{0};
  };

  static void setNonBlocking(int socket) {
    const auto flags = fcntl(socket, F_GETFL, 0);
    if (flags < 0 || fcntl(socket, F_SETFL, flags | O_NONBLOCK) < 0) {
      throw std::runtime_error("failed to set C1 socket nonblocking");
    }
  }

  static bool containsInsensitive(const std::string& text, const char* value) {
    auto lower = text;
    std::transform(
        lower.begin(), lower.end(), lower.begin(), [](char character) {
          return static_cast<char>(
              std::tolower(static_cast<unsigned char>(character)));
        });
    return lower.find(value) != std::string::npos;
  }

  void closeConnection(Connection& connection) noexcept {
    if (connection.socket >= 0) {
      shutdown(connection.socket, SHUT_RDWR);
      close(connection.socket);
      connection.socket = -1;
    }
  }

  void failConnection(Connection& connection) noexcept {
    {
      std::lock_guard lock(mutex_);
      if (connection.requestComplete && active_ != 0) {
        --active_;
      }
      failed_ = true;
    }
    closeConnection(connection);
    condition_.notify_all();
  }

  void failServer() noexcept {
    {
      std::lock_guard lock(mutex_);
      failed_ = true;
    }
    condition_.notify_all();
  }

  bool canSendHeaders() {
    std::lock_guard lock(mutex_);
    return releaseHeaders_ && !stopping_;
  }

  void markRequest(Connection& connection) {
    const bool validGet = connection.request.rfind("GET ", 0) == 0;
    const bool validRange =
        containsInsensitive(connection.request, "range: bytes=0-4194303");
    {
      std::lock_guard lock(mutex_);
      allGetsValid_ = allGetsValid_ && validGet;
      allRangesValid_ = allRangesValid_ && validRange;
      ++active_;
      peakActive_ = std::max(peakActive_, active_);
    }
    condition_.notify_all();
    connection.requestComplete = true;
  }

  void finishResponse(Connection& connection) {
    {
      std::lock_guard lock(mutex_);
      if (active_ != 0) {
        --active_;
      }
      ++completed_;
    }
    connection.request.clear();
    connection.requestComplete = false;
    connection.responseStarted = false;
    connection.bodySent = 0;
    connection.headerSent = 0;
    condition_.notify_all();
  }

  void readRequests(std::vector<Connection>& connections) {
    for (auto& connection : connections) {
      if (connection.socket < 0 || connection.requestComplete) {
        continue;
      }
      std::array<char, 4'096> buffer{};
      while (true) {
        const auto result =
            recv(connection.socket, buffer.data(), buffer.size(), 0);
        if (result > 0) {
          connection.request.append(buffer.data(), static_cast<size_t>(result));
          if (connection.request.size() > 8 * 1'024) {
            failConnection(connection);
            break;
          }
          if (connection.request.find("\r\n\r\n") != std::string::npos) {
            markRequest(connection);
            break;
          }
          continue;
        }
        if (result == 0 || (result < 0 && errno != EAGAIN && errno != EINTR)) {
          failConnection(connection);
        }
        break;
      }
    }
  }

  void writeResponses(std::vector<Connection>& connections) {
    static constexpr char headers[] =
        "HTTP/1.1 206 Partial Content\r\n"
        "Content-Length: 4194304\r\n"
        "Content-Range: bytes 0-4194303/4194304\r\n"
        "Accept-Ranges: bytes\r\n"
        "ETag: \"c1-etag\"\r\n"
        "Last-Modified: Wed, 01 Jan 2020 00:00:00 GMT\r\n"
        "Content-Type: application/octet-stream\r\n"
        "x-ms-request-id: c1-request\r\n"
        "x-ms-version: 2020-10-02\r\n"
        "x-ms-blob-type: BlockBlob\r\n"
        "x-ms-creation-time: Wed, 01 Jan 2020 00:00:00 GMT\r\n"
        "x-ms-server-encrypted: true\r\n"
        "Date: Wed, 01 Jan 2020 00:00:00 GMT\r\n"
        "Connection: keep-alive\r\n\r\n";
    static const std::array<uint8_t, kPatternBytes> pattern = [] {
      std::array<uint8_t, kPatternBytes> value{};
      for (size_t index = 0; index < value.size(); ++index) {
        value[index] = bodyByte(index);
      }
      return value;
    }();

    for (auto& connection : connections) {
      if (connection.socket < 0 || !connection.requestComplete) {
        continue;
      }
      if (!connection.responseStarted) {
        if (!canSendHeaders()) {
          continue;
        }
        const auto result = send(
            connection.socket,
            headers + connection.headerSent,
            sizeof(headers) - 1 - connection.headerSent,
            MSG_NOSIGNAL);
        if (result > 0) {
          connection.headerSent += static_cast<size_t>(result);
          if (connection.headerSent == sizeof(headers) - 1) {
            connection.responseStarted = true;
            std::lock_guard lock(mutex_);
            ++headersSent_;
          }
        } else if (result < 0 && errno != EAGAIN && errno != EINTR) {
          failConnection(connection);
        }
        continue;
      }
      const auto remaining = kBodyBytes - connection.bodySent;
      if (remaining == 0) {
        finishResponse(connection);
        continue;
      }
      const auto offset = connection.bodySent % pattern.size();
      const auto count = std::min(remaining, pattern.size() - offset);
      const auto result =
          send(connection.socket, pattern.data() + offset, count, MSG_NOSIGNAL);
      if (result > 0) {
        connection.bodySent += static_cast<size_t>(result);
      } else if (result < 0 && errno != EAGAIN && errno != EINTR) {
        failConnection(connection);
      }
    }
  }

  void run() noexcept {
    try {
      std::vector<Connection> connections;
      while (true) {
        {
          std::lock_guard lock(mutex_);
          if (stopping_ || completed_ == kRequestCount) {
            break;
          }
        }
        pollfd listener{listenSocket_, POLLIN, 0};
        const auto pollResult = poll(&listener, 1, 10);
        if (pollResult < 0) {
          if (errno == EINTR) {
            continue;
          }
          failServer();
          break;
        }
        if (listener.revents & (POLLERR | POLLHUP | POLLNVAL)) {
          failServer();
          break;
        }
        if (listener.revents & POLLIN) {
          while (connections.size() < kConnectionLimit) {
            const auto socket = accept(listenSocket_, nullptr, nullptr);
            if (socket < 0) {
              if (errno == EINTR) {
                continue;
              }
              if (errno != EAGAIN && errno != EWOULDBLOCK) {
                failServer();
              }
              break;
            }
            setNonBlocking(socket);
            connections.push_back(Connection{socket, {}, false, false, 0, 0});
            std::lock_guard lock(mutex_);
            ++acceptedConnections_;
          }
        }
        readRequests(connections);
        writeResponses(connections);
        connections.erase(
            std::remove_if(
                connections.begin(),
                connections.end(),
                [](const Connection& connection) {
                  return connection.socket < 0;
                }),
            connections.end());
      }
      for (auto& connection : connections) {
        closeConnection(connection);
      }
    } catch (...) {
      failServer();
    }
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_{false};
  bool releaseHeaders_{false};
  bool failed_{false};
  bool allRangesValid_{true};
  bool allGetsValid_{true};
  size_t acceptedConnections_{0};
  size_t active_{0};
  size_t peakActive_{0};
  size_t completed_{0};
  size_t headersSent_{0};
};

class RssSampler {
 public:
  void start() {
    sample();
    thread_ = std::thread([this] {
      while (!stopping_) {
        sample();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
      sample();
    });
  }

  void stop() {
    stopping_ = true;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  size_t baseline() const {
    return baseline_.load();
  }

  size_t peak() const {
    return peak_.load();
  }

 private:
  void sample() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmRSS:", 0) != 0) {
        continue;
      }
      const auto firstDigit = line.find_first_of("0123456789");
      if (firstDigit == std::string::npos) {
        return;
      }
      size_t value{0};
      try {
        value = std::stoull(line.substr(firstDigit));
      } catch (...) {
        return;
      }
      auto expected = peak_.load();
      while (value > expected &&
             !peak_.compare_exchange_weak(expected, value)) {
      }
      size_t currentBaseline = baseline_.load();
      if (currentBaseline == 0) {
        baseline_.compare_exchange_strong(currentBaseline, value);
      }
      return;
    }
  }

  std::atomic<bool> stopping_{false};
  std::atomic<size_t> baseline_{0};
  std::atomic<size_t> peak_{0};
  std::thread thread_;
};

struct DownloadResult {
  uint64_t checksum{0};
  size_t bytes{0};
  std::thread::id runtimeThread;
};

struct MetricsResult {
  FollyHttpTransport::PoolMetrics pool;
  std::set<std::thread::id> runtimeThreads;
};

TEST(FollyHttpTransportC1Test, SixtyFourDownloadsOneEventBase) {
  C1Server server;
  RssSampler sampler;
  folly::ScopedEventBaseThread eventBaseThread("abfs-c1-event");
  auto* eventBase = eventBaseThread.getEventBase();
  folly::fibers::FiberManager::Options fiberOptions;
  fiberOptions.stackSize = kFiberStackBytes;
  fiberOptions.stackSizeMultiplier = 1;
  folly::fibers::FiberManager* fiberManager{nullptr};
  std::mutex observationsMutex;
  std::set<std::thread::id> runtimeThreads;
  auto transport = std::make_shared<std::shared_ptr<FollyHttpTransport>>();
  std::vector<folly::SemiFuture<DownloadResult>> futures;
  futures.reserve(kRequestCount);
  const auto serverPort = server.address().getPort();
  const auto submitterThread = std::this_thread::get_id();

  sampler.start();
  server.start();
  for (size_t requestIndex = 0; requestIndex < kRequestCount; ++requestIndex) {
    auto contract = folly::makePromiseContract<DownloadResult>();
    auto promise = std::make_shared<folly::Promise<DownloadResult>>(
        std::move(contract.promise));
    futures.push_back(std::move(contract.future));
    eventBase->runInEventBaseThread([eventBase,
                                     &fiberManager,
                                     &fiberOptions,
                                     &observationsMutex,
                                     &runtimeThreads,
                                     &server,
                                     serverPort,
                                     transport,
                                     promise = std::move(promise)]() mutable {
      if (fiberManager == nullptr) {
        fiberManager =
            &folly::fibers::getFiberManager(*eventBase, fiberOptions);
        AsyncChannelEndpoint endpoint;
        endpoint.connectAddress = server.address();
        endpoint.serverName = "127.0.0.1";
        auto factory = std::make_shared<EventSocketChannelFactory>(*eventBase);
        *transport = std::make_shared<FollyHttpTransport>(
            std::move(factory),
            endpoint,
            HttpLimits{},
            HttpTimeouts{},
            kConnectionLimit);
      }
      fiberManager->add([&observationsMutex,
                         &runtimeThreads,
                         serverPort,
                         transport,
                         promise = std::move(promise)]() mutable {
        try {
          const auto runtimeThread = std::this_thread::get_id();
          {
            std::lock_guard lock(observationsMutex);
            runtimeThreads.insert(runtimeThread);
          }
          Azure::Storage::Blobs::BlobClientOptions clientOptions;
          clientOptions.Transport.Transport = *transport;
          clientOptions.Retry.MaxRetries = 0;
          Azure::Storage::Blobs::BlobClient client(
              "http://127.0.0.1:" + std::to_string(serverPort) +
                  "/c1/blob?sig=dummy",
              clientOptions);
          Azure::Storage::Blobs::DownloadBlobOptions downloadOptions;
          Azure::Core::Http::HttpRange range;
          range.Offset = 0;
          range.Length = static_cast<int64_t>(kBodyBytes);
          downloadOptions.Range = range;
          auto response = client.Download(downloadOptions);
          if (response.Value.BlobSize != kBodyBytes ||
              response.Value.ContentRange.Offset != 0 ||
              !response.Value.ContentRange.Length.HasValue() ||
              response.Value.ContentRange.Length.Value() != kBodyBytes) {
            throw std::runtime_error("C1 Azure range metadata mismatch");
          }
          std::array<uint8_t, kPatternBytes> buffer{};
          uint64_t checksum{0};
          size_t bytes{0};
          while (true) {
            const auto count =
                response.Value.BodyStream->Read(buffer.data(), buffer.size());
            if (count == 0) {
              break;
            }
            for (size_t index = 0; index < count; ++index) {
              checksum = checksum * 1'000'003 + buffer[index];
            }
            bytes += count;
          }
          if (bytes != kBodyBytes || checksum != kExpectedChecksum) {
            throw std::runtime_error("C1 body verification mismatch");
          }
          promise->setValue(DownloadResult{checksum, bytes, runtimeThread});
        } catch (...) {
          promise->setException(
              folly::exception_wrapper(std::current_exception()));
        }
      });
    });
  }

  const bool reachedPeak =
      server.waitForPeak(std::chrono::seconds(15)) && !server.failed();
  const bool headersStillPending = server.metrics().headersSent == 0;
  const bool futuresPending =
      std::all_of(futures.begin(), futures.end(), [](const auto& future) {
        return !future.isReady();
      });
  server.releaseHeaders();
  const bool completed = server.waitForCompletion(kWaitTimeout);
  std::vector<DownloadResult> results;
  std::exception_ptr failure;
  for (auto& future : futures) {
    try {
      results.push_back(std::move(future).get(kWaitTimeout));
    } catch (...) {
      failure = std::current_exception();
    }
  }

  auto metricsContract = folly::makePromiseContract<MetricsResult>();
  auto metricsPromise = std::make_shared<folly::Promise<MetricsResult>>(
      std::move(metricsContract.promise));
  auto metricsFuture = std::move(metricsContract.future);
  eventBase->runInEventBaseThread([&observationsMutex,
                                   &runtimeThreads,
                                   transport,
                                   promise =
                                       std::move(metricsPromise)]() mutable {
    try {
      std::set<std::thread::id> threads;
      {
        std::lock_guard lock(observationsMutex);
        threads = runtimeThreads;
      }
      if (!*transport) {
        throw std::runtime_error("C1 transport initialization failed");
      }
      auto result =
          MetricsResult{(*transport)->poolMetrics(), std::move(threads)};
      promise->setValue(std::move(result));
    } catch (...) {
      promise->setException(folly::exception_wrapper(std::current_exception()));
    }
  });
  MetricsResult metrics;
  try {
    metrics = std::move(metricsFuture).get(kWaitTimeout);
  } catch (...) {
    failure = std::current_exception();
  }
  server.stop();
  sampler.stop();

  const auto serverMetrics = server.metrics();
  const auto rssBaseline = sampler.baseline();
  const auto rssPeak = sampler.peak();
  const auto rssGrowth = rssPeak >= rssBaseline ? rssPeak - rssBaseline : 0;
  const auto modeledBound =
      (kRequestCount * kFiberStackBytes + kConnectionLimit * kIngressBytes) /
      1'024;
  std::cout << "C1_METRICS requests=64 connections="
            << serverMetrics.acceptedConnections
            << " peak_active=" << serverMetrics.peakActive
            << " runtime_threads=1 body_bytes=" << (kRequestCount * kBodyBytes)
            << " body_bytes_each=" << kBodyBytes
            << " rss_baseline_kib=" << rssBaseline
            << " rss_peak_kib=" << rssPeak << " rss_growth_kib=" << rssGrowth
            << " modeled_bound_kib=" << modeledBound
            << " full_buffer_kib=" << (kRequestCount * kBodyBytes) / 1'024
            << '\n';

  if (failure != nullptr) {
    try {
      std::rethrow_exception(failure);
    } catch (const std::exception&) {
      ADD_FAILURE()
          << "C1 harness failure during download or metrics collection";
    } catch (...) {
      ADD_FAILURE()
          << "C1 harness failure during download or metrics collection";
    }
  }
  ASSERT_GT(rssBaseline, 0);
  ASSERT_GE(rssPeak, rssBaseline);
  ASSERT_TRUE(reachedPeak);
  ASSERT_FALSE(server.failed());
  ASSERT_TRUE(headersStillPending);
  ASSERT_TRUE(futuresPending);
  ASSERT_TRUE(completed);
  ASSERT_FALSE(serverMetrics.headersSent == 0);
  ASSERT_TRUE(serverMetrics.allGetsValid);
  ASSERT_TRUE(serverMetrics.allRangesValid);
  ASSERT_EQ(serverMetrics.completed, kRequestCount);
  ASSERT_EQ(serverMetrics.acceptedConnections, kConnectionLimit);
  ASSERT_GE(serverMetrics.peakActive, kConnectionLimit);
  ASSERT_EQ(results.size(), kRequestCount);
  ASSERT_EQ(metrics.runtimeThreads.size(), 1);
  ASSERT_NE(*metrics.runtimeThreads.begin(), submitterThread);
  ASSERT_EQ(metrics.pool.maxConnections, kConnectionLimit);
  ASSERT_GE(metrics.pool.peakLeasedConnections, kConnectionLimit);
  ASSERT_EQ(metrics.pool.waitingFibers, 0);
  ASSERT_EQ(metrics.pool.leasedConnections, 0);
  ASSERT_LE(metrics.pool.totalConnections, kConnectionLimit);
  ASSERT_LE(metrics.pool.idleConnections, kConnectionLimit);
  ASSERT_EQ(metrics.pool.peakLeasedConnections, kConnectionLimit);
  ASSERT_LT(rssGrowth, (kRequestCount * kBodyBytes) / (2 * 1'024));
  for (const auto& result : results) {
    ASSERT_EQ(result.bytes, kBodyBytes);
    ASSERT_EQ(result.checksum, kExpectedChecksum);
    ASSERT_NE(result.runtimeThread, submitterThread);
  }
}

} // namespace
} // namespace facebook::velox::filesystems

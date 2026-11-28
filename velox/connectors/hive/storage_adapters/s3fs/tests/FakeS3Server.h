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

#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"

#include <fmt/format.h>
#include <folly/String.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace facebook::velox::filesystems::test {

/// An HTTP server that answers the HeadObject and ranged GetObject requests the
/// AWS SDK sends for one object, so a test can script responses an S3 service
/// does not produce on demand, such as a 503 whose body is longer than the
/// requested range. Every response closes its connection.
class FakeS3Server {
 public:
  /// A GetObject request: its 1-based count and the inclusive byte range.
  struct GetRequest {
    int number;
    uint64_t begin;
    uint64_t end;

    uint64_t length() const {
      return end - begin + 1;
    }
  };

  struct Response {
    int status;
    std::string reason;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
  };

  using GetHandler = std::function<Response(const GetRequest&)>;

  /// Serves 'object'; GETs get the requested range unless onGet() set a
  /// handler.
  explicit FakeS3Server(std::string object) : object_(std::move(object)) {}

  ~FakeS3Server() {
    stop();
  }

  void onGet(GetHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    getHandler_ = std::move(handler);
  }

  /// The bytes 'get' asks for, as a 206.
  Response rangeOf(const GetRequest& get) const {
    return {
        206,
        "Partial Content",
        object_.substr(get.begin, get.length()),
        {{"Content-Range",
          fmt::format("bytes {}-{}/{}", get.begin, get.end, object_.size())}}};
  }

  /// The whole object as a 200, as a server that ignores the range sends it.
  Response wholeObject() const {
    return {200, "OK", object_, {}};
  }

  /// A 503 with an S3 SlowDown XML body of at least 'minBodyBytes'.
  static Response slowDown(size_t minBodyBytes) {
    static constexpr std::string_view kHead =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>SlowDown</Code>"
        "<Message>Please reduce your request rate.</Message><HostId>";
    static constexpr std::string_view kTail = "</HostId></Error>";
    std::string body(kHead);
    if (minBodyBytes > kHead.size() + kTail.size()) {
      body.append(minBodyBytes - kHead.size() - kTail.size(), 'x');
    }
    body.append(kTail);
    return {503, "Service Unavailable", std::move(body), {}};
  }

  void start() {
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    VELOX_CHECK_GE(listenFd_, 0, "socket: {}", folly::errnoStr(errno));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    VELOX_CHECK_EQ(
        ::bind(
            listenFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
        0,
        "bind: {}",
        folly::errnoStr(errno));
    socklen_t addressLength = sizeof(address);
    VELOX_CHECK_EQ(
        ::getsockname(
            listenFd_, reinterpret_cast<sockaddr*>(&address), &addressLength),
        0);
    port_ = ntohs(address.sin_port);
    VELOX_CHECK_EQ(
        ::listen(listenFd_, 16), 0, "listen: {}", folly::errnoStr(errno));
    thread_ = std::thread([this]() { serve(); });
  }

  void stop() {
    if (listenFd_ < 0) {
      return;
    }
    stopped_ = true;
    if (thread_.joinable()) {
      thread_.join();
    }
    ::close(listenFd_);
    listenFd_ = -1;
  }

  /// The S3 connector config that points at this server.
  std::shared_ptr<const config::ConfigBase> s3Config(
      std::unordered_map<std::string, std::string> overrides = {}) const {
    std::unordered_map<std::string, std::string> config{
        {"s3.aws-access-key", "fake"},
        {"s3.aws-secret-key", "fake"},
        {"s3.endpoint", fmt::format("127.0.0.1:{}", port_)},
        {"s3.ssl.enabled", "false"},
        {"s3.path-style-access", "true"},
    };
    for (auto& [name, value] : overrides) {
      config[name] = std::move(value);
    }
    return std::make_shared<const config::ConfigBase>(std::move(config));
  }

  /// GETs answered so far.
  int gets() const {
    return gets_;
  }

 private:
  void serve() {
    const int listenFd = listenFd_;
    while (!stopped_) {
      // Polls so that stop() is noticed between connections.
      pollfd listener{listenFd, POLLIN, 0};
      const int ready = ::poll(&listener, 1, 50);
      if (ready < 0 && errno != EINTR) {
        return;
      }
      if (ready <= 0) {
        continue;
      }
      const int fd = ::accept(listenFd, nullptr, nullptr);
      if (fd < 0) {
        continue;
      }
      try {
        handle(fd);
      } catch (const std::exception& e) {
        // A scripting mistake fails the client's request instead of the test
        // binary.
        send(fd, "GET", {500, "Internal Server Error", e.what(), {}});
      }
      ::close(fd);
    }
  }

  // Reads one request's header block and answers it.
  void handle(int fd) {
    timeval timeout{.tv_sec = 5, .tv_usec = 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    std::string request;
    char chunk[4096];
    while (request.find("\r\n\r\n") == std::string::npos) {
      const auto n = ::recv(fd, chunk, sizeof(chunk), 0);
      if (n <= 0) {
        return;
      }
      request.append(chunk, n);
    }

    const auto method = request.substr(0, request.find(' '));
    Response response;
    if (method == "HEAD") {
      response = {200, "OK", "", {{"Accept-Ranges", "bytes"}}};
      response.headers.emplace_back(
          "Content-Length", std::to_string(object_.size()));
    } else {
      const GetRequest get{++gets_, 0, object_.size() - 1};
      const auto ranged = parseRange(request, get);
      GetHandler handler;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        handler = getHandler_;
      }
      response = handler ? handler(ranged) : rangeOf(ranged);
    }
    send(fd, method, std::move(response));
  }

  // Writes 'response' to a request of 'method'.
  static void send(int fd, const std::string& method, Response response) {
#ifdef MSG_NOSIGNAL
    constexpr int kSendFlags = MSG_NOSIGNAL;
#else
    // macOS has no MSG_NOSIGNAL; the socket option keeps a closed peer from
    // raising SIGPIPE instead.
    constexpr int kSendFlags = 0;
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    std::string wire = fmt::format(
        "HTTP/1.1 {} {}\r\nConnection: close\r\n",
        response.status,
        response.reason);
    if (method != "HEAD") {
      response.headers.emplace_back(
          "Content-Length", std::to_string(response.body.size()));
    }
    for (const auto& [name, value] : response.headers) {
      wire += fmt::format("{}: {}\r\n", name, value);
    }
    wire += "\r\n";
    if (method != "HEAD") {
      wire += response.body;
    }
    size_t sent = 0;
    while (sent < wire.size()) {
      const auto n =
          ::send(fd, wire.data() + sent, wire.size() - sent, kSendFlags);
      if (n <= 0) {
        return;
      }
      sent += n;
    }
  }

  // Applies the request's "Range: bytes=begin-end" header, if any, to 'get'.
  static GetRequest parseRange(const std::string& request, GetRequest get) {
    static constexpr std::string_view kRange = "\r\nrange: bytes=";
    std::string lower = request;
    folly::toLowerAscii(lower);
    const auto start = lower.find(kRange);
    if (start == std::string::npos) {
      return get;
    }
    const auto spec = lower.substr(
        start + kRange.size(),
        lower.find("\r\n", start + kRange.size()) - start - kRange.size());
    const auto dash = spec.find('-');
    get.begin = std::stoull(spec.substr(0, dash));
    if (dash + 1 < spec.size()) {
      get.end = std::stoull(spec.substr(dash + 1));
    }
    return get;
  }

  const std::string object_;
  mutable std::mutex mutex_;
  GetHandler getHandler_;
  int listenFd_{-1};
  int port_{0};
  std::thread thread_;
  std::atomic<bool> stopped_{false};
  std::atomic<int> gets_{0};
};

} // namespace facebook::velox::filesystems::test

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
#include "velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.h"
#include "velox/connectors/hive/storage_adapters/abfs/HttpConnection.h"

#include <azure/core/http/http.hpp>
#include <azure/core/url.hpp>
#include <azure/storage/blobs.hpp>

#include <folly/ScopeGuard.h>
#include <folly/SocketAddress.h>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManagerMap.h>
#include <folly/futures/Promise.h>
#include <folly/io/async/AsyncSocketException.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/portability/OpenSSL.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace facebook::velox::filesystems {
namespace {

constexpr std::array<char, 6> kMarker{'A', 'B', 'F', 'S', '!'};

class LoopbackMarkerServer {
 public:
  LoopbackMarkerServer() {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create loopback listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 1) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind loopback listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect loopback listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~LoopbackMarkerServer() {
    stop();
  }

  LoopbackMarkerServer(const LoopbackMarkerServer&) = delete;
  LoopbackMarkerServer& operator=(const LoopbackMarkerServer&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  bool failed() const noexcept {
    return failed_;
  }

  void stop() noexcept {
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
  void run() noexcept {
    pollfd pollDescriptor{listenSocket_, POLLIN, 0};
    if (poll(&pollDescriptor, 1, 2'000) <= 0 ||
        !(pollDescriptor.revents & POLLIN)) {
      failed_ = true;
      return;
    }
    const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
    if (clientSocket < 0) {
      failed_ = true;
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(75));
    size_t bytesSent{0};
    while (bytesSent < kMarker.size() - 1) {
      const auto result = send(
          clientSocket,
          kMarker.data() + bytesSent,
          kMarker.size() - 1 - bytesSent,
          MSG_NOSIGNAL);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        failed_ = true;
        break;
      }
      bytesSent += static_cast<size_t>(result);
    }
    shutdown(clientSocket, SHUT_RDWR);
    close(clientSocket);
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  bool failed_{false};
  std::thread thread_;
};

enum class MarkerTerminalState { kPending, kSuccess, kFailure };

class MarkerReadCallback final : public folly::AsyncTransport::ReadCallback {
 public:
  explicit MarkerReadCallback(folly::fibers::Baton* baton) : baton_(baton) {}

  void getReadBuffer(void** buffer, size_t* length) noexcept override {
    const auto bufferOffset = std::min(bytesRead_, buffer_.size() - 1);
    *buffer = buffer_.data() + bufferOffset;
    *length = buffer_.size() - bufferOffset;
  }

  void readDataAvailable(size_t length) noexcept override {
    try {
      if (state_ != MarkerTerminalState::kPending ||
          bytesRead_ > kMarker.size() - 1 ||
          length > kMarker.size() - 1 - bytesRead_) {
        complete(MarkerTerminalState::kFailure);
        return;
      }
      bytesRead_ += length;
      if (bytesRead_ >= kMarker.size() - 1) {
        complete(
            std::memcmp(buffer_.data(), kMarker.data(), kMarker.size() - 1) == 0
                ? MarkerTerminalState::kSuccess
                : MarkerTerminalState::kFailure);
      }
    } catch (...) {
      complete(MarkerTerminalState::kFailure);
    }
  }

  void readEOF() noexcept override {
    complete(MarkerTerminalState::kFailure);
  }

  void readErr(const folly::AsyncSocketException&) noexcept override {
    complete(MarkerTerminalState::kFailure);
  }

  bool markerReceived() const noexcept {
    return state_ == MarkerTerminalState::kSuccess;
  }

  bool failed() const noexcept {
    return state_ == MarkerTerminalState::kFailure;
  }

 private:
  void complete(MarkerTerminalState state) noexcept {
    if (state_ != MarkerTerminalState::kPending) {
      return;
    }
    state_ = state;
    postOnce();
  }

  void postOnce() noexcept {
    if (!posted_) {
      posted_ = true;
      baton_->post();
    }
  }

  folly::fibers::Baton* baton_;
  std::array<char, kMarker.size()> buffer_{};
  size_t bytesRead_{0};
  MarkerTerminalState state_{MarkerTerminalState::kPending};
  bool posted_{false};
};

class BoundLoopbackSocket {
 public:
  BoundLoopbackSocket() {
    socketDescriptor_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socketDescriptor_ < 0) {
      throw std::runtime_error("failed to create reserved-port socket");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            socketDescriptor_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0) {
      close(socketDescriptor_);
      socketDescriptor_ = -1;
      throw std::runtime_error("failed to reserve loopback port");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            socketDescriptor_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(socketDescriptor_);
      socketDescriptor_ = -1;
      throw std::runtime_error("failed to inspect reserved loopback port");
    }
    port_ = ntohs(address.sin_port);
  }

  ~BoundLoopbackSocket() {
    if (socketDescriptor_ >= 0) {
      close(socketDescriptor_);
    }
  }

  BoundLoopbackSocket(const BoundLoopbackSocket&) = delete;
  BoundLoopbackSocket& operator=(const BoundLoopbackSocket&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

 private:
  int socketDescriptor_{-1};
  uint16_t port_{0};
};

class LoopbackTlsHttpServer {
 public:
  LoopbackTlsHttpServer(
      std::string certificatePath,
      std::string keyPath,
      bool stallHandshake = false)
      : certificatePath_(std::move(certificatePath)),
        keyPath_(std::move(keyPath)),
        stallHandshake_(stallHandshake) {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create TLS loopback listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 1) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind TLS loopback listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect TLS loopback listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~LoopbackTlsHttpServer() {
    stop();
  }

  LoopbackTlsHttpServer(const LoopbackTlsHttpServer&) = delete;
  LoopbackTlsHttpServer& operator=(const LoopbackTlsHttpServer&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  bool failed() const noexcept {
    return failed_;
  }

  bool served() const noexcept {
    return served_;
  }

  void stop() noexcept {
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
  static bool setNonBlocking(int descriptor) noexcept {
    const auto flags = fcntl(descriptor, F_GETFL, 0);
    return flags >= 0 && fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0;
  }

  static bool waitForSocket(int descriptor, short events) noexcept {
    pollfd descriptorState{descriptor, events, 0};
    while (true) {
      const auto result = poll(&descriptorState, 1, 2'000);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      return result > 0 && (descriptorState.revents & events) != 0;
    }
  }

  bool handshake(SSL* ssl, int clientSocket) noexcept {
    while (true) {
      const auto result = SSL_accept(ssl);
      if (result == 1) {
        return true;
      }
      const auto error = SSL_get_error(ssl, result);
      if (error == SSL_ERROR_WANT_READ && waitForSocket(clientSocket, POLLIN)) {
        continue;
      }
      if (error == SSL_ERROR_WANT_WRITE &&
          waitForSocket(clientSocket, POLLOUT)) {
        continue;
      }
      return false;
    }
  }

  bool readRequest(SSL* ssl, int clientSocket) noexcept {
    constexpr size_t kRequestLimit{4 * 1'024};
    constexpr std::array<char, 4> kHeaderEnd{'\r', '\n', '\r', '\n'};
    std::array<char, kRequestLimit> request{};
    size_t requestSize{0};
    while (requestSize < request.size()) {
      const auto bytes = SSL_read(
          ssl, request.data() + requestSize, request.size() - requestSize);
      if (bytes > 0) {
        requestSize += static_cast<size_t>(bytes);
        if (std::search(
                request.begin(),
                request.begin() + requestSize,
                kHeaderEnd.begin(),
                kHeaderEnd.end()) != request.begin() + requestSize) {
          return true;
        }
        continue;
      }
      const auto error = SSL_get_error(ssl, bytes);
      if (error == SSL_ERROR_WANT_READ && waitForSocket(clientSocket, POLLIN)) {
        continue;
      }
      if (error == SSL_ERROR_WANT_WRITE &&
          waitForSocket(clientSocket, POLLOUT)) {
        continue;
      }
      return false;
    }
    return false;
  }

  bool sendResponse(SSL* ssl, int clientSocket) noexcept {
    static constexpr char kResponse[] =
        "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\n"
        "tls-response";
    size_t sent{0};
    while (sent < sizeof(kResponse) - 1) {
      const auto bytes =
          SSL_write(ssl, kResponse + sent, sizeof(kResponse) - 1 - sent);
      if (bytes > 0) {
        sent += static_cast<size_t>(bytes);
        continue;
      }
      const auto error = SSL_get_error(ssl, bytes);
      if (error == SSL_ERROR_WANT_READ && waitForSocket(clientSocket, POLLIN)) {
        continue;
      }
      if (error == SSL_ERROR_WANT_WRITE &&
          waitForSocket(clientSocket, POLLOUT)) {
        continue;
      }
      return false;
    }
    return true;
  }

  void run() noexcept {
    sigset_t blockedSignals;
    sigemptyset(&blockedSignals);
    sigaddset(&blockedSignals, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &blockedSignals, nullptr);

    if (!waitForSocket(listenSocket_, POLLIN)) {
      failed_ = true;
      return;
    }
    const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
    if (clientSocket < 0) {
      failed_ = true;
      return;
    }
    auto closeClientSocket = folly::makeGuard([&] { close(clientSocket); });
    if (stallHandshake_) {
      waitForSocket(clientSocket, POLLIN);
      return;
    }
    if (!setNonBlocking(clientSocket)) {
      failed_ = true;
      return;
    }

    SSL_CTX* context = SSL_CTX_new(TLS_server_method());
    auto freeContext = folly::makeGuard([&] { SSL_CTX_free(context); });
    SSL* ssl = nullptr;
    if (context != nullptr) {
      SSL_CTX_set_min_proto_version(context, TLS1_2_VERSION);
      if (SSL_CTX_use_certificate_file(
              context, certificatePath_.c_str(), SSL_FILETYPE_PEM) == 1 &&
          SSL_CTX_use_PrivateKey_file(
              context, keyPath_.c_str(), SSL_FILETYPE_PEM) == 1 &&
          SSL_CTX_check_private_key(context) == 1) {
        ssl = SSL_new(context);
      }
    }
    if (ssl == nullptr || SSL_set_fd(ssl, clientSocket) != 1) {
      failed_ = true;
      SSL_free(ssl);
      return;
    }
    auto freeSsl = folly::makeGuard([&] { SSL_free(ssl); });
    if (handshake(ssl, clientSocket)) {
      served_ =
          readRequest(ssl, clientSocket) && sendResponse(ssl, clientSocket);
    }
    if (served_) {
      SSL_shutdown(ssl);
    }
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::string certificatePath_;
  std::string keyPath_;
  bool stallHandshake_{false};
  bool failed_{false};
  bool served_{false};
  std::thread thread_;
};

class LoopbackHttpServer {
 public:
  LoopbackHttpServer(
      std::string response,
      std::vector<size_t> fragments,
      std::chrono::milliseconds delay,
      bool closeClient,
      bool resetClient = false)
      : response_(std::move(response)),
        fragments_(std::move(fragments)),
        delay_(delay),
        closeClient_(closeClient),
        resetClient_(resetClient) {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create HTTP loopback listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 1) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind HTTP loopback listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect HTTP loopback listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~LoopbackHttpServer() {
    stop();
  }

  LoopbackHttpServer(const LoopbackHttpServer&) = delete;
  LoopbackHttpServer& operator=(const LoopbackHttpServer&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  bool failed() const noexcept {
    return failed_;
  }

  void stop() noexcept {
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
  void sendBytes(int clientSocket, const char* data, size_t size) noexcept {
    size_t sent{0};
    while (sent < size) {
      const auto result =
          send(clientSocket, data + sent, size - sent, MSG_NOSIGNAL);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        failed_ = true;
        return;
      }
      sent += static_cast<size_t>(result);
    }
  }

  void run() noexcept {
    pollfd pollDescriptor{listenSocket_, POLLIN, 0};
    if (poll(&pollDescriptor, 1, 2'000) <= 0 ||
        !(pollDescriptor.revents & POLLIN)) {
      failed_ = true;
      return;
    }
    const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
    if (clientSocket < 0) {
      failed_ = true;
      return;
    }
    std::array<char, 4 * 1'024> request{};
    size_t requestSize{0};
    constexpr std::array<char, 4> kHeaderEnd{'\r', '\n', '\r', '\n'};
    while (std::search(
               request.begin(),
               request.begin() + requestSize,
               kHeaderEnd.begin(),
               kHeaderEnd.end()) == request.begin() + requestSize) {
      pollfd clientPoll{clientSocket, POLLIN, 0};
      if (poll(&clientPoll, 1, 2'000) <= 0 || !(clientPoll.revents & POLLIN) ||
          requestSize == request.size()) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      const auto bytes = recv(
          clientSocket,
          request.data() + requestSize,
          request.size() - requestSize,
          0);
      if (bytes <= 0) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      requestSize += static_cast<size_t>(bytes);
    }
    size_t responseOffset{0};
    if (fragments_.empty()) {
      sendBytes(clientSocket, response_.data(), response_.size());
    } else {
      for (const auto fragmentSize : fragments_) {
        if (responseOffset >= response_.size()) {
          break;
        }
        const auto bytes =
            std::min(fragmentSize, response_.size() - responseOffset);
        sendBytes(clientSocket, response_.data() + responseOffset, bytes);
        responseOffset += bytes;
        if (delay_.count() > 0) {
          std::this_thread::sleep_for(delay_);
        }
      }
      if (responseOffset < response_.size()) {
        sendBytes(
            clientSocket,
            response_.data() + responseOffset,
            response_.size() - responseOffset);
      }
    }
    if (resetClient_) {
      linger reset{1, 0};
      setsockopt(clientSocket, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset));
      close(clientSocket);
      return;
    }
    if (closeClient_) {
      shutdown(clientSocket, SHUT_RDWR);
    }
    close(clientSocket);
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::string response_;
  std::vector<size_t> fragments_;
  std::chrono::milliseconds delay_;
  bool closeClient_{false};
  bool resetClient_{false};
  bool failed_{false};
  std::thread thread_;
};

class CapturingLoopbackHttpServer {
 public:
  explicit CapturingLoopbackHttpServer(std::string response)
      : response_(std::move(response)) {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create request capture listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 1) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind request capture listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect request capture listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~CapturingLoopbackHttpServer() {
    stop();
  }

  CapturingLoopbackHttpServer(const CapturingLoopbackHttpServer&) = delete;
  CapturingLoopbackHttpServer& operator=(const CapturingLoopbackHttpServer&) =
      delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  const std::string& request() const noexcept {
    return request_;
  }

  bool failed() const noexcept {
    return failed_;
  }

  void stop() noexcept {
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
  void sendResponse(int clientSocket) noexcept {
    size_t sent{0};
    while (sent < response_.size()) {
      const auto result = send(
          clientSocket,
          response_.data() + sent,
          response_.size() - sent,
          MSG_NOSIGNAL);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        failed_ = true;
        return;
      }
      sent += static_cast<size_t>(result);
    }
  }

  void run() noexcept {
    pollfd listenPoll{listenSocket_, POLLIN, 0};
    if (poll(&listenPoll, 1, 2'000) <= 0 || !(listenPoll.revents & POLLIN)) {
      failed_ = true;
      return;
    }
    const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
    if (clientSocket < 0) {
      failed_ = true;
      return;
    }
    constexpr size_t kCaptureLimit{8 * 1'024};
    while (request_.find("\r\n\r\n") == std::string::npos) {
      pollfd clientPoll{clientSocket, POLLIN, 0};
      if (poll(&clientPoll, 1, 2'000) <= 0 || !(clientPoll.revents & POLLIN)) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      std::array<char, 512> buffer{};
      const auto bytes = recv(clientSocket, buffer.data(), buffer.size(), 0);
      if (bytes <= 0 ||
          request_.size() + static_cast<size_t>(bytes) > kCaptureLimit) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      request_.append(buffer.data(), static_cast<size_t>(bytes));
    }
    const auto headerEnd = request_.find("\r\n\r\n") + 4;
    size_t bodyLength{0};
    const auto lengthStart = request_.find("Content-Length: ");
    if (lengthStart != std::string::npos) {
      const auto valueStart = lengthStart + std::strlen("Content-Length: ");
      const auto valueEnd = request_.find("\r\n", valueStart);
      const auto value = request_.substr(valueStart, valueEnd - valueStart);
      const auto parseResult = std::from_chars(
          value.data(), value.data() + value.size(), bodyLength);
      if (parseResult.ec != std::errc{} ||
          parseResult.ptr != value.data() + value.size()) {
        failed_ = true;
        close(clientSocket);
        return;
      }
    }
    while (request_.size() < headerEnd + bodyLength) {
      pollfd clientPoll{clientSocket, POLLIN, 0};
      if (poll(&clientPoll, 1, 2'000) <= 0 || !(clientPoll.revents & POLLIN)) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      std::array<char, 512> buffer{};
      const auto bytes = recv(clientSocket, buffer.data(), buffer.size(), 0);
      if (bytes <= 0 ||
          request_.size() + static_cast<size_t>(bytes) > kCaptureLimit) {
        failed_ = true;
        close(clientSocket);
        return;
      }
      request_.append(buffer.data(), static_cast<size_t>(bytes));
    }
    sendResponse(clientSocket);
    shutdown(clientSocket, SHUT_RDWR);
    close(clientSocket);
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::string response_;
  std::string request_;
  bool failed_{false};
  std::thread thread_;
};

class CountingChannelFactory final : public AsyncChannelFactory {
 public:
  folly::AsyncTransportWrapper::UniquePtr connect(
      const AsyncChannelEndpoint&) override {
    ++connectCount;
    throw std::runtime_error("counting factory must not connect");
  }

  size_t connectCount{0};
};

class KeepAliveLoopbackHttpServer {
 public:
  explicit KeepAliveLoopbackHttpServer(
      std::chrono::milliseconds firstResponseDelay = {},
      bool requireNewConnection = false,
      size_t expectedRequests = 2)
      : firstResponseDelay_(firstResponseDelay),
        requireNewConnection_(requireNewConnection),
        expectedRequests_(expectedRequests) {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create keep-alive listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 2) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind keep-alive listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect keep-alive listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~KeepAliveLoopbackHttpServer() {
    stop();
  }

  KeepAliveLoopbackHttpServer(const KeepAliveLoopbackHttpServer&) = delete;
  KeepAliveLoopbackHttpServer& operator=(const KeepAliveLoopbackHttpServer&) =
      delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  void stop() noexcept {
    if (listenSocket_ >= 0) {
      shutdown(listenSocket_, SHUT_RDWR);
    }
    const auto clientSocket = activeClientSocket_.load();
    if (clientSocket >= 0) {
      shutdown(clientSocket, SHUT_RDWR);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    if (listenSocket_ >= 0) {
      close(listenSocket_);
      listenSocket_ = -1;
    }
  }

  bool failed() const noexcept {
    return failed_;
  }

  size_t acceptedConnections() const noexcept {
    return acceptedConnections_;
  }

  size_t requests() const noexcept {
    return requests_;
  }

  bool clientClosed() const noexcept {
    return clientClosed_;
  }

 private:
  bool readRequest(int clientSocket) noexcept {
    std::array<char, 4 * 1'024> buffer{};
    size_t bytesRead{0};
    constexpr std::array<char, 4> kHeaderEnd{'\r', '\n', '\r', '\n'};
    while (bytesRead < buffer.size()) {
      pollfd clientPoll{clientSocket, POLLIN, 0};
      if (poll(&clientPoll, 1, 2'000) <= 0 || !(clientPoll.revents & POLLIN)) {
        return false;
      }
      const auto bytes = recv(
          clientSocket,
          buffer.data() + bytesRead,
          buffer.size() - bytesRead,
          0);
      if (bytes == 0) {
        clientClosed_ = true;
        return false;
      }
      if (bytes < 0) {
        return false;
      }
      bytesRead += static_cast<size_t>(bytes);
      if (std::search(
              buffer.begin(),
              buffer.begin() + bytesRead,
              kHeaderEnd.begin(),
              kHeaderEnd.end()) != buffer.begin() + bytesRead) {
        return true;
      }
    }
    return false;
  }

  static bool sendResponse(int clientSocket, bool keepAlive) noexcept {
    const auto response = keepAlive
        ? std::string(
              "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: keep-alive\r\n\r\nbody")
        : std::string(
              "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbody");
    size_t bytesSent{0};
    while (bytesSent < response.size()) {
      const auto bytes = send(
          clientSocket,
          response.data() + bytesSent,
          response.size() - bytesSent,
          MSG_NOSIGNAL);
      if (bytes < 0 && errno == EINTR) {
        continue;
      }
      if (bytes <= 0) {
        return false;
      }
      bytesSent += static_cast<size_t>(bytes);
    }
    return true;
  }

  void run() noexcept {
    pollfd listener{listenSocket_, POLLIN, 0};
    if (poll(&listener, 1, 2'000) <= 0 || !(listener.revents & POLLIN)) {
      failed_ = true;
      return;
    }
    while (requests_ < expectedRequests_) {
      const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
      if (clientSocket < 0) {
        failed_ = true;
        return;
      }
      activeClientSocket_ = clientSocket;
      ++acceptedConnections_;
      for (size_t requestNumber{0}; requestNumber < expectedRequests_;
           ++requestNumber) {
        if (!readRequest(clientSocket)) {
          close(clientSocket);
          if (requireNewConnection_) {
            break;
          }
          return;
        }
        ++requests_;
        if (requests_ == 1 && firstResponseDelay_.count() > 0) {
          std::this_thread::sleep_for(firstResponseDelay_);
        }
        if (!sendResponse(clientSocket, true)) {
          close(clientSocket);
          return;
        }
        if (requireNewConnection_) {
          break;
        }
      }
      shutdown(clientSocket, SHUT_RDWR);
      close(clientSocket);
      activeClientSocket_ = -1;
    }
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::chrono::milliseconds firstResponseDelay_;
  bool requireNewConnection_{false};
  size_t expectedRequests_{2};
  std::atomic<int> activeClientSocket_{-1};
  std::atomic<bool> failed_{false};
  std::atomic<bool> clientClosed_{false};
  std::atomic<size_t> acceptedConnections_{0};
  std::atomic<size_t> requests_{0};
  std::thread thread_;
};

class BoundedBlobDownloadServer {
 public:
  explicit BoundedBlobDownloadServer(bool holdBody = false)
      : holdBody_(holdBody) {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create blob listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 1) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind blob listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect blob listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~BoundedBlobDownloadServer() {
    stop();
  }

  BoundedBlobDownloadServer(const BoundedBlobDownloadServer&) = delete;
  BoundedBlobDownloadServer& operator=(const BoundedBlobDownloadServer&) =
      delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  void stop() noexcept {
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

  bool failed() const noexcept {
    return failed_;
  }

  bool sawGet() const noexcept {
    return sawGet_;
  }

  bool sawRange() const noexcept {
    return sawRange_;
  }

  bool headersSent() const noexcept {
    return headersSent_;
  }

  bool bodyFullySent() const noexcept {
    return bodyFullySent_;
  }

  bool clientClosed() const noexcept {
    return clientClosed_;
  }

 private:
  static bool sendAll(int descriptor, const char* data, size_t size) noexcept {
    size_t sent{0};
    while (sent < size) {
      const auto bytes =
          send(descriptor, data + sent, size - sent, MSG_NOSIGNAL);
      if (bytes < 0 && errno == EINTR) {
        continue;
      }
      if (bytes <= 0) {
        return false;
      }
      sent += static_cast<size_t>(bytes);
    }
    return true;
  }

  void run() noexcept {
    try {
      pollfd listener{listenSocket_, POLLIN, 0};
      if (poll(&listener, 1, 2'000) <= 0 || !(listener.revents & POLLIN)) {
        failed_ = true;
        return;
      }
      const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
      if (clientSocket < 0) {
        failed_ = true;
        return;
      }
      auto closeClient = folly::makeGuard([&] { close(clientSocket); });
      constexpr size_t kHeaderLimit{8 * 1'024};
      std::string request;
      while (request.find("\r\n\r\n") == std::string::npos &&
             request.size() < kHeaderLimit) {
        pollfd clientPoll{clientSocket, POLLIN, 0};
        if (poll(&clientPoll, 1, 2'000) <= 0 ||
            !(clientPoll.revents & POLLIN)) {
          failed_ = true;
          return;
        }
        std::array<char, 512> buffer{};
        const auto bytes = recv(clientSocket, buffer.data(), buffer.size(), 0);
        if (bytes <= 0 ||
            request.size() + static_cast<size_t>(bytes) > kHeaderLimit) {
          failed_ = true;
          return;
        }
        request.append(buffer.data(), static_cast<size_t>(bytes));
      }
      if (request.find("\r\n\r\n") == std::string::npos) {
        failed_ = true;
        return;
      }
      sawGet_ = request.find("GET ") == 0;
      std::transform(
          request.begin(),
          request.end(),
          request.begin(),
          [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
          });
      sawRange_ = request.find("range: bytes=5-8\r\n") != std::string::npos;
      static constexpr char kHeaders[] =
          "HTTP/1.1 206 Partial Content\r\n"
          "Content-Length: 4\r\n"
          "Content-Range: bytes 5-8/20\r\n"
          "Accept-Ranges: bytes\r\n"
          "ETag: \"deterministic-etag\"\r\n"
          "Last-Modified: Mon, 01 Jan 2024 00:00:00 GMT\r\n"
          "Content-Type: application/octet-stream\r\n"
          "x-ms-request-id: deterministic-request\r\n"
          "x-ms-version: 2023-11-03\r\n"
          "x-ms-blob-type: BlockBlob\r\n"
          "x-ms-creation-time: Mon, 01 Jan 2024 00:00:00 GMT\r\n"
          "x-ms-server-encrypted: true\r\n"
          "Date: Mon, 01 Jan 2024 00:00:00 GMT\r\n"
          "Connection: close\r\n\r\n";
      if (!sendAll(clientSocket, kHeaders, sizeof(kHeaders) - 1)) {
        failed_ = true;
        return;
      }
      headersSent_ = true;
      if (holdBody_) {
        pollfd clientPoll{clientSocket, POLLIN, 0};
        if (poll(&clientPoll, 1, 2'000) > 0 &&
            (clientPoll.revents & (POLLIN | POLLHUP | POLLERR))) {
          char byte{};
          clientClosed_ = recv(clientSocket, &byte, 1, 0) == 0;
        }
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      static constexpr char kBody[] = "5678";
      for (size_t offset = 0; offset < sizeof(kBody) - 1; ++offset) {
        if (!sendAll(clientSocket, kBody + offset, 1)) {
          failed_ = true;
          return;
        }
      }
      bodyFullySent_ = true;
    } catch (...) {
      failed_ = true;
    }
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::atomic<bool> failed_{false};
  std::atomic<bool> sawGet_{false};
  std::atomic<bool> sawRange_{false};
  std::atomic<bool> headersSent_{false};
  std::atomic<bool> bodyFullySent_{false};
  std::atomic<bool> clientClosed_{false};
  bool holdBody_{false};
  std::thread thread_;
};

class TestBodyTransaction final : public HttpBodyTransaction {
 public:
  explicit TestBodyTransaction(
      std::string body,
      bool failOnRead = false,
      std::shared_ptr<size_t> sharedAbandonCount = nullptr)
      : body_(std::move(body)),
        failOnRead_(failOnRead),
        sharedAbandonCount_(std::move(sharedAbandonCount)) {}

  size_t read(uint8_t* buffer, size_t size, std::chrono::milliseconds)
      override {
    if (failOnRead_) {
      throw std::runtime_error("test body read failed");
    }
    const auto bytes = std::min(size, body_.size() - offset_);
    std::memcpy(buffer, body_.data() + offset_, bytes);
    offset_ += bytes;
    return bytes;
  }

  bool complete() const noexcept override {
    return offset_ == body_.size() || abandoned_;
  }

  void abandon() noexcept override {
    abandoned_ = true;
    ++abandonCount;
    if (sharedAbandonCount_ != nullptr) {
      ++*sharedAbandonCount_;
    }
  }

  size_t abandonCount{0};

 private:
  std::string body_;
  size_t offset_{0};
  bool failOnRead_{false};
  bool abandoned_{false};
  std::shared_ptr<size_t> sharedAbandonCount_;
};

template <typename Server>
void expectServerSucceeded(const Server& server) {
  EXPECT_FALSE(server.failed());
}

std::string testFixturePath(const char* filename) {
  return std::string(VELOX_ABFS_TEST_DATA_DIR) + "/" + filename;
}

std::string readCompleteBody(HttpResponseTransaction& transaction) {
  std::array<uint8_t, 3> buffer{};
  std::string body;
  while (!transaction.body->complete()) {
    const auto bytes = transaction.body->read(
        buffer.data(), buffer.size(), std::chrono::seconds(2));
    body.append(reinterpret_cast<const char*>(buffer.data()), bytes);
  }
  return body;
}

void expectNoBodyResponse(int statusCode) {
  const std::string response =
      "HTTP/1.1 " + std::to_string(statusCode) + " No Body\r\n\r\n";
  LoopbackHttpServer server(response, {response.size()}, {}, false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kClosed};
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      EXPECT_TRUE(transaction.body->complete());
      transaction.body->read(nullptr, 0, std::chrono::seconds(1));
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(FollyHttpTransportTest, BatonPostBeforeWait) {
  folly::fibers::Baton baton;
  baton.post();
  EXPECT_TRUE(baton.try_wait());
}

TEST(FollyHttpTransportTest, ContractDefaults) {
  const HttpRequest request;
  const HttpLimits limits;

  EXPECT_EQ(request.method, HttpMethod::kGet);
  EXPECT_EQ(request.responseBodyMode, HttpResponseBodyMode::kParse);
  const auto headRequest = HttpRequest::head("/file");
  EXPECT_EQ(headRequest.responseBodyMode, HttpResponseBodyMode::kSkip);
  EXPECT_EQ(limits.maxIngressBytes, 64 * 1'024);
  EXPECT_EQ(limits.maxInformationalResponses, 8);
}

TEST(FollyHttpTransportTest, RejectsZeroPoolConnectionLimit) {
  auto factory = std::make_shared<CountingChannelFactory>();
  AsyncChannelEndpoint endpoint;
  endpoint.connectAddress = folly::SocketAddress("127.0.0.1", 80);
  endpoint.serverName = "127.0.0.1";

  EXPECT_THROW(
      FollyHttpTransport(factory, endpoint, HttpLimits{}, HttpTimeouts{}, 0),
      std::invalid_argument);
}

TEST(FollyHttpTransportTest, RejectsNonPositiveConnectionPoolTimeouts) {
  auto factory = std::make_shared<CountingChannelFactory>();
  AsyncChannelEndpoint endpoint;
  endpoint.connectAddress = folly::SocketAddress("127.0.0.1", 80);
  endpoint.serverName = "127.0.0.1";
  HttpTimeouts timeouts;
  timeouts.connectionAcquire = std::chrono::milliseconds(0);

  EXPECT_THROW(
      FollyHttpTransport(factory, endpoint, HttpLimits{}, timeouts, 1),
      std::invalid_argument);

  timeouts.connectionAcquire = std::chrono::seconds(1);
  timeouts.connectionIdle = std::chrono::milliseconds(0);

  EXPECT_THROW(
      FollyHttpTransport(factory, endpoint, HttpLimits{}, timeouts, 1),
      std::invalid_argument);
}

TEST(FollyHttpTransportTest, RejectsAzureEndpointMismatchesBeforeConnect) {
  for (const auto& caseData : {
           std::pair<std::string, std::string>{
               "https://127.0.0.1:80/container/blob", "scheme"},
           std::pair<std::string, std::string>{
               "http://localhost:80/container/blob", "host"},
           std::pair<std::string, std::string>{
               "http://127.0.0.1:81/container/blob", "port"},
           std::pair<std::string, std::string>{
               "HTTP://127.0.0.1:80", "relative target"},
       }) {
    auto factory = std::make_shared<CountingChannelFactory>();
    AsyncChannelEndpoint endpoint;
    endpoint.connectAddress = folly::SocketAddress("127.0.0.1", 80);
    endpoint.serverName = "127.0.0.1";
    FollyHttpTransport transport(
        factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
    Azure::Core::Http::Request request(
        Azure::Core::Http::HttpMethod::Get, Azure::Core::Url(caseData.first));

    EXPECT_THROW(
        transport.Send(request, Azure::Core::Context{}),
        Azure::Core::Http::TransportException)
        << caseData.second;
    EXPECT_EQ(factory->connectCount, 0U) << caseData.second;
  }
}

TEST(FollyHttpTransportTest, RejectsAzureSendOutsideFiberClearly) {
  auto factory = std::make_shared<CountingChannelFactory>();
  AsyncChannelEndpoint endpoint;
  endpoint.connectAddress = folly::SocketAddress("127.0.0.1", 80);
  endpoint.serverName = "127.0.0.1";
  FollyHttpTransport transport(
      factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
  Azure::Core::Http::Request request(
      Azure::Core::Http::HttpMethod::Get,
      Azure::Core::Url("http://127.0.0.1:80/container/blob"));

  EXPECT_THROW(
      transport.Send(request, Azure::Core::Context{}),
      Azure::Core::Http::TransportException);
  EXPECT_EQ(factory->connectCount, 0U);
}

TEST(FollyHttpTransportTest, BuffersDefaultAzureResponseBody) {
  const std::string body = "buffered-loopback-body";
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\nConnection: close\r\n\r\n" + body;
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "HTTP://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/buffered"));
      auto responseResult = transport.Send(request, Azure::Core::Context{});
      if (responseResult == nullptr) {
        throw std::runtime_error("buffered response was null");
      }
      EXPECT_EQ(
          responseResult->GetBody(),
          std::vector<uint8_t>(body.begin(), body.end()));
      EXPECT_EQ(responseResult->ExtractBodyStream(), nullptr);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
}

TEST(FollyHttpTransportTest, RejectsBufferedResponseOverflowAndClosesSocket) {
  const std::string body = "response-too-large";
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\nConnection: close\r\n\r\n" + body;
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  HttpLimits limits;
  limits.maxBufferedResponseBodyBytes = 3;
  bool failed{false};
  std::exception_ptr unexpectedFailure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, limits, HttpTimeouts{}, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/overflow"));
      transport.Send(request, Azure::Core::Context{});
    } catch (const Azure::Core::Http::TransportException&) {
      failed = true;
    } catch (...) {
      unexpectedFailure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (unexpectedFailure) {
    std::rethrow_exception(unexpectedFailure);
  }
  EXPECT_TRUE(failed);
}

TEST(FollyHttpTransportTest, SendsAzurePostBodyAndHeadersOverLoopback) {
  const std::string body = "azure-post-body";
  CapturingLoopbackHttpServer server("HTTP/1.1 204 No Content\r\n\r\n");
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Core::IO::MemoryBodyStream bodyStream(
          reinterpret_cast<const uint8_t*>(body.data()), body.size());
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Post,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/upload"),
          &bodyStream);
      request.SetHeader("X-Test-Header", "deterministic");
      auto response = transport.Send(request, Azure::Core::Context{});
      if (response == nullptr) {
        throw std::runtime_error("post response was null");
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  const auto requestLineEnd = server.request().find("\r\n");
  ASSERT_NE(requestLineEnd, std::string::npos);
  std::istringstream requestLine(server.request().substr(0, requestLineEnd));
  std::string method;
  std::string target;
  std::string version;
  requestLine >> method >> target >> version;
  EXPECT_EQ(method, "POST");
  EXPECT_EQ(target, "/upload");
  EXPECT_EQ(version, "HTTP/1.1");
  auto normalizedRequest = server.request();
  std::transform(
      normalizedRequest.begin(),
      normalizedRequest.end(),
      normalizedRequest.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
      });
  EXPECT_NE(
      normalizedRequest.find("x-test-header: deterministic\r\n"),
      std::string::npos);
  EXPECT_NE(
      normalizedRequest.find("content-length: " + std::to_string(body.size())),
      std::string::npos);
  EXPECT_EQ(
      server.request().substr(server.request().find("\r\n\r\n") + 4), body);
}

TEST(FollyHttpTransportTest, RejectsAzurePostBodyBeforeConnectAtLimit) {
  auto factory = std::make_shared<CountingChannelFactory>();
  AsyncChannelEndpoint endpoint;
  endpoint.connectAddress = folly::SocketAddress("127.0.0.1", 80);
  endpoint.serverName = "127.0.0.1";
  HttpLimits limits;
  limits.maxRequestBodyBytes = 3;
  FollyHttpTransport transport(factory, endpoint, limits, HttpTimeouts{}, 1);
  const std::string body = "too-large";
  Azure::Core::IO::MemoryBodyStream bodyStream(
      reinterpret_cast<const uint8_t*>(body.data()), body.size());
  Azure::Core::Http::Request request(
      Azure::Core::Http::HttpMethod::Post,
      Azure::Core::Url("http://127.0.0.1:80/upload"),
      &bodyStream);

  folly::EventBase eventBase;
  std::exception_ptr failure;
  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      transport.Send(request, Azure::Core::Context{});
    } catch (const Azure::Core::Http::TransportException&) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  EXPECT_TRUE(failure != nullptr);
  EXPECT_EQ(factory->connectCount, 0U);
}

TEST(FollyHttpTransportTest, AzureBodyStreamExposesLengthAndReads) {
  auto transaction = std::make_unique<TestBodyTransaction>("body");
  auto* transactionPointer = transaction.get();
  FollyResponseBodyStream stream(
      std::move(transaction), int64_t{4}, std::chrono::seconds(1));
  std::array<uint8_t, 2> buffer{};

  EXPECT_EQ(stream.Length(), 4);
  EXPECT_EQ(
      stream.Read(buffer.data(), buffer.size(), Azure::Core::Context{}), 2);
  EXPECT_EQ(
      stream.Read(buffer.data(), buffer.size(), Azure::Core::Context{}), 2);
  EXPECT_TRUE(
      stream.Read(buffer.data(), buffer.size(), Azure::Core::Context{}) == 0);
  EXPECT_EQ(transactionPointer->abandonCount, 0U);
}

TEST(FollyHttpTransportTest, AzureBodyStreamMapsFailureAndAbandonsOnce) {
  auto transaction = std::make_unique<TestBodyTransaction>("body", true);
  auto* transactionPointer = transaction.get();
  FollyResponseBodyStream stream(
      std::move(transaction), int64_t{4}, std::chrono::seconds(1));

  EXPECT_THROW(
      stream.Read(std::array<uint8_t, 4>{}.data(), 4, Azure::Core::Context{}),
      Azure::Core::Http::TransportException);
  EXPECT_EQ(transactionPointer->abandonCount, 1U);
}

TEST(FollyHttpTransportTest, AzureBodyStreamAbandonsUnreadBody) {
  auto abandonCount = std::make_shared<size_t>(0);
  auto transaction =
      std::make_unique<TestBodyTransaction>("body", false, abandonCount);
  {
    FollyResponseBodyStream stream(
        std::move(transaction), int64_t{4}, std::chrono::seconds(1));
    EXPECT_EQ(stream.Length(), 4);
  }
  EXPECT_EQ(*abandonCount, 1U);
}

TEST(FollyHttpTransportTest, AzureTransportStreamsRealLoopbackResponse) {
  const std::string body = "azure-loopback-body";
  const std::string response =
      "HTTP/1.1 206 Partial Content\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
  LoopbackHttpServer server(
      response,
      {response.find("\r\n\r\n") + 4, 1},
      std::chrono::milliseconds(25),
      true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/container/blob?sig=dummy"),
          false);
      auto responseResult = transport.Send(request, Azure::Core::Context{});
      ASSERT_NE(responseResult, nullptr);
      auto bodyStream = responseResult->ExtractBodyStream();
      ASSERT_NE(bodyStream, nullptr);
      EXPECT_EQ(bodyStream->Length(), body.size());
      std::array<uint8_t, 3> buffer{};
      while (true) {
        const auto bytes = bodyStream->Read(
            buffer.data(), buffer.size(), Azure::Core::Context{});
        if (bytes == 0) {
          break;
        }
        received.append(reinterpret_cast<const char*>(buffer.data()), bytes);
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
}

TEST(FollyHttpTransportTest, StreamsFragmentedResponseAndRetainsBodySpill) {
  const std::string body = "fragmented-body-larger-than-the-read-buffer";
  const std::string response =
      "HTTP/1.1 200 OK\r\nX-Trace: first\r\nX-Trace: second\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  std::vector<size_t> fragments(response.size(), 1);
  LoopbackHttpServer server(
      response, std::move(fragments), std::chrono::milliseconds(1), false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kClosed};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.connectTimeout = std::chrono::seconds(1);
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpRequest request;
      auto transaction = connection->send(
          request,
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      EXPECT_EQ(transaction.head.statusCode, 200);
      EXPECT_EQ(transaction.head.headers.size(), 3);
      EXPECT_EQ(transaction.head.contentLength, body.size());
      std::array<uint8_t, 5> buffer{};
      while (!transaction.body->complete()) {
        const auto bytes = transaction.body->read(
            buffer.data(), buffer.size(), std::chrono::seconds(2));
        received.append(reinterpret_cast<const char*>(buffer.data()), bytes);
      }
      transaction.body->read(
          buffer.data(), buffer.size(), std::chrono::seconds(2));
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(
    FollyHttpTransportTest,
    SerializesPostWithOrderedDuplicateHeadersAndLength) {
  const std::string body = "post-body";
  CapturingLoopbackHttpServer server("HTTP/1.1 204 No Content\r\n\r\n");
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpRequest request;
      request.method = HttpMethod::kPost;
      request.target = "/container/file?sig=a%2Fb";
      request.headers = {{"X-Signed", "first"}, {"X-Signed", "Second"}};
      request.body = body;
      auto transaction = connection->send(
          request,
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome outcome) {
            ++releaseCount;
            EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
          });
      EXPECT_TRUE(transaction.body->complete());
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(
      server.request(),
      "POST /container/file?sig=a%2Fb HTTP/1.1\r\n"
      "X-Signed: first\r\n"
      "X-Signed: Second\r\n"
      "Content-Length: 9\r\n\r\npost-body");
  EXPECT_EQ(releaseCount, 1);
}

TEST(FollyHttpTransportTest, RejectsRequestBodyBeforeNetworkWrite) {
  CapturingLoopbackHttpServer server("HTTP/1.1 204 No Content\r\n\r\n");
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpRequest request;
      request.method = HttpMethod::kPost;
      request.target = "/limited";
      request.body = "too-large";
      HttpLimits limits;
      limits.maxRequestBodyBytes = 3;
      connection->send(
          request,
          limits,
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::invalid_argument&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, DecodesFragmentedChunkedResponse) {
  const std::string body = "chunked-response-body";
  const std::string response =
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
      "7\r\nchunked\r\n"
      "-response-body";
  const auto separator = response.find("-response-body");
  const std::string framed =
      response.substr(0, separator) + "e\r\n-response-body\r\n0\r\n\r\n";
  std::vector<size_t> fragments(framed.size(), 1);
  LoopbackHttpServer server(framed, std::move(fragments), {}, false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kClosed};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      received = readCompleteBody(transaction);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(FollyHttpTransportTest, RejectsMalformedChunkAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
      "not-a-size\r\nbody\r\n0\r\n\r\n";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 8> buffer{};
      transaction.body->read(
          buffer.data(), buffer.size(), std::chrono::seconds(2));
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsPrematureChunkedEofAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
      "5\r\nabc";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 8> buffer{};
      while (true) {
        transaction.body->read(
            buffer.data(), buffer.size(), std::chrono::seconds(2));
      }
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, StreamsCloseDelimitedResponseAndCloses) {
  const std::string body = "close-delimited";
  const std::string response =
      "HTTP/1.0 200 OK\r\nX-Trace: one\r\nX-Trace: two\r\nConnection: close\r\n\r\n" +
      body;
  LoopbackHttpServer server(response, {response.size() - 2, 2}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      EXPECT_EQ(transaction.head.version.major, 1);
      EXPECT_EQ(transaction.head.version.minor, 0);
      EXPECT_EQ(transaction.head.reason, "OK");
      EXPECT_EQ(transaction.head.headers.size(), 3);
      EXPECT_EQ(
          transaction.head.headers[0],
          std::make_pair(std::string("X-Trace"), std::string("one")));
      EXPECT_EQ(
          transaction.head.headers[1],
          std::make_pair(std::string("X-Trace"), std::string("two")));
      received = readCompleteBody(transaction);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kClosed);
}

TEST(
    FollyHttpTransportTest,
    DiscardsInformationalResponsesAndPreservesFinalSpill) {
  const std::string body = "final-body";
  const std::string response =
      "HTTP/1.1 100 Continue\r\n\r\n"
      "HTTP/1.1 103 Early Hints\r\nX-Hint: one\r\n\r\n"
      "HTTP/1.1 200 Final\r\nX-Trace: one\r\nX-Trace: two\r\nContent-Length: 10\r\n\r\n" +
      body;
  LoopbackHttpServer server(response, {response.size()}, {}, false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome outcome) {
            ++releaseCount;
            EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
          });
      EXPECT_EQ(transaction.head.statusCode, 200);
      EXPECT_EQ(transaction.head.reason, "Final");
      EXPECT_EQ(transaction.head.informationalResponseCount, 2);
      received = readCompleteBody(transaction);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
}

TEST(FollyHttpTransportTest, RejectsInformationalOverflowAndUpgrade) {
  for (const auto& response :
       {std::string("HTTP/1.1 100 Continue\r\n\r\n") +
            "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\n\r\n",
        std::string("HTTP/1.1 101 Switching Protocols\r\n\r\n")}) {
    LoopbackHttpServer server(response, {}, {}, true);
    server.start();
    folly::EventBase eventBase;
    EventSocketChannelFactory factory(eventBase);
    size_t releaseCount{0};
    HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
    bool failed{false};

    folly::fibers::getFiberManager(eventBase).add([&] {
      try {
        AsyncChannelEndpoint endpoint;
        endpoint.connectAddress = server.address();
        auto connection =
            std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
        HttpLimits limits;
        limits.maxInformationalResponses = 1;
        connection->send(
            HttpRequest{},
            limits,
            HttpTimeouts{},
            [&](HttpTransactionOutcome releaseOutcome) {
              ++releaseCount;
              outcome = releaseOutcome;
            });
      } catch (const std::exception&) {
        failed = true;
      }
    });
    eventBase.loop();

    server.stop();
    EXPECT_TRUE(failed);
    EXPECT_EQ(releaseCount, 1);
    EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
  }
}

TEST(FollyHttpTransportTest, SkipsNoBodyFinalResponses) {
  expectNoBodyResponse(204);
  expectNoBodyResponse(304);
}

TEST(FollyHttpTransportTest, RejectsSpillAfterNoBodyFinalResponse) {
  const std::string response = "HTTP/1.1 204 No Content\r\n\r\nspill";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsResponseHeaderLimitAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nX-Long-Header: 01234567890123456789\r\n\r\n";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpLimits limits;
      limits.maxHeaderBytes = 32;
      connection->send(
          HttpRequest{},
          limits,
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, StreamsBodyWhenHeadersAndBodyShareServerWrite) {
  const std::string body = "coalesced-body";
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\n\r\n" + body;
  LoopbackHttpServer server(response, {}, {}, false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kClosed};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 4> buffer{};
      while (!transaction.body->complete()) {
        const auto bytes = transaction.body->read(
            buffer.data(), buffer.size(), std::chrono::seconds(2));
        received.append(reinterpret_cast<const char*>(buffer.data()), bytes);
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(FollyHttpTransportTest, CompleteFramedResponseWinsOverSocketReset) {
  const std::string body = "complete-before-reset";
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\n\r\n" + body;
  LoopbackHttpServer server(response, {}, {}, false, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kFailed};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      received = readCompleteBody(transaction);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(received, body);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(FollyHttpTransportTest, RejectsExtraBytesAfterDeclaredResponseBody) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nbodyspill";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 8> buffer{};
      transaction.body->read(
          buffer.data(), buffer.size(), std::chrono::seconds(2));
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, SkipsHeadResponseBody) {
  const std::string response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
  LoopbackHttpServer server(response, {response.size()}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  std::exception_ptr failure;
  bool complete{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest::head("/head"),
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            EXPECT_EQ(releaseOutcome, HttpTransactionOutcome::kReusable);
          });
      complete = transaction.body->complete();
      transaction.body->read(nullptr, 0, std::chrono::seconds(1));
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(complete);
  EXPECT_EQ(releaseCount, 1);
}

TEST(FollyHttpTransportTest, DerivesHeadBodySkipFromRequestMethod) {
  const std::string response = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kClosed};
  std::exception_ptr failure;
  bool complete{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpRequest request;
      request.method = HttpMethod::kHead;
      request.target = "/head-direct";
      auto transaction = connection->send(
          request,
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      complete = transaction.body->complete();
      transaction.body->read(nullptr, 0, std::chrono::seconds(1));
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(complete);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kReusable);
}

TEST(FollyHttpTransportTest, ReportsEarlyContentLengthEofAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nabc";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  std::exception_ptr failure;
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 8> buffer{};
      while (true) {
        transaction.body->read(
            buffer.data(), buffer.size(), std::chrono::seconds(2));
      }
    } catch (const std::runtime_error&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsIngressBoundAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\n0123456789abcdef";
  LoopbackHttpServer server(response, {response.size()}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpLimits limits;
      limits.maxIngressBytes = 0;
      connection->send(
          HttpRequest{},
          limits,
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsActualIngressOverflowAndReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: 32\r\n\r\n01234567890123456789012345678901";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpLimits limits;
      limits.maxIngressBytes = 8;
      connection->send(
          HttpRequest{},
          limits,
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsInvalidTimeoutAndReleasesOnce) {
  CapturingLoopbackHttpServer server("HTTP/1.1 204 No Content\r\n\r\n");
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpTimeouts timeouts;
      timeouts.total = std::chrono::milliseconds(0);
      connection->send(
          HttpRequest{},
          HttpLimits{},
          timeouts,
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_TRUE(server.request().empty());
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, RejectsStatusLineAtExactByteLimit) {
  const std::string response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpLimits limits;
      limits.maxStatusLineBytes = 16;
      connection->send(
          HttpRequest{}, limits, HttpTimeouts{}, [&](HttpTransactionOutcome) {
            ++releaseCount;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
}

TEST(FollyHttpTransportTest, RejectsOversizedInformationalStatusLine) {
  const std::string response =
      "HTTP/1.1 103 This-reason-phrase-is-too-long\r\n\r\n"
      "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
  LoopbackHttpServer server(response, {}, {}, true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool failed{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpLimits limits;
      limits.maxStatusLineBytes = 16;
      connection->send(
          HttpRequest{},
          limits,
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kFailed);
}

TEST(FollyHttpTransportTest, BodyIdleTimeoutClosesAndReleasesOnce) {
  const std::string header = "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n";
  const std::string response = header + "data";
  LoopbackHttpServer server(
      response, {header.size(), 4}, std::chrono::milliseconds(150), true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  bool timedOut{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      HttpTimeouts timeouts;
      timeouts.bodyIdle = std::chrono::milliseconds(25);
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          timeouts,
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      std::array<uint8_t, 4> buffer{};
      transaction.body->read(
          buffer.data(), buffer.size(), std::chrono::seconds(1));
    } catch (const std::runtime_error&) {
      timedOut = true;
    }
  });
  eventBase.loop();

  server.stop();
  EXPECT_TRUE(timedOut);
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kTimedOut);
}

TEST(FollyHttpTransportTest, AbandonmentReleasesOnce) {
  const std::string response =
      "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\ndata";
  LoopbackHttpServer server(response, {}, {}, false);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kReusable};
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      transaction.body->abandon();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kAbandoned);
}

TEST(FollyHttpTransportTest, ReusesBufferedResponseOnOneKeepAliveConnection) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  FollyHttpTransport::PoolMetrics metrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      const std::vector<uint8_t> expectedBody{'b', 'o', 'd', 'y'};
      for (size_t requestNumber{0}; requestNumber < 2; ++requestNumber) {
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url(
                "http://127.0.0.1:" +
                std::to_string(server.address().getPort()) + "/reuse"));
        auto response = transport.Send(request, Azure::Core::Context{});
        ASSERT_NE(response, nullptr);
        EXPECT_EQ(response->GetBody(), expectedBody);
      }
      metrics = transport.poolMetrics();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 2);
  EXPECT_EQ(metrics.totalConnections, 1);
  EXPECT_EQ(metrics.idleConnections, 1);
  EXPECT_EQ(metrics.peakLeasedConnections, 1);
}

TEST(FollyHttpTransportTest, EvictsIdleConnectionAtTimeout) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  FollyHttpTransport::PoolMetrics idleMetrics;
  FollyHttpTransport::PoolMetrics evictedMetrics;
  bool clientClosedAfterEviction{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      HttpTimeouts timeouts;
      timeouts.connectionIdle = std::chrono::milliseconds(20);
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, timeouts, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/idle-timeout"));
      auto response = transport.Send(request, Azure::Core::Context{});
      ASSERT_NE(response, nullptr);
      EXPECT_EQ(
          response->GetBody(), (std::vector<uint8_t>{'b', 'o', 'd', 'y'}));
      idleMetrics = transport.poolMetrics();

      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      do {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
        evictedMetrics = transport.poolMetrics();
      } while (
          (evictedMetrics.idleConnections != 0 || !server.clientClosed()) &&
          std::chrono::steady_clock::now() < deadline);
      clientClosedAfterEviction = server.clientClosed();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(clientClosedAfterEviction);
  EXPECT_EQ(idleMetrics.totalConnections, 1);
  EXPECT_EQ(idleMetrics.leasedConnections, 0);
  EXPECT_EQ(idleMetrics.idleConnections, 1);
  EXPECT_EQ(evictedMetrics.totalConnections, 0);
  EXPECT_EQ(evictedMetrics.leasedConnections, 0);
  EXPECT_EQ(evictedMetrics.idleConnections, 0);
  EXPECT_EQ(evictedMetrics.idleConnectionEvictions, 1);
}

TEST(FollyHttpTransportTest, ReusesFullyConsumedStreamingResponse) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      for (size_t requestNumber{0}; requestNumber < 2; ++requestNumber) {
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url(
                "http://127.0.0.1:" +
                std::to_string(server.address().getPort()) + "/stream"),
            false);
        auto response = transport.Send(request, Azure::Core::Context{});
        auto body = response->ExtractBodyStream();
        ASSERT_NE(body, nullptr);
        std::array<uint8_t, 2> buffer{};
        std::string received;
        while (true) {
          const auto bytes =
              body->Read(buffer.data(), buffer.size(), Azure::Core::Context{});
          if (bytes == 0) {
            break;
          }
          received.append(reinterpret_cast<const char*>(buffer.data()), bytes);
        }
        EXPECT_EQ(received, "body");
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 2);
}

TEST(FollyHttpTransportTest, AbandonedResponseIsNotReused) {
  KeepAliveLoopbackHttpServer server({}, true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  FollyHttpTransport::PoolMetrics metrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/abandon"),
          false);
      auto response = transport.Send(request, Azure::Core::Context{});
      auto body = response->ExtractBodyStream();
      ASSERT_NE(body, nullptr);
      body.reset();
      response.reset();

      auto secondResponse = transport.Send(request, Azure::Core::Context{});
      ASSERT_NE(secondResponse, nullptr);
      auto secondBody = secondResponse->ExtractBodyStream();
      ASSERT_NE(secondBody, nullptr);
      std::array<uint8_t, 4> buffer{};
      EXPECT_EQ(
          secondBody->Read(
              buffer.data(), buffer.size(), Azure::Core::Context{}),
          buffer.size());
      EXPECT_EQ(
          secondBody->Read(
              buffer.data(), buffer.size(), Azure::Core::Context{}),
          0);
      metrics = transport.poolMetrics();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(server.acceptedConnections(), 2);
  EXPECT_EQ(server.requests(), 2);
  EXPECT_EQ(metrics.totalConnections, 1);
  EXPECT_EQ(metrics.idleConnections, 1);
}

TEST(FollyHttpTransportTest, PoolCapSuspendsFiberUntilBodyRelease) {
  KeepAliveLoopbackHttpServer server(std::chrono::milliseconds(75));
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  folly::fibers::Baton firstResponseReady;
  folly::fibers::Baton firstBodyReleased;
  bool secondStarted{false};
  bool secondCompleted{false};
  bool secondWasPending{false};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      const auto requestUrl =
          "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
          "/cap";
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(requestUrl),
          false);
      folly::fibers::getFiberManager(eventBase).add([&, transport, requestUrl] {
        try {
          secondStarted = true;
          Azure::Core::Http::Request secondRequest(
              Azure::Core::Http::HttpMethod::Get,
              Azure::Core::Url(requestUrl),
              false);
          auto response =
              transport->Send(secondRequest, Azure::Core::Context{});
          secondCompleted = response != nullptr;
          firstBodyReleased.post();
        } catch (...) {
          failure = std::current_exception();
          firstBodyReleased.post();
        }
      });
      auto firstResponse = transport->Send(request, Azure::Core::Context{});
      auto firstBody = firstResponse->ExtractBodyStream();
      ASSERT_NE(firstBody, nullptr);
      firstResponseReady.post();
      firstBodyReleased.timed_wait(std::chrono::milliseconds(50));
      secondWasPending = secondStarted && !secondCompleted;
      std::array<uint8_t, 4> buffer{};
      EXPECT_EQ(
          firstBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          4);
      firstBody.reset();
    } catch (...) {
      failure = std::current_exception();
      firstBodyReleased.post();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(secondWasPending);
  EXPECT_TRUE(secondCompleted);
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 2);
}

TEST(FollyHttpTransportTest, ConnectionReturnAtWaiterTimeoutBoundary) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  folly::fibers::Baton secondFinished;
  bool secondCompleted{false};
  size_t secondCompletions{0};
  FollyHttpTransport::PoolMetrics evictedMetrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      HttpTimeouts timeouts;
      timeouts.connectionAcquire = std::chrono::milliseconds(50);
      timeouts.connectionIdle = std::chrono::milliseconds(20);
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, timeouts, 1);
      const auto requestUrl =
          "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
          "/timeout-boundary";
      Azure::Core::Http::Request firstRequest(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(requestUrl),
          false);
      auto firstResponse =
          transport->Send(firstRequest, Azure::Core::Context{});
      auto firstBody = firstResponse->ExtractBodyStream();
      ASSERT_NE(firstBody, nullptr);

      folly::fibers::getFiberManager(eventBase).add([&] {
        folly::fibers::Baton boundaryDelay;
        boundaryDelay.timed_wait(timeouts.connectionAcquire);
        try {
          std::array<uint8_t, 4> buffer{};
          EXPECT_EQ(
              firstBody->Read(
                  buffer.data(), buffer.size(), Azure::Core::Context{}),
              buffer.size());
          EXPECT_EQ(
              firstBody->Read(
                  buffer.data(), buffer.size(), Azure::Core::Context{}),
              0);
        } catch (...) {
          failure = std::current_exception();
        }
      });
      folly::fibers::getFiberManager(eventBase).add([&] {
        try {
          Azure::Core::Http::Request secondRequest(
              Azure::Core::Http::HttpMethod::Get, Azure::Core::Url(requestUrl));
          auto secondResponse =
              transport->Send(secondRequest, Azure::Core::Context{});
          secondCompleted = secondResponse != nullptr &&
              secondResponse->GetBody() ==
                  (std::vector<uint8_t>{'b', 'o', 'd', 'y'});
          ++secondCompletions;
        } catch (...) {
          failure = std::current_exception();
        }
        secondFinished.post();
      });
      EXPECT_TRUE(secondFinished.timed_wait(std::chrono::seconds(1)));

      const auto evictionDeadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      do {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
        evictedMetrics = transport->poolMetrics();
      } while (evictedMetrics.idleConnections != 0 &&
               std::chrono::steady_clock::now() < evictionDeadline);
    } catch (...) {
      failure = std::current_exception();
      secondFinished.post();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(secondCompleted);
  EXPECT_EQ(secondCompletions, 1);
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 2);
  EXPECT_EQ(evictedMetrics.totalConnections, 0);
  EXPECT_EQ(evictedMetrics.leasedConnections, 0);
  EXPECT_EQ(evictedMetrics.idleConnections, 0);
  EXPECT_EQ(evictedMetrics.waitingFibers, 0);
  EXPECT_EQ(evictedMetrics.idleConnectionEvictions, 1);
}

TEST(FollyHttpTransportTest, PoolWaitersCompleteInFifoOrder) {
  KeepAliveLoopbackHttpServer server({}, false, 3);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  folly::fibers::Baton firstReady;
  folly::fibers::Baton secondStarted;
  folly::fibers::Baton thirdStarted;
  size_t completedWaiters{0};
  std::vector<int> completionOrder;
  FollyHttpTransport::PoolMetrics metrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      const auto requestUrl =
          "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
          "/fifo";
      auto makeRequest = [&] {
        return Azure::Core::Http::Request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url(requestUrl),
            false);
      };

      auto firstRequest = makeRequest();
      auto firstResponse =
          transport->Send(firstRequest, Azure::Core::Context{});
      auto firstBody = firstResponse->ExtractBodyStream();
      ASSERT_NE(firstBody, nullptr);
      firstReady.post();

      folly::fibers::getFiberManager(eventBase).add([&] {
        try {
          secondStarted.post();
          auto request = makeRequest();
          auto response = transport->Send(request, Azure::Core::Context{});
          completionOrder.push_back(2);
          auto body = response->ExtractBodyStream();
          ASSERT_NE(body, nullptr);
          std::array<uint8_t, 4> buffer{};
          EXPECT_EQ(
              body->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
              buffer.size());
          EXPECT_EQ(
              body->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
              0);
        } catch (...) {
          failure = std::current_exception();
        }
        ++completedWaiters;
      });
      secondStarted.timed_wait(std::chrono::seconds(1));
      while (transport->poolMetrics().waitingFibers < 1) {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
      }

      folly::fibers::getFiberManager(eventBase).add([&] {
        try {
          thirdStarted.post();
          auto request = makeRequest();
          auto response = transport->Send(request, Azure::Core::Context{});
          completionOrder.push_back(3);
          auto body = response->ExtractBodyStream();
          ASSERT_NE(body, nullptr);
          std::array<uint8_t, 4> buffer{};
          EXPECT_EQ(
              body->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
              buffer.size());
          EXPECT_EQ(
              body->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
              0);
        } catch (...) {
          failure = std::current_exception();
        }
        ++completedWaiters;
      });
      thirdStarted.timed_wait(std::chrono::seconds(1));
      while (transport->poolMetrics().waitingFibers < 2) {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
      }

      std::array<uint8_t, 4> buffer{};
      EXPECT_EQ(
          firstBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          buffer.size());
      firstBody.reset();
      const auto completionDeadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (completedWaiters < 2 &&
             std::chrono::steady_clock::now() < completionDeadline) {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
      }
      EXPECT_EQ(completedWaiters, 2);
      metrics = transport->poolMetrics();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(completionOrder, (std::vector<int>{2, 3}));
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 3);
  EXPECT_EQ(metrics.totalConnections, 1);
  EXPECT_EQ(metrics.leasedConnections, 0);
  EXPECT_EQ(metrics.idleConnections, 1);
  EXPECT_EQ(metrics.waitingFibers, 0);
}

TEST(FollyHttpTransportTest, PoolAcquireTimeoutRemovesWaiter) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  folly::fibers::Baton secondFinished;
  bool secondTimedOut{false};
  FollyHttpTransport::PoolMetrics idleMetrics;
  FollyHttpTransport::PoolMetrics evictedMetrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      HttpTimeouts timeouts;
      timeouts.connectionAcquire = std::chrono::milliseconds(50);
      timeouts.connectionIdle = std::chrono::milliseconds(20);
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, timeouts, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/timeout"),
          false);
      auto firstResponse = transport->Send(request, Azure::Core::Context{});
      auto firstBody = firstResponse->ExtractBodyStream();
      ASSERT_NE(firstBody, nullptr);
      folly::fibers::getFiberManager(eventBase).add([&] {
        try {
          transport->Send(request, Azure::Core::Context{});
        } catch (const Azure::Core::Http::TransportException&) {
          secondTimedOut = true;
        } catch (...) {
          failure = std::current_exception();
        }
        secondFinished.post();
      });
      secondFinished.timed_wait(std::chrono::seconds(1));
      std::array<uint8_t, 4> buffer{};
      EXPECT_EQ(
          firstBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          buffer.size());
      EXPECT_EQ(
          firstBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          0);
      auto thirdResponse = transport->Send(request, Azure::Core::Context{});
      ASSERT_NE(thirdResponse, nullptr);
      auto thirdBody = thirdResponse->ExtractBodyStream();
      ASSERT_NE(thirdBody, nullptr);
      EXPECT_EQ(
          thirdBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          buffer.size());
      EXPECT_EQ(
          thirdBody->Read(buffer.data(), buffer.size(), Azure::Core::Context{}),
          0);
      idleMetrics = transport->poolMetrics();
      const auto evictionDeadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      do {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
        evictedMetrics = transport->poolMetrics();
      } while (evictedMetrics.idleConnections != 0 &&
               std::chrono::steady_clock::now() < evictionDeadline);
    } catch (...) {
      failure = std::current_exception();
      secondFinished.post();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(secondTimedOut);
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 2);
  EXPECT_EQ(idleMetrics.totalConnections, 1);
  EXPECT_EQ(idleMetrics.leasedConnections, 0);
  EXPECT_EQ(idleMetrics.idleConnections, 1);
  EXPECT_EQ(idleMetrics.waitingFibers, 0);
  EXPECT_EQ(evictedMetrics.totalConnections, 0);
  EXPECT_EQ(evictedMetrics.leasedConnections, 0);
  EXPECT_EQ(evictedMetrics.idleConnections, 0);
  EXPECT_EQ(evictedMetrics.waitingFibers, 0);
  EXPECT_EQ(evictedMetrics.idleConnectionEvictions, 1);
}

TEST(FollyHttpTransportTest, DestroyingTransportKeepsActiveBodySafe) {
  BoundedBlobDownloadServer server(true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::unique_ptr<Azure::Core::IO::BodyStream> bodyStream;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      {
        FollyHttpTransport transport(
            factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url(
                "http://127.0.0.1:" +
                std::to_string(server.address().getPort()) + "/lifetime"),
            false);
        auto response = transport.Send(request, Azure::Core::Context{});
        bodyStream = response->ExtractBodyStream();
      }
      bodyStream.reset();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(server.headersSent());
  EXPECT_TRUE(server.clientClosed());
}

TEST(FollyHttpTransportTest, DestroysLeasedWaitingAndIdlePoolState) {
  KeepAliveLoopbackHttpServer server({}, true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;
  folly::fibers::Baton secondFinished;
  bool secondCompleted{false};
  FollyHttpTransport::PoolMetrics pendingMetrics;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      const auto requestUrl =
          "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
          "/destroy-state";
      Azure::Core::Http::Request firstRequest(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(requestUrl),
          false);
      auto firstResponse =
          transport->Send(firstRequest, Azure::Core::Context{});
      auto firstBody = firstResponse->ExtractBodyStream();
      ASSERT_NE(firstBody, nullptr);

      auto waitingTransport = transport;
      folly::fibers::getFiberManager(eventBase).add(
          [&, waitingTransport = std::move(waitingTransport)]() mutable {
            try {
              Azure::Core::Http::Request secondRequest(
                  Azure::Core::Http::HttpMethod::Get,
                  Azure::Core::Url(requestUrl));
              auto secondResponse =
                  waitingTransport->Send(secondRequest, Azure::Core::Context{});
              secondCompleted = secondResponse != nullptr &&
                  secondResponse->GetBody() ==
                      (std::vector<uint8_t>{'b', 'o', 'd', 'y'});
            } catch (...) {
              failure = std::current_exception();
            }
            secondFinished.post();
          });

      const auto waiterDeadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      do {
        folly::fibers::Baton delay;
        delay.timed_wait(std::chrono::milliseconds(1));
        pendingMetrics = transport->poolMetrics();
      } while (pendingMetrics.waitingFibers != 1 &&
               std::chrono::steady_clock::now() < waiterDeadline);
      EXPECT_EQ(pendingMetrics.totalConnections, 1);
      EXPECT_EQ(pendingMetrics.leasedConnections, 1);
      EXPECT_EQ(pendingMetrics.waitingFibers, 1);

      transport.reset();
      firstBody.reset();
      firstResponse.reset();
      EXPECT_TRUE(secondFinished.timed_wait(std::chrono::seconds(1)));
    } catch (...) {
      failure = std::current_exception();
      secondFinished.post();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(secondCompleted);
  EXPECT_EQ(server.acceptedConnections(), 2);
  EXPECT_EQ(server.requests(), 2);
}

TEST(FollyHttpTransportTest, DestroysIdlePoolOnOwningEventBase) {
  KeepAliveLoopbackHttpServer server;
  server.start();
  folly::ScopedEventBaseThread eventBaseThread("abfs-pool-destroy");
  auto* eventBase = eventBaseThread.getEventBase();
  AsyncChannelEndpoint endpoint;
  endpoint.connectAddress = server.address();
  endpoint.serverName = "127.0.0.1";
  auto factory = std::make_shared<EventSocketChannelFactory>(*eventBase);
  auto transport = std::make_shared<FollyHttpTransport>(
      factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
  const auto requestUrl =
      "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
      "/idle";
  auto contract = folly::makePromiseContract<folly::Unit>();
  auto promise = std::make_shared<folly::Promise<folly::Unit>>(
      std::move(contract.promise));
  auto future = std::move(contract.future);
  eventBase->runInEventBaseThread(
      [eventBase, transport, promise, requestUrl]() mutable {
        folly::fibers::getFiberManager(*eventBase)
            .add([transport, promise, requestUrl]() mutable {
              try {
                Azure::Core::Http::Request request(
                    Azure::Core::Http::HttpMethod::Get,
                    Azure::Core::Url(requestUrl));
                auto response =
                    transport->Send(request, Azure::Core::Context{});
                if (response == nullptr || response->GetBody().size() != 4) {
                  throw std::runtime_error("idle pool response was invalid");
                }
                promise->setValue();
              } catch (...) {
                promise->setException(
                    folly::exception_wrapper(std::current_exception()));
              }
            });
      });
  std::move(future).get(std::chrono::seconds(2));
  transport.reset();
  server.stop();

  EXPECT_FALSE(server.failed());
  EXPECT_TRUE(server.clientClosed());
  EXPECT_EQ(server.acceptedConnections(), 1);
  EXPECT_EQ(server.requests(), 1);
}

TEST(FollyHttpTransportTest, ConnectsAndReadsDelayedMarkerOnFiber) {
  LoopbackMarkerServer server;
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  std::exception_ptr failure;
  bool markerReceived = false;
  std::chrono::milliseconds elapsed{0};

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.connectTimeout = std::chrono::seconds(1);
      const auto start = std::chrono::steady_clock::now();
      auto transport = factory.connect(endpoint);
      folly::fibers::Baton readBaton;
      MarkerReadCallback readCallback(&readBaton);
      auto clearReadCallback =
          folly::makeGuard([&] { transport->setReadCB(nullptr); });
      transport->setReadCB(&readCallback);
      const auto completed = readBaton.timed_wait(std::chrono::seconds(2));
      elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      if (!completed || readCallback.failed()) {
        throw std::runtime_error("marker read did not complete");
      }
      markerReceived = readCallback.markerReceived();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(markerReceived);
  EXPECT_GE(elapsed.count(), 25);
}

TEST(FollyHttpTransportTest, DownloadsBlobRangeThroughAzureSdkStreamingPath) {
  BoundedBlobDownloadServer server;
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  bool headersWereSent{false};
  bool bodyWasPending{false};
  bool bodyWasRead{false};
  bool blobMetadataWasParsed{false};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::FiberManager::Options fiberOptions;
  fiberOptions.stackSize = 256 * 1'024;
  fiberOptions.stackSizeMultiplier = 1;
  folly::fibers::getFiberManager(eventBase, fiberOptions).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      auto transport = std::make_shared<FollyHttpTransport>(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Storage::Blobs::BlobClientOptions clientOptions;
      clientOptions.Transport.Transport = transport;
      clientOptions.Retry.MaxRetries = 0;
      const auto blobUrl =
          "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
          "/container/blob?sig=dummy";
      Azure::Storage::Blobs::BlobClient client(blobUrl, clientOptions);
      Azure::Storage::Blobs::DownloadBlobOptions downloadOptions;
      Azure::Core::Http::HttpRange range;
      range.Offset = 5;
      range.Length = 4;
      downloadOptions.Range = range;
      auto response = client.Download(downloadOptions);
      headersWereSent = server.headersSent();
      bodyWasPending = !server.bodyFullySent();
      if (response.Value.BlobSize == 20 &&
          response.Value.ContentRange.Offset == 5 &&
          response.Value.ContentRange.Length.HasValue() &&
          response.Value.ContentRange.Length.Value() == 4) {
        blobMetadataWasParsed = true;
      }
      std::array<uint8_t, 2> buffer{};
      while (true) {
        const auto bytes =
            response.Value.BodyStream->Read(buffer.data(), buffer.size());
        if (bytes == 0) {
          break;
        }
        received.append(reinterpret_cast<const char*>(buffer.data()), bytes);
      }
      bodyWasRead = server.bodyFullySent();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(server.sawGet());
  EXPECT_TRUE(server.sawRange());
  EXPECT_TRUE(headersWereSent);
  EXPECT_TRUE(bodyWasPending);
  EXPECT_TRUE(bodyWasRead);
  EXPECT_TRUE(blobMetadataWasParsed);
  EXPECT_EQ(received, "5678");
}

TEST(FollyHttpTransportTest, DestroyingStreamingResponseClosesPeerBeforeEof) {
  BoundedBlobDownloadServer server(true);
  server.start();
  folly::EventBase eventBase;
  auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "127.0.0.1";
      FollyHttpTransport transport(
          factory, endpoint, HttpLimits{}, HttpTimeouts{}, 1);
      Azure::Core::Http::Request request(
          Azure::Core::Http::HttpMethod::Get,
          Azure::Core::Url(
              "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
              "/stream"),
          false);
      auto response = transport.Send(request, Azure::Core::Context{});
      auto bodyStream = response->ExtractBodyStream();
      if (bodyStream == nullptr) {
        throw std::runtime_error("streaming response body was null");
      }
      bodyStream.reset();
      response.reset();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(server.headersSent());
  EXPECT_TRUE(server.clientClosed());
}

TEST(FollyHttpTransportTest, ConnectFailureWakesFiber) {
  BoundLoopbackSocket reservedSocket;
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  bool failed = false;
  std::chrono::milliseconds elapsed{0};

  folly::fibers::getFiberManager(eventBase).add([&] {
    const auto start = std::chrono::steady_clock::now();
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = reservedSocket.address();
      endpoint.connectTimeout = std::chrono::seconds(1);
      factory.connect(endpoint);
    } catch (const std::exception&) {
      failed = true;
    }
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  });
  eventBase.loop();

  EXPECT_TRUE(failed);
  EXPECT_LT(elapsed.count(), 2'000);
}

TEST(FollyHttpTransportTest, EstablishesVerifiedTlsAndServesHttp) {
  LoopbackTlsHttpServer server(
      testFixturePath("test-server.crt"), testFixturePath("test-server.key"));
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  size_t releaseCount{0};
  HttpTransactionOutcome outcome{HttpTransactionOutcome::kFailed};
  std::string received;
  std::exception_ptr failure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "localhost";
      endpoint.security = AsyncChannelSecurity::kTls;
      endpoint.connectTimeout = std::chrono::seconds(1);
      endpoint.tlsHandshakeTimeout = std::chrono::seconds(1);
      endpoint.additionalTrustedCaPath = testFixturePath("test-ca-bundle.pem");
      auto connection =
          std::make_unique<FollyHttpConnection>(factory.connect(endpoint));
      auto transaction = connection->send(
          HttpRequest{},
          HttpLimits{},
          HttpTimeouts{},
          [&](HttpTransactionOutcome releaseOutcome) {
            ++releaseCount;
            outcome = releaseOutcome;
          });
      received = readCompleteBody(transaction);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(server.served());
  EXPECT_EQ(received, "tls-response");
  EXPECT_EQ(releaseCount, 1);
  EXPECT_EQ(outcome, HttpTransactionOutcome::kClosed);
}

TEST(FollyHttpTransportTest, RejectsTlsUnknownCertificateAuthority) {
  LoopbackTlsHttpServer server(
      testFixturePath("test-server.crt"), testFixturePath("test-server.key"));
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  bool failed{false};
  std::exception_ptr unexpectedFailure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "localhost";
      endpoint.security = AsyncChannelSecurity::kTls;
      endpoint.connectTimeout = std::chrono::seconds(1);
      endpoint.tlsHandshakeTimeout = std::chrono::seconds(1);
      factory.connect(endpoint);
    } catch (const std::exception&) {
      failed = true;
    } catch (...) {
      unexpectedFailure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (unexpectedFailure) {
    std::rethrow_exception(unexpectedFailure);
  }
  EXPECT_TRUE(failed);
  EXPECT_FALSE(server.served());
}

TEST(FollyHttpTransportTest, RejectsTlsHostnameMismatchWithTrustedChain) {
  LoopbackTlsHttpServer server(
      testFixturePath("test-wrong-host.crt"),
      testFixturePath("test-wrong-host.key"));
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  bool failed{false};
  std::exception_ptr unexpectedFailure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "localhost";
      endpoint.security = AsyncChannelSecurity::kTls;
      endpoint.connectTimeout = std::chrono::seconds(1);
      endpoint.tlsHandshakeTimeout = std::chrono::seconds(1);
      endpoint.additionalTrustedCaPath = testFixturePath("test-ca-bundle.pem");
      factory.connect(endpoint);
    } catch (const std::exception&) {
      failed = true;
    } catch (...) {
      unexpectedFailure = std::current_exception();
    }
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (unexpectedFailure) {
    std::rethrow_exception(unexpectedFailure);
  }
  EXPECT_TRUE(failed);
  EXPECT_FALSE(server.served());
}

TEST(FollyHttpTransportTest, TlsHandshakeTimeoutWakesFiber) {
  LoopbackTlsHttpServer server(
      testFixturePath("test-server.crt"),
      testFixturePath("test-server.key"),
      true);
  server.start();
  folly::EventBase eventBase;
  EventSocketChannelFactory factory(eventBase);
  bool failed{false};
  std::chrono::milliseconds elapsed{0};
  std::exception_ptr unexpectedFailure;

  folly::fibers::getFiberManager(eventBase).add([&] {
    const auto start = std::chrono::steady_clock::now();
    try {
      AsyncChannelEndpoint endpoint;
      endpoint.connectAddress = server.address();
      endpoint.serverName = "localhost";
      endpoint.security = AsyncChannelSecurity::kTls;
      endpoint.connectTimeout = std::chrono::seconds(1);
      endpoint.tlsHandshakeTimeout = std::chrono::milliseconds(50);
      endpoint.additionalTrustedCaPath = testFixturePath("test-ca-bundle.pem");
      factory.connect(endpoint);
    } catch (const std::exception&) {
      failed = true;
    } catch (...) {
      unexpectedFailure = std::current_exception();
    }
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  });
  eventBase.loop();

  server.stop();
  expectServerSucceeded(server);
  if (unexpectedFailure) {
    std::rethrow_exception(unexpectedFailure);
  }
  EXPECT_TRUE(failed);
  EXPECT_FALSE(server.served());
  EXPECT_LT(elapsed.count(), 2'000);
}

} // namespace
} // namespace facebook::velox::filesystems

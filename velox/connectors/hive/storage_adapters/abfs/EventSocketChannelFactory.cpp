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

#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/io/async/AsyncSSLSocket.h>
#include <folly/io/async/AsyncSocket.h>
#include <folly/io/async/AsyncSocketException.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/SSLContext.h>
#include <folly/portability/OpenSSL.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace facebook::velox::filesystems {
namespace {

enum class ConnectTerminalState : uint8_t { kPending, kSuccess, kError };

class ConnectCallback final : public folly::AsyncSocket::ConnectCallback {
 public:
  explicit ConnectCallback(folly::fibers::Baton* baton) : baton_(baton) {}

  void connectSuccess() noexcept override {
    complete(ConnectTerminalState::kSuccess, 0);
  }

  void connectErr(
      const folly::AsyncSocketException& exception) noexcept override {
    try {
      complete(
          ConnectTerminalState::kError, static_cast<int>(exception.getErrno()));
    } catch (...) {
      complete(ConnectTerminalState::kError, 0);
    }
  }

  ConnectTerminalState state() const noexcept {
    return state_;
  }

  int errorNumber() const noexcept {
    return errorNumber_;
  }

 private:
  void complete(ConnectTerminalState state, int errorNumber) noexcept {
    if (state_ != ConnectTerminalState::kPending) {
      return;
    }
    state_ = state;
    errorNumber_ = errorNumber;
    baton_->post();
  }

  folly::fibers::Baton* baton_;
  ConnectTerminalState state_{ConnectTerminalState::kPending};
  int errorNumber_{0};
};

enum class HandshakeTerminalState : uint8_t { kPending, kSuccess, kError };

class HandshakeCallback final : public folly::AsyncSSLSocket::HandshakeCB {
 public:
  explicit HandshakeCallback(folly::fibers::Baton* baton) : baton_(baton) {}

  void handshakeSuc(folly::AsyncSSLSocket*) noexcept override {
    complete(HandshakeTerminalState::kSuccess, 0);
  }

  void handshakeErr(
      folly::AsyncSSLSocket*,
      const folly::AsyncSocketException& exception) noexcept override {
    try {
      complete(
          HandshakeTerminalState::kError,
          static_cast<int>(exception.getErrno()));
    } catch (...) {
      complete(HandshakeTerminalState::kError, 0);
    }
  }

  HandshakeTerminalState state() const noexcept {
    return state_;
  }

  int errorNumber() const noexcept {
    return errorNumber_;
  }

 private:
  void complete(HandshakeTerminalState state, int errorNumber) noexcept {
    if (state_ != HandshakeTerminalState::kPending) {
      return;
    }
    state_ = state;
    errorNumber_ = errorNumber;
    baton_->post();
  }

  folly::fibers::Baton* baton_;
  HandshakeTerminalState state_{HandshakeTerminalState::kPending};
  int errorNumber_{0};
};

uint32_t validateTimeout(
    std::chrono::milliseconds timeout,
    const char* description) {
  if (timeout.count() <= 0 ||
      static_cast<uint64_t>(timeout.count()) >
          std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(description);
  }
  return static_cast<uint32_t>(timeout.count());
}

} // namespace

EventSocketChannelFactory::EventSocketChannelFactory(
    folly::EventBase* eventBase)
    : eventBase_(eventBase) {
  if (eventBase_ == nullptr) {
    throw std::invalid_argument(
        "EventSocketChannelFactory requires an EventBase");
  }
}

EventSocketChannelFactory::EventSocketChannelFactory(
    folly::EventBase& eventBase)
    : EventSocketChannelFactory(&eventBase) {}

folly::AsyncTransportWrapper::UniquePtr EventSocketChannelFactory::connect(
    const AsyncChannelEndpoint& endpoint) {
  if (!eventBase_->isInEventBaseThread()) {
    throw std::logic_error(
        "ABFS socket connect must run on its EventBase thread");
  }
  if (!folly::fibers::onFiber()) {
    throw std::logic_error("ABFS socket connect must run inside a fiber");
  }
  if (endpoint.security == AsyncChannelSecurity::kPlaintext) {
    if (endpoint.connectTimeout.count() <= 0 ||
        static_cast<uint64_t>(endpoint.connectTimeout.count()) >
            std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument(
          "ABFS socket connect timeout is out of range");
    }

    auto socket = folly::AsyncSocket::newSocket(eventBase_);
    folly::fibers::Baton baton;
    ConnectCallback callback(&baton);
    socket->connect(
        &callback,
        endpoint.connectAddress,
        static_cast<uint32_t>(endpoint.connectTimeout.count()));
    baton.wait();

    if (callback.state() == ConnectTerminalState::kError) {
      throw std::runtime_error(
          std::string("ABFS socket connect failed, errno ") +
          std::to_string(callback.errorNumber()));
    }
    if (callback.state() != ConnectTerminalState::kSuccess) {
      throw std::runtime_error(
          "ABFS socket connect completed without a result");
    }
    return socket;
  }

  if (endpoint.security != AsyncChannelSecurity::kTls) {
    throw std::invalid_argument("ABFS socket security is unsupported");
  }
  if (endpoint.serverName.empty()) {
    throw std::invalid_argument("ABFS TLS endpoint requires a server name");
  }
  const auto connectTimeout = validateTimeout(
      endpoint.connectTimeout, "ABFS socket connect timeout is out of range");
  const auto tlsHandshakeTimeout = validateTimeout(
      endpoint.tlsHandshakeTimeout,
      "ABFS TLS handshake timeout is out of range");

  auto context =
      std::make_shared<folly::SSLContext>(folly::SSLContext::TLSv1_2);
  if (SSL_CTX_set_default_verify_paths(context->getSSLCtx()) != 1) {
    throw std::runtime_error("ABFS TLS system trust roots could not be loaded");
  }
  if (!endpoint.additionalTrustedCaPath.empty()) {
    context->loadTrustedCertificates(endpoint.additionalTrustedCaPath.c_str());
  }
  context->authenticate(true, true, endpoint.serverName);
  if (X509_VERIFY_PARAM_set1_host(
          SSL_CTX_get0_param(context->getSSLCtx()),
          endpoint.serverName.c_str(),
          endpoint.serverName.size()) != 1) {
    throw std::runtime_error(
        "ABFS TLS hostname verification could not be configured");
  }

  folly::AsyncSSLSocket::Options options;
  options.deferSecurityNegotiation = true;
  options.isServer = false;
  options.serverName = endpoint.serverName;
  auto socket = folly::AsyncSSLSocket::newSocket(
      std::move(context), eventBase_, std::move(options));

  folly::fibers::Baton connectBaton;
  ConnectCallback connectCallback(&connectBaton);
  socket->connect(&connectCallback, endpoint.connectAddress, connectTimeout);
  connectBaton.wait();
  if (connectCallback.state() != ConnectTerminalState::kSuccess) {
    socket->closeNow();
    if (connectCallback.state() == ConnectTerminalState::kError) {
      throw std::runtime_error(
          std::string("ABFS TLS TCP connect failed, errno ") +
          std::to_string(connectCallback.errorNumber()));
    }
    throw std::runtime_error("ABFS TLS TCP connect completed without a result");
  }

  folly::fibers::Baton handshakeBaton;
  HandshakeCallback handshakeCallback(&handshakeBaton);
  socket->sslConn(
      &handshakeCallback,
      std::chrono::milliseconds(tlsHandshakeTimeout),
      folly::SSLContext::SSLVerifyPeerEnum::USE_CTX);
  handshakeBaton.wait();
  if (handshakeCallback.state() != HandshakeTerminalState::kSuccess) {
    socket->closeNow();
    if (handshakeCallback.state() == HandshakeTerminalState::kError) {
      throw std::runtime_error(
          std::string("ABFS TLS handshake failed, errno ") +
          std::to_string(handshakeCallback.errorNumber()));
    }
    throw std::runtime_error("ABFS TLS handshake completed without a result");
  }
  return socket;
}

} // namespace facebook::velox::filesystems

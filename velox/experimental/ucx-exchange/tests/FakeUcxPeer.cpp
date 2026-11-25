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
#include "velox/experimental/ucx-exchange/tests/FakeUcxPeer.h"

#include <future>
#include <string>

#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {
namespace {

// The receiver callback of the handshake. It is the same in every protocol
// version, so it is spelled out here rather than taken from the Communicator.
const ucxx::AmReceiverCallbackOwnerType kAmCallbackOwner{"velox"};
constexpr ucxx::AmReceiverCallbackIdType kAmCallbackId{123};

// How long the progress thread may block without a UCX event or a new action.
constexpr int kEpollTimeoutMs{10};

// How long a test waits for the progress thread to run an action, for an
// active message to be sent, and for a cancelled request to complete.
constexpr std::chrono::milliseconds kInternalTimeout{5'000};

} // namespace

// progressWorkerEvent() waits on the worker's event file descriptor, which
// needs UCP_FEATURE_WAKEUP.
FakeUcxPeer::FakeUcxPeer()
    : context_(
          ucxx::contextBuilder(
              UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_WAKEUP)
              .build()),
      worker_(context_->workerBuilder().build()) {
  worker_->initBlockingProgressMode();
  worker_->registerAmReceiverCallback(
      ucxx::AmReceiverCallbackInfo(kAmCallbackOwner, kAmCallbackId),
      [this](std::shared_ptr<ucxx::Request> request, ucp_ep_h /*endpoint*/) {
        onActiveMessage(request);
      });
}

// static
std::unique_ptr<FakeUcxPeer> FakeUcxPeer::listen(uint16_t port) {
  std::unique_ptr<FakeUcxPeer> peer(new FakeUcxPeer());
  peer->listener_ =
      peer->worker_
          ->listenerBuilder(port, &FakeUcxPeer::onConnectionRequest, peer.get())
          .build();
  peer->startProgressThread();
  return peer;
}

// static
std::unique_ptr<FakeUcxPeer> FakeUcxPeer::connect(
    std::string_view host,
    uint16_t port) {
  std::unique_ptr<FakeUcxPeer> peer(new FakeUcxPeer());
  peer->endpoint_ = peer->worker_->endpointBuilder(std::string(host), port)
                        .endpointErrorHandling(true)
                        .build();
  peer->startProgressThread();
  return peer;
}

FakeUcxPeer::~FakeUcxPeer() {
  running_ = false;
  worker_->signal();
  progressThread_.join();
  // The progress thread has stopped, so closeBlocking() progresses the worker
  // on this thread.
  if (endpoint_ != nullptr) {
    endpoint_->closeBlocking();
  }
  endpoint_.reset();
  listener_.reset();
  worker_.reset();
  context_.reset();
}

// static
void FakeUcxPeer::onConnectionRequest(ucp_conn_request_h request, void* arg) {
  auto* peer = static_cast<FakeUcxPeer*>(arg);
  if (peer->endpoint_ != nullptr) {
    LOG(ERROR) << "FakeUcxPeer rejects a second connection";
    ucp_listener_reject(peer->listener_->getHandle(), request);
    return;
  }
  peer->endpoint_ = peer->listener_->endpointBuilder(request)
                        .endpointErrorHandling(true)
                        .build();
}

void FakeUcxPeer::onActiveMessage(
    const std::shared_ptr<ucxx::Request>& request) {
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  if (buffer == nullptr) {
    LOG(ERROR) << "FakeUcxPeer received an active message without a buffer";
    return;
  }
  const auto* data = static_cast<const uint8_t*>(buffer->data());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    handshakes_.emplace_back(data, data + buffer->getSize());
  }
  handshakeArrived_.notify_all();
}

void FakeUcxPeer::startProgressThread() {
  progressThread_ = std::thread([this] { progressLoop(); });
}

void FakeUcxPeer::progressLoop() {
  while (running_) {
    std::deque<std::function<void()>> actions;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      actions.swap(actions_);
    }
    for (auto& action : actions) {
      action();
    }
    worker_->progressWorkerEvent(kEpollTimeoutMs);
  }
}

void FakeUcxPeer::runOnProgressThread(std::function<void()> action) {
  auto done = std::make_shared<std::promise<void>>();
  auto future = done->get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    actions_.emplace_back([action = std::move(action), done] {
      try {
        action();
        done->set_value();
      } catch (...) {
        done->set_exception(std::current_exception());
      }
    });
  }
  worker_->signal();
  VELOX_CHECK(
      future.wait_for(kInternalTimeout) == std::future_status::ready,
      "FakeUcxPeer's progress thread did not run an action in time");
  future.get();
}

ucs_status_t FakeUcxPeer::waitForRequest(
    const std::shared_ptr<ucxx::Request>& request,
    std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!request->isCompleted() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (request->isCompleted()) {
    return request->getStatus();
  }
  runOnProgressThread([request] { request->cancel(); });
  // The request owns its buffer, so a late completion cannot write to freed
  // memory. Waiting keeps a cancelled request from completing in a later step.
  const auto cancelDeadline =
      std::chrono::steady_clock::now() + kInternalTimeout;
  while (!request->isCompleted() &&
         std::chrono::steady_clock::now() < cancelDeadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return UCS_ERR_TIMED_OUT;
}

std::vector<uint8_t> FakeUcxPeer::waitForHandshake(
    std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!handshakeArrived_.wait_for(
          lock, timeout, [this] { return !handshakes_.empty(); })) {
    return {};
  }
  auto handshake = std::move(handshakes_.front());
  handshakes_.pop_front();
  return handshake;
}

void FakeUcxPeer::sendAm(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  auto buffer = std::make_shared<std::vector<uint8_t>>(bytes, bytes + size);
  std::shared_ptr<ucxx::Request> request;
  runOnProgressThread([this, buffer, &request] {
    VELOX_CHECK_NOT_NULL(endpoint_, "FakeUcxPeer has no connection");
    request =
        endpoint_
            ->amSendBuilder(
                buffer->data(), buffer->size(), UCS_MEMORY_TYPE_HOST)
            .receiverCallbackInfo(
                ucxx::AmReceiverCallbackInfo(kAmCallbackOwner, kAmCallbackId))
            .callbackData(buffer)
            .build();
  });
  const auto status = waitForRequest(request, kInternalTimeout);
  VELOX_CHECK(
      status == UCS_OK,
      "FakeUcxPeer failed to send an active message: {}",
      ucs_status_string(status));
}

ucs_status_t FakeUcxPeer::sendTag(
    uint64_t tag,
    const void* data,
    size_t size,
    std::chrono::milliseconds timeout) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  auto buffer = std::make_shared<std::vector<uint8_t>>(bytes, bytes + size);
  std::shared_ptr<ucxx::Request> request;
  runOnProgressThread([this, buffer, tag, &request] {
    VELOX_CHECK_NOT_NULL(endpoint_, "FakeUcxPeer has no connection");
    request =
        endpoint_
            ->tagSendBuilder(buffer->data(), buffer->size(), ucxx::Tag{tag})
            .callbackData(buffer)
            .build();
  });
  return waitForRequest(request, timeout);
}

std::pair<ucs_status_t, std::vector<uint8_t>> FakeUcxPeer::receiveTag(
    uint64_t tag,
    size_t bufferSize,
    std::chrono::milliseconds timeout) {
  auto buffer = std::make_shared<std::vector<uint8_t>>(bufferSize, 0);
  std::shared_ptr<ucxx::Request> request;
  runOnProgressThread([this, buffer, tag, &request] {
    request = worker_
                  ->tagRecvBuilder(
                      buffer->data(),
                      buffer->size(),
                      ucxx::Tag{tag},
                      ucxx::TagMaskFull)
                  .callbackData(buffer)
                  .build();
  });
  const auto status = waitForRequest(request, timeout);
  if (status == UCS_ERR_TIMED_OUT) {
    return {status, {}};
  }
  return {status, *buffer};
}

} // namespace facebook::velox::ucx_exchange

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

#include <ucxx/api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// A UCX peer that a test drives byte by byte, to play a producer or a consumer
/// of another protocol version. It has its own UCX context and worker, apart
/// from the Communicator, and progresses them on its own thread. Every UCX call
/// runs on that thread; the methods below hand their work to it and wait.
class FakeUcxPeer {
 public:
  /// Returns a fake producer that listens on 'port'. It accepts one connection
  /// and captures each handshake that arrives as an active message for the
  /// UCX exchange's receiver callback.
  static std::unique_ptr<FakeUcxPeer> listen(uint16_t port);

  /// Returns a fake consumer connected to the listener at 'host':'port'.
  static std::unique_ptr<FakeUcxPeer> connect(
      std::string_view host,
      uint16_t port);

  ~FakeUcxPeer();

  FakeUcxPeer(const FakeUcxPeer&) = delete;
  FakeUcxPeer& operator=(const FakeUcxPeer&) = delete;

  /// Returns the bytes of the next captured handshake, or no bytes if none
  /// arrives within 'timeout'.
  std::vector<uint8_t> waitForHandshake(std::chrono::milliseconds timeout);

  /// Sends 'size' bytes at 'data' as an active message for the UCX exchange's
  /// receiver callback, as a source sends its handshake. Throws if the send
  /// fails or does not complete within a few seconds.
  void sendAm(const void* data, size_t size);

  /// Sends 'size' bytes at 'data' with 'tag' on the connection. Returns the
  /// send's status, or UCS_ERR_TIMED_OUT after cancelling the send if it does
  /// not complete within 'timeout'.
  ucs_status_t sendTag(
      uint64_t tag,
      const void* data,
      size_t size,
      std::chrono::milliseconds timeout);

  /// Receives a message with exactly 'tag' into a zero-filled buffer of
  /// 'bufferSize' bytes. Returns the receive's status and the whole buffer, or
  /// UCS_ERR_TIMED_OUT and no bytes after cancelling the receive if no message
  /// arrives within 'timeout'.
  std::pair<ucs_status_t, std::vector<uint8_t>> receiveTag(
      uint64_t tag,
      size_t bufferSize,
      std::chrono::milliseconds timeout);

 private:
  FakeUcxPeer();

  static void onConnectionRequest(ucp_conn_request_h request, void* arg);

  // Captures the bytes of a handshake. Runs on the progress thread.
  void onActiveMessage(const std::shared_ptr<ucxx::Request>& request);

  // Starts the progress thread. Called once the worker, and the listener or
  // the endpoint, exist.
  void startProgressThread();

  void progressLoop();

  // Runs 'action' on the progress thread and waits for it. Rethrows what
  // 'action' throws.
  void runOnProgressThread(std::function<void()> action);

  // Waits for 'request' to complete and returns its status. Cancels it and
  // returns UCS_ERR_TIMED_OUT if it does not complete within 'timeout'.
  ucs_status_t waitForRequest(
      const std::shared_ptr<ucxx::Request>& request,
      std::chrono::milliseconds timeout);

  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Listener> listener_;

  // The connection: accepted by a fake producer, opened by a fake consumer.
  // Written and read only on the progress thread once it runs.
  std::shared_ptr<ucxx::Endpoint> endpoint_;

  std::mutex mutex_;
  std::condition_variable handshakeArrived_;
  // Guarded by 'mutex_'.
  std::deque<std::vector<uint8_t>> handshakes_;
  // Guarded by 'mutex_'.
  std::deque<std::function<void()>> actions_;

  std::atomic<bool> running_{true};
  std::thread progressThread_;
};

} // namespace facebook::velox::ucx_exchange

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
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include "velox/common/future/VeloxPromise.h"
#include "velox/experimental/ucx-exchange/Acceptor.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/WorkQueue.h"

#include <gflags/gflags.h>

DECLARE_bool(velox_ucx_exchange);

namespace facebook::velox::ucx_exchange {

struct HostPort {
  std::string hostname;
  uint16_t port = 0;

  // Default and parameterized constructors
  HostPort() = default;
  HostPort(std::string h, uint16_t p) : hostname(std::move(h)), port(p) {}

  // Strict weak ordering for std::map
  bool operator<(HostPort const& other) const noexcept {
    if (hostname < other.hostname)
      return true;
    if (other.hostname < hostname)
      return false;
    return port < other.port;
  }
};

class Communicator {
 public:
  const ucxx::AmReceiverCallbackOwnerType kAmCallbackOwner = "velox";
  const ucxx::AmReceiverCallbackIdType kAmCallbackId = 123;

  /// @brief Method to initialize the communicator and get a reference to it.
  /// @param port The port to listen on.
  /// @param coordinatorURL The URL of the coordinator
  /// @param future An optional future that will be set when the communicator
  /// is running and ready to accept connections.
  [[nodiscard]] static std::shared_ptr<Communicator> initAndGet(
      uint16_t port,
      std::string_view coordinatorURL,
      ContinueFuture* future = nullptr);

  /// @brief Method to get the Communicator reference
  static std::shared_ptr<Communicator> getInstance();

  /// @brief Destructor.
  ~Communicator();

  /// @brief Starts the Communicator
  void run();

  /// @brief Stops the Communicator
  void stop();

  /// @brief Registers a communication element with the communicator.
  /// This also automatically puts the element into the work queue.
  /// @param comms The element to register.
  void registerCommElement(std::shared_ptr<CommElement> comms);

  /// @brief Adds an already registered communication element to the work queue
  /// such that "process" will be called on it.
  /// @param comms The element to be added to the work queue.
  void addToWorkQueue(std::shared_ptr<CommElement> comms);

  /// @brief Unregisters a communication element
  /// @brief comms The communication element.
  void unregister(std::shared_ptr<CommElement> comms);

  /// @brief Associates a CommElement with an endpoint that connects to the
  /// given host and port. If no endpoint exists, then a new connected endpoint
  /// is created.
  /// @param commElement The commElement that will be associated with the
  /// endpoint reference on success.
  /// @param hostPort Identifies and existing endpoint or is used to create a
  /// new endpoint if none exists for this host and port.
  /// @returns The endpoint reference or null if no connection was possible.
  [[nodiscard]] std::shared_ptr<EndpointRef> assocEndpointRef(
      std::shared_ptr<CommElement> commElement,
      HostPort hostPort);

  /// @brief Removes an endpoint from the communicator. This is required when
  /// the endpoint has become stale since the other side has disappeared.
  /// NOTE: This should NOT be called from within a UCX callback. Use
  /// deferEndpointCleanup() instead.
  void removeEndpointRef(std::shared_ptr<EndpointRef> ep);

  /// @brief Defers endpoint cleanup to the main progress loop.
  /// This MUST be used instead of removeEndpointRef when called from
  /// within a UCX callback, because UCX callbacks cannot call progress
  /// functions (which closeBlocking() does internally).
  /// @param ep The endpoint to be cleaned up later.
  void deferEndpointCleanup(std::shared_ptr<EndpointRef> ep);

  /// @brief Defers cleanup of a cancelled UCXX request to the main loop.
  /// The request (and the GPU buffers it references via its arg) will be
  /// held alive until UCX has fully processed the cancellation.
  /// Must only be called from the Communicator thread.
  void deferRequestCleanup(std::shared_ptr<ucxx::Request> request);

  /// Returns the URL of the coordinator.
  [[nodiscard]] const std::string& getCoordinatorUrl();

  /// @brief Get the listener's bound IP address.
  /// Used for same-node detection in intra-node transfer optimization.
  /// @returns The IP address string from the UCXX listener, or empty if not
  /// initialized.
  [[nodiscard]] std::string getListenerIp() const;

  /// @brief Get the listener's bound port.
  /// Used for same-node detection in intra-node transfer optimization.
  /// @returns The port number from the UCXX listener.
  [[nodiscard]] uint16_t getListenerPort() const;

  /// @brief Get the unique worker ID for this Communicator instance.
  /// Used for intra-process detection: if two endpoints share the same
  /// workerId, they are in the same process and can use
  /// IntraNodeTransferRegistry instead of UCXX.
  uint64_t getWorkerId() const {
    return workerId_;
  }

  /// Looks up the EndpointRef associated with a raw UCP endpoint handle.
  /// Used by the Acceptor's active-message callback to resolve the endpoint
  /// that received a handshake request.
  /// @returns The EndpointRef or nullptr if no mapping exists for the handle.
  [[nodiscard]] std::shared_ptr<EndpointRef> findEndpointRefByHandle(
      ucp_ep_h handle);

 private:
  Communicator() =
      default; // Private constructor to prevent direct instantiation
  // delete some methods to make the Communicator a singleton.
  Communicator(const Communicator&) = delete; // Prevent copying
  Communicator& operator=(const Communicator&) = delete; // Prevent assignment

  // Invoked when a client connects to the listener.
  void listenerCallback(ucp_conn_request_h conn_request);

  // Wrapper to map the C-style callback to the listener method.
  static void cStyleListenerCallback(
      ucp_conn_request_h conn_request,
      void* arg);

  static std::once_flag onceFlag; // Flag for thread-safe initialization
  static std::shared_ptr<Communicator> instancePtr_;

  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Listener> listener_;
  uint16_t port_;
  std::string coordinatorURL_;
  std::atomic<bool> running_;
  Acceptor acceptor_;
  ContinuePromise promise_{"Communicator::run"};

  // the set of elements known to the communicator.
  // The elements_ set makes sure that there exists a shared_ptr to
  // the senders and receivers as long as they are active and not unregistered.

  // Comparator that compares the stored raw pointers
  struct PtrAddressLess {
    bool operator()(
        std::shared_ptr<CommElement> const& a,
        std::shared_ptr<CommElement> const& b) const noexcept {
      return a.get() < b.get();
    }
  };
  // Comparison function is the raw pointer, not the shared pointer.
  std::set<std::shared_ptr<CommElement>, PtrAddressLess> elements_;
  // protect elements_ by a mutex. Needs to be mutable if called from a const
  // function.
  std::mutex elemMutex_;

  // The work queue for communication elements that need to do things on
  // the communicator thread.
  WorkQueue<CommElement> workQueue_;

  // Protects endpoints_. Must be recursive because removeEndpointRef() holds
  // the lock and calls closeBlocking(), which progresses the worker and can
  // fire listenerCallback() re-entrantly — that callback also locks this mutex.
  std::recursive_mutex endpointsMutex_;

  // Shared endpoints keyed by remote host:port.
  std::map<HostPort, std::shared_ptr<EndpointRef>> endpoints_;

  // Signals the UCXX worker to wake up from a blocking
  // progressWorkerEvent() call. Thread-safe. No-op if worker_ is null
  // or if not in blocking progress mode.
  void signalWorker();

  // A random unique identifier for this Communicator instance (process).
  // Generated once at initialization. Used by the Acceptor to determine
  // if a connecting source is in the same process (intra-node transfer).
  uint64_t workerId_{0};

  // Queue of endpoints that need cleanup, populated by callbacks.
  // UCX callbacks cannot call progress functions (like closeBlocking),
  // so they defer cleanup to the main loop via this queue.
  WorkQueue<EndpointRef> deferredEndpointCleanup_;

  // Cancelled UCXX requests whose GPU buffers may still be referenced by
  // UCX internals. Held alive here until isCompleted() returns true,
  // ensuring the GPU buffers (owned via the request's arg shared_ptr)
  // are not freed prematurely.
  std::vector<std::shared_ptr<ucxx::Request>> deferredRequests_;

  // Heartbeat state for diagnostic logging.
  std::chrono::steady_clock::time_point lastHeartbeat_{
      std::chrono::steady_clock::now()};
  uint64_t workItemsProcessed_{0};
};

} // namespace facebook::velox::ucx_exchange

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
#include "velox/experimental/ucx-exchange/Communicator.h"
#ifdef VELOX_ENABLE_CUDF
#include <cuda_runtime.h>
#endif
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowAcceptor.h"
#include "velox/experimental/ucx-exchange/UcxExchangeModules.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#ifdef VELOX_ENABLE_CUDF
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"
#endif

#include <glog/logging.h>

// gflag for whether the UCX exchange is active.
DEFINE_bool(velox_ucx_exchange, true, "Enable UCX exchange");

namespace {

int readIntEnv(const char* name, int defaultValue, int minValue, int maxValue) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') {
    return defaultValue;
  }

  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || errno != 0) {
    LOG(WARNING) << "Ignoring invalid " << name << "=" << env;
    return defaultValue;
  }
  return static_cast<int>(
      std::clamp<long>(parsed, static_cast<long>(minValue), maxValue));
}

#ifndef VELOX_ENABLE_CUDF
bool readBoolEnv(const char* name, bool defaultValue) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') {
    return defaultValue;
  }
  if (env[0] == '0' && env[1] == '\0') {
    return false;
  }
  if (env[0] == '1' && env[1] == '\0') {
    return true;
  }
  LOG(WARNING) << "Ignoring invalid " << name << "=" << env
               << "; expected 0 or 1";
  return defaultValue;
}

bool readBoolEnv(
    const char* primaryName,
    const char* fallbackName,
    bool defaultValue) {
  if (const char* env = std::getenv(primaryName);
      env != nullptr && *env != '\0') {
    return readBoolEnv(primaryName, defaultValue);
  }
  return readBoolEnv(fallbackName, defaultValue);
}
#endif

int maxWorkItemsPerDrain() {
  static const int value =
      readIntEnv(
          "VELOX_UCX_MAX_WORK_ITEMS_PER_DRAIN", 256, 1, 1 << 20);
  return value;
}
} // namespace

namespace facebook::velox::ucx_exchange {

namespace {
// Config knobs used by both the cudf and CPU-row exchange paths. cudf builds
// use CudfConfig so existing config.properties continue to control UCX
// exchange. CPU-only builds do not compile CudfConfig, so keep env overrides
// for low-level transport diagnostics there.
struct UcxConfigView {
#ifdef VELOX_ENABLE_CUDF
  static int exchangeLogLevel() {
    return cudf_velox::CudfConfig::getInstance().exchangeLogLevel;
  }
  static bool ucxxBlockingPolling() {
    return cudf_velox::CudfConfig::getInstance().ucxxBlockingPolling;
  }
  static bool ucxxErrorHandling() {
    return cudf_velox::CudfConfig::getInstance().ucxxErrorHandling;
  }
#else
  static int exchangeLogLevel() {
    return readIntEnv("VELOX_UCX_LOG_LEVEL", 0, 0, 10);
  }
  static bool ucxxBlockingPolling() {
    return readBoolEnv("VELOX_UCX_BLOCKING_POLLING", false);
  }
  static bool ucxxErrorHandling() {
    return readBoolEnv(
        "VELOX_UCX_ERROR_HANDLING", "VELOX_UCX_CPU_ERROR_HANDLING", true);
  }
#endif
};

std::string readLocalHostIdentity() {
  if (const char* env = std::getenv("VELOX_UCX_CPU_HOST_ID")) {
    if (*env != '\0') {
      return std::string{"env:"} + env;
    }
  }

  std::ifstream bootId("/proc/sys/kernel/random/boot_id");
  std::string value;
  if (bootId >> value && !value.empty()) {
    value = std::string{"boot:"} + value;
  }

  if (value.empty()) {
    std::array<char, 256> hostname{};
    if (::gethostname(hostname.data(), hostname.size() - 1) == 0 &&
        hostname[0] != '\0') {
      value = std::string{"host:"} + hostname.data();
    }
  }

  struct stat ipcNs {};
  struct stat pidNs {};
  if (value.empty() || ::stat("/proc/self/ns/ipc", &ipcNs) != 0 ||
      ::stat("/proc/self/ns/pid", &pidNs) != 0) {
    return "";
  }

  std::ostringstream out;
  out << value << "|ipc:" << ipcNs.st_dev << ":" << ipcNs.st_ino
      << "|pid:" << pidNs.st_dev << ":" << pidNs.st_ino;
  return out.str();
}

uint32_t getLocalHostIdHash() {
  const auto identity = readLocalHostIdentity();
  if (identity.empty()) {
    return 0;
  }
  return fnv1a_32(identity);
}

bool isSameKnownHost(uint32_t localHostIdHash, uint32_t peerHostIdHash) {
  return localHostIdHash != 0 && peerHostIdHash != 0 &&
      localHostIdHash == peerHostIdHash;
}
} // namespace

// static
std::once_flag Communicator::onceFlag;
std::shared_ptr<Communicator> Communicator::instancePtr_ = nullptr;

/* static */
std::shared_ptr<Communicator> Communicator::initAndGet(
    uint16_t port,
    std::string_view coordinatorURL,
    ContinueFuture* future) {
  if (!FLAGS_velox_ucx_exchange) {
    return nullptr;
  }
  std::call_once(onceFlag, [&] {
    instancePtr_ = std::shared_ptr<Communicator>(new Communicator());
    instancePtr_->port_ = port;
    instancePtr_->coordinatorURL_ = coordinatorURL;
    // Generate a random unique worker ID for same-process detection.
    // std::random_device reads from /dev/urandom on Linux (non-blocking).
    // A 64-bit random value has negligible collision probability.
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    instancePtr_->workerId_ = dist(gen);
    LOG(INFO) << "Communicator workerId=" << instancePtr_->workerId_;
    instancePtr_->hostIdHash_ = getLocalHostIdHash();
    LOG(INFO) << "Communicator hostIdHash=" << instancePtr_->hostIdHash_;
    auto logLevel = UcxConfigView::exchangeLogLevel();
    LOG(INFO) << "ucx-exchange VLOG level set to " << logLevel;
    if (logLevel > 0) {
      // Set VLOG level for all ucx-exchange source files.
      // kUcxExchangeModules is auto-generated by CMake from the source list.
      for (const char* module : kUcxExchangeModules) {
        google::SetVLOGLevel(module, logLevel);
      }
    }
    if (future) {
      *future = instancePtr_->promise_.getSemiFuture();
    }
  });
  VELOX_CHECK_EQ(
      instancePtr_->port_,
      port,
      "Cannot initialize communicator again with different port");
  return instancePtr_;
}

/* static */
std::shared_ptr<Communicator> Communicator::getInstance() {
  VELOX_CHECK_NOT_NULL(
      instancePtr_, "Communicator not initialized. Call init(port) first.");
  return instancePtr_;
}

/* static */ void Communicator::cStyleListenerCallback(
    ucp_conn_request_h conn_request,
    void* arg) {
  // cast the argument back to our instance variable:
  Communicator* instance = static_cast<Communicator*>(arg);
  instance->listenerCallback(conn_request);
}

Communicator::~Communicator() {
  listener_.reset();
  // Note: worker_->flush() was removed - it only applies to RMA (Remote Memory
  // Access) operations like ucp_put/ucp_get, which this code doesn't use.
  // This code only uses tag send/recv and active messages.
  worker_.reset();
  context_.reset();
  VLOG(3) << "Communicator destructed";
}

/// @brief Run doesn't return until stop() is called.
/// The thread that calls run owns communicator state dispatch. UCX worker
/// progress is owned by the dedicated progress thread started by run().
void Communicator::run() {
  VLOG(3) << "Using error handling mode: " << UcxConfigView::ucxxErrorHandling()
          << std::endl;
  VLOG(3) << "Using blocking progress mode: "
          << UcxConfigView::ucxxBlockingPolling() << std::endl;

  running_.store(true);
#ifdef VELOX_ENABLE_CUDF
  // Force CUDA context creation. CPU-only builds skip this entirely:
  // there is no GPU to initialize.
  auto cudaStatus = cudaFree(0);
  VELOX_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to initialize CUDA context: {}",
      cudaGetErrorString(cudaStatus));
  int cudaDevice = 0;
  cudaStatus = cudaGetDevice(&cudaDevice);
  VELOX_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to get CUDA device: {} ({})",
      static_cast<int>(cudaStatus),
      cudaGetErrorString(cudaStatus));
#endif

  // create the UCXX context, worker, listener-context etc.
  if (UcxConfigView::ucxxBlockingPolling()) {
    context_ = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  } else {
    // Keep wakeup enabled so external signals can interrupt a blocking worker
    // and so ucxx can wake the progress thread when needed.
    context_ = ucxx::createContext(
        {}, UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_WAKEUP);
  }

  worker_ = context_->createWorker(false);

  listener_ = worker_->createListener(
      port_, Communicator::cStyleListenerCallback, this);

#ifdef VELOX_ENABLE_CUDF
  // Setup the active message callback that handles the
  // initial handshake and creates the senders for the cudf path.
  // CPU-only builds skip this: Acceptor.cpp / UcxExchangeServer.cpp
  // aren't compiled in.
  ucxx::AmReceiverCallbackInfo info(kAmCallbackOwner, kAmCallbackId);
  worker_->registerAmReceiverCallback(info, &Acceptor::cStyleAMCallback);
#endif

  // Parallel callback for the CPU-row exchange path. The CPU acceptor
  // creates a UcxCpuRowExchangeServer in response to its own handshakes.
  ucxx::AmReceiverCallbackInfo cpuInfo(kAmCallbackOwner, kAmCpuCallbackId);
  worker_->registerAmReceiverCallback(
      cpuInfo, &UcxCpuRowAcceptor::cStyleAMCallback);

  // The thread that called run() owns CommElement::process() /
  // state-machine dispatch. The UCXX progress thread owns worker progress and
  // callback execution so callbacks have one producer thread.
  const bool blockingMode = UcxConfigView::ucxxBlockingPolling();
  LOG(INFO) << "Communicator starting one UCXX progress thread "
            << "(maxWorkItemsPerDrain=" << maxWorkItemsPerDrain() << ")";
#ifdef VELOX_ENABLE_CUDF
  worker_->setProgressThreadStartCallback(
      [](void* arg) {
        const int device = *static_cast<int*>(arg);
        auto cudaStatus = cudaSetDevice(device);
        VELOX_CHECK(
            cudaStatus == cudaSuccess,
            "Failed to set CUDA device on UCXX progress thread: {} ({})",
            static_cast<int>(cudaStatus),
            cudaGetErrorString(cudaStatus));
        cudaStatus = cudaFree(0);
        VELOX_CHECK(
            cudaStatus == cudaSuccess,
            "Failed to initialize CUDA context on UCXX progress thread: {} ({})",
            static_cast<int>(cudaStatus),
            cudaGetErrorString(cudaStatus));
      },
      &cudaDevice);
#endif
  worker_->startProgressThread(!blockingMode, blockingMode ? 0 : 1);

  promise_.setValue();

  VLOG(3) << "Communicator running.";
  while (running_) {
    try {
      // Periodic heartbeat for diagnostic logging.
      auto now = std::chrono::steady_clock::now();
      if (now - lastHeartbeat_ >= std::chrono::seconds(5)) {
        std::lock_guard<std::mutex> lock(elemMutex_);

#ifdef VELOX_ENABLE_CUDF
        // GPU memory usage via CUDA runtime.
        size_t gpuFree = 0, gpuTotal = 0;
        auto memStatus = cudaMemGetInfo(&gpuFree, &gpuTotal);
        if (memStatus != cudaSuccess) {
          LOG(WARNING) << "cudaMemGetInfo failed: "
                       << cudaGetErrorString(memStatus);
        }
        size_t gpuUsedMB = (gpuTotal - gpuFree) / (1024 * 1024);
        size_t gpuFreeMB = gpuFree / (1024 * 1024);
        size_t gpuTotalMB = gpuTotal / (1024 * 1024);

        // Count ExSrv vs ExSrc elements (cudf path only; the CPU path
        // uses different CommElement subclasses and we don't have a
        // typed dynamic_cast hook here for them yet).
        int numServers = 0, numSources = 0;
        for (const auto& elem : elements_) {
          if (dynamic_cast<UcxExchangeServer*>(elem.get())) {
            ++numServers;
          } else if (dynamic_cast<UcxExchangeSource*>(elem.get())) {
            ++numSources;
          }
        }
#endif

        size_t numEndpoints;
        {
          std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
          numEndpoints = endpoints_.size();
        }
        VLOG(2) << "[COMM-HEARTBEAT] workQueue=" << workQueue_.size()
                << " elements=" << elements_.size()
#ifdef VELOX_ENABLE_CUDF
                << " (servers=" << numServers << " sources=" << numSources
                << ")"
#endif
                << " endpoints=" << numEndpoints
                << " deferredCleanup=" << deferredEndpointCleanup_.size()
                << " deferredRequests=" << deferredRequests_.size()
                << " workItemsProcessed="
                << workItemsProcessed_.exchange(0, std::memory_order_relaxed)
#ifdef VELOX_ENABLE_CUDF
                << " GPU=" << gpuUsedMB << "/" << gpuTotalMB << "MB"
                << " (free=" << gpuFreeMB << "MB)"
#endif
            ;
        // workItemsProcessed_ was already reset by exchange() above.
        lastHeartbeat_ = now;
      }

      // Process deferred endpoint cleanups from callbacks.
      // UCX callbacks cannot call closeBlocking() (which progresses the
      // worker) or iterate communicators_, so they defer cleanup to
      // this main loop via deferEndpointCleanup().
      while (auto ep = deferredEndpointCleanup_.pop()) {
        VLOG(3) << "Processing deferred endpoint cleanup";
        // First, close all communicators associated with this endpoint.
        // This must happen before removeEndpointRef() which may destroy
        // the endpoint.
        ep->closeAndDrainCommunicators();
        removeEndpointRef(ep);
      }

      // Drain primary-thread work. The UCXX progress thread is the only
      // thread that progresses the worker.
      drainWorkQueue();

      // Clean up deferred requests that UCX has fully processed.
      // These are cancelled requests whose GPU buffers needed to stay
      // alive until UCX finished any in-flight operations on them.
      // The communicator thread sweeps this list; UCX callbacks append to it
      // via deferRequestCleanup(), under the mutex.
      {
        std::lock_guard<std::mutex> lock(deferredRequestsMutex_);
        if (!deferredRequests_.empty()) {
          deferredRequests_.erase(
              std::remove_if(
                  deferredRequests_.begin(),
                  deferredRequests_.end(),
                  [](const auto& req) { return req->isCompleted(); }),
              deferredRequests_.end());
        }
      }
      std::this_thread::yield();
    } catch (ucxx::IOError& e) {
      LOG(ERROR) << "In Communicator main loop UCXX Exception: " << e.what();
      throw;
    }
  }
  VLOG(3) << "Communicator stopping.";

  if (worker_) {
    worker_->stopProgressThread();
  }
}

/// @brief Stops the communicator, called from an outside thread.
void Communicator::stop() {
  running_.store(false);
  signalWorker();
  size_t elementsSize;
  {
    std::lock_guard<std::mutex> lock(elemMutex_);
    elementsSize = elements_.size();
  }
  // endpoints_ is only accessed from the Communicator thread, so reading it
  // here would be a data race.
  VLOG(3) << "In Communicator::stop "
          << " elements_.size(): " << elementsSize
          << " workQueue_.size(): " << workQueue_.size();
}

void Communicator::registerCommElement(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  auto ret = elements_.insert(comms);
  VELOX_CHECK(ret.second, "CommElement already registered!");
  // Also put the comms element into the work queue.
  workQueue_.push(comms);
  signalWorker();
}

void Communicator::signalWorker() {
  if (worker_ && UcxConfigView::ucxxBlockingPolling()) {
    worker_->signal();
  }
}

void Communicator::addToWorkQueue(std::shared_ptr<CommElement> comms) {
  if (!comms) {
    return;
  }
  workQueue_.push(comms);
  signalWorker();
}

void Communicator::unregister(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  if (!comms) {
    return;
  }
  workQueue_.erase(comms);
  elements_.erase(comms);
}

std::shared_ptr<EndpointRef> Communicator::assocEndpointRef(
    std::shared_ptr<CommElement> comms,
    HostPort hostPort) {
  {
    std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
    auto it = endpoints_.find(hostPort);
    if (it != endpoints_.end()) {
      std::shared_ptr<EndpointRef> ep = it->second;
      ep->addCommElem(comms);
      return ep;
    }
  }

  // Endpoint doesn't exist; connect and register it. With error handling
  // enabled, UCXX reports connection failures through request callbacks
  // instead of leaving exchange sources waiting indefinitely.
  const bool errorHandling = UcxConfigView::ucxxErrorHandling();
  std::shared_ptr<ucxx::Endpoint> ep;
  try {
    ep = worker_->createEndpointFromHostname(
        hostPort.hostname, hostPort.port, errorHandling);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create UCX endpoint to " << hostPort.hostname
                 << ":" << hostPort.port << ": " << e.what();
    return nullptr;
  }
  if (ep == nullptr) {
    return nullptr;
  }

  auto epRef = std::make_shared<EndpointRef>(ep);
  epRef->addCommElem(comms);

  {
    std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
    auto [it, inserted] = endpoints_.insert(std::pair{hostPort, epRef});
    if (!inserted) {
      std::shared_ptr<EndpointRef> existing = it->second;
      existing->addCommElem(comms);
      return existing;
    }
  }
  if (errorHandling) {
    ep->setCloseCallback(EndpointRef::onClose, epRef);
  }
  return epRef;
}

bool Communicator::hasSameHostTransportIdentity(uint32_t peerHostIdHash) const {
  return isSameKnownHost(hostIdHash_, peerHostIdHash);
}

std::shared_ptr<EndpointRef>
Communicator::createSameHostEndpointRefFromWorkerAddress(
    std::string_view workerAddress,
    std::string peerIp,
    uint32_t peerHostIdHash) {
  if (workerAddress.empty()) {
    return nullptr;
  }

  if (!hasSameHostTransportIdentity(peerHostIdHash)) {
    VLOG(1) << "[CPU-UCX] not creating same-host worker-address endpoint "
            << "peerIp=" << peerIp << " localHostIdHash=" << hostIdHash_
            << " peerHostIdHash=" << peerHostIdHash;
    return nullptr;
  }

  std::string key{workerAddress};
  {
    std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
    auto it = workerAddressEndpoints_.find(key);
    if (it != workerAddressEndpoints_.end()) {
      return it->second;
    }
  }

  // This is a separate worker-address data connection, not the listener
  // bootstrap connection. UCX wireup carries this endpoint's err_mode to the
  // peer; the remote internal endpoint is created with the same mode. Do not
  // mix modes across the two sides of a single listener/conn_request pair.
  constexpr bool kErrorHandling = false;
  VLOG(1) << "[CPU-UCX] creating same-host worker-address endpoint peerIp="
          << peerIp << " localHostIdHash=" << hostIdHash_
          << " peerHostIdHash=" << peerHostIdHash
          << " errorHandling=" << kErrorHandling;
  std::shared_ptr<ucxx::Endpoint> ep;
  try {
    auto address = ucxx::createAddressFromString(key);
    ep = worker_->createEndpointFromWorkerAddress(address, kErrorHandling);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create UCX endpoint from worker address: "
                 << e.what();
    return nullptr;
  }
  if (ep == nullptr) {
    return nullptr;
  }

  auto epRef = std::make_shared<EndpointRef>(ep, std::move(peerIp));

  {
    std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
    auto [it, inserted] = workerAddressEndpoints_.emplace(key, epRef);
    if (!inserted) {
      return it->second;
    }
  }
  return epRef;
}

std::string Communicator::getWorkerAddress() const {
  VELOX_CHECK_NOT_NULL(worker_, "Communicator worker is not initialized");
  std::shared_ptr<ucxx::Address> address;
  const bool success = worker_->registerGenericPre(
      [&]() { address = worker_->getAddress(); }, 3000000000);
  VELOX_CHECK(success, "Timed out reading UCX worker address");
  VELOX_CHECK_NOT_NULL(address, "Communicator worker address is null");
  return address->getString();
}

void Communicator::removeEndpointRef(std::shared_ptr<EndpointRef> ep) {
  std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
  VLOG(3) << "In Communicator::removeEndpointRef for Communicator with port = "
          << Communicator::getInstance()->port_;

  // Close the endpoint if it's still alive.
  // With the UCXX progress thread running, closeBlocking() schedules close
  // work on that thread and waits for completion. Do not call it from a UCX
  // callback; callbacks defer cleanup to this communicator loop instead.
  if (ep->endpoint_ && ep->endpoint_->isAlive()) {
    VLOG(3) << "In Communicator::removeEndpointRef call closeBlocking";
    ep->endpoint_->closeBlocking();
  }
  for (auto it = endpoints_.begin(); it != endpoints_.end();) {
    if (it->second == ep) {
      it = endpoints_.erase(it);
    } else {
      ++it;
    }
  }
  VLOG(3) << "- Communicator::removeEndpointRef";
}

void Communicator::deferEndpointCleanup(std::shared_ptr<EndpointRef> ep) {
  // This method is safe to call from UCX callbacks because it doesn't
  // call any blocking/progress functions. The actual cleanup happens
  // in the main run() loop.
  VLOG(3) << "Deferring endpoint cleanup to main loop";
  deferredEndpointCleanup_.push(ep);
  signalWorker();
}

void Communicator::deferRequestCleanup(std::shared_ptr<ucxx::Request> request) {
  if (request) {
    std::lock_guard<std::mutex> lock(deferredRequestsMutex_);
    deferredRequests_.push_back(std::move(request));
  }
}

void Communicator::drainWorkQueue() {
  // Primary-thread work dispatch. The UCXX progress thread does not run
  // CommElement::process(). Keep the try-lock path because close() can
  // still race from driver / endpoint-cleanup paths.
  std::vector<std::shared_ptr<CommElement>> deferred;
  bool anyProcessed = false;
  int poppedItems = 0;
  while (poppedItems < maxWorkItemsPerDrain()) {
    auto comms = workQueue_.pop();
    if (!comms) {
      break;
    }
    ++poppedItems;

    std::unique_lock<std::recursive_mutex> lock(
        comms->processMutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      deferred.push_back(std::move(comms));
      continue;
    }
    comms->process();
    workItemsProcessed_.fetch_add(1, std::memory_order_relaxed);
    anyProcessed = true;
  }
  for (auto& c : deferred) {
    workQueue_.push(std::move(c));
  }
  // If the only thing we did this round was push deferred items back,
  // yield rather than spin; the lock-holder's progress (or another
  // thread's progress) needs CPU time to make forward motion.
  if (!anyProcessed && !deferred.empty()) {
    std::this_thread::yield();
  }
}

std::shared_ptr<EndpointRef> Communicator::findEndpointRefByHandle(
    ucp_ep_h handle) {
  std::lock_guard<std::mutex> lock(acceptor_.mutex_);
  auto it = acceptor_.handleToEndpointRef_.find(handle);
  if (it != acceptor_.handleToEndpointRef_.end()) {
    return it->second;
  }
  return nullptr;
}

const std::string& Communicator::getCoordinatorUrl() {
  return coordinatorURL_;
}

std::string Communicator::getListenerIp() const {
  if (listener_) {
    return listener_->getIp();
  }
  return "";
}

uint16_t Communicator::getListenerPort() const {
  if (listener_) {
    return listener_->getPort();
  }
  return port_;
}

/// @brief The callback method that is invoked when a client connects.
void Communicator::listenerCallback(ucp_conn_request_h conn_request) {
  char ip_str[INET6_ADDRSTRLEN];
  char port_str[INET6_ADDRSTRLEN];
  ucp_conn_request_attr_t attr{};

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(
      &attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
  VLOG(3)
      << "Communicator received a connection request from client at address "
      << ip_str << ":" << port_str;

  // incoming endpoints are not shared. Outgoing endpoints to the same node are
  // shared. This guarantees that between any two nodes, there will be at most 2
  // endpoints, one per direction. For compatibility reasons, both incoming and
  // outgoing endpoints are represented using the EndpointRef.
  auto endpoint = listener_->createEndpointFromConnRequest(
      conn_request, UcxConfigView::ucxxErrorHandling());
  // Pass the peer's actual IP to EndpointRef for reliable intra-node detection.
  auto epRef = std::make_shared<EndpointRef>(endpoint, std::string(ip_str));
  if (UcxConfigView::ucxxErrorHandling()) {
    endpoint->setCloseCallback(EndpointRef::onClose, epRef);
  }
  // Add this endpoint reference to the list of endpoints.
  // NOTE: This runs inside a UCX listener callback (during worker progress),
  // so we must not throw. Throwing from a UCX callback is undefined behavior.
  unsigned long val = std::strtoul(port_str, nullptr, 10);
  if (val > static_cast<unsigned long>(std::numeric_limits<uint16_t>::max())) {
    LOG(ERROR) << "listenerCallback: port out of range for uint16_t: " << val;
    return;
  }

  uint16_t port = static_cast<uint16_t>(val);
  HostPort hp(ip_str, port);
  {
    std::lock_guard<std::recursive_mutex> lock(endpointsMutex_);
    auto res = endpoints_.insert(std::pair{hp, epRef});
    if (!res.second) {
      LOG(ERROR) << "listenerCallback: endpoint already exists for " << ip_str
                 << ":" << port;
      return;
    }
  }
  acceptor_.registerEndpointRef(epRef);
}

} // namespace facebook::velox::ucx_exchange

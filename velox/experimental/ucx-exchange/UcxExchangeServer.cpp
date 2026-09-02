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
#include <rmm/cuda_stream.hpp>

#include <glog/logging.h>
#include <rmm/cuda_stream_view.hpp>
#include <functional>
#include <mutex>
#include <unordered_map>
#include "cuda_runtime.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxCodecPipeline.h"
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"
#include "velox/experimental/ucx-exchange/UcxCompression.h"
#include "velox/experimental/ucx-exchange/UcxCompressionCostModel.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxExchangeServer::ServerState::Created, "Created"},
              {UcxExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxExchangeServer::ServerState::WaitingForCompression,
               "WaitingForCompression"},
              {UcxExchangeServer::ServerState::CompressionReady,
               "CompressionReady"},
              {UcxExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxExchangeServer::ServerState::WaitingForIntraNodeRetrieve,
               "WaitingForIntraNodeRetrieve"},
              {UcxExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}

bool meetsCompressionMinimum(std::size_t bytes) {
  const auto minimum =
      cudf_velox::CudfConfig::getInstance().exchangeCompressionMinBytes;
  return minimum <= 0 || bytes >= static_cast<std::size_t>(minimum);
}

constexpr std::size_t kLargeAdvancedCodecMinBytes = 128u << 20;

bool isPackedCompressionMode(std::string_view mode) {
  return mode == "column" || mode == "column-adaptive" || mode == "for" ||
      mode == "column-dict-pfor" || mode == "column-adaptive-dict-pfor" ||
      mode == "column-freq-pfor" || mode == "column-adaptive-freq-pfor" ||
      mode == "column-freq-pfor-min128" ||
      mode == "column-adaptive-freq-pfor-min128";
}

bool isAdaptiveCompressionMode(std::string_view mode) {
  return mode == "column-adaptive" || mode == "column-adaptive-dict-pfor" ||
      mode == "column-adaptive-freq-pfor" ||
      mode == "column-adaptive-freq-pfor-min128";
}

bool enablesAdvancedCodecs(std::string_view mode) {
  return mode == "column-dict-pfor" || mode == "column-adaptive-dict-pfor" ||
      mode == "column-freq-pfor" || mode == "column-adaptive-freq-pfor" ||
      mode == "column-freq-pfor-min128" ||
      mode == "column-adaptive-freq-pfor-min128";
}

std::size_t advancedCodecMinBytes(std::string_view mode) {
  return mode == "column-freq-pfor-min128" ||
          mode == "column-adaptive-freq-pfor-min128"
      ? kLargeAdvancedCodecMinBytes
      : 0;
}

UcxCompressionCostModel& compressionCostModel() {
  return UcxCompressionCostModel::instance(
      cudf_velox::CudfConfig::getInstance().exchangeCompressionSafetyMargin);
}

void logPackedCompressionAttempt(
    uint64_t workerId,
    const std::string& taskId,
    uint32_t destination,
    uint32_t sequenceNumber,
    const std::string& mode,
    const PackedCompressResult::Stats& stats,
    bool accepted,
    double seconds) {
  if (!VLOG_IS_ON(1)) {
    return;
  }
  const auto regions = stats.raw.regions + stats.byteRans.regions +
      stats.frameOfReference.regions + stats.deltaFrameOfReference.regions +
      stats.dictionaryPfor.regions + stats.frequencyPfor.regions +
      stats.deltaFrequencyPfor.regions + stats.float64Alp.regions +
      stats.float64ExponentRans.regions;
  const auto wireBytes = accepted ? stats.candidateBytes : stats.inputBytes;
  VLOG(1) << "[UCX-CODEC-ATTEMPT] worker=" << workerId << " task=" << taskId
          << " destination=" << destination << " seq=" << sequenceNumber
          << " mode=" << mode << " attempted=" << stats.attempted
          << " accepted=" << accepted << " inputBytes=" << stats.inputBytes
          << " candidateBytes=" << stats.candidateBytes
          << " wireBytes=" << wireBytes << " seconds=" << seconds
          << " advancedProbeAttempts=" << stats.advancedRegionProbeAttempts
          << " advancedProbeSkips=" << stats.advancedRegionProbeSkips
          << " regions=" << regions << " rawRegions=" << stats.raw.regions
          << " rawInputBytes=" << stats.raw.inputBytes
          << " rawCandidateBytes=" << stats.raw.candidateBytes
          << " ransRegions=" << stats.byteRans.regions
          << " ransInputBytes=" << stats.byteRans.inputBytes
          << " ransCandidateBytes=" << stats.byteRans.candidateBytes
          << " forRegions=" << stats.frameOfReference.regions
          << " forInputBytes=" << stats.frameOfReference.inputBytes
          << " forCandidateBytes=" << stats.frameOfReference.candidateBytes
          << " deltaRegions=" << stats.deltaFrameOfReference.regions
          << " deltaInputBytes=" << stats.deltaFrameOfReference.inputBytes
          << " deltaCandidateBytes="
          << stats.deltaFrameOfReference.candidateBytes
          << " dictPforRegions=" << stats.dictionaryPfor.regions
          << " dictPforInputBytes=" << stats.dictionaryPfor.inputBytes
          << " dictPforCandidateBytes=" << stats.dictionaryPfor.candidateBytes
          << " freqPforRegions=" << stats.frequencyPfor.regions
          << " freqPforInputBytes=" << stats.frequencyPfor.inputBytes
          << " freqPforCandidateBytes=" << stats.frequencyPfor.candidateBytes
          << " deltaFreqPforRegions=" << stats.deltaFrequencyPfor.regions
          << " deltaFreqPforInputBytes=" << stats.deltaFrequencyPfor.inputBytes
          << " deltaFreqPforCandidateBytes="
          << stats.deltaFrequencyPfor.candidateBytes
          << " float64AlpRegions=" << stats.float64Alp.regions
          << " float64AlpInputBytes=" << stats.float64Alp.inputBytes
          << " float64AlpCandidateBytes=" << stats.float64Alp.candidateBytes
          << " float64ExponentRansRegions=" << stats.float64ExponentRans.regions
          << " float64ExponentRansInputBytes="
          << stats.float64ExponentRans.inputBytes
          << " float64ExponentRansCandidateBytes="
          << stats.float64ExponentRans.candidateBytes
          << " residualRansAttempts=" << stats.residualRansAttempts
          << " residualRansAccepted=" << stats.residualRansAccepted
          << " residualRansInputBytes=" << stats.residualRansInputBytes
          << " residualRansCandidateBytes=" << stats.residualRansCandidateBytes;
}

void logBlobCompressionAttempt(
    uint64_t workerId,
    const std::string& taskId,
    uint32_t destination,
    uint32_t sequenceNumber,
    const CompressResult::Stats& stats,
    bool accepted,
    double seconds) {
  if (!VLOG_IS_ON(1)) {
    return;
  }
  const auto wireBytes = accepted ? stats.candidateBytes : stats.inputBytes;
  VLOG(1) << "[UCX-CODEC-ATTEMPT] worker=" << workerId << " task=" << taskId
          << " destination=" << destination << " seq=" << sequenceNumber
          << " mode=ans"
          << " attempted=" << stats.attempted << " accepted=" << accepted
          << " inputBytes=" << stats.inputBytes
          << " candidateBytes=" << stats.candidateBytes
          << " wireBytes=" << wireBytes << " seconds=" << seconds;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeServer,
    ServerState,
    serverStateNames)

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime (which should be freed promptly after DMA completes).
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory. The context remains alive as an empty shell for the
// lifetime of the Request, which is safe and costs negligible memory.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  // Compressed payload when exchange compression kicked in; kept alive with
  // the context until the DMA completes. When set, the send transfers this
  // buffer instead of data->gpu_data.
  std::shared_ptr<rmm::device_buffer> compressedData;
};

struct UcxExchangeServer::SharedCompressionWork {
  using Completion =
      std::function<void(std::shared_ptr<const AsyncCompressionResult>)>;

  explicit SharedCompressionWork(
      const std::shared_ptr<cudf::packed_columns>& input)
      : input(input) {}

  void subscribe(Completion completion) {
    std::shared_ptr<const AsyncCompressionResult> readyResult;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (result) {
        readyResult = result;
      } else {
        completions.push_back(std::move(completion));
        return;
      }
    }
    completion(std::move(readyResult));
  }

  void complete(std::shared_ptr<const AsyncCompressionResult> value) {
    std::vector<Completion> readyCompletions;
    {
      std::lock_guard<std::mutex> lock(mutex);
      VELOX_CHECK_NULL(result);
      result = value;
      readyCompletions = std::move(completions);
    }
    for (auto& completion : readyCompletions) {
      completion(value);
    }
  }

  std::weak_ptr<cudf::packed_columns> input;
  std::mutex mutex;
  std::shared_ptr<const AsyncCompressionResult> result;
  std::vector<Completion> completions;
};

void UcxExchangeServer::setState(ServerState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private
UcxExchangeServer::UcxExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer)
    : CommElement(communicator, endpointRef),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      isIntraNodeTransfer_(isIntraNodeTransfer),
      queueMgr_(UcxOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);

  if (isIntraNodeTransfer_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Detected same-node source (intra-node transfer) for "
            << partitionKey_.toString();
  }
}

// static
std::shared_ptr<UcxExchangeServer> UcxExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  auto ptr = std::shared_ptr<UcxExchangeServer>(new UcxExchangeServer(
      communicator, endpointRef, key, isIntraNodeTransfer));
  return ptr;
}

void UcxExchangeServer::process() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  switch (state_) {
    case ServerState::Created:
      setState(ServerState::ReadyToTransfer);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::ReadyToTransfer: {
      // Fetch the data from UcxQueueManager and store it in the dataPtr_;
      setState(ServerState::WaitingForDataFromQueue);
      // Register the callback with the destination queue to get data.
      // If the queue doesn't exist yet, getData will create an empty
      // queue and the callback will be triggered once the corresponding
      // source task has initialized the queue and added data to it.
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
      queueMgr_->getData(
          partitionKey_.taskId,
          partitionKey_.destination,
          [weakQueue](
              std::shared_ptr<cudf::packed_columns> data,
              std::vector<int64_t> remainingBytes) {
            auto self = weakQueue.lock();
            if (!self) {
              return; // Object was destroyed, safe to ignore
            }
            // Check if close() was called - avoid processing if we're shutting
            // down
            if (self->closed_.load(std::memory_order_acquire)) {
              VLOG(3) << "@" << self->partitionKey_.taskId
                      << " getData callback called after close, ignoring";
              return;
            }
            // This upcall may be called from another thread than the
            // communicator thread. It is called
            // when data on the queue becomes available.
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " Found data for client: "
                    << self->partitionKey_.toString();
            std::lock_guard<std::recursive_mutex> lock(self->dataMutex_);
            VELOX_CHECK_NULL(
                self->dataPtr_, "Data pointer exists: Illegal state!");
            self->dataPtr_ = std::move(data);
            self->setState(ServerState::DataReady);
            self->communicator_->addToWorkQueue(self);
          });
      this->communicator_->addToWorkQueue(getSelfPtr());
    } break;
    case ServerState::WaitingForDataFromQueue:
      // Waiting for data is handled by an upcall from the data queue. Nothing
      // to do
      break;
    case ServerState::DataReady:
      if (shouldPipelineCompression()) {
        startCompression();
      } else {
        sendData();
      }
      break;
    case ServerState::WaitingForCompression:
      // Completion is published by the codec executor.
      break;
    case ServerState::CompressionReady:
      sendData();
      break;
    case ServerState::WaitingForSendComplete:
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
      break;
    case ServerState::WaitingForIntraNodeRetrieve:
      // Intra-node transfer: check if the source has retrieved the data
      if (intraNodeRetrieveFuture_.valid()) {
        auto status =
            intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
          intraNodeRetrieveFuture_.get(); // Clear the future
          intraNodePollCount_ = 0;
          onIntraNodeRetrieveComplete();
        } else {
          // Not ready yet, re-queue to check later
          ++intraNodePollCount_;
          if (intraNodePollCount_ % 100 == 0) {
            VLOG(2) << "[INTRA] [ExSrv " << partitionKey_.toString()
                    << " seq=" << sequenceNumber_
                    << "] still waiting for source retrieval, polls="
                    << intraNodePollCount_;
          }
          communicator_->addToWorkQueue(getSelfPtr());
        }
      }
      break;
    case ServerState::Done:
      close();
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      break;
  };
}

void UcxExchangeServer::close() {
  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }
  VLOG(3) << "@" << partitionKey_.taskId
          << " Close UcxExchangeServer to remote " << partitionKey_.toString();

  // Cancel any outstanding requests. With weak_ptr callbacks, the callbacks
  // will safely no-op if we're destroyed before they complete.
  if (metaRequest_ && !metaRequest_->isCompleted()) {
    metaRequest_->cancel();
  }
  if (dataRequest_ && !dataRequest_->isCompleted()) {
    dataRequest_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  if (communicator_) {
    if (metaRequest_) {
      communicator_->deferRequestCleanup(std::move(metaRequest_));
    }
    if (dataRequest_) {
      communicator_->deferRequestCleanup(std::move(dataRequest_));
    }
    for (auto& req : completedRequests_) {
      communicator_->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  communicator_->unregister(getSelfPtr());
}

std::string UcxExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxExchangeServer> UcxExchangeServer::getSelfPtr() {
  return shared_from_this();
}

bool UcxExchangeServer::endpointAllowsCompression() {
  VELOX_CHECK_NOT_NULL(endpointRef_);
  if (!cudaIpcTransport_) {
    cudaIpcTransport_ = endpointRef_->usesTransport("cuda_ipc");
  }
  const bool allowed = cudaIpcTransport_.has_value() && !*cudaIpcTransport_;
  VLOG(1) << "[UCX-COMPRESSION-TRANSPORT] cudaIpcKnown="
          << cudaIpcTransport_.has_value()
          << " cudaIpc=" << cudaIpcTransport_.value_or(true)
          << " allowed=" << allowed;
  return allowed;
}

const UcxCompressionCostModel::Decision&
UcxExchangeServer::compressionDecision() {
  VELOX_CHECK_NOT_NULL(dataPtr_);
  if (!compressionDecision_) {
    compressionDecision_ = compressionCostModel().decide(
        partitionKey_.taskId, dataPtr_->gpu_data->size());
    const auto& decision = *compressionDecision_;
    VLOG(1) << "[UCX-COMPRESSION-DECISION] worker="
            << communicator_->getWorkerId() << " task=" << partitionKey_.taskId
            << " destination=" << partitionKey_.destination
            << " seq=" << sequenceNumber_ << " action="
            << UcxCompressionCostModel::actionName(decision.action)
            << " rawBytes=" << dataPtr_->gpu_data->size()
            << " encodeSamples=" << decision.encodeSamples
            << " transferSamples=" << decision.transferSamples
            << " decodeSamples=" << decision.decodeSamples
            << " candidateRatio=" << decision.candidateRatio
            << " effectiveTransferBps="
            << decision.effectiveTransferBytesPerSecond
            << " transferSavedSeconds="
            << decision.estimatedTransferSavedSeconds
            << " codecSeconds=" << decision.estimatedCodecSeconds;
  }
  return *compressionDecision_;
}

std::pair<std::shared_ptr<UcxExchangeServer::SharedCompressionWork>, bool>
UcxExchangeServer::acquireSharedCompressionWork(
    const std::shared_ptr<cudf::packed_columns>& input) {
  static std::mutex registryMutex;
  static std::unordered_map<
      const cudf::packed_columns*,
      std::weak_ptr<SharedCompressionWork>>
      registry;
  static std::size_t acquisitions{0};

  std::lock_guard<std::mutex> lock(registryMutex);
  if (++acquisitions % 1024 == 0) {
    for (auto it = registry.begin(); it != registry.end();) {
      if (it->second.expired()) {
        it = registry.erase(it);
      } else {
        ++it;
      }
    }
  }

  const auto key = input.get();
  auto it = registry.find(key);
  if (it != registry.end()) {
    if (auto work = it->second.lock()) {
      if (work->input.lock() == input) {
        return {std::move(work), false};
      }
    }
    registry.erase(it);
  }

  auto work = std::make_shared<SharedCompressionWork>(input);
  registry.emplace(key, work);
  return {std::move(work), true};
}

bool UcxExchangeServer::shouldPipelineCompression() {
  if (!codecPipelineEnabled() || isIntraNodeTransfer_ || !dataPtr_ ||
      !dataPtr_->gpu_data || dataPtr_->gpu_data->size() == 0 ||
      !meetsCompressionMinimum(dataPtr_->gpu_data->size())) {
    return false;
  }

  const auto& mode = cudf_velox::CudfConfig::getInstance().exchangeCompression;
  if (!isPackedCompressionMode(mode) && mode != "ans") {
    return false;
  }

  if (!endpointAllowsCompression()) {
    return false;
  }

  return !isAdaptiveCompressionMode(mode) ||
      compressionDecision().action != UcxCompressionCostModel::Action::kRaw;
}

void UcxExchangeServer::startCompression() {
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);
  VELOX_CHECK(getState() == ServerState::DataReady);
  VELOX_CHECK_NOT_NULL(dataPtr_);

  auto input = dataPtr_;
  const auto mode = cudf_velox::CudfConfig::getInstance().exchangeCompression;
  int device = 0;
  auto cudaStatus = cudaGetDevice(&device);
  VELOX_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to get codec CUDA device: {}",
      cudaGetErrorString(cudaStatus));

  setState(ServerState::WaitingForCompression);
  std::weak_ptr<UcxExchangeServer> weak = weak_from_this();
  auto [work, ownsCompression] = acquireSharedCompressionWork(input);
  compressionWork_ = work;
  work->subscribe(
      [weak, input](std::shared_ptr<const AsyncCompressionResult> result) {
        if (auto self = weak.lock()) {
          self->onCompressionComplete(input, std::move(result));
        }
      });
  if (!ownsCompression) {
    VLOG(1) << "@" << partitionKey_.taskId
            << " reusing broadcast compression for destination "
            << partitionKey_.destination << " sequence " << sequenceNumber_;
    return;
  }

  const auto taskId = partitionKey_.taskId;
  const auto destination = partitionKey_.destination;
  const auto sequenceNumber = sequenceNumber_;
  const auto workerId = communicator_->getWorkerId();
  double effectiveLinkBytesPerSecond = 0.0;
  if (isAdaptiveCompressionMode(mode)) {
    const auto observed = compressionDecision().effectiveTransferBytesPerSecond;
    if (observed > 0.0) {
      effectiveLinkBytesPerSecond = observed;
    }
  }
  submitCodecTask([work,
                   input,
                   mode,
                   device,
                   workerId,
                   taskId,
                   destination,
                   sequenceNumber,
                   effectiveLinkBytesPerSecond]() mutable {
    auto result = std::make_shared<AsyncCompressionResult>();
    PackedCompressResult::Stats packedStats;
    CompressResult::Stats blobStats;
    bool packedMode = false;
    bool accepted = false;
    const auto start = std::chrono::steady_clock::now();
    try {
      auto status = cudaSetDevice(device);
      VELOX_CHECK(
          status == cudaSuccess,
          "Failed to set codec CUDA device {}: {}",
          device,
          cudaGetErrorString(status));
      static thread_local rmm::cuda_stream codecStream;

      if (isPackedCompressionMode(mode)) {
        packedMode = true;
        auto packed = mode == "for" ? compressPackedFor(
                                          input->gpu_data->data(),
                                          input->gpu_data->size(),
                                          codecStream.view())
                                    : compressPacked(
                                          input->metadata->data(),
                                          input->gpu_data->data(),
                                          input->gpu_data->size(),
                                          codecStream.view(),
                                          0.02,
                                          enablesAdvancedCodecs(mode),
                                          advancedCodecMinBytes(mode),
                                          effectiveLinkBytesPerSecond);
        packedStats = packed.stats;
        accepted = packed.used;
        if (packed.used) {
          serializeRegions(packed, input->gpu_data->size(), result->descriptor);
          result->data =
              std::make_shared<rmm::device_buffer>(std::move(packed.data));
        }
      } else if (mode == "ans") {
        auto compressed = compressBlob(
            input->gpu_data->data(),
            input->gpu_data->size(),
            codecStream.view());
        blobStats = compressed.stats;
        accepted = compressed.used;
        if (compressed.used) {
          result->data =
              std::make_shared<rmm::device_buffer>(std::move(compressed.data));
          result->descriptor.reserve(2 + compressed.segSizes.size());
          result->descriptor.push_back(
              static_cast<int64_t>(ExchangeCodec::kByteRans));
          result->descriptor.push_back(
              static_cast<int64_t>(input->gpu_data->size()));
          for (auto size : compressed.segSizes) {
            result->descriptor.push_back(size);
          }
        }
      }
    } catch (...) {
      result->error = std::current_exception();
    }
    result->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (!result->error) {
      if (packedMode) {
        if (isAdaptiveCompressionMode(mode)) {
          compressionCostModel().recordEncode(
              taskId,
              packedStats.inputBytes,
              packedStats.candidateBytes,
              result->seconds);
        }
        logPackedCompressionAttempt(
            workerId,
            taskId,
            destination,
            sequenceNumber,
            mode,
            packedStats,
            accepted,
            result->seconds);
      } else {
        logBlobCompressionAttempt(
            workerId,
            taskId,
            destination,
            sequenceNumber,
            blobStats,
            accepted,
            result->seconds);
      }
    }

    work->complete(std::move(result));
  });
}

void UcxExchangeServer::onCompressionComplete(
    const std::shared_ptr<cudf::packed_columns>& input,
    std::shared_ptr<const AsyncCompressionResult> result) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    if (getState() != ServerState::WaitingForCompression || dataPtr_ != input) {
      return;
    }
    compressionResult_ = std::move(result);
    setState(ServerState::CompressionReady);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::sendData() {
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);

  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_
          << "] sendData hasData=" << (dataPtr_ != nullptr)
          << (dataPtr_ && dataPtr_->gpu_data
                  ? " size=" + std::to_string(dataPtr_->gpu_data->size())
                  : "");

  if (isIntraNodeTransfer_) {
    // INTRA-NODE TRANSFER PATH: Use registry for all communication, no UCXX
    // needed
    sendStart_ = std::chrono::high_resolution_clock::now();

    if (dataPtr_) {
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, dataPtr_, /*atEnd=*/false);
      dataPtr_.reset();
      intraNodeAtEndPublished_ = false;

      // Transition to WaitingForIntraNodeRetrieve state
      setState(ServerState::WaitingForIntraNodeRetrieve);
      communicator_->addToWorkQueue(getSelfPtr());
    } else {
      // Data pointer is null, so no more data will be coming.
      // Publish atEnd marker to registry
      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing atEnd for sequence "
              << sequenceNumber_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, nullptr, /*atEnd=*/true);
      intraNodeAtEndPublished_ = true;

      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);

      // Wait for source to acknowledge atEnd before finishing
      setState(ServerState::WaitingForIntraNodeRetrieve);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  } else {
    // REMOTE EXCHANGE PATH: Use UCXX for metadata and data transfer
    std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

    // Compressed payload for this chunk, when compression is enabled and
    // pays. Wire descriptor rides in remainingBytes:
    // [codecId, uncompressedBytes, segSize0, segSize1, ...].
    std::shared_ptr<rmm::device_buffer> compressedData;

    std::shared_ptr<const AsyncCompressionResult> prepared;
    if (getState() == ServerState::CompressionReady) {
      prepared = std::move(compressionResult_);
      VELOX_CHECK_NOT_NULL(prepared);
      if (prepared->error) {
        std::rethrow_exception(prepared->error);
      }
      compressedData = prepared->data;
    }

    if (dataPtr_) {
      // Copy metadata (not move) because in broadcast mode, the same
      // packed_columns may be shared across multiple destination queues.
      // Metadata is small (CPU-side), so copying is negligible.
      metadataMsg->cudfMetadata =
          std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
      metadataMsg->remainingBytes = {};
      if (prepared) {
        metadataMsg->remainingBytes = prepared->descriptor;
        if (compressedData) {
          VLOG(1) << "@" << partitionKey_.taskId << " encodeGBps="
                  << dataPtr_->gpu_data->size() /
                  std::max(prepared->seconds, 1e-9) / 1e9
                  << " bytes=" << dataPtr_->gpu_data->size() << " pipelined=1";
          VLOG(1) << "@" << partitionKey_.taskId
                  << " pipeline-compressed chunk " << sequenceNumber_ << ": "
                  << dataPtr_->gpu_data->size() << " -> "
                  << compressedData->size() << " bytes";
        }
      } else {
        const auto& compressionMode =
            cudf_velox::CudfConfig::getInstance().exchangeCompression;
        const auto inputBytes = dataPtr_->gpu_data->size();
        const bool configuredPacked = isPackedCompressionMode(compressionMode);
        const bool configuredBlob = compressionMode == "ans";
        const bool compressionAllowed =
            (configuredPacked || configuredBlob) && endpointAllowsCompression();
        const bool packedMode = compressionAllowed && configuredPacked;
        const bool blobMode = compressionAllowed && configuredBlob;
        const bool eligible = meetsCompressionMinimum(inputBytes);
        const bool adaptive = isAdaptiveCompressionMode(compressionMode);
        const auto* decision = adaptive && eligible && inputBytes > 0
            ? &compressionDecision()
            : nullptr;
        const bool selectedRaw = decision != nullptr &&
            decision->action == UcxCompressionCostModel::Action::kRaw;
        double effectiveLinkBytesPerSecond = 0.0;
        if (decision != nullptr &&
            decision->effectiveTransferBytesPerSecond > 0.0) {
          effectiveLinkBytesPerSecond =
              decision->effectiveTransferBytesPerSecond;
        }
        if (inputBytes > 0 && (!eligible || selectedRaw) &&
            (packedMode || blobMode)) {
          if (packedMode) {
            PackedCompressResult::Stats stats;
            stats.inputBytes = inputBytes;
            logPackedCompressionAttempt(
                communicator_->getWorkerId(),
                partitionKey_.taskId,
                partitionKey_.destination,
                sequenceNumber_,
                compressionMode,
                stats,
                false,
                0.0);
          } else {
            CompressResult::Stats stats;
            stats.inputBytes = inputBytes;
            logBlobCompressionAttempt(
                communicator_->getWorkerId(),
                partitionKey_.taskId,
                partitionKey_.destination,
                sequenceNumber_,
                stats,
                false,
                0.0);
          }
        } else if (packedMode && inputBytes > 0) {
          static rmm::cuda_stream columnStream;
          auto encodeStart = std::chrono::steady_clock::now();
          auto packed = compressionMode == "for"
              ? compressPackedFor(
                    dataPtr_->gpu_data->data(),
                    dataPtr_->gpu_data->size(),
                    columnStream.view())
              : compressPacked(
                    dataPtr_->metadata->data(),
                    dataPtr_->gpu_data->data(),
                    dataPtr_->gpu_data->size(),
                    columnStream.view(),
                    0.02,
                    enablesAdvancedCodecs(compressionMode),
                    advancedCodecMinBytes(compressionMode),
                    effectiveLinkBytesPerSecond);
          const double encSeconds =
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - encodeStart)
                  .count();
          if (adaptive) {
            compressionCostModel().recordEncode(
                partitionKey_.taskId,
                packed.stats.inputBytes,
                packed.stats.candidateBytes,
                encSeconds);
          }
          logPackedCompressionAttempt(
              communicator_->getWorkerId(),
              partitionKey_.taskId,
              partitionKey_.destination,
              sequenceNumber_,
              compressionMode,
              packed.stats,
              packed.used,
              encSeconds);
          if (packed.used) {
            VLOG(1) << "@" << partitionKey_.taskId << " encodeGBps="
                    << dataPtr_->gpu_data->size() / encSeconds / 1e9
                    << " bytes=" << dataPtr_->gpu_data->size();
            compressedData =
                std::make_shared<rmm::device_buffer>(std::move(packed.data));
            serializeRegions(
                packed,
                dataPtr_->gpu_data->size(),
                metadataMsg->remainingBytes);
            VLOG(1) << "@" << partitionKey_.taskId
                    << " column-compressed chunk " << sequenceNumber_ << ": "
                    << dataPtr_->gpu_data->size() << " -> "
                    << compressedData->size() << " bytes ("
                    << packed.regions.size() << " regions)";
          }
        } else if (blobMode && inputBytes > 0) {
          // Dedicated stream: the blob is already synchronized by the producer
          // and compressBlob synchronizes before returning, so the compressed
          // buffer is settled before the UCX hand-off below.
          static rmm::cuda_stream compressionStream;
          auto encodeStart = std::chrono::steady_clock::now();
          auto compressed = compressBlob(
              dataPtr_->gpu_data->data(),
              dataPtr_->gpu_data->size(),
              compressionStream.view());
          const double encSeconds =
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - encodeStart)
                  .count();
          logBlobCompressionAttempt(
              communicator_->getWorkerId(),
              partitionKey_.taskId,
              partitionKey_.destination,
              sequenceNumber_,
              compressed.stats,
              compressed.used,
              encSeconds);
          if (compressed.used) {
            compressedData = std::make_shared<rmm::device_buffer>(
                std::move(compressed.data));
            metadataMsg->remainingBytes.reserve(2 + compressed.segSizes.size());
            metadataMsg->remainingBytes.push_back(
                static_cast<int64_t>(ExchangeCodec::kByteRans));
            metadataMsg->remainingBytes.push_back(
                static_cast<int64_t>(dataPtr_->gpu_data->size()));
            for (auto segSize : compressed.segSizes) {
              metadataMsg->remainingBytes.push_back(segSize);
            }
            VLOG(1) << "@" << partitionKey_.taskId << " compressed chunk "
                    << sequenceNumber_ << ": " << dataPtr_->gpu_data->size()
                    << " -> " << compressedData->size() << " bytes";
          }
        }
      }
      metadataMsg->dataSizeBytes =
          compressedData ? compressedData->size() : dataPtr_->gpu_data->size();
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = true;
    }

    auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

    // send metadata.
    uint64_t metadataTag =
        getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
    if (metaRequest_) {
      completedRequests_.push_back(std::move(metaRequest_));
    }

    // Wrap the serialized metadata in a context so the callback can release
    // it after the send completes, while the Request (and context shell)
    // stays alive for UCP wireup replay.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;

    metaRequest_ = endpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta](
            ucs_status_t status, std::shared_ptr<void> arg) {
          // Release the metadata buffer from the context. The context
          // shell stays alive with the Request; only the payload is freed.
          auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata); // release CPU buffer

          auto self = weakMeta.lock();
          if (!self) {
            return; // Object was destroyed, safe to ignore
          }
          // Check if close() was called
          if (self->closed_.load(std::memory_order_acquire)) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata send callback called after close, ignoring";
            return;
          }
          if (status == UCS_OK) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata successfully sent to " << tid
                    << " with tag: " << std::hex << metadataTag;
          } else {
            VLOG(0) << "@" << self->partitionKey_.taskId
                    << " Error in sendData, send metadata "
                    << ucs_status_string(status) << " failed for task: " << tid;
            self->setState(ServerState::Done);
            self->communicator_->addToWorkQueue(self);
          }
        },
        metaCtx);

    // send the data chunk (if any)
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ =
          compressedData ? compressedData->size() : dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Sending rmm::buffer: " << std::hex
              << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      setState(ServerState::WaitingForSendComplete);
      uint64_t dataTag =
          getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
      if (dataRequest_) {
        completedRequests_.push_back(std::move(dataRequest_));
      }

      // Wrap the GPU data buffer in a context so the callback can release
      // it after the DMA completes, while the Request (and context shell)
      // stays alive for UCP wireup replay.
      auto dataCtx = std::make_shared<DataSendContext>();
      dataCtx->data = dataPtr_;
      dataCtx->compressedData = compressedData;

      void* sendPtr = compressedData ? compressedData->data()
                                     : dataCtx->data->gpu_data->data();
      const std::size_t sendBytes = compressedData
          ? compressedData->size()
          : dataCtx->data->gpu_data->size();
      dataRequest_ = endpointRef_->endpoint_->tagSend(
          sendPtr,
          sendBytes,
          ucxx::Tag{dataTag},
          false,
          [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
            // Release the GPU data buffer from the context. The DMA has
            // completed by the time this callback fires, so the buffer is
            // safe to free. The context shell stays alive with the Request.
            auto ctx = std::static_pointer_cast<DataSendContext>(arg);
            auto dataHolder = std::move(ctx->data);
            auto compressedHolder = std::move(ctx->compressedData);

            if (auto self = weakData.lock()) {
              self->sendComplete(status, arg);
            }
            // dataHolder is destroyed here, releasing the GPU buffer if
            // sendComplete() already reset the server's dataPtr_.
          },
          dataCtx);
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "@" << partitionKey_.taskId
              << " Finished transferring partition for task "
              << partitionKey_.toString();
      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  }
}

void UcxExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " sendComplete called after close, ignoring";
    return;
  }
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    VELOX_CHECK_NOT_NULL(dataPtr_, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    const double seconds = std::chrono::duration<double>(duration).count();
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

    VLOG(1) << "[UCX-TRANSFER-SAMPLE] worker=" << communicator_->getWorkerId()
            << " task=" << partitionKey_.taskId
            << " destination=" << partitionKey_.destination
            << " seq=" << sequenceNumber_
            << " rawBytes=" << dataPtr_->gpu_data->size()
            << " wireBytes=" << bytes_ << " seconds=" << seconds;

    const auto& compressionMode =
        cudf_velox::CudfConfig::getInstance().exchangeCompression;
    if (isAdaptiveCompressionMode(compressionMode) &&
        meetsCompressionMinimum(dataPtr_->gpu_data->size())) {
      compressionCostModel().recordTransfer(
          partitionKey_.taskId, bytes_, seconds);
    }

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";

    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    compressionWork_.reset();
    compressionDecision_.reset();
    VLOG(3) << "@" << partitionKey_.taskId
            << " Releasing dataPtr_ in sendComplete.";
    setState(ServerState::ReadyToTransfer);
  } else {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Error in sendComplete, send complete "
            << ucs_status_string(status);
    setState(ServerState::Done);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::onIntraNodeRetrieveComplete() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " onIntraNodeRetrieveComplete called after close, ignoring";
    return;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - sendStart_;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

  VLOG(3)
      << "@" << partitionKey_.taskId << " Intra-node transfer duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer throughput: " << throughput << " MByte/s";

  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer complete for sequence " << sequenceNumber_;

  if (intraNodeAtEndPublished_) {
    // This was the final atEnd marker, we're done
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: atEnd acknowledged, finishing";
    setState(ServerState::Done);
  } else {
    // More data may be coming, continue transfer loop
    this->sequenceNumber_++;
    setState(ServerState::ReadyToTransfer);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

} // namespace facebook::velox::ucx_exchange

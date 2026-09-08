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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include <glog/logging.h>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <folly/ScopeGuard.h>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
uint64_t multiplySaturated(uint64_t value, uint64_t multiplier) {
  if (value == 0) {
    return 0;
  }
  if (multiplier > std::numeric_limits<uint64_t>::max() / value) {
    return std::numeric_limits<uint64_t>::max();
  }
  return value * multiplier;
}

uint64_t addSaturated(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

uint64_t divideCeil(uint64_t value, uint64_t divisor) {
  VELOX_CHECK_GT(divisor, 0);
  return value / divisor + (value % divisor == 0 ? 0 : 1);
}

constexpr uint64_t kTransferWindowMultiplier = 4;

std::atomic<uint64_t>& inProcessGpuMaterializationReservationBytes() {
  static std::atomic<uint64_t> bytes{0};
  return bytes;
}

uint64_t effectiveFreeBytes(const CudfDeviceMemoryInfo& info) {
  return addSaturated(info.freeBytes, info.poolReusableBytes);
}
} // namespace

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const std::vector<std::string>& n1,
    const std::vector<std::string>& n2,
    std::vector<uint32_t>& remap) {
  std::unordered_map<std::string, uint32_t> lookup;
  for (uint32_t i = 0; i < n1.size(); ++i) {
    lookup[n1[i]] = i;
  }

  remap.clear();
  remap.reserve(n2.size());
  for (const auto& key : n2) {
    remap.push_back(lookup.at(key));
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(UcxOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      targetRowsPerChunk_(ctx->queryConfig().get<int64_t>(
          core::QueryConfig::kUcxPartitionedOutputBatchRows,
          CudfConfig::getInstance().partitionedOutputBatchRows)),
      initialPayloadBytes_(
          std::max<uint64_t>(ctx->queryConfig().maxOutputBufferSize(), 1)) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(inNames, outNames, remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");

  const auto inputBytes = input->estimateFlatSize();
  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(inputBytes, input->size());
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingBytes_ += inputBytes;
  pendingInputs_.push_back(std::move(cudfVector));

  if (shouldFlushPending()) {
    flushPending();
  }
}

void UcxPartitionedOutput::recordAllocationPressure(uint64_t waitBytes) {
  auto queueManager = sharedQueueManager();
  queueManager->recordFullTransferCongestion(this->taskId());

  const auto destinationBaseBytes = std::max<uint64_t>(
      initialPayloadBytes_,
      divideCeil(
          std::max<uint64_t>(waitBytes, 1),
          std::max<uint64_t>(static_cast<uint64_t>(numPartitions_), 1)));
  const auto clampedBaseBytes = static_cast<int64_t>(std::min<uint64_t>(
      destinationBaseBytes,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    queueManager->recordTransferCongestion(
        this->taskId(), partition, clampedBaseBytes);
  }

  const auto clampedWaitBytes = static_cast<int64_t>(std::min<uint64_t>(
      std::max<uint64_t>(waitBytes, 1),
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  if (queueManager->waitForFullTransferCapacity(
          this->taskId(), clampedWaitBytes, &future_) ||
      queueManager->checkBlocked(this->taskId(), &future_)) {
    blockingReason_ = exec::BlockingReason::kWaitForConsumer;
    return;
  }

  blockingReason_ = exec::BlockingReason::kNotBlocked;
}

void UcxPartitionedOutput::partitionPendingInputBatch() {
  VELOX_CHECK(pendingInputBatch_.has_value());

  auto& input = pendingInputBatch_.value();
  const auto estimatedBytes = std::max<int64_t>(input.estimatedBytes, 1);

  uint64_t materializationReservationBytes = 0;
  auto releaseMaterializationReservation = folly::makeGuard([&]() {
    if (materializationReservationBytes == 0) {
      return;
    }
    inProcessGpuMaterializationReservationBytes().fetch_sub(
        materializationReservationBytes, std::memory_order_acq_rel);
  });

  const bool needsPartitionMaterialization =
      numPartitions_ > 1 && usesHashPartitioning();
  if (needsPartitionMaterialization) {
    const auto estimatedMaterializationBytes =
        static_cast<uint64_t>(estimatedBytes);
    const auto producerHeadroomBytes = std::max<uint64_t>(
        estimatedMaterializationBytes, initialPayloadBytes_);
    if (const auto memoryInfo = currentDeviceMemoryInfo()) {
      const auto freeBytes = effectiveFreeBytes(*memoryInfo);
      const auto inProcessReserved =
          inProcessGpuMaterializationReservationBytes().load(
              std::memory_order_acquire);
      const auto requiredBytes = addSaturated(
          addSaturated(inProcessReserved, estimatedMaterializationBytes),
          producerHeadroomBytes);
      if (freeBytes <= requiredBytes) {
        recordAllocationPressure(estimatedMaterializationBytes);
        return;
      }

      inProcessGpuMaterializationReservationBytes().fetch_add(
          estimatedMaterializationBytes, std::memory_order_acq_rel);
      materializationReservationBytes = estimatedMaterializationBytes;
    }
  }

  try {
    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (usesHashPartitioning()) {
        hashPartition(input.tableView, input.stream, estimatedBytes);
      } else {
        equalPartition(
            input.tableView,
            input.stream,
            std::move(input.tableOwner),
            std::move(input.vectorOwners),
            estimatedBytes);
      }

      pendingInputBatch_.reset();
      drainPendingPartitionedBatch();
      return;
    }

    auto packedCols = cudf::pack(
        input.tableView, input.stream, cudf::get_current_device_resource_ref());
    input.stream.synchronize();
    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(packedCols.metadata), std::move(packedCols.gpu_data));
    queueManager->enqueue(
        this->taskId(),
        0,
        std::move(packedColsPtr),
        input.tableView.num_rows());

    pendingInputBatch_.reset();
    const auto blocked = queueManager->checkBlocked(this->taskId(), &future_);
    blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                              : exec::BlockingReason::kNotBlocked;
  } catch (const std::bad_alloc&) {
    recordAllocationPressure(static_cast<uint64_t>(estimatedBytes));
  }
}

void UcxPartitionedOutput::flushPending() {
  if (pendingPartitionedBatch_) {
    drainPendingPartitionedBatch();
    return;
  }

  if (pendingInputBatch_) {
    partitionPendingInputBatch();
    return;
  }

  if (pendingInputs_.empty()) {
    return;
  }

  const auto estimatedBytes = std::max<int64_t>(pendingBytes_, 1);

  try {
    cudf::table_view tableView;
    rmm::cuda_stream_view stream = pendingInputs_.back()->stream();
    // Keeps the merged table alive while tableView references it.
    std::unique_ptr<cudf::table> mergedTable;
    std::vector<CudfVectorPtr> vectorOwners;

    if (pendingInputs_.size() == 1) {
      // Fast path: use the single input's view directly (no GPU alloc).
      vectorOwners = std::move(pendingInputs_);
      auto& cv = vectorOwners[0];
      stream = cv->stream();
      tableView = remap_.empty()
          ? cv->getTableView()
          : cv->getTableView().select(remap_.begin(), remap_.end());
    } else {
      // Collect (remapped) table views.
      std::vector<cudf::table_view> views;
      std::vector<rmm::cuda_stream_view> inputStreams;
      views.reserve(pendingInputs_.size());
      inputStreams.reserve(pendingInputs_.size());
      for (auto& v : pendingInputs_) {
        inputStreams.push_back(v->stream());
        views.push_back(
            remap_.empty()
                ? v->getTableView()
                : v->getTableView().select(remap_.begin(), remap_.end()));
      }

      cudf::detail::join_streams(inputStreams, stream);
      mergedTable = cudf::concatenate(
          views, stream, cudf::get_current_device_resource_ref());

      orderCudfVectorDeallocationsAfterStream(
          pendingInputs_, inputStreams, stream);

      // Free input GPU memory before partitioning (peak = 2x -> 1x).
      pendingInputs_.clear();

      tableView = mergedTable->view();
    }

    pendingInputBatch_.emplace(PendingInputBatch{
        std::move(mergedTable),
        std::move(vectorOwners),
        tableView,
        estimatedBytes,
        stream});
    pendingRows_ = 0;
    pendingBytes_ = 0;
    partitionPendingInputBatch();
  } catch (const std::bad_alloc&) {
    recordAllocationPressure(static_cast<uint64_t>(estimatedBytes));
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  if (shouldDrainPending()) {
    return exec::BlockingReason::kNotBlocked;
  }
  if (!finished_ &&
      sharedQueueManager()->checkBlocked(this->taskId(), future)) {
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  if (shouldDrainPending()) {
    flushPending();
    if (shouldDrainPending() ||
        blockingReason_ != exec::BlockingReason::kNotBlocked) {
      return nullptr;
    }
  }
  if (noMoreInput_) {
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

bool UcxPartitionedOutput::shouldFlushPending() const {
  return !pendingInputs_.empty() &&
      (targetRowsPerChunk_ <= 0 || pendingRows_ >= targetRowsPerChunk_);
}

bool UcxPartitionedOutput::shouldDrainPending() const {
  return pendingPartitionedBatch_.has_value() || pendingInputBatch_.has_value() ||
      shouldFlushPending() || (noMoreInput_ && !pendingInputs_.empty());
}

bool UcxPartitionedOutput::usesHashPartitioning() const {
  return !partitionKeyIndices_.empty() || isGatherPartition_;
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  isGatherPartition_ =
      dynamic_cast<const core::GatherPartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec()) != nullptr;
  spec_ = planNode->partitionFunctionSpec().toString();

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream,
    int64_t estimatedBytes) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  auto partitionedView = partitionedTable->view();
  pendingPartitionedBatch_.emplace(PendingPartitionedBatch{
      std::move(partitionedTable),
      {},
      partitionedView,
      std::move(partitionOffsets),
      estimatedBytes,
      false,
      stream,
      0,
      {},
      {},
      {},
      0});
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream,
    std::unique_ptr<cudf::table> tableOwner,
    std::vector<CudfVectorPtr> vectorOwners,
    int64_t estimatedBytes) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  offsets.reserve(numPartitions_ + 1);
  offsets.push_back(0);
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  offsets.push_back(size);

  pendingPartitionedBatch_.emplace(PendingPartitionedBatch{
      std::move(tableOwner),
      std::move(vectorOwners),
      tableView,
      std::move(offsets),
      estimatedBytes,
      false,
      stream,
      0,
      {},
      {},
      {},
      0});
}

// Estimate payload bytes for a partition using the reservation estimate. The
// contiguous fast path rechecks real packed sizes before enqueueing if this
// estimate still does not fit.
uint64_t UcxPartitionedOutput::estimatedPartitionBytes(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end) const {
  const auto partitionRows = end - start;
  const auto totalRows = batch.tableView.num_rows();
  if (partitionRows <= 0 || totalRows <= 0 || batch.estimatedBytes <= 0) {
    return 0;
  }

  const auto estimatedBytes =
      static_cast<long double>(batch.estimatedBytes) *
      static_cast<long double>(partitionRows) /
      static_cast<long double>(totalRows);
  if (estimatedBytes >=
      static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }

  const auto wholeBytes = static_cast<uint64_t>(estimatedBytes);
  return estimatedBytes > static_cast<long double>(wholeBytes)
      ? wholeBytes + 1
      : wholeBytes;
}

uint64_t UcxPartitionedOutput::exactPartitionBytes(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end) const {
  if (start == end) {
    return 0;
  }

  std::vector<cudf::size_type> offsets{start, end};
  auto views = cudf::slice(batch.tableView, offsets);
  VELOX_CHECK_EQ(views.size(), 1);
  return cudf::packed_size(
      views[0], batch.stream, cudf::get_current_device_resource_ref());
}

// Estimate a row chunk size that keeps each packed UCX payload under the given
// byte target. The caller increases the target geometrically, so large
// partitions do not become many tiny messages, while the first allocations stay
// small enough to avoid an immediate memory spike.
cudf::size_type UcxPartitionedOutput::rowsForPayloadTarget(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end,
    uint64_t targetBytes) const {
  const auto partitionRows = end - start;
  if (partitionRows <= 0) {
    return 0;
  }

  const auto totalRows = batch.tableView.num_rows();
  if (totalRows <= 0 || batch.estimatedBytes <= 0 || targetBytes == 0) {
    return partitionRows;
  }

  const auto partitionBytes = estimatedPartitionBytes(batch, start, end);
  if (partitionBytes <= targetBytes) {
    return partitionRows;
  }

  const auto rows = static_cast<cudf::size_type>(
      static_cast<long double>(partitionRows) *
      static_cast<long double>(targetBytes) /
      static_cast<long double>(partitionBytes));
  return std::max<cudf::size_type>(rows, 1);
}

bool UcxPartitionedOutput::shouldSplitPayload(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end) const {
  return estimatedPartitionBytes(batch, start, end) >
      maxTransferWindowBytes(batch);
}

uint64_t UcxPartitionedOutput::baseTransferWindowBytes(
    const PendingPartitionedBatch& batch) const {
  if (batch.estimatedBytes <= 0 || numPartitions_ == 0) {
    return initialPayloadBytes_;
  }

  const auto averagePartitionBytes = divideCeil(
      static_cast<uint64_t>(batch.estimatedBytes),
      static_cast<uint64_t>(numPartitions_));
  return std::max<uint64_t>(initialPayloadBytes_, averagePartitionBytes);
}

uint64_t UcxPartitionedOutput::maxTransferWindowBytes(
    const PendingPartitionedBatch& batch) const {
  return std::min<uint64_t>(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      multiplySaturated(
          baseTransferWindowBytes(batch), kTransferWindowMultiplier));
}

uint64_t UcxPartitionedOutput::normalTransferWindowBytes(
    const PendingPartitionedBatch& batch) const {
  return std::min<uint64_t>(
      maxTransferWindowBytes(batch),
      multiplySaturated(baseTransferWindowBytes(batch), 2));
}

uint64_t UcxPartitionedOutput::transferWindowBytes(
    const PendingPartitionedBatch& batch,
    int destination,
    const std::shared_ptr<UcxOutputQueueManager>& queueManager) const {
  const auto baseWindow = baseTransferWindowBytes(batch);
  const auto normalWindow = normalTransferWindowBytes(batch);
  const auto maxWindow = maxTransferWindowBytes(batch);
  if (maxWindow <= baseWindow) {
    return baseWindow;
  }

  return static_cast<uint64_t>(queueManager->transferWindowBytes(
      this->taskId(),
      destination,
      static_cast<int64_t>(baseWindow),
      static_cast<int64_t>(normalWindow),
      static_cast<int64_t>(maxWindow)));
}

void UcxPartitionedOutput::initializePartitionDrainState(
    PendingPartitionedBatch& batch) const {
  if (!batch.nextRows.empty()) {
    return;
  }

  VELOX_CHECK_EQ(
      batch.offsets.size(), numPartitions_ + 1, "mismatch in numPartitions_");

  const auto transferWindow = batch.conservativeChunkSizing
      ? initialPayloadBytes_
      : normalTransferWindowBytes(batch);
  batch.nextRows.resize(numPartitions_);
  batch.nextPayloadBytes.resize(numPartitions_, 0);
  batch.drainDeficits.resize(numPartitions_, 0);
  batch.remainingPartitions = 0;
  batch.nextPartition = 0;

  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    const auto start = batch.offsets[partition];
    const auto end = batch.offsets[partition + 1];
    VELOX_CHECK_LE(start, end);

    batch.nextRows[partition] = start;
    if (start == end) {
      continue;
    }

    batch.nextPayloadBytes[partition] =
        (batch.conservativeChunkSizing || shouldSplitPayload(batch, start, end))
        ? initialPayloadBytes_
        : transferWindow;
    ++batch.remainingPartitions;
  }
}

bool UcxPartitionedOutput::tryDrainWithContiguousSplit(
    PendingPartitionedBatch& batch) {
  if (!batch.nextRows.empty() || batch.tableView.num_rows() == 0) {
    return false;
  }
  if (batch.conservativeChunkSizing) {
    return false;
  }

  VELOX_CHECK_EQ(
      batch.offsets.size(), numPartitions_ + 1, "mismatch in numPartitions_");

  auto queueManager = sharedQueueManager();
  auto recordContiguousSplitPressure = [&]() {
    batch.conservativeChunkSizing = true;
    queueManager->recordFullTransferCongestion(this->taskId());
    for (size_t partition = 0; partition < numPartitions_; ++partition) {
      if (batch.offsets[partition] < batch.offsets[partition + 1]) {
        queueManager->recordTransferCongestion(
            this->taskId(),
            partition,
            static_cast<int64_t>(baseTransferWindowBytes(batch)));
      }
    }
  };
  std::vector<bool> eligible(numPartitions_, false);
  std::vector<uint64_t> transferReservations(numPartitions_, 0);
  std::vector<uint64_t> transferWindows(numPartitions_, 0);
  auto releaseReservations = folly::makeGuard([&]() {
    for (size_t partition = 0; partition < transferReservations.size();
         ++partition) {
      if (transferReservations[partition] == 0) {
        continue;
      }
      queueManager->releaseTransferReservation(
          this->taskId(),
          partition,
          static_cast<int64_t>(transferReservations[partition]));
      transferReservations[partition] = 0;
    }
  });
  auto releaseReservation = [&](size_t partition) {
    if (transferReservations[partition] == 0) {
      return;
    }
    queueManager->releaseTransferReservation(
        this->taskId(),
        partition,
        static_cast<int64_t>(transferReservations[partition]));
    transferReservations[partition] = 0;
  };
  size_t eligiblePartitions = 0;

  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    const auto start = batch.offsets[partition];
    const auto end = batch.offsets[partition + 1];
    VELOX_CHECK_LE(start, end);
    if (start == end) {
      continue;
    }

    auto transferWindow = transferWindowBytes(batch, partition, queueManager);
    auto partitionBytes = estimatedPartitionBytes(batch, start, end);
    if (partitionBytes > transferWindow) {
      try {
        partitionBytes = exactPartitionBytes(batch, start, end);
      } catch (const std::bad_alloc&) {
        recordContiguousSplitPressure();
        return false;
      }
      if (partitionBytes > transferWindow) {
        queueManager->recordTransferDemand(
            this->taskId(),
            partition,
            static_cast<int64_t>(std::min<uint64_t>(
                partitionBytes,
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))),
            static_cast<int64_t>(baseTransferWindowBytes(batch)),
            static_cast<int64_t>(maxTransferWindowBytes(batch)));
        transferWindow = transferWindowBytes(batch, partition, queueManager);
        if (partitionBytes > transferWindow) {
          continue;
        }
      }
    }

    const auto reservationBytes = std::max<uint64_t>(partitionBytes, 1);
    if (queueManager->reserveTransferBytes(
            this->taskId(),
            partition,
            static_cast<int64_t>(reservationBytes),
            static_cast<int64_t>(transferWindow),
            nullptr)) {
      continue;
    }

    transferReservations[partition] = reservationBytes;
    transferWindows[partition] = transferWindow;
    eligible[partition] = true;
    ++eligiblePartitions;
  }

  if (eligiblePartitions == 0) {
    return false;
  }

  initializePartitionDrainState(batch);

  bool madeProgress = false;
  for (size_t runStart = 0; runStart < numPartitions_;) {
    while (runStart < numPartitions_ && !eligible[runStart]) {
      ++runStart;
    }
    if (runStart == numPartitions_) {
      break;
    }

    auto runEnd = runStart;
    while (runEnd + 1 < numPartitions_ && eligible[runEnd + 1]) {
      ++runEnd;
    }

    const auto sliceStart = batch.offsets[runStart];
    const auto sliceEnd = batch.offsets[runEnd + 1];
    VELOX_CHECK_LT(sliceStart, sliceEnd);

    std::vector<cudf::size_type> sliceOffsets{sliceStart, sliceEnd};
    auto slices = cudf::slice(batch.tableView, sliceOffsets, batch.stream);
    VELOX_CHECK_EQ(slices.size(), 1);

    std::vector<cudf::size_type> splitOffsets;
    splitOffsets.reserve(runEnd - runStart);
    for (size_t partition = runStart + 1; partition <= runEnd; ++partition) {
      splitOffsets.push_back(batch.offsets[partition] - sliceStart);
    }

    std::vector<cudf::packed_table> contiguousTables;
    try {
      contiguousTables = cudf::contiguous_split(
          slices[0],
          splitOffsets,
          batch.stream,
          cudf::get_current_device_resource_ref());
    } catch (const std::bad_alloc&) {
      recordContiguousSplitPressure();
      return false;
    }

    // UCXX/UCX is not stream-aware, so the packed payloads must be complete
    // before exposing their raw device pointers to the exchange server.
    batch.stream.synchronize();

    VELOX_CHECK_EQ(contiguousTables.size(), runEnd - runStart + 1);
    for (size_t index = 0; index < contiguousTables.size(); ++index) {
      const auto partition = runStart + index;
      auto& partitionTable = contiguousTables[index];
      if (partitionTable.table.num_rows() == 0) {
        releaseReservation(partition);
        continue;
      }

      auto transferWindow = transferWindows[partition];
      const auto actualBytes = partitionTable.data.gpu_data->size();
      if (actualBytes > transferWindow) {
        queueManager->recordTransferDemand(
            this->taskId(),
            partition,
            static_cast<int64_t>(std::min<uint64_t>(
                actualBytes,
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))),
            static_cast<int64_t>(baseTransferWindowBytes(batch)),
            static_cast<int64_t>(maxTransferWindowBytes(batch)));
        transferWindow = transferWindowBytes(batch, partition, queueManager);
        transferWindows[partition] = transferWindow;
        if (actualBytes > transferWindow) {
          batch.conservativeChunkSizing = true;
          batch.nextPayloadBytes[partition] = initialPayloadBytes_;
          batch.drainDeficits[partition] = 0;
          queueManager->recordTransferCongestion(
              this->taskId(),
              partition,
              static_cast<int64_t>(baseTransferWindowBytes(batch)));
          releaseReservation(partition);
          return false;
        }
      }

      if (actualBytes > transferReservations[partition]) {
        const auto reservationDelta =
            actualBytes - transferReservations[partition];
        if (queueManager->reserveTransferBytes(
                this->taskId(),
                partition,
                static_cast<int64_t>(reservationDelta),
                static_cast<int64_t>(transferWindow),
                nullptr)) {
          batch.conservativeChunkSizing = true;
          batch.nextPayloadBytes[partition] = initialPayloadBytes_;
          batch.drainDeficits[partition] = 0;
          queueManager->recordTransferCongestion(
              this->taskId(),
              partition,
              static_cast<int64_t>(baseTransferWindowBytes(batch)));
          releaseReservation(partition);
          return false;
        }
        transferReservations[partition] += reservationDelta;
      }

      const auto transferReservationBytes =
          static_cast<int64_t>(transferReservations[partition]);

      auto packedColsPtr = std::make_unique<cudf::packed_columns>(
          std::move(partitionTable.data.metadata),
          std::move(partitionTable.data.gpu_data));

      queueManager->enqueue(
          this->taskId(),
          partition,
          std::move(packedColsPtr),
          partitionTable.table.num_rows(),
          transferReservationBytes);
      transferReservations[partition] = 0;

      madeProgress = true;
      if (batch.nextRows[partition] < batch.offsets[partition + 1]) {
        batch.nextRows[partition] = batch.offsets[partition + 1];
        VELOX_CHECK_GT(batch.remainingPartitions, 0);
        --batch.remainingPartitions;
      }
      batch.nextPayloadBytes[partition] = 0;
      batch.drainDeficits[partition] = 0;
    }

    runStart = runEnd + 1;
  }

  if (!madeProgress) {
    return false;
  }

  if (batch.remainingPartitions > 0) {
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return false;
  }

  pendingPartitionedBatch_.reset();
  pendingRows_ = 0;
  pendingBytes_ = 0;
  blockingReason_ = exec::BlockingReason::kNotBlocked;
  releaseReservations.dismiss();
  return true;
}

bool UcxPartitionedOutput::tryDrainWithFullContiguousSplit(
    PendingPartitionedBatch& batch) {
  if (!batch.nextRows.empty() || batch.tableView.num_rows() == 0) {
    return false;
  }

  VELOX_CHECK_EQ(
      batch.offsets.size(), numPartitions_ + 1, "mismatch in numPartitions_");

  auto queueManager = sharedQueueManager();
  std::vector<uint64_t> transferReservations(numPartitions_, 0);
  auto releaseReservations = folly::makeGuard([&]() {
    for (size_t partition = 0; partition < transferReservations.size();
         ++partition) {
      if (transferReservations[partition] == 0) {
        continue;
      }
      queueManager->releaseTransferReservation(
          this->taskId(),
          partition,
          static_cast<int64_t>(transferReservations[partition]));
      transferReservations[partition] = 0;
    }
  });

  const auto maxReservation =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  auto recordFullSplitPressure = [&]() {
    batch.conservativeChunkSizing = true;
    queueManager->recordFullTransferCongestion(this->taskId());
    for (size_t partition = 0; partition < numPartitions_; ++partition) {
      if (batch.offsets[partition] < batch.offsets[partition + 1]) {
        queueManager->recordTransferCongestion(
            this->taskId(),
            partition,
            static_cast<int64_t>(baseTransferWindowBytes(batch)));
      }
    }
  };

  // Build split offsets first, then check live device headroom before
  // materializing the full fanout.
  std::vector<cudf::size_type> splitOffsets;
  splitOffsets.reserve(numPartitions_ > 0 ? numPartitions_ - 1 : 0);
  for (size_t partition = 1; partition < numPartitions_; ++partition) {
    splitOffsets.push_back(batch.offsets[partition]);
  }

  uint64_t deviceReservationBytes = 0;
  auto releaseDeviceReservation = folly::makeGuard([&]() {
    if (deviceReservationBytes == 0) {
      return;
    }
    inProcessGpuMaterializationReservationBytes().fetch_sub(
        deviceReservationBytes, std::memory_order_acq_rel);
  });

  const auto estimatedPayloadBytes = std::max<uint64_t>(
      static_cast<uint64_t>(std::max<int64_t>(batch.estimatedBytes, 1)), 1);
  // A full fanout is only safe when the device can hold the fanout and still
  // leave room for another producer-side allocation of comparable scale. This
  // keeps small/free queries on the fast path, but prevents exchange-retained
  // payloads from being the first actor to discover the GPU allocation cliff.
  const auto producerHeadroomBytes =
      std::max<uint64_t>(estimatedPayloadBytes, baseTransferWindowBytes(batch));
  if (const auto memoryInfo = currentDeviceMemoryInfo()) {
    const auto freeBytes = effectiveFreeBytes(*memoryInfo);
    const auto inProcessReserved =
        inProcessGpuMaterializationReservationBytes().load(
            std::memory_order_acquire);
    const auto requiredBytes = addSaturated(
        addSaturated(inProcessReserved, estimatedPayloadBytes),
        producerHeadroomBytes);
    if (freeBytes <= requiredBytes) {
      recordFullSplitPressure();
      return false;
    }

    inProcessGpuMaterializationReservationBytes().fetch_add(
        estimatedPayloadBytes, std::memory_order_acq_rel);
    deviceReservationBytes = estimatedPayloadBytes;
  }

  std::vector<cudf::packed_table> contiguousTables;
  try {
    contiguousTables = cudf::contiguous_split(
        batch.tableView,
        splitOffsets,
        batch.stream,
        cudf::get_current_device_resource_ref());
  } catch (const std::bad_alloc&) {
    recordFullSplitPressure();
    return false;
  }

  // UCXX/UCX is not stream-aware, so the packed payloads must be complete
  // before exposing their raw device pointers to the exchange server.
  batch.stream.synchronize();

  VELOX_CHECK_EQ(contiguousTables.size(), numPartitions_);
  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    auto& partitionTable = contiguousTables[partition];
    if (partitionTable.table.num_rows() == 0) {
      continue;
    }

    auto actualReservationBytes = transferReservations[partition];
    const auto actualBytes = partitionTable.data.gpu_data->size();
    if (actualBytes > actualReservationBytes) {
      auto reservationDelta = actualBytes - actualReservationBytes;
      reservationDelta = std::min<uint64_t>(reservationDelta, maxReservation);
      const auto blocked = queueManager->reserveFullTransferBytes(
          this->taskId(),
          partition,
          static_cast<int64_t>(reservationDelta),
          nullptr);
      if (blocked) {
        recordFullSplitPressure();
        return false;
      }
      actualReservationBytes += reservationDelta;
      transferReservations[partition] = actualReservationBytes;
    }
  }

  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    auto& partitionTable = contiguousTables[partition];
    if (partitionTable.table.num_rows() == 0) {
      if (transferReservations[partition] > 0) {
        queueManager->releaseTransferReservation(
            this->taskId(),
            partition,
            static_cast<int64_t>(transferReservations[partition]));
        transferReservations[partition] = 0;
      }
      continue;
    }

    const auto actualReservationBytes = transferReservations[partition];
    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(partitionTable.data.metadata),
        std::move(partitionTable.data.gpu_data));

    queueManager->enqueue(
        this->taskId(),
        partition,
        std::move(packedColsPtr),
        partitionTable.table.num_rows(),
        static_cast<int64_t>(actualReservationBytes));
    transferReservations[partition] = 0;
  }

  pendingPartitionedBatch_.reset();
  pendingRows_ = 0;
  pendingBytes_ = 0;
  blockingReason_ = exec::BlockingReason::kNotBlocked;
  releaseReservations.dismiss();
  return true;
}

bool UcxPartitionedOutput::drainPendingPartitionedBatch() {
  VELOX_CHECK(pendingPartitionedBatch_.has_value());

  auto queueManager = sharedQueueManager();
  auto& batch = pendingPartitionedBatch_.value();
  if (tryDrainWithFullContiguousSplit(batch)) {
    return true;
  }
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    return false;
  }
  if (tryDrainWithContiguousSplit(batch)) {
    return true;
  }
  initializePartitionDrainState(batch);

  VELOX_CHECK_EQ(
      batch.offsets.size(), numPartitions_ + 1, "mismatch in numPartitions_");

  while (batch.remainingPartitions > 0) {
    int waitPartition = -1;
    bool madeProgress = false;

    for (size_t visited = 0; visited < numPartitions_; ++visited) {
      const auto partition = batch.nextPartition;
      batch.nextPartition = (batch.nextPartition + 1) % numPartitions_;

      if (batch.nextRows[partition] >= batch.offsets[partition + 1]) {
        continue;
      }

      const auto transferWindow =
          transferWindowBytes(batch, partition, queueManager);
      batch.drainDeficits[partition] = std::min<uint64_t>(
          transferWindow,
          addSaturated(batch.drainDeficits[partition], transferWindow));
      batch.nextPayloadBytes[partition] =
          std::min<uint64_t>(batch.nextPayloadBytes[partition], transferWindow);

      while (batch.nextRows[partition] < batch.offsets[partition + 1]) {
        if (queueManager->checkTransferCapacity(
                this->taskId(), partition, transferWindow, nullptr)) {
          if (waitPartition < 0) {
            waitPartition = partition;
          }
          break;
        }

        if (batch.drainDeficits[partition] <
            batch.nextPayloadBytes[partition]) {
          break;
        }

        madeProgress = true;
        const auto start = batch.offsets[partition];
        const auto end = batch.offsets[partition + 1];
        VELOX_CHECK_LE(start, end);
        VELOX_CHECK_LT(batch.nextRows[partition], end);

        const auto chunkStart = batch.nextRows[partition];
        const auto rowsPerChunk = rowsForPayloadTarget(
            batch, chunkStart, end, batch.nextPayloadBytes[partition]);
        VELOX_CHECK_GT(rowsPerChunk, 0);
        auto chunkEnd =
            std::min<cudf::size_type>(end, chunkStart + rowsPerChunk);
        auto recordChunkAllocationPressure = [&](uint64_t waitBytes) {
          queueManager->recordFullTransferCongestion(this->taskId());
          queueManager->recordTransferCongestion(
              this->taskId(),
              partition,
              static_cast<int64_t>(baseTransferWindowBytes(batch)));
          batch.conservativeChunkSizing = true;
          batch.nextPayloadBytes[partition] = initialPayloadBytes_;
          batch.drainDeficits[partition] = 0;
          if (queueManager->waitForFullTransferCapacity(
                  this->taskId(),
                  static_cast<int64_t>(
                      std::min<uint64_t>(
                          std::max<uint64_t>(waitBytes, 1),
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()))),
                  &future_)) {
            blockingReason_ = exec::BlockingReason::kWaitForConsumer;
          }
        };

        uint64_t exactBytes = 0;
        try {
          exactBytes = exactPartitionBytes(batch, chunkStart, chunkEnd);
          while (exactBytes > transferWindow && chunkEnd - chunkStart > 1) {
            const auto currentRows = chunkEnd - chunkStart;
            auto adjustedRows = static_cast<cudf::size_type>(
                static_cast<long double>(currentRows) *
                static_cast<long double>(transferWindow) /
                static_cast<long double>(exactBytes));
            adjustedRows = std::clamp<cudf::size_type>(
                adjustedRows, 1, currentRows - 1);
            chunkEnd = chunkStart + adjustedRows;
            exactBytes = exactPartitionBytes(batch, chunkStart, chunkEnd);
          }
        } catch (const std::bad_alloc&) {
          recordChunkAllocationPressure(batch.nextPayloadBytes[partition]);
          return false;
        }

        auto transferReservationBytes = std::max<uint64_t>(exactBytes, 1);
        auto reservationLimit =
            std::max<uint64_t>(transferWindow, transferReservationBytes);
        const auto maxReservation =
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        transferReservationBytes =
            std::min<uint64_t>(transferReservationBytes, maxReservation);
        reservationLimit =
            std::min<uint64_t>(reservationLimit, maxReservation);

        if (queueManager->reserveTransferBytes(
                this->taskId(),
                partition,
                static_cast<int64_t>(transferReservationBytes),
                static_cast<int64_t>(reservationLimit),
                &future_)) {
          blockingReason_ = exec::BlockingReason::kWaitForConsumer;
          return false;
        }
        auto releaseReservation = folly::makeGuard([&]() {
          queueManager->releaseTransferReservation(
              this->taskId(),
              partition,
              static_cast<int64_t>(transferReservationBytes));
        });

        std::vector<cudf::size_type> sliceOffsets{chunkStart, chunkEnd};
        auto tableSlices = cudf::slice(batch.tableView, sliceOffsets);
        VELOX_CHECK_EQ(tableSlices.size(), 1);

        std::optional<cudf::packed_columns> packedCols;
        try {
          packedCols.emplace(cudf::pack(
              tableSlices[0],
              batch.stream,
              cudf::get_current_device_resource_ref()));
        } catch (const std::bad_alloc&) {
          recordChunkAllocationPressure(transferReservationBytes);
          return false;
        }

        // UCXX/UCX is not stream-aware, so the packed payload must be complete
        // before exposing its raw device pointer to the exchange server.
        batch.stream.synchronize();

        auto packedColsPtr = std::make_unique<cudf::packed_columns>(
            std::move(packedCols->metadata), std::move(packedCols->gpu_data));
        const auto packedBytes = packedColsPtr->gpu_data->size();
        if (packedBytes > transferReservationBytes) {
          const auto reservationDelta = packedBytes - transferReservationBytes;
          const auto actualReservationLimit = std::min<uint64_t>(
              std::max<uint64_t>(reservationLimit, packedBytes),
              maxReservation);
          if (queueManager->reserveTransferBytes(
                  this->taskId(),
                  partition,
                  static_cast<int64_t>(reservationDelta),
                  static_cast<int64_t>(actualReservationLimit),
                  &future_)) {
            blockingReason_ = exec::BlockingReason::kWaitForConsumer;
            return false;
          }
          transferReservationBytes += reservationDelta;
        }
        queueManager->enqueue(
            this->taskId(),
            partition,
            std::move(packedColsPtr),
            chunkEnd - chunkStart,
            static_cast<int64_t>(transferReservationBytes));
        releaseReservation.dismiss();

        if (packedBytes >= batch.drainDeficits[partition]) {
          batch.drainDeficits[partition] = 0;
        } else {
          batch.drainDeficits[partition] -= packedBytes;
        }

        batch.nextRows[partition] = chunkEnd;
        if (chunkEnd == end) {
          --batch.remainingPartitions;
          batch.nextPayloadBytes[partition] = 0;
          batch.drainDeficits[partition] = 0;
          break;
        }

        batch.nextPayloadBytes[partition] = std::min<uint64_t>(
            transferWindow,
            multiplySaturated(batch.nextPayloadBytes[partition], 2));
      }
    }

    if (madeProgress) {
      continue;
    }

    VELOX_CHECK_GE(waitPartition, 0);
    const auto transferWindow =
        transferWindowBytes(batch, waitPartition, queueManager);
    if (queueManager->checkTransferCapacity(
            this->taskId(), waitPartition, transferWindow, &future_)) {
      blockingReason_ = exec::BlockingReason::kWaitForConsumer;
      return false;
    }
  }

  pendingPartitionedBatch_.reset();
  pendingRows_ = 0;
  pendingBytes_ = 0;
  blockingReason_ = exec::BlockingReason::kNotBlocked;
  return true;
}

} // namespace facebook::velox::ucx_exchange

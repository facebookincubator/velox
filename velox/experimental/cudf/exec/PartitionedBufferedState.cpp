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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/column/column.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/memory_pool>

#include <algorithm>
#include <limits>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

// Avoid spinning forever when a single partition key cannot be split.
constexpr uint32_t kMaxSplitSeedAttempts = 4;

bool referencesSameColumn(
    const cudf::column_view& left,
    const cudf::column_view& right) {
  if (left.type() != right.type() || left.size() != right.size() ||
      left.offset() != right.offset() ||
      left.head<void>() != right.head<void>() ||
      left.null_mask() != right.null_mask() ||
      left.num_children() != right.num_children()) {
    return false;
  }

  for (cudf::size_type i = 0; i < left.num_children(); ++i) {
    if (!referencesSameColumn(left.child(i), right.child(i))) {
      return false;
    }
  }
  return true;
}

bool nodeEmpty(const PartitionedBufferedState::Node& node) {
  if (node.isLeaf()) {
    return node.leafState == nullptr;
  }

  for (const auto& child : node.children) {
    if (child && !nodeEmpty(*child)) {
      return false;
    }
  }
  return true;
}

size_t countNonEmptyChildren(
    const std::vector<InputChunk>& storedPartitions,
    const std::vector<InputChunk>& incomingPartitions) {
  VELOX_CHECK_EQ(storedPartitions.size(), incomingPartitions.size());

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if (!storedPartitions[i].empty() || !incomingPartitions[i].empty()) {
      ++nonEmptyChildren;
    }
  }
  return nonEmptyChildren;
}

struct SplitLeafAttempt {
  PartitionSpec spec;
  std::vector<InputChunk> storedPartitions;
  std::vector<InputChunk> incomingPartitions;
};

class AccountedPinnedBuffer {
 public:
  AccountedPinnedBuffer() = default;

  AccountedPinnedBuffer(size_t bytes, memory::MemoryPool* pool)
      : pool_(pool), bytes_(bytes) {
    if (bytes_ == 0) {
      return;
    }

    VELOX_CHECK_NOT_NULL(pool_);
    pool_->reportExternalAllocation(bytes_);
    try {
      data_ = resource().allocate_sync(bytes_);
    } catch (...) {
      pool_->reportExternalFree(bytes_);
      pool_ = nullptr;
      bytes_ = 0;
      throw;
    }
  }

  ~AccountedPinnedBuffer() {
    reset();
  }

  AccountedPinnedBuffer(AccountedPinnedBuffer&& other) noexcept
      : pool_(std::exchange(other.pool_, nullptr)),
        data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}

  AccountedPinnedBuffer& operator=(AccountedPinnedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      pool_ = std::exchange(other.pool_, nullptr);
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  AccountedPinnedBuffer(const AccountedPinnedBuffer&) = delete;
  AccountedPinnedBuffer& operator=(const AccountedPinnedBuffer&) = delete;

  void* data() {
    return data_;
  }

  const void* data() const {
    return data_;
  }

  size_t size() const {
    return bytes_;
  }

 private:
  static cuda::pinned_memory_pool_ref& resource() {
    return cuda::pinned_default_memory_pool();
  }

  void reset() noexcept {
    if (data_ == nullptr) {
      return;
    }
    resource().deallocate_sync(data_, bytes_);
    pool_->reportExternalFree(bytes_);
    pool_ = nullptr;
    data_ = nullptr;
    bytes_ = 0;
  }

  memory::MemoryPool* pool_{nullptr};
  void* data_{nullptr};
  size_t bytes_{0};
};

struct SpilledColumnStorage;

struct DeviceColumnStorage {
  explicit DeviceColumnStorage(std::unique_ptr<cudf::column> column)
      : type(column->type()),
        size(column->size()),
        nullCount(column->null_count()) {
    auto contents = column->release();
    data = std::move(*contents.data);
    nullMask = std::move(*contents.null_mask);
    children.reserve(contents.children.size());
    for (auto& child : contents.children) {
      children.emplace_back(std::move(child));
    }
  }

  DeviceColumnStorage(
      const SpilledColumnStorage& host,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type nullCount;
  rmm::device_buffer data;
  rmm::device_buffer nullMask;
  std::vector<DeviceColumnStorage> children;

  uint64_t bytes() const {
    uint64_t result = data.size() + nullMask.size();
    for (const auto& child : children) {
      result += child.bytes();
    }
    return result;
  }

  std::unique_ptr<cudf::column> releaseColumn() && {
    std::vector<std::unique_ptr<cudf::column>> releasedChildren;
    releasedChildren.reserve(children.size());
    for (auto& child : children) {
      releasedChildren.push_back(std::move(child).releaseColumn());
    }
    return std::make_unique<cudf::column>(
        type,
        size,
        std::move(data),
        std::move(nullMask),
        nullCount,
        std::move(releasedChildren));
  }
};

struct SpilledColumnStorage {
  SpilledColumnStorage(
      const DeviceColumnStorage& device,
      memory::MemoryPool* hostPool)
      : type(device.type),
        size(device.size),
        nullCount(device.nullCount),
        data(device.data.size(), hostPool),
        nullMask(device.nullMask.size(), hostPool) {
    children.reserve(device.children.size());
    for (const auto& child : device.children) {
      children.emplace_back(child, hostPool);
    }
  }

  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type nullCount;
  AccountedPinnedBuffer data;
  AccountedPinnedBuffer nullMask;
  std::vector<SpilledColumnStorage> children;

  uint64_t bytes() const {
    uint64_t result = data.size() + nullMask.size();
    for (const auto& child : children) {
      result += child.bytes();
    }
    return result;
  }

  void copyFromDevice(
      const DeviceColumnStorage& device,
      rmm::cuda_stream_view stream) {
    VELOX_CHECK_EQ(data.size(), device.data.size());
    VELOX_CHECK_EQ(nullMask.size(), device.nullMask.size());
    VELOX_CHECK_EQ(children.size(), device.children.size());
    if (data.size() != 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          data.data(),
          device.data.data(),
          data.size(),
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    if (nullMask.size() != 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          nullMask.data(),
          device.nullMask.data(),
          nullMask.size(),
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    for (size_t i = 0; i < children.size(); ++i) {
      children[i].copyFromDevice(device.children[i], stream);
    }
  }
};

DeviceColumnStorage::DeviceColumnStorage(
    const SpilledColumnStorage& host,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
    : type(host.type),
      size(host.size),
      nullCount(host.nullCount),
      data(host.data.size(), stream, mr),
      nullMask(host.nullMask.size(), stream, mr) {
  children.reserve(host.children.size());
  for (const auto& child : host.children) {
    children.emplace_back(child, stream, mr);
  }
}

void copyToDevice(
    const SpilledColumnStorage& host,
    DeviceColumnStorage& device,
    rmm::cuda_stream_view stream) {
  if (host.data.size() != 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        device.data.data(),
        host.data.data(),
        host.data.size(),
        cudaMemcpyHostToDevice,
        stream.value()));
  }
  if (host.nullMask.size() != 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        device.nullMask.data(),
        host.nullMask.data(),
        host.nullMask.size(),
        cudaMemcpyHostToDevice,
        stream.value()));
  }
  VELOX_CHECK_EQ(host.children.size(), device.children.size());
  for (size_t i = 0; i < host.children.size(); ++i) {
    copyToDevice(host.children[i], device.children[i], stream);
  }
}

} // namespace

struct SpilledCudfVector::Impl {
  memory::MemoryPool* vectorPool;
  TypePtr type;
  vector_size_t numRows;
  rmm::cuda_stream_view stream;
  uint64_t deviceBytes;
  std::vector<SpilledColumnStorage> columns;
};

SpilledCudfVector::SpilledCudfVector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
  VELOX_CHECK_NOT_NULL(impl_);
}

SpilledCudfVector::~SpilledCudfVector() = default;

SpilledCudfVector::SpilledCudfVector(SpilledCudfVector&&) noexcept = default;

SpilledCudfVector& SpilledCudfVector::operator=(SpilledCudfVector&&) noexcept =
    default;

SpilledCudfVector SpilledCudfVector::spill(
    CudfVectorPtr& resident,
    memory::MemoryPool* hostPool) {
  VELOX_CHECK_NOT_NULL(resident);
  VELOX_CHECK_NOT_NULL(hostPool);
  VELOX_CHECK_EQ(
      resident.use_count(),
      1,
      "Only an independently owned CudfVector can be spilled");

  const auto vectorPool = resident->pool();
  const auto type = resident->type();
  const auto numRows = resident->size();
  const auto stream = resident->stream();
  VELOX_CHECK(
      resident->rebindStream(stream),
      "Spilling requires an independently owned cudf::table");

  auto table = resident->release();
  resident.reset();
  auto releasedColumns = table->release();
  std::vector<DeviceColumnStorage> deviceColumns;
  deviceColumns.reserve(releasedColumns.size());
  for (auto& column : releasedColumns) {
    deviceColumns.emplace_back(std::move(column));
  }

  auto restoreResident = [&]() {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(deviceColumns.size());
    for (auto& column : deviceColumns) {
      columns.push_back(std::move(column).releaseColumn());
    }
    auto restoredTable = std::make_unique<cudf::table>(std::move(columns));
    resident = std::make_shared<CudfVector>(
        vectorPool, type, numRows, std::move(restoredTable), stream);
  };

  auto impl =
      std::make_unique<Impl>(Impl{vectorPool, type, numRows, stream, 0, {}});
  try {
    impl->columns.reserve(deviceColumns.size());
    for (const auto& column : deviceColumns) {
      impl->columns.emplace_back(column, hostPool);
      impl->deviceBytes += column.bytes();
    }
    for (size_t i = 0; i < impl->columns.size(); ++i) {
      impl->columns[i].copyFromDevice(deviceColumns[i], stream);
    }
    stream.synchronize();

    deviceColumns.clear();
    stream.synchronize();
    return SpilledCudfVector(std::move(impl));
  } catch (...) {
    // D2H may have been queued before a CUDA failure. Keep any pinned
    // destinations alive until the stream is no longer using them.
    try {
      stream.synchronize();
    } catch (...) {
    }
    restoreResident();
    throw;
  }
}

CudfVectorPtr SpilledCudfVector::restore(
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_NOT_NULL(impl_);
  std::vector<DeviceColumnStorage> deviceColumns;
  deviceColumns.reserve(impl_->columns.size());
  for (const auto& column : impl_->columns) {
    deviceColumns.emplace_back(column, impl_->stream, mr);
  }
  for (size_t i = 0; i < impl_->columns.size(); ++i) {
    copyToDevice(impl_->columns[i], deviceColumns[i], impl_->stream);
  }
  impl_->stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(deviceColumns.size());
  for (auto& column : deviceColumns) {
    columns.push_back(std::move(column).releaseColumn());
  }
  auto table = std::make_unique<cudf::table>(std::move(columns));
  return std::make_shared<CudfVector>(
      impl_->vectorPool,
      impl_->type,
      impl_->numRows,
      std::move(table),
      impl_->stream);
}

uint64_t SpilledCudfVector::deviceBytes() const {
  VELOX_CHECK_NOT_NULL(impl_);
  return impl_->deviceBytes;
}

uint64_t SpilledCudfVector::hostBytes() const {
  VELOX_CHECK_NOT_NULL(impl_);
  uint64_t bytes = 0;
  for (const auto& column : impl_->columns) {
    bytes += column.bytes();
  }
  return bytes;
}

bool InputChunk::ownsFullTable() const {
  if (storage != InputChunkStorage::kOwned || owner == nullptr ||
      tableOwner != nullptr) {
    return false;
  }

  const auto ownerView = owner->getTableView();
  if (view.num_rows() != ownerView.num_rows() ||
      view.num_columns() != ownerView.num_columns()) {
    return false;
  }
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    if (!referencesSameColumn(view.column(i), ownerView.column(i))) {
      return false;
    }
  }
  return true;
}

InputChunk InputChunk::materialize(rmm::device_async_resource_ref mr) && {
  if (empty()) {
    return InputChunk{};
  }

  if (storage == InputChunkStorage::kOwned) {
    VELOX_CHECK(
        ownsFullTable(),
        "An owned InputChunk must reference its owner's complete table");
    if (owner.use_count() == 1) {
      return std::move(*this);
    }
  }

  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(type);
  VELOX_CHECK(
      owner != nullptr || tableOwner != nullptr,
      "A borrowed InputChunk must retain its source storage");

  auto materializedTable = std::make_unique<cudf::table>(view, stream, mr);
  auto materialized = std::make_shared<CudfVector>(
      pool,
      type,
      materializedTable->num_rows(),
      std::move(materializedTable),
      stream);
  auto result = InputChunk{
      materialized->pool(),
      type,
      materialized->getTableView(),
      materialized->stream(),
      std::move(materialized),
      nullptr,
      InputChunkStorage::kOwned};
  owner.reset();
  tableOwner.reset();
  return result;
}

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t maxRowsPerLeaf,
    uint32_t initialHashSeed)
    : ops_(std::move(ops)),
      maxRowsPerLeaf_(maxRowsPerLeaf),
      root_(std::make_unique<Node>()),
      nextHashSeed_(initialHashSeed) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(maxRowsPerLeaf_, 0);
}

PartitionedBufferedState::ActiveLeafGuard::ActiveLeafGuard(
    PartitionedBufferedState& owner,
    Node& node)
    : owner_(owner), previous_(owner.activeLeaf_) {
  owner_.activeLeaf_ = &node;
}

PartitionedBufferedState::ActiveLeafGuard::~ActiveLeafGuard() {
  owner_.activeLeaf_ = previous_;
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto compacted = ops_->prepareInput(std::move(rawInput));
  if (compacted.empty()) {
    return;
  }

  insert(*root_, std::move(compacted));
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput() {
  return drainNextOutput(*root_);
}

bool PartitionedBufferedState::empty() const {
  return nodeEmpty(*root_);
}

uint64_t PartitionedBufferedState::reclaimableBytes() const {
  std::vector<std::pair<Node*, uint64_t>> leaves;
  collectReclaimableLeaves(*root_, leaves);
  uint64_t bytes = 0;
  for (const auto& [node, leafBytes] : leaves) {
    if (std::numeric_limits<uint64_t>::max() - bytes < leafBytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    bytes += leafBytes;
  }
  return bytes;
}

uint64_t PartitionedBufferedState::reclaim(uint64_t targetBytes) {
  std::vector<std::pair<Node*, uint64_t>> leaves;
  collectReclaimableLeaves(*root_, leaves);
  std::sort(
      leaves.begin(), leaves.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
      });

  uint64_t reclaimedBytes = 0;
  for (const auto& [node, leafBytes] : leaves) {
    if (targetBytes != 0 && reclaimedBytes >= targetBytes) {
      break;
    }
    VELOX_CHECK_NOT_NULL(node->leafState);
    ops_->spillLeaf(*node->leafState);
    VELOX_CHECK_EQ(
        ops_->leafReclaimableBytes(*node->leafState),
        0,
        "A spilled PBS leaf must not retain reclaimable device bytes");
    reclaimedBytes += leafBytes;
  }
  return reclaimedBytes;
}

void PartitionedBufferedState::collectReclaimableLeaves(
    Node& node,
    std::vector<std::pair<Node*, uint64_t>>& leaves) const {
  if (!node.isLeaf()) {
    for (auto& child : node.children) {
      collectReclaimableLeaves(*child, leaves);
    }
    return;
  }

  if (node.leafState && &node != activeLeaf_) {
    const auto bytes = ops_->leafReclaimableBytes(*node.leafState);
    if (bytes != 0) {
      leaves.emplace_back(&node, bytes);
    }
  }
}

void PartitionedBufferedState::insert(Node& node, InputChunk bufferedInput) {
  if (bufferedInput.empty()) {
    return;
  }

  // If node is not a leaf, it is an internal node and any input has to be
  // partitioned and inserted into child nodes.
  if (!node.isLeaf()) {
    auto partitions = partitionInput(bufferedInput, *node.partitionSpec);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  // If node is a leaf but empty, initialize it with the input.
  if (!node.leafState) {
    node.leafState = ops_->createLeaf(std::move(bufferedInput));
    if (node.leafState) {
      node.leafRows = ops_->leafRowCount(*node.leafState);
      ensureLeafWithinLimit(node);
    }
    return;
  }

  ActiveLeafGuard activeLeaf(*this, node);
  restoreLeaf(node);
  // If node is a leaf and not empty, check if the input can be added to the
  // leaf. If not, split the leaf and insert the input into the new subtree.
  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*node.leafState, bufferedInput);
  if (projectedRows > maxRowsPerLeaf_) {
    splitLeafAndAddInput(node, std::move(bufferedInput));
    return;
  }

  ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
  node.leafRows = ops_->leafRowCount(*node.leafState);
  ensureLeafWithinLimit(node);
}

void PartitionedBufferedState::splitLeaf(Node& node) {
  splitLeafAndAddInput(node, InputChunk{});
}

void PartitionedBufferedState::splitLeafAndAddInput(
    Node& node,
    InputChunk bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(node.leafState || !bufferedInput.empty());
  ActiveLeafGuard activeLeaf(*this, node);
  restoreLeaf(node);

  const auto totalRows = node.leafRows + bufferedInput.size();
  auto splitAttempt = [&]() {
    for (uint32_t attempt = 1;; ++attempt) {
      auto spec = makePartitionSpec(totalRows);
      auto storedPartitions = node.leafState
          ? ops_->partitionLeaf(*node.leafState, spec)
          : std::vector<InputChunk>(spec.numPartitions);
      auto incomingPartitions = partitionInput(bufferedInput, spec);
      const auto nonEmptyChildren =
          countNonEmptyChildren(storedPartitions, incomingPartitions);

      if (nonEmptyChildren > 1) {
        // Found a partition spec that can split the leaf into more than one
        // child. If nonEmptyChildren = 1, then the partition spec could not
        // split and we'd have no meaningful progress.
        return SplitLeafAttempt{
            std::move(spec),
            std::move(storedPartitions),
            std::move(incomingPartitions)};
      }

      VELOX_CHECK_LT(
          attempt,
          kMaxSplitSeedAttempts,
          "Partitioning buffered state made no progress after {} hash seed "
          "attempts: {} rows exceeded the per-leaf limit of {} rows using {} "
          "hash partitions. This can happen when every row has the same "
          "partition key.",
          kMaxSplitSeedAttempts,
          totalRows,
          maxRowsPerLeaf_,
          spec.numPartitions);
    }
  }();

  node.leafRows = 0;
  node.leafState.reset();
  node.partitionSpec = splitAttempt.spec;
  node.children.clear();
  node.children.reserve(splitAttempt.spec.numPartitions);
  for (int32_t i = 0; i < splitAttempt.spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    auto& child = *node.children[i];
    auto& stored = splitAttempt.storedPartitions[i];
    auto& incoming = splitAttempt.incomingPartitions[i];
    if (!stored.empty() && !incoming.empty()) {
      child.leafState =
          ops_->createLeafFromInputs(std::move(stored), std::move(incoming));
    } else if (!stored.empty()) {
      child.leafState = ops_->createLeaf(std::move(stored));
    } else if (!incoming.empty()) {
      child.leafState = ops_->createLeaf(std::move(incoming));
    }

    if (child.leafState) {
      child.leafRows = ops_->leafRowCount(*child.leafState);
      ensureLeafWithinLimit(child);
    }
  }
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput(Node& node) {
  if (!node.isLeaf()) {
    for (auto& child : node.children) {
      if (child) {
        auto output = drainNextOutput(*child);
        if (output) {
          return output;
        }
      }
    }
    return nullptr;
  }

  if (!node.leafState) {
    return nullptr;
  }

  ActiveLeafGuard activeLeaf(*this, node);
  restoreLeaf(node);
  node.leafRows = 0;
  auto leaf = std::move(node.leafState);
  return ops_->finalizeLeaf(std::move(leaf));
}

PartitionSpec PartitionedBufferedState::makePartitionSpec(size_t totalRows) {
  VELOX_CHECK_GT(totalRows, maxRowsPerLeaf_);

  const auto requiredPartitions =
      (totalRows + maxRowsPerLeaf_ - 1) / maxRowsPerLeaf_;
  const auto numPartitions =
      std::max<size_t>(2, std::min(requiredPartitions, totalRows));
  VELOX_CHECK_LE(numPartitions, std::numeric_limits<int32_t>::max());

  return PartitionSpec{
      static_cast<int32_t>(numPartitions),
      ops_->keyIndices(),
      cudf::hash_id::HASH_MURMUR3,
      nextHashSeed_++};
}

void PartitionedBufferedState::ensureLeafWithinLimit(Node& node) {
  if (node.isLeaf() && node.leafState && node.leafRows > maxRowsPerLeaf_) {
    splitLeaf(node);
  }
}

std::vector<InputChunk> PartitionedBufferedState::partitionInput(
    const InputChunk& input,
    const PartitionSpec& spec) {
  return input.empty() ? std::vector<InputChunk>(spec.numPartitions)
                       : ops_->partitionInput(input, spec);
}

void PartitionedBufferedState::restoreLeaf(Node& node) {
  if (node.leafState) {
    ops_->restoreLeaf(*node.leafState);
  }
}

FlushableBufferedState::FlushableBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t flushRowLimit,
    uint64_t flushByteLimit)
    : ops_(std::move(ops)),
      flushRowLimit_(flushRowLimit),
      flushByteLimit_(flushByteLimit) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(flushRowLimit_, 0);
}

void FlushableBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto chunk = ops_->prepareInput(std::move(rawInput));
  if (chunk.empty()) {
    return;
  }

  if (!currentLeaf_) {
    currentLeaf_ = ops_->createLeaf(std::move(chunk));
    if (currentLeaf_) {
      currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
      if (currentLeafRows_ > flushRowLimit_) {
        finalizeActiveLeaf();
      }
    }
    return;
  }

  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*currentLeaf_, chunk);
  if (projectedRows > flushRowLimit_) {
    finalizeActiveLeaf();
    currentLeaf_ = ops_->createLeaf(std::move(chunk));
    if (currentLeaf_) {
      currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
      if (currentLeafRows_ > flushRowLimit_) {
        finalizeActiveLeaf();
      }
    }
    return;
  }

  ops_->addInputToLeaf(*currentLeaf_, std::move(chunk));
  currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
  if (currentLeafRows_ > flushRowLimit_) {
    finalizeActiveLeaf();
  }
}

bool FlushableBufferedState::shouldFlushActiveLeaf() const {
  return currentLeaf_ && ops_->leafFlatSize(*currentLeaf_) > flushByteLimit_;
}

CudfVectorPtr FlushableBufferedState::getOutput(bool noMoreInput) {
  if (auto output = popPendingOutput()) {
    return output;
  }

  if (shouldFlushActiveLeaf()) {
    finalizeActiveLeaf();
    return popPendingOutput();
  }

  if (noMoreInput && currentLeaf_) {
    finalizeActiveLeaf();
    return popPendingOutput();
  }

  return nullptr;
}

bool FlushableBufferedState::empty() const {
  return !currentLeaf_ && pendingOutputs_.empty();
}

CudfVectorPtr FlushableBufferedState::popPendingOutput() {
  if (pendingOutputs_.empty()) {
    return nullptr;
  }

  auto output = std::move(pendingOutputs_.front());
  pendingOutputs_.pop_front();
  return output;
}

void FlushableBufferedState::finalizeActiveLeaf() {
  if (!currentLeaf_) {
    return;
  }

  currentLeafRows_ = 0;
  auto output = ops_->finalizeLeaf(std::move(currentLeaf_));
  if (output) {
    pendingOutputs_.push_back(std::move(output));
  }
}

} // namespace facebook::velox::cudf_velox

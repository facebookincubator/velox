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

#include "velox/common/memory/MemoryPool.h"

#include <set>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::memory {
namespace {
#define VELOX_MEM_CAP_EXCEEDED(errorMessage)                        \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemCapExceeded.c_str(),       \
      /* isRetriable */ true,                                       \
      "{}",                                                         \
      errorMessage);

static constexpr size_t kCapMessageIndentSize = 4;

std::vector<MemoryPoolImpl::MemoryUsage> fromMemoryUsageHeap(
    MemoryPoolImpl::MemoryUsageHeap& heap) {
  std::vector<MemoryPoolImpl::MemoryUsage> usages;
  usages.reserve(heap.size());
  while (!heap.empty()) {
    usages.push_back(heap.top());
    heap.pop();
  }
  std::reverse(usages.begin(), usages.end());
  return usages;
}
} // namespace

std::string MemoryPool::Stats::toString() const {
  return fmt::format(
      "peakBytes:{} cumulativeBytes:{} numAllocs:{} numFrees:{} numReserves:{} numReleases:{} numShrinks:{} numReclaims:{} numCollisions:{} numChildren:{}",
      peakBytes,
      cumulativeBytes,
      numAllocs,
      numFrees,
      numReserves,
      numReleases,
      numShrinks,
      numReclaims,
      numCollisions,
      numChildren);
}

bool MemoryPool::Stats::operator==(const MemoryPool::Stats& other) const {
  return std::tie(
             peakBytes,
             cumulativeBytes,
             numAllocs,
             numFrees,
             numReserves,
             numReleases,
             numCollisions,
             numChildren) ==
      std::tie(
             other.peakBytes,
             other.cumulativeBytes,
             other.numAllocs,
             other.numFrees,
             other.numReserves,
             other.numReleases,
             other.numCollisions,
             other.numChildren);
}

std::ostream& operator<<(std::ostream& os, const MemoryPool::Stats& stats) {
  return os << stats.toString();
}

MemoryPool::MemoryPool(
    const std::string& name,
    Kind kind,
    std::shared_ptr<MemoryPool> parent,
    const Options& options)
    : name_(name),
      alignment_(options.alignment),
      kind_(kind),
      parent_(std::move(parent)),
      reclaimer_(std::move(options.reclaimer)) {
  VELOX_CHECK(parent_ != nullptr || kind_ != Kind::kLeaf);
  MemoryAllocator::alignmentCheck(0, alignment_);
}

MemoryPool::~MemoryPool() {
  VELOX_CHECK(children_.empty());
  if (parent_ != nullptr) {
    parent_->dropChild(this);
  }
}

std::string MemoryPool::kindString(Kind kind) {
  switch (kind) {
    case Kind::kLeaf:
      return "LEAF";
    case Kind::kAggregate:
      return "AGGREGATE";
    default:
      return fmt::format("UNKNOWN_{}", static_cast<int>(kind));
  }
}

std::ostream& operator<<(std::ostream& out, MemoryPool::Kind kind) {
  return out << MemoryPool::kindString(kind);
}

const std::string& MemoryPool::name() const {
  return name_;
}

MemoryPool::Kind MemoryPool::kind() const {
  return kind_;
}

bool MemoryPool::canReclaim() const {
  if (reclaimer_ == nullptr) {
    return false;
  }
  return reclaimer_->canReclaim(*this);
}

int64_t MemoryPool::reclaimableBytes(bool& reclaimable) const {
  if (reclaimer_ == nullptr) {
    reclaimable = false;
    return 0;
  }
  return reclaimer_->reclaimableBytes(*this, reclaimable);
}

int64_t MemoryPool::reclaim(uint64_t targetBytes) {
  if (reclaimer_ == nullptr) {
    return 0;
  }
  return reclaimer_->reclaim(this, targetBytes);
}

void MemoryPool::enterArbitration() {
  if (reclaimer_ != nullptr) {
    reclaimer_->enterArbitration();
  }
}

void MemoryPool::leaveArbitration() {
  if (reclaimer_ != nullptr) {
    reclaimer_->leaveArbitration();
  }
}

MemoryPool* MemoryPool::parent() const {
  return parent_.get();
}

MemoryPool* MemoryPool::root() const {
  MemoryPool* root = const_cast<MemoryPool*>(this);
  while (root->parent_ != nullptr) {
    root = root->parent_.get();
  }
  return root;
}

size_t MemoryPool::childrenCount() const {
  folly::SharedMutex::ReadHolder guard{childMutex_};
  return children_.size();
}

void MemoryPool::visitChildren(std::function<bool(MemoryPool*)> visitor) const {
  std::vector<std::shared_ptr<MemoryPool>> children;
  {
    folly::SharedMutex::ReadHolder guard{childMutex_};
    children.reserve(children_.size());
    for (const auto& entry : children_) {
      auto child = entry.second.lock();
      if (child != nullptr) {
        children.push_back(std::move(child));
      }
    }
  }

  // NOTE: we should call 'visitor' on child pool object out of
  // 'childrenMutex_' to avoid potential recursive locking issues. Firstly, the
  // user provided 'visitor' might try to acquire this memory pool lock again.
  // Secondly, the shared child pool reference created from the weak pointer
  // might be the last reference if some other threads drop all the external
  // references during this time window. Then drop of this last shared reference
  // after 'visitor' call will trigger child memory pool destruction in that
  // case. The child memory pool destructor will remove its weak pointer
  // reference from the parent pool which needs to acquire this memory pool lock
  // again.
  for (const auto& child : children) {
    if (!visitor(child.get())) {
      break;
    }
  }
}

std::shared_ptr<MemoryPool> MemoryPool::addChild(
    const std::string& name,
    Kind kind,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  VELOX_CHECK_EQ(
      kind_,
      Kind::kAggregate,
      "Add child is only allowed on aggregate memory pool: {}",
      toString());
  folly::SharedMutex::WriteHolder guard{childMutex_};
  VELOX_CHECK_EQ(
      children_.count(name),
      0,
      "Child memory pool {} already exists in memory pool {}",
      name,
      name_);
  auto child = genChild(shared_from_this(), name, kind, std::move(reclaimer));
  children_.emplace(name, child);
  return child;
}

void MemoryPool::dropChild(const MemoryPool* child) {
  folly::SharedMutex::WriteHolder guard{childMutex_};
  const auto ret = children_.erase(child->name());
  VELOX_CHECK_EQ(
      ret,
      1,
      "Child memory pool {} doesn't exist in memory pool {}",
      child->name(),
      name_);
}

size_t MemoryPool::preferredSize(size_t size) {
  if (size < 8) {
    return 8;
  }
  int32_t bits = 63 - bits::countLeadingZeros(size);
  size_t lower = 1ULL << bits;
  // Size is a power of 2.
  if (lower == size) {
    return size;
  }
  // If size is below 1.5 * previous power of two, return 1.5 *
  // the previous power of two, else the next power of 2.
  if (lower + (lower / 2) >= size) {
    return lower + (lower / 2);
  }
  return lower * 2;
}

MemoryPoolImpl::MemoryPoolImpl(
    MemoryManager* manager,
    const std::string& name,
    Kind kind,
    std::shared_ptr<MemoryPool> parent,
    DestructionCallback destructionCb,
    const Options& options)
    : MemoryPool(name, kind, std::move(parent), options),
      memoryManager_{manager},
      allocator_{&memoryManager_->allocator()},
      destructionCb_(std::move(destructionCb)),
      capacity_(parent_ == nullptr ? options.capacity : kMaxMemory) {}

MemoryPoolImpl::~MemoryPoolImpl() {
  VELOX_CHECK(
      (usedReservationBytes_ == 0) && (reservationBytes_ == 0) &&
          (minReservationBytes_ == 0),
      "Bad memory pool state on destruction: {}",
      toString());
  if (destructionCb_ != nullptr) {
    destructionCb_(this);
  }
}

std::shared_ptr<MemoryPool> MemoryPoolImpl::genChild(
    std::shared_ptr<MemoryPool> parent,
    const std::string& name,
    Kind kind,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  return std::make_shared<MemoryPoolImpl>(
      memoryManager_,
      name,
      kind,
      parent,
      nullptr,
      Options{.alignment = alignment_, .reclaimer = std::move(reclaimer)});
}

MemoryPool::Stats MemoryPoolImpl::stats() const {
  std::lock_guard<std::mutex> l(mutex_);
  Stats stats;
  stats.peakBytes = peakBytes_;
  stats.cumulativeBytes = cumulativeBytes_;
  stats.numAllocs = numAllocs_;
  stats.numFrees = numFrees_;
  stats.numReserves = numReserves_;
  stats.numReleases = numReleases_;
  stats.numCollisions = numCollisions_;
  return stats;
}

int64_t MemoryPoolImpl::capacity() const {
  if (parent_ != nullptr) {
    return parent_->capacity();
  }
  std::lock_guard<std::mutex> l(mutex_);
  return capacity_;
}

void MemoryPoolImpl::sanityCheckLocked() const {
  VELOX_CHECK(
      (reservationBytes_ >= usedReservationBytes_) &&
          (reservationBytes_ >= minReservationBytes_),
      "Bad tracker state: {}",
      toStringLocked());
  VELOX_DCHECK_GE(usedReservationBytes_, 0);
  if (usedReservationBytes_ < 0) {
    VELOX_MEM_LOG_EVERY_MS(ERROR, 1000)
        << "used reservation is negative " << toStringLocked();
  }
}

bool MemoryPoolImpl::maybeReserve(uint64_t increment) {
  checkAllocation();
  TestValue::adjust(
      "facebook::velox::memory::MemoryPoolImpl::maybeReserve", this);
  constexpr int32_t kGrowthQuantum = 8 << 20;
  const auto reservationToAdd = bits::roundUp(increment, kGrowthQuantum);
  try {
    reserve(reservationToAdd);
  } catch (const std::exception& e) {
    return false;
  }
  return true;
}

void MemoryPoolImpl::release() {
  checkAllocation();
  release(0);
}

void MemoryPoolImpl::reserve(uint64_t size) {
  checkAllocation();
  reserve(size, false);
}

void MemoryPoolImpl::reserve(uint64_t size, bool reserveOnly) {
  VELOX_CHECK_GT(size, 0);
  ++numReserves_;

  int32_t numAttempts = 0;
  int64_t increment = 0;
  for (;; ++numAttempts) {
    {
      std::lock_guard<std::mutex> l(mutex_);
      increment = reservationSizeLocked(size);
      if (increment == 0) {
        if (reserveOnly) {
          ++numReserves_;
          minReservationBytes_ = reservationBytes_;
        } else {
          usedReservationBytes_ += size;
        }
        sanityCheckLocked();
        break;
      }
    }
    TestValue::adjust("facebook::velox::memory::MemoryPoolImpl::reserve", this);
    incrementReservation(this, increment);
  }

  // NOTE: in case of concurrent reserve and release requests, we might see
  // potential conflicts as the quantized memory release might free up extra
  // reservation bytes so reserve might go extra round to reserve more bytes.
  // This should happen rarely in production as the leaf tracker updates are
  // mostly single thread executed.
  if (numAttempts > 1) {
    numCollisions_ += numAttempts - 1;
  }
}

int64_t MemoryPoolImpl::reservationSizeLocked(int64_t size) {
  const int64_t neededSize = size - (reservationBytes_ - usedReservationBytes_);
  if (neededSize <= 0) {
    return 0;
  }
  return roundedDelta(reservationBytes_, neededSize);
}

bool MemoryPoolImpl::incrementReservation(
    MemoryPool* requestor,
    uint64_t size) {
  VELOX_CHECK_GT(size, 0);

  // Update parent first. If one of the ancestor's limits are exceeded, it will
  // throw MEM_CAP_EXCEEDED exception. This exception will be caught and
  // re-thrown with an additional message appended to it if a
  // capExceededMessageMaker_ is set.
  if (parent_ != nullptr) {
    if (!fromPool(parent_)->incrementReservation(requestor, size)) {
      return false;
    }
  }

  {
    std::lock_guard<std::mutex> l(mutex_);
    if (parent_ != nullptr || (reservationBytes_ + size <= capacity_)) {
      reservationBytes_ += size;
      cumulativeBytes_ += size;
      maybeUpdatePeakBytesLocked(reservationBytes_);
      return true;
    }
  }

  if (memoryManager_->growPool(requestor, size)) {
    return false;
  }

  VELOX_MEM_CAP_EXCEEDED(capExceedingMessage(size));
}

std::string MemoryPoolImpl::capExceedingMessage(uint64_t incrementBytes) {
  VELOX_CHECK_NULL(parent_);
  std::stringstream out;
  {
    std::lock_guard<std::mutex> l(mutex_);
    out << "\nExceeded memory cap of " << succinctBytes(capacity_)
        << " when requesting " << succinctBytes(incrementBytes) << "\n";
    out << memoryUsageLocked().toString() << "\n";
  }

  MemoryUsageHeap topLeafMemUsages;
  visitChildren([&, indent = kCapMessageIndentSize](MemoryPool* pool) {
    fromPool(pool)->capExceedingMessage(indent, topLeafMemUsages, out);
    return true;
  });

  if (!topLeafMemUsages.empty()) {
    out << "\nTop " << topLeafMemUsages.size() << " leaf memory pool usages:\n";
    std::vector<MemoryUsage> usages = fromMemoryUsageHeap(topLeafMemUsages);
    for (const auto& usage : usages) {
      out << std::string(kCapMessageIndentSize, ' ') << usage.toString()
          << "\n";
    }
  }
  return out.str();
}

void MemoryPoolImpl::capExceedingMessage(
    size_t indent,
    MemoryUsageHeap& topLeafMemUsages,
    std::stringstream& out) {
  const MemoryUsage usage = memoryUsage();
  out << std::string(indent, ' ') << usage.toString() << "\n";

  if (kind_ == Kind::kLeaf) {
    static const size_t kTopNLeafMessages = 10;
    topLeafMemUsages.push(usage);
    if (topLeafMemUsages.size() > kTopNLeafMessages) {
      topLeafMemUsages.pop();
    }
    return;
  }
  visitChildren([&, indent = indent + kCapMessageIndentSize](MemoryPool* pool) {
    fromPool(pool)->capExceedingMessage(indent, topLeafMemUsages, out);
    return true;
  });
}

MemoryPoolImpl::MemoryUsage MemoryPoolImpl::memoryUsage() const {
  std::lock_guard<std::mutex> l(mutex_);
  return memoryUsageLocked();
}

MemoryPoolImpl::MemoryUsage MemoryPoolImpl::memoryUsageLocked() const {
  MemoryUsage usage;
  usage.name = name_;
  usage.currentUsage = currentBytesLocked();
  usage.peakUsage = peakBytes_;
  return usage;
}

void MemoryPoolImpl::release(uint64_t size) {
  ++numReleases_;

  int64_t freeable = 0;
  {
    std::lock_guard<std::mutex> l(mutex_);
    int64_t newQuantized;
    if (size == 0) {
      if (minReservationBytes_ == 0) {
        return;
      }
      newQuantized = quantizedSize(usedReservationBytes_);
      minReservationBytes_ = 0;
    } else {
      usedReservationBytes_ -= size;
      const int64_t newCap =
          std::max(minReservationBytes_, usedReservationBytes_);
      newQuantized = quantizedSize(newCap);
    }
    freeable = reservationBytes_ - newQuantized;
    if (freeable > 0) {
      reservationBytes_ = newQuantized;
    }
    sanityCheckLocked();
  }
  if (freeable > 0) {
    fromPool(parent_)->decrementReservation(freeable);
  }
}

void MemoryPoolImpl::decrementReservation(uint64_t size) noexcept {
  VELOX_CHECK_GT(size, 0);

  if (parent_ != nullptr) {
    fromPool(parent_)->decrementReservation(size);
  }
  std::lock_guard<std::mutex> l(mutex_);
  reservationBytes_ -= size;
  VELOX_CHECK_GE(reservationBytes_, 0, "release size {}", size);
}

void MemoryPoolImpl::maybeUpdatePeakBytesLocked(int64_t newPeak) {
  peakBytes_ = std::max(peakBytes_, newPeak);
}

/* static */
int64_t MemoryPoolImpl::sizeAlign(int64_t size) {
  const auto remainder = size % alignment_;
  return (remainder == 0) ? size : (size + alignment_ - remainder);
}

void* MemoryPoolImpl::allocate(int64_t size) {
  checkAllocation();
  const auto alignedSize = sizeAlign(size);
  reserve(alignedSize);
  void* buffer = allocator_->allocateBytes(alignedSize, alignment_);
  if (FOLLY_UNLIKELY(buffer == nullptr)) {
    release(alignedSize);
    VELOX_MEM_ALLOC_ERROR(fmt::format(
        "{} failed with {} bytes from {}", __FUNCTION__, size, toString()));
  }
  return buffer;
}

void* MemoryPoolImpl::allocateZeroFilled(int64_t numEntries, int64_t sizeEach) {
  checkAllocation();
  const auto alignedSize = sizeAlign(sizeEach * numEntries);
  reserve(alignedSize);
  void* buffer = allocator_->allocateZeroFilled(alignedSize);
  if (FOLLY_UNLIKELY(buffer == nullptr)) {
    release(alignedSize);
    VELOX_MEM_ALLOC_ERROR(fmt::format(
        "{} failed with {} entries and {} bytes each from {}",
        __FUNCTION__,
        numEntries,
        sizeEach,
        toString()));
  }
  return buffer;
}

void* MemoryPoolImpl::reallocate(
    void* FOLLY_NULLABLE p,
    int64_t size,
    int64_t newSize) {
  checkAllocation();
  auto alignedSize = sizeAlign(size);
  auto alignedNewSize = sizeAlign(newSize);
  const int64_t difference = alignedNewSize - alignedSize;
  reserve(difference);
  void* newP =
      allocator_->reallocateBytes(p, alignedSize, alignedNewSize, alignment_);
  if (FOLLY_UNLIKELY(newP == nullptr)) {
    free(p, alignedSize);
    release(alignedNewSize);
    VELOX_MEM_ALLOC_ERROR(fmt::format(
        "{} failed with {} new bytes and {} old bytes from {}",
        __FUNCTION__,
        newSize,
        size,
        toString()));
  }
  return newP;
}

void MemoryPoolImpl::free(void* p, int64_t size) {
  checkAllocation();
  const auto alignedSize = sizeAlign(size);
  allocator_->freeBytes(p, alignedSize);
  release(alignedSize);
}

void MemoryPoolImpl::allocateNonContiguous(
    MachinePageCount numPages,
    Allocation& out,
    MachinePageCount minSizeClass) {
  checkAllocation();
  VELOX_CHECK_GT(numPages, 0);

  if (!allocator_->allocateNonContiguous(
          numPages,
          out,
          [this](int64_t allocBytes, bool preAllocate) {
            if (preAllocate) {
              reserve(allocBytes);
            } else {
              release(allocBytes);
            }
          },
          minSizeClass)) {
    VELOX_CHECK(out.empty());
    VELOX_MEM_ALLOC_ERROR(fmt::format(
        "{} failed with {} pages from {}", __FUNCTION__, numPages, toString()));
  }
  VELOX_CHECK(!out.empty());
  VELOX_CHECK_NULL(out.pool());
  out.setPool(this);
}

void MemoryPoolImpl::freeNonContiguous(Allocation& allocation) {
  checkAllocation();
  const int64_t freedBytes = allocator_->freeNonContiguous(allocation);
  VELOX_CHECK(allocation.empty());
  release(freedBytes);
}

MachinePageCount MemoryPoolImpl::largestSizeClass() const {
  return allocator_->largestSizeClass();
}

const std::vector<MachinePageCount>& MemoryPoolImpl::sizeClasses() const {
  return allocator_->sizeClasses();
}

void MemoryPoolImpl::allocateContiguous(
    MachinePageCount numPages,
    ContiguousAllocation& out) {
  checkAllocation();
  VELOX_CHECK_GT(numPages, 0);

  if (!allocator_->allocateContiguous(
          numPages, nullptr, out, [this](int64_t allocBytes, bool preAlloc) {
            if (preAlloc) {
              reserve(allocBytes);
            } else {
              release(allocBytes);
            }
          })) {
    VELOX_CHECK(out.empty());
    VELOX_MEM_ALLOC_ERROR(fmt::format(
        "{} failed with {} pages from {}", __FUNCTION__, numPages, toString()));
  }
  VELOX_CHECK(!out.empty());
  VELOX_CHECK_NULL(out.pool());
  out.setPool(this);
}

void MemoryPoolImpl::freeContiguous(ContiguousAllocation& allocation) {
  checkAllocation();
  const int64_t bytesToFree = allocation.size();
  allocator_->freeContiguous(allocation);
  VELOX_CHECK(allocation.empty());
  release(bytesToFree);
}

int64_t MemoryPoolImpl::currentBytes() const {
  std::lock_guard<std::mutex> l(mutex_);
  return currentBytesLocked();
}

int64_t MemoryPoolImpl::currentBytesLocked() const {
  return (kind_ == Kind::kLeaf) ? usedReservationBytes_ : reservationBytes_;
}

int64_t MemoryPoolImpl::availableReservation() const {
  std::lock_guard<std::mutex> l(mutex_);
  return availableReservationLocked();
}

int64_t MemoryPoolImpl::availableReservationLocked() const {
  return kind_ != Kind::kLeaf
      ? 0
      : std::max<int64_t>(0, reservationBytes_ - usedReservationBytes_);
}

int64_t MemoryPoolImpl::reservedBytes() const {
  std::lock_guard<std::mutex> l(mutex_);
  return reservationBytes_;
}

std::string MemoryPoolImpl::toStringLocked() const {
  std::stringstream out;
  out << "Memory Pool[" << name_ << " " << kindString(kind_) << " "
      << MemoryAllocator::kindString(allocator_->kind()) << " "
      << childrenCount() << " child pools]<used "
      << succinctBytes(currentBytesLocked()) << " available "
      << succinctBytes(availableReservationLocked());
  if (capacity_ != kMaxMemory) {
    out << " limit " << succinctBytes(capacity_);
  }
  out << " reservation [used " << succinctBytes(usedReservationBytes_)
      << ", reserved " << succinctBytes(reservationBytes_) << ", min "
      << succinctBytes(minReservationBytes_);
  out << "] counters [allocs " << numAllocs_ << ", frees " << numFrees_
      << ", reserves " << numReserves_ << ", releases " << numReleases_
      << ", collisions " << numCollisions_ << "])";
  out << ">]";
  return out.str();
}

uint64_t MemoryPoolImpl::shrinkableBytes() const {
  if (parent_ != nullptr) {
    return parent_->shrinkableBytes();
  }
  uint64_t shrinkableBytes{0};
  std::lock_guard<std::mutex> l(mutex_);
  if (capacity_ == kMaxMemory) {
    return shrinkableBytes;
  }
  shrinkableBytes = std::max<uint64_t>(0, capacity_ - reservationBytes_);
  return shrinkableBytes;
}

uint64_t MemoryPoolImpl::shrink(uint64_t targetBytes) {
  if (parent_ != nullptr) {
    return parent_->shrink(targetBytes);
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (capacity_ == kMaxMemory) {
    return 0;
  }
  uint64_t freeBytes = std::max<uint64_t>(0, capacity_ - reservationBytes_);
  if (targetBytes != 0) {
    freeBytes = std::min(targetBytes, freeBytes);
  }
  capacity_ -= freeBytes;
  return freeBytes;
}

uint64_t MemoryPoolImpl::grow(int64_t bytes) {
  if (parent_ != nullptr) {
    return parent_->grow(bytes);
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (capacity_ == kMaxMemory) {
    return capacity_;
  }
  capacity_ += bytes;
  VELOX_CHECK_GE(capacity_, bytes);
  return capacity_;
}

void MemoryPoolImpl::testingSetCapacity(int64_t bytes) {
  if (parent_ != nullptr) {
    return fromPool(parent_)->testingSetCapacity(bytes);
  }
  std::lock_guard<std::mutex> l(mutex_);
  capacity_ = bytes;
}
} // namespace facebook::velox::memory

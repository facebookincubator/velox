#include "ArrowMemoryPool.h"
#include <arrow/type_fwd.h>
#include <velox/common/base/Exceptions.h>

namespace velox4j {

namespace {
#define ROUND_TO_LINE(n, round) (((n) + (round) - 1) & ~((round) - 1))
} // namespace

bool ListenableMemoryAllocator::allocate(int64_t size, void** out) {
  updateUsage(size);
  bool succeed = delegated_->allocate(size, out);
  if (!succeed) {
    updateUsage(-size);
  }
  return succeed;
}

bool ListenableMemoryAllocator::allocateZeroFilled(
    int64_t nmemb,
    int64_t size,
    void** out) {
  updateUsage(size * nmemb);
  bool succeed = delegated_->allocateZeroFilled(nmemb, size, out);
  if (!succeed) {
    updateUsage(-size * nmemb);
  }
  return succeed;
}

bool ListenableMemoryAllocator::allocateAligned(
    uint64_t alignment,
    int64_t size,
    void** out) {
  updateUsage(size);
  bool succeed = delegated_->allocateAligned(alignment, size, out);
  if (!succeed) {
    updateUsage(-size);
  }
  return succeed;
}

bool ListenableMemoryAllocator::reallocate(
    void* p,
    int64_t size,
    int64_t newSize,
    void** out) {
  int64_t diff = newSize - size;
  if (diff >= 0) {
    updateUsage(diff);
    bool succeed = delegated_->reallocate(p, size, newSize, out);
    if (!succeed) {
      updateUsage(-diff);
    }
    return succeed;
  } else {
    bool succeed = delegated_->reallocate(p, size, newSize, out);
    if (succeed) {
      updateUsage(diff);
    }
    return succeed;
  }
}

bool ListenableMemoryAllocator::reallocateAligned(
    void* p,
    uint64_t alignment,
    int64_t size,
    int64_t newSize,
    void** out) {
  int64_t diff = newSize - size;
  if (diff >= 0) {
    updateUsage(diff);
    bool succeed =
        delegated_->reallocateAligned(p, alignment, size, newSize, out);
    if (!succeed) {
      updateUsage(-diff);
    }
    return succeed;
  } else {
    bool succeed =
        delegated_->reallocateAligned(p, alignment, size, newSize, out);
    if (succeed) {
      updateUsage(diff);
    }
    return succeed;
  }
}

bool ListenableMemoryAllocator::free(void* p, int64_t size) {
  bool succeed = delegated_->free(p, size);
  if (succeed) {
    updateUsage(-size);
  }
  return succeed;
}

int64_t ListenableMemoryAllocator::getBytes() const {
  return usedBytes_;
}

int64_t ListenableMemoryAllocator::peakBytes() const {
  return peakBytes_;
}

void ListenableMemoryAllocator::updateUsage(int64_t size) {
  listener_->allocationChanged(size);
  usedBytes_ += size;
  while (true) {
    int64_t savedPeakBytes = peakBytes_;
    if (usedBytes_ <= savedPeakBytes) {
      break;
    }
    // usedBytes_ > savedPeakBytes, update peak
    if (peakBytes_.compare_exchange_weak(savedPeakBytes, usedBytes_)) {
      break;
    }
  }
}

bool StdMemoryAllocator::allocate(int64_t size, void** out) {
  VELOX_CHECK(size >= 0, "size is less than 0");
  *out = std::malloc(size);
  if (*out == nullptr) {
    return false;
  }
  bytes_ += size;
  return true;
}

bool StdMemoryAllocator::allocateZeroFilled(
    int64_t nmemb,
    int64_t size,
    void** out) {
  VELOX_CHECK(nmemb >= 0, "nmemb is less than 0");
  VELOX_CHECK(size >= 0, "size is less than 0");
  *out = std::calloc(nmemb, size);
  if (*out == nullptr) {
    return false;
  }
  bytes_ += size;
  return true;
}

bool StdMemoryAllocator::allocateAligned(
    uint64_t alignment,
    int64_t size,
    void** out) {
  VELOX_CHECK(size >= 0, "size is less than 0");
  *out = aligned_alloc(alignment, size);
  if (*out == nullptr) {
    return false;
  }
  bytes_ += size;
  return true;
}

bool StdMemoryAllocator::reallocate(
    void* p,
    int64_t size,
    int64_t newSize,
    void** out) {
  *out = std::realloc(p, newSize);
  if (*out == nullptr) {
    return false;
  }
  bytes_ += (newSize - size);
  return true;
}

bool StdMemoryAllocator::reallocateAligned(
    void* p,
    uint64_t alignment,
    int64_t size,
    int64_t newSize,
    void** out) {
  VELOX_CHECK(p != nullptr, "reallocate with nullptr");
  if (newSize <= 0) {
    return false;
  }
  if (newSize <= size) {
    auto aligned = ROUND_TO_LINE(static_cast<uint64_t>(newSize), alignment);
    if (aligned <= size) {
      // shrink-to-fit
      return reallocate(p, size, aligned, out);
    }
  }
  void* reallocatedP = std::aligned_alloc(alignment, newSize);
  if (reallocatedP == nullptr) {
    return false;
  }
  memcpy(reallocatedP, p, std::min(size, newSize));
  std::free(p);
  *out = reallocatedP;
  bytes_ += (newSize - size);
  return true;
}

bool StdMemoryAllocator::free(void* p, int64_t size) {
  VELOX_CHECK(p != nullptr, "free with nullptr");
  std::free(p);
  bytes_ -= size;
  return true;
}

int64_t StdMemoryAllocator::getBytes() const {
  return bytes_;
}

int64_t StdMemoryAllocator::peakBytes() const {
  return 0;
}

arrow::Status
ArrowMemoryPool::Allocate(int64_t size, int64_t alignment, uint8_t** out) {
  if (!allocator_->allocateAligned(
          alignment, size, reinterpret_cast<void**>(out))) {
    return arrow::Status::Invalid(
        "WrappedMemoryPool: Error allocating " + std::to_string(size) +
        " bytes");
  }
  return arrow::Status::OK();
}

arrow::Status ArrowMemoryPool::Reallocate(
    int64_t oldSize,
    int64_t newSize,
    int64_t alignment,
    uint8_t** ptr) {
  if (!allocator_->reallocateAligned(
          *ptr, alignment, oldSize, newSize, reinterpret_cast<void**>(ptr))) {
    return arrow::Status::Invalid(
        "WrappedMemoryPool: Error reallocating " + std::to_string(newSize) +
        " bytes");
  }
  return arrow::Status::OK();
}

void ArrowMemoryPool::Free(uint8_t* buffer, int64_t size, int64_t alignment) {
  allocator_->free(buffer, size);
}

int64_t ArrowMemoryPool::bytes_allocated() const {
  // fixme use self accountant
  return allocator_->getBytes();
}

int64_t ArrowMemoryPool::total_bytes_allocated() const {
  VELOX_NYI("Not implement");
}

int64_t ArrowMemoryPool::num_allocations() const {
  VELOX_NYI("Not implement");
}

std::string ArrowMemoryPool::backend_name() const {
  return "velox4j";
}

std::shared_ptr<MemoryAllocator> defaultMemoryAllocator() {
  static std::shared_ptr<MemoryAllocator> alloc =
      std::make_shared<StdMemoryAllocator>();
  return alloc;
}

std::shared_ptr<arrow::MemoryPool> defaultArrowMemoryPool() {
  static auto staticPool =
      std::make_shared<ArrowMemoryPool>(defaultMemoryAllocator().get());
  return staticPool;
}
} // namespace velox4j

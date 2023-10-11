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

#include <functional>
#include <memory>
/// Contains wrappers for common Cuda objects. Wave does not directly
/// include Cuda headers because of interference with BitUtils.h and
/// SimdUtils.h.
namespace facebook::velox::wave {

struct Device {
  explicit Device(int32_t id) : deviceId(id) {}

  int32_t deviceId;
};

/// Checks that the machine has the right capability and returns a Device
/// struct. If 'preferredId' is given tries to return  a Device on that device
/// id.
Device* getDevice(int32_t preferredId = -1);
/// Binds subsequent Cuda operations of the calling thread to 'device'.
void setDevice(Device* device);

struct StreamImpl;

class Stream {
 public:
  Stream();
  virtual ~Stream();

  /// Waits  until the stream is completed.
  void wait();

  /// Enqueus a prefetch. Prefetches to host if 'device' is nullptr, otherwise
  /// to 'device'.
  void prefetch(Device* device, void* address, size_t size);

  // Enqueues a copy from host to device.
  void
  hostToDeviceAsync(void* deviceAddress, const void* hostAddress, size_t size);

  // Enqueues a copy from device to host.
  void
  deviceToHostAsync(void* hostAddress, const void* deviceAddress, size_t size);

  /// Adds a callback to be invoked after pending processing is done.
  void addCallback(std::function<void()> callback);

  auto stream() const {
    return stream_.get();
  }

  /// Mutable reference to arbitrary non-owned user data. Can be used for
  /// tagging streams when scheduling.
  void*& userData() {
    return userData_;
  }

 protected:
  std::unique_ptr<StreamImpl> stream_;
  void* userData_{nullptr};

  friend class Event;
};

struct EventImpl;

/// Wrapper on Cuda Event.
class Event {
 public:
  Event(bool withTime = false);

  ~Event();

  ///  Recirds event on 'stream'. This must be called before other member
  ///  functions.
  void record(Stream&);

  /// Returns true if the work captured by 'this' is complete. Throws for Cuda
  /// error.
  bool query() const;

  /// Calling host thread waits  for work recorded by 'this' to complete.
  void wait();

  /// 'stream' will wait for the work recorded by 'this' to complete before
  /// executing work enqueued after this call to wait()..
  void wait(Stream& stream);

  /// Returns time in ms betweene 'this' and an earlier 'start'. Both events
  /// must enable timing.
  float elapsedTime(const Event& start) const;

 private:
  std::unique_ptr<EventImpl> event_;
  const bool hasTiming_;
  bool recorded_{false};
};

// Abstract class wrapping device or universal address memory allocation.
class GpuAllocator {
 public:
  virtual ~GpuAllocator() = default;

  // Returns a pointer to at least 'bytes' of universal or device memory,
  // depending on specific allocator. The size can be rounded up. The alignment
  // is to 8 bytes.
  virtual void* allocate(size_t bytes) = 0;

  /// Frees a pointer from allocate(). 'size' must correspond to the size given
  /// to allocate(). A Memory must be freed to the same allocator it came from.
  virtual void free(void* ptr, size_t bytes) = 0;

  class Deleter;

  template <typename T>
  using UniquePtr = std::unique_ptr<T, GpuAllocator::Deleter>;

  /// Convenient method to do allocation with automatic life cycle management.
  template <typename T>
  UniquePtr<T> allocate();

  /// Convenient method to do allocation with automatic life cycle management.
  template <typename T>
  UniquePtr<T[]> allocate(size_t n);
};

// Returns an allocator that produces unified memory.
GpuAllocator* getAllocator(Device* device);

// Returns an allocator that produces device memory on current device.
GpuAllocator* getDeviceAllocator(Device* device);

/// Returns an allocator that produces pinned host memory.
GpuAllocator* getHostAllocator(Device* device);

class GpuAllocator::Deleter {
 public:
  Deleter() = default;

  Deleter(GpuAllocator* allocator, size_t bytes)
      : allocator_(allocator), bytes_(bytes) {}

  void operator()(void* ptr) const {
    if (ptr) {
      allocator_->free(ptr, bytes_);
    }
  }

 private:
  GpuAllocator* allocator_;
  size_t bytes_;
};

template <typename T>
GpuAllocator::UniquePtr<T> GpuAllocator::allocate() {
  static_assert(std::is_trivially_destructible_v<T>);
  auto bytes = sizeof(T);
  T* ptr = static_cast<T*>(allocate(bytes));
  return UniquePtr<T>(ptr, Deleter(this, bytes));
}

template <typename T>
GpuAllocator::UniquePtr<T[]> GpuAllocator::allocate(size_t n) {
  static_assert(std::is_trivially_destructible_v<T>);
  auto bytes = n * sizeof(T);
  T* ptr = static_cast<T*>(allocate(bytes));
  return UniquePtr<T[]>(ptr, Deleter(this, bytes));
}

} // namespace facebook::velox::wave

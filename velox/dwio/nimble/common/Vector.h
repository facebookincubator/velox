/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace facebook::nimble {

/// A vector-like container similar to std::vector, but without the edge case
/// for booleans. Unlike std::vector<bool>, data() returns T* for all T,
/// allowing implicit conversion to std::span for all T.
template <typename T>
class Vector {
  using InnerType =
      typename std::conditional<std::is_same_v<T, bool>, uint8_t, T>::type;

 public:
  /// Constructs a vector with the given size, filled with the specified value.
  Vector(velox::memory::MemoryPool* pool, size_t size, T value) : pool_{pool} {
    init(size);
    std::fill(dataRawPtr_, dataRawPtr_ + size_, value);
  }

  /// Constructs a vector with the given size, with uninitialized elements.
  Vector(velox::memory::MemoryPool* pool, size_t size) : pool_{pool} {
    init(size);
  }

  /// Constructs an empty vector.
  explicit Vector(velox::memory::MemoryPool* pool) : pool_{pool} {
    capacity_ = 0;
    size_ = 0;
    data_ = nullptr;
    dataRawPtr_ = nullptr;
#ifndef NDEBUG
    dataRawPtr_ = placeholder_.data();
#endif
  }

  /// Constructs a vector by adopting an existing buffer. The pool is extracted
  /// from the buffer for future allocations. Size is set to 0 (no valid data).
  explicit Vector(velox::BufferPtr buf)
      : pool_{buf->pool()},
        data_{std::move(buf)},
        capacity_{data_->capacity() / sizeof(InnerType)},
        size_{0},
        dataRawPtr_{reinterpret_cast<T*>(data_->asMutable<InnerType>())} {}

  /// Constructs a vector from an iterator range.
  template <typename It>
  Vector(velox::memory::MemoryPool* pool, It first, It last) : pool_{pool} {
    auto size = last - first;
    init(size);
    std::copy(first, last, dataRawPtr_);
  }

  /// Copy constructor.
  Vector(const Vector& other) {
    *this = other;
  }

  /// Copy assignment operator.
  Vector& operator=(const Vector& other) {
    if (this != &other) {
      size_ = other.size();
      capacity_ = other.capacity_;
      pool_ = other.pool_;
      allocateBuffer();
      std::copy(other.dataRawPtr_, other.dataRawPtr_ + size_, dataRawPtr_);
    }
    return *this;
  }

  /// Move constructor.
  Vector(Vector&& other) noexcept {
    *this = std::move(other);
  }

  /// Move assignment operator.
  Vector& operator=(Vector&& other) noexcept {
    if (this != &other) {
      size_ = other.size();
      capacity_ = other.capacity_;
      data_ = std::move(other.data_);
#ifndef NDEBUG
      dataRawPtr_ = placeholder_.data();
#endif
      if (data_ != nullptr) {
        dataRawPtr_ = reinterpret_cast<T*>(data_->asMutable<InnerType>());
      }
      pool_ = other.pool_;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  /// Constructs a vector from an initializer list.
  Vector(velox::memory::MemoryPool* pool, std::initializer_list<T> l)
      : pool_{pool} {
    init(l.size());
    std::copy(l.begin(), l.end(), dataRawPtr_);
  }

  /// Returns the memory pool used by this vector.
  inline velox::memory::MemoryPool* pool() {
    return pool_;
  }

  /// Returns the number of elements in the vector.
  uint64_t size() const {
    return size_;
  }

  /// Returns true if the vector is empty.
  bool empty() const {
    return size_ == 0;
  }

  /// Returns the current capacity of the vector.
  uint64_t capacity() const {
    return capacity_;
  }
  /// Returns a reference to the element at the given index.
  T& operator[](uint64_t i) {
    return dataRawPtr_[i];
  }

  /// Returns a const reference to the element at the given index.
  const T& operator[](uint64_t i) const {
    return dataRawPtr_[i];
  }

  /// Returns a pointer to the first element.
  T* begin() {
    return dataRawPtr_;
  }

  /// Returns a pointer past the last element.
  T* end() {
    return dataRawPtr_ + size_;
  }

  /// Returns a const pointer to the first element.
  const T* begin() const {
    return dataRawPtr_;
  }

  /// Returns a const pointer past the last element.
  const T* end() const {
    return dataRawPtr_ + size_;
  }

  /// Returns a reference to the last element.
  T& back() {
    return dataRawPtr_[size_ - 1];
  }

  /// Returns a const reference to the last element.
  const T& back() const {
    return dataRawPtr_[size_ - 1];
  }

  /// Directly updates the size to the given value.
  /// Useful if you've filled in data directly using the underlying raw
  /// pointers.
  void update_size(uint64_t size) {
    size_ = size;
  }

  /// Fills all elements with the default value T().
  void zero_out() {
    std::fill(dataRawPtr_, dataRawPtr_ + size_, T());
  }

  /// Fills all elements with the given value.
  void fill(T value) {
    std::fill(dataRawPtr_, dataRawPtr_ + size_, value);
  }

  /// Resets the vector to an empty state, releasing allocated memory.
  void clear() {
    capacity_ = 0;
    size_ = 0;
    data_.reset();
    dataRawPtr_ = nullptr;
  }

  /// Inserts elements from [inputStart, inputEnd) at the given output
  /// position.
  void insert(T* output, const T* inputStart, const T* inputEnd) {
    const uint64_t inputSize = inputEnd - inputStart;
    const uint64_t distanceToEnd = end() - output;
    if (inputSize > distanceToEnd) {
      const uint64_t sizeChange = inputSize - distanceToEnd;
      const uint64_t distanceFromBegin = output - begin();
      resize(size_ + sizeChange);
      std::move(inputStart, inputEnd, begin() + distanceFromBegin);
    } else {
      std::move(inputStart, inputEnd, output);
    }
  }

  /// Appends the given number of copies of value to the end of the vector.
  void extend(uint64_t copies, T value) {
    reserve(size_ + copies);
    std::fill(end(), end() + copies, value);
    size_ += copies;
  }

  /// Returns a pointer to the underlying data.
  T* data() noexcept {
    return dataRawPtr_;
  }

  /// Returns a const pointer to the underlying data.
  const T* data() const noexcept {
    return dataRawPtr_;
  }

  /// Appends the given value to the end of the vector.
  void push_back(T value) {
    if (size_ == capacity_) {
      reserve(calculateNewSize(capacity_));
    }
    dataRawPtr_[size_] = value;
    ++size_;
  }

  /// Constructs an element in-place at the end of the vector.
  template <typename... Args>
  void emplace_back(Args&&... args) {
    if (size_ == capacity_) {
      reserve(calculateNewSize(capacity_));
    }
    // use placement new to construct the object.
    new (dataRawPtr_ + size_) InnerType(std::forward<Args>(args)...);
    ++size_;
  }

  /// Ensures the vector can hold at least the given number of elements.
  /// Does NOT shrink if size is less than current size, and does NOT
  /// initialize any new elements.
  void reserve(uint64_t size) {
    if (size > capacity_) {
      auto newData =
          velox::AlignedBuffer::allocateExact<InnerType>(size, pool_);
      // AlignedBuffer can allocate a bit more than requested for the alignment
      // purpose, let's leverage that by using its true capacity.
      capacity_ = newData->capacity() / sizeof(InnerType);
      NIMBLE_DCHECK_GE(
          capacity_, size, "Allocated capacity is smaller than requested");
      if (data_ != nullptr && size_ > 0) {
        std::move(
            dataRawPtr_,
            dataRawPtr_ + size_,
            reinterpret_cast<T*>(newData->template asMutable<InnerType>()));
      }
      data_ = std::move(newData);
      dataRawPtr_ = reinterpret_cast<T*>(data_->asMutable<InnerType>());
    }
  }

  /// Changes the size of the vector.
  /// Does NOT shrink capacity, and does NOT initialize new elements.
  void resize(uint64_t size) {
    reserve(size);
    size_ = size;
  }

  /// Changes the size of the vector, initializing new elements to value.
  /// Does NOT shrink capacity.
  void resize(uint64_t newSize, const T& value) {
    auto initialSize = size_;
    resize(newSize);

    if (size_ > initialSize) {
      std::fill(dataRawPtr_ + initialSize, end(), value);
    }
  }

  /// Releases ownership of the underlying buffer and returns it.
  /// Sets the buffer's size to match the vector's logical size.
  /// The vector is left in an empty state after this call.
  velox::BufferPtr releaseOwnership() {
    velox::BufferPtr tmp = std::move(data_);
    tmp->setSize(size_);
    capacity_ = 0;
    size_ = 0;
    data_ = nullptr;
    dataRawPtr_ = nullptr;
#ifndef NDEBUG
    dataRawPtr_ = placeholder_.data();
#endif
    return tmp;
  }

  /// Releases the underlying buffer without setting its size.
  /// Returns nullptr if the vector has no buffer.
  /// The vector is left in an empty state after this call.
  velox::BufferPtr releaseBuffer() {
    auto tmp = std::move(data_);
    capacity_ = 0;
    size_ = 0;
    dataRawPtr_ = nullptr;
#ifndef NDEBUG
    dataRawPtr_ = placeholder_.data();
#endif
    return tmp;
  }

  const velox::BufferPtr& testingBuffer() const {
    return data_;
  }

 private:
  inline void init(size_t size) {
    capacity_ = size;
    size_ = size;
    allocateBuffer();
  }

  inline size_t calculateNewSize(size_t size) {
    auto newSize = size <<= 1;
    if (newSize == 0) {
      return velox::AlignedBuffer::kSizeofAlignedBuffer;
    }

    return newSize;
  }

  inline void allocateBuffer() {
    data_ = velox::AlignedBuffer::allocateExact<InnerType>(capacity_, pool_);
    dataRawPtr_ = reinterpret_cast<T*>(data_->asMutable<InnerType>());
    uint64_t newCapacity = data_->capacity() / sizeof(InnerType);
    NIMBLE_DCHECK_GE(
        newCapacity, capacity_, "Allocated capacity is smaller than requested");
    capacity_ = newCapacity;
  }

  velox::memory::MemoryPool* pool_;
  velox::BufferPtr data_;
  uint64_t capacity_;
  uint64_t size_;
  T* dataRawPtr_;
#ifndef NDEBUG
  inline static std::array<T, sizeof(size_t)> placeholder_;
#endif
};

/// Owns a temporary Vector and returns its allocation to a BufferPool.
template <typename V>
class ScopedVector {
 public:
  ScopedVector(
      uint64_t size,
      velox::memory::MemoryPool* pool,
      velox::BufferPool* bufferPool)
      : bufferPool_{bufferPool},
        vector_{acquireVectorBuffer(size, pool, bufferPool)} {
    vector_.resize(size);
  }

  ~ScopedVector() {
    if (bufferPool_ != nullptr) {
      auto buffer = vector_.releaseBuffer();
      if (buffer != nullptr) {
        bufferPool_->release(std::move(buffer));
      }
    }
  }

  ScopedVector(const ScopedVector&) = delete;
  ScopedVector& operator=(const ScopedVector&) = delete;

  operator Vector<V>&() {
    return vector_;
  }

  operator const Vector<V>&() const {
    return vector_;
  }

  Vector<V>* operator->() {
    return &vector_;
  }

  const Vector<V>* operator->() const {
    return &vector_;
  }

  Vector<V>& operator*() {
    return vector_;
  }

  const Vector<V>& operator*() const {
    return vector_;
  }

  V& operator[](uint64_t i) {
    return vector_[i];
  }

  const V& operator[](uint64_t i) const {
    return vector_[i];
  }

  V* begin() {
    return vector_.begin();
  }

  V* end() {
    return vector_.end();
  }

  const V* begin() const {
    return vector_.begin();
  }

  const V* end() const {
    return vector_.end();
  }

  V* data() {
    return vector_.data();
  }

  const V* data() const {
    return vector_.data();
  }

  uint64_t size() const {
    return vector_.size();
  }

  bool empty() const {
    return vector_.empty();
  }

  uint64_t capacity() const {
    return vector_.capacity();
  }

  void reserve(uint64_t size) {
    vector_.reserve(size);
  }

  void resize(uint64_t size) {
    vector_.resize(size);
  }

  void push_back(V value) {
    vector_.push_back(value);
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    vector_.emplace_back(std::forward<Args>(args)...);
  }

 private:
  static Vector<V> acquireVectorBuffer(
      uint64_t size,
      velox::memory::MemoryPool* pool,
      velox::BufferPool* bufferPool) {
    using InnerType =
        typename std::conditional<std::is_same_v<V, bool>, uint8_t, V>::type;
    if (bufferPool != nullptr && size > 0) {
      if (auto buffer = bufferPool->get(size * sizeof(InnerType))) {
        return Vector<V>{std::move(buffer)};
      }
    }
    return Vector<V>{pool};
  }

  velox::BufferPool* const bufferPool_;
  Vector<V> vector_;
};

} // namespace facebook::nimble

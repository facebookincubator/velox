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

#include <algorithm>
#include <cstring>
#include <span>
#include <string_view>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"

namespace facebook::nimble {

/// Stream data is a generic interface representing a stream of data, allowing
/// generic access to the content to be used by writers
class StreamData {
 public:
  explicit StreamData(const StreamDescriptorBuilder& descriptor)
      : descriptor_{descriptor} {}

  StreamData(const StreamData&) = delete;
  StreamData(StreamData&&) = delete;
  StreamData& operator=(const StreamData&) = delete;
  StreamData& operator=(StreamData&&) = delete;
  virtual ~StreamData() = default;

  virtual std::string_view data() const = 0;

  virtual std::span<const bool> nonNulls() const = 0;
  virtual bool hasNulls() const = 0;

  /// Returns true only when the nonNulls bitmap contains at least one null
  /// value. This differs from hasNulls(), which only says that a validity
  /// bitmap is present.
  static bool hasNullValues(std::span<const bool> nonNulls) {
    return std::any_of(nonNulls.begin(), nonNulls.end(), [](bool notNull) {
      return !notNull;
    });
  }

  /// Returns true only when this stream has a validity bitmap and that bitmap
  /// contains at least one null value.
  bool hasNullValues() const {
    return hasNulls() && hasNullValues(nonNulls());
  }

  /// Number of null (false) entries in the validity bitmap; 0 when absent.
  /// Chunk views override this to return the count precomputed by the chunker.
  virtual uint64_t numNulls() const {
    if (!hasNulls()) {
      return 0;
    }
    const auto validity = nonNulls();
    return static_cast<uint64_t>(
        std::count(validity.begin(), validity.end(), false));
  }

  virtual bool empty() const = 0;

  virtual uint64_t memoryUsed() const = 0;

  /// Returns the number of rows in the stream.
  virtual uint32_t rowCount() const = 0;

  /// Returns true for null-only streams. These streams carry boolean
  /// non-null bits and may omit storage when all rows are non-null.
  virtual bool isNullStream() const {
    return false;
  }

  virtual void reset() = 0;

  virtual void materialize() {}

  const StreamDescriptorBuilder& descriptor() const {
    return descriptor_;
  }

 private:
  const StreamDescriptorBuilder& descriptor_;
};

class MutableStreamData : public StreamData {
 protected:
  MutableStreamData(
      const StreamDescriptorBuilder& descriptor,
      const InputBufferGrowthPolicy& growthPolicy)
      : StreamData(descriptor), growthPolicy_{growthPolicy} {}

  virtual ~MutableStreamData() = default;

  template <typename T>
  void ensureDataCapacity(Vector<T>& data, uint64_t newSize) {
    if (newSize > data.capacity()) {
      const auto newCapacity =
          growthPolicy_.getExtendedCapacity(newSize, data.capacity());
      data.reserve(newCapacity);
    }
  }

 private:
  const InputBufferGrowthPolicy& growthPolicy_;
};

/// Provides a lightweight, non-owning view into a portion of stream data,
/// containing references to the data content and null indicators for efficient
/// processing of large streams without copying data.
class StreamDataView final : public StreamData {
 public:
  StreamDataView(
      const StreamDescriptorBuilder& descriptor,
      std::string_view data,
      uint32_t rowCount,
      std::optional<std::span<const bool>> nonNulls = std::nullopt,
      uint32_t nullCount = 0)
      : StreamData(descriptor),
        data_{data},
        rowCount_{rowCount},
        nonNulls_{nonNulls},
        nullCount_{nullCount} {}

  StreamDataView(StreamDataView&& other) noexcept
      : StreamData(other.descriptor()),
        data_{other.data_},
        rowCount_{other.rowCount_},
        nonNulls_{other.nonNulls_},
        nullCount_{other.nullCount_} {}

  StreamDataView(const StreamDataView&) = delete;

  StreamDataView& operator=(const StreamDataView&) = delete;

  StreamDataView& operator=(StreamDataView&& other) noexcept = delete;

  ~StreamDataView() override = default;

  uint32_t rowCount() const override {
    return rowCount_;
  }

  std::string_view data() const override {
    return data_;
  }

  std::span<const bool> nonNulls() const override {
    return nonNulls_.value_or(std::span<const bool>{});
  }

  bool hasNulls() const override {
    return nonNulls_.has_value();
  }

  uint64_t numNulls() const override {
    return nullCount_;
  }

  uint64_t memoryUsed() const override {
    NIMBLE_UNREACHABLE("StreamDataView is non-owning");
  }

  bool empty() const override {
    NIMBLE_UNREACHABLE("StreamDataView is non-owning");
  }

  void reset() override {
    NIMBLE_UNREACHABLE("StreamDataView is non-owning");
  }

 private:
  const std::string_view data_;
  const uint32_t rowCount_;
  const std::optional<std::span<const bool>> nonNulls_;
  const uint32_t nullCount_;
};

/// Content only data stream.
/// Used when a stream doesn't contain nulls.
template <typename T>
class ContentStreamData final : public MutableStreamData {
 public:
  ContentStreamData(
      velox::memory::MemoryPool& pool,
      const StreamDescriptorBuilder& descriptor,
      const InputBufferGrowthPolicy& growthPolicy)
      : MutableStreamData(descriptor, growthPolicy), data_{&pool} {}

  inline uint32_t rowCount() const override {
    return data_.size();
  }

  inline std::string_view data() const override {
    return {
        reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(T)};
  }

  inline std::span<const bool> nonNulls() const override {
    return {};
  }

  inline bool hasNulls() const override {
    return false;
  }

  inline bool empty() const override {
    return data_.empty();
  }

  inline uint64_t memoryUsed() const override {
    return (data_.size() * sizeof(T)) + extraMemory_;
  }

  inline Vector<T>& mutableData() {
    return data_;
  }

  void ensureMutableDataCapacity(uint64_t newSize) {
    this->ensureDataCapacity(data_, newSize);
  }

  inline uint64_t& extraMemory() {
    return extraMemory_;
  }

  inline virtual void reset() override {
    data_.clear();
    extraMemory_ = 0;
  }

 private:
  Vector<T> data_;
  uint64_t extraMemory_{0};
};

/// Nulls only data stream.
/// Used in cases where boolean data (representing nulls) is needed.
///
/// NOTE: ContentStreamData<bool> can also be used to represent these data
/// streams, however, for these "null streams", we have special optimizations,
/// where if all data is non-null, we omit the stream. This class
/// specialization helps with reusing enabling this optimization.
class NullsStreamData : public MutableStreamData {
 public:
  NullsStreamData(
      velox::memory::MemoryPool& pool,
      const StreamDescriptorBuilder& descriptor,
      const InputBufferGrowthPolicy& growthPolicy)
      : MutableStreamData(descriptor, growthPolicy), nonNulls_{&pool} {}

  inline uint32_t rowCount() const override {
    return bufferedCount_;
  }

  inline std::string_view data() const override {
    return {};
  }

  inline std::span<const bool> nonNulls() const override {
    return nonNulls_;
  }

  inline bool hasNulls() const override {
    return hasNulls_;
  }

  inline bool empty() const override {
    return nonNulls_.empty() && bufferedCount_ == 0;
  }

  bool isNullStream() const override {
    return true;
  }

  inline uint64_t memoryUsed() const override {
    return nonNulls_.size();
  }

  inline Vector<bool>& mutableNonNulls() {
    return nonNulls_;
  }

  inline void reset() override {
    nonNulls_.clear();
    hasNulls_ = false;
    bufferedCount_ = 0;
  }

  void materialize() override {
    const auto offset = nonNulls_.size();
    NIMBLE_CHECK_LE(offset, bufferedCount_);
    if (offset >= bufferedCount_) {
      return;
    }
    nonNulls_.resize(bufferedCount_);
    std::fill(
        nonNulls_.data() + offset, nonNulls_.data() + bufferedCount_, true);
  }

  void ensureAdditionalNullsCapacity(
      bool mayHaveNulls,
      uint64_t additionalSize);

 protected:
  Vector<bool> nonNulls_;
  bool hasNulls_{false};
  uint32_t bufferedCount_{0};
};

/// Nullable content data stream.
/// Used for scalar content streams where data_ stores row-aligned payload
/// values and NullsStreamData::nonNulls_ stores the corresponding validity
/// bits. Payload values for null rows are not semantically meaningful.
template <typename T>
class NullableContentStreamData final : public NullsStreamData {
 public:
  NullableContentStreamData(
      velox::memory::MemoryPool& memoryPool,
      const StreamDescriptorBuilder& descriptor,
      const InputBufferGrowthPolicy& growthPolicy)
      : NullsStreamData(memoryPool, descriptor, growthPolicy),
        data_{&memoryPool},
        extraMemory_{0} {}

  inline std::string_view data() const override {
    return {
        reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(T)};
  }

  inline bool empty() const override {
    return NullsStreamData::empty() && data_.empty();
  }

  inline uint64_t memoryUsed() const override {
    return (data_.size() * sizeof(T)) + extraMemory_ +
        NullsStreamData::memoryUsed();
  }

  bool isNullStream() const override {
    return false;
  }

  inline Vector<T>& mutableData() {
    return data_;
  }

  void ensureMutableDataCapacity(uint64_t newSize) {
    this->ensureDataCapacity(data_, newSize);
  }

  inline uint64_t& extraMemory() {
    return extraMemory_;
  }

  inline void reset() override {
    NullsStreamData::reset();
    data_.clear();
    extraMemory_ = 0;
  }

 private:
  Vector<T> data_;
  uint64_t extraMemory_;
};

struct StringBuffer {
  Vector<char>& buffer;
  Vector<size_t>& lengths;
};

class NullableContentStringStreamData final : public NullsStreamData {
 public:
  NullableContentStringStreamData(
      velox::memory::MemoryPool& memoryPool,
      const StreamDescriptorBuilder& descriptor,
      const InputBufferGrowthPolicy& lengthsGrowthPolicy,
      const InputBufferGrowthPolicy& bufferGrowthPolicy)
      : NullsStreamData(memoryPool, descriptor, lengthsGrowthPolicy),
        buffer_{&memoryPool},
        lengths_{&memoryPool},
        stringViews_{&memoryPool},
        bufferGrowthPolicy_{bufferGrowthPolicy},
        lengthGrowthPolicy_{lengthsGrowthPolicy} {}

  inline std::string_view data() const override {
    NIMBLE_DCHECK(materialized_, "Data must be materialized before access");
    return std::string_view{
        reinterpret_cast<const char*>(stringViews_.data()),
        stringViews_.size() * sizeof(std::string_view)};
  }

  inline bool empty() const override {
    return NullsStreamData::empty() && lengths_.empty();
  }

  inline uint64_t memoryUsed() const override {
    // NOTE: This is not a bug, we multiply lengths_.size() by
    // sizeof(std::string_view) instead of sizeof(uint64_t) to prevent reader
    // regressions with the current default writer settings.
    return NullsStreamData::memoryUsed() + buffer_.size() +
        lengths_.size() * sizeof(std::string_view) +
        stringViews_.size() * sizeof(std::string_view);
  }

  inline uint64_t bufferSize() const {
    return lengths_.size() * sizeof(std::string_view) + buffer_.size();
  }

  bool isNullStream() const override {
    return false;
  }

  inline StringBuffer mutableData() {
    // When mutableData() is called, it returns direct references to buffer_ and
    // lengths_, allowing external code to modify them. Since dataChunkVector_
    // contains string_view objects pointing into buffer_, any buffer
    // modifications invalidate these views. Therefore, materialize() must be
    // called before the next access to rebuild dataChunkVector_ with valid
    // string_view references to the updated buffer content.
    materialized_ = false;
    return {buffer_, lengths_};
  }

  inline Vector<std::string_view>& mutableStringViews() {
    materialized_ = false;
    return stringViews_;
  }

  void materializeNulls() {
    NullsStreamData::materialize();
  }

  void materialize() override {
    stringViews_.resize(lengths_.size());
    auto runningOffset = 0;
    for (auto i = 0; i < lengths_.size(); ++i) {
      stringViews_[i] =
          std::string_view(buffer_.data() + runningOffset, lengths_[i]);
      runningOffset += lengths_[i];
    }
    materializeNulls();
    materialized_ = true;
  }

  void ensureStringBufferCapacity(
      uint64_t additionalStrings,
      uint64_t additionalBytes) {
    if (additionalStrings == 0) {
      return;
    }

    const auto newLengthsSize = lengths_.size() + additionalStrings;
    if (newLengthsSize > lengths_.capacity()) {
      const auto newCapacity = lengthGrowthPolicy_.getExtendedCapacity(
          newLengthsSize, lengths_.capacity());
      lengths_.reserve(newCapacity);
    }

    if (additionalBytes > 0) {
      const auto newBufferSize = buffer_.size() + additionalBytes;
      if (newBufferSize > buffer_.capacity()) {
        const auto newCapacity = bufferGrowthPolicy_.getExtendedCapacity(
            newBufferSize, buffer_.capacity());
        buffer_.reserve(newCapacity);
      }
    }
  }

  inline void reset() override {
    NullsStreamData::reset();
    buffer_.clear();
    lengths_.clear();
    stringViews_.clear();
  }

 private:
  Vector<char> buffer_;
  Vector<uint64_t> lengths_;
  Vector<std::string_view> stringViews_;
  const InputBufferGrowthPolicy& bufferGrowthPolicy_;
  const InputBufferGrowthPolicy& lengthGrowthPolicy_;

  bool materialized_{false};
};

/// Returns true if the boolean stream data is constant — all bytes are the
/// same value (all-true or all-false), or the data is empty.
inline bool isConstantBoolStream(std::string_view data) {
  if (data.empty()) {
    return true;
  }
  // Check all-true: no zero byte found.
  if (::memchr(data.data(), 0, data.size()) == nullptr) {
    return true;
  }
  // Check all-false: no non-zero byte found.
  return ::memchr(data.data(), 1, data.size()) == nullptr;
}

} // namespace facebook::nimble

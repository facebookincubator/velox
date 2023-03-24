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

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/dynamic.h>
#include <gflags/gflags_declare.h>

#include "velox/common/base/SimdUtil.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/BuilderTypeUtils.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox {

// FlatVector is marked final to allow for inlining on virtual methods called
// on a pointer that has the static type FlatVector<T>; this can be a
// significant performance win when these methods are called in loops.
template <typename T>
class FlatVector final : public SimpleVector<T> {
 public:
  using value_type = T;

  static constexpr bool can_simd =
      (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t> ||
       std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> ||
       std::is_same_v<T, bool> || std::is_same_v<T, size_t>);

  // Minimum size of a string buffer. 32 KB value is chosen to ensure that a
  // single buffer is sufficient for a "typical" vector: 1K rows, medium size
  // strings.
  static constexpr vector_size_t kInitialStringSize =
      (32 * 1024) - sizeof(AlignedBuffer);
  /// Maximum size of a string buffer to re-use (see
  /// BaseVector::prepareForReuse): 1MB.
  static constexpr vector_size_t kMaxStringSizeForReuse =
      (1 << 20) - sizeof(AlignedBuffer);

  FlatVector(
      velox::memory::MemoryPool* pool,
      const TypePtr& type,
      BufferPtr nulls,
      size_t length,
      BufferPtr values,
      std::vector<BufferPtr>&& stringBuffers,
      const SimpleVectorStats<T>& stats = {},
      std::optional<vector_size_t> distinctValueCount = std::nullopt,
      std::optional<vector_size_t> nullCount = std::nullopt,
      std::optional<bool> isSorted = std::nullopt,
      std::optional<ByteCount> representedBytes = std::nullopt,
      std::optional<ByteCount> storageByteCount = std::nullopt)
      : SimpleVector<T>(
            pool,
            type,
            VectorEncoding::Simple::FLAT,
            std::move(nulls),
            length,
            stats,
            distinctValueCount,
            nullCount,
            isSorted,
            representedBytes,
            storageByteCount),
        values_(std::move(values)),
        rawValues_(values_.get() ? const_cast<T*>(values_->as<T>()) : nullptr) {
    setStringBuffers(std::move(stringBuffers));
    VELOX_DCHECK_GE(stringBuffers_.size(), stringBufferSet_.size());
    VELOX_DCHECK_EQ(
        (std::is_same_v<T, UnscaledShortDecimal>), type->isShortDecimal());
    VELOX_DCHECK_EQ(
        (std::is_same_v<T, UnscaledLongDecimal>), type->isLongDecimal());
    VELOX_CHECK(
        values_ || BaseVector::nulls_,
        "FlatVector needs to either have values or nulls");
    if (!values_) {
      return;
    }
    auto byteSize = BaseVector::byteSize<T>(BaseVector::length_);
    VELOX_CHECK_GE(values_->capacity(), byteSize);
    if (values_->size() < byteSize) {
      // If values_ is resized, this guarantees that elements below
      // 'length_' get preserved. If the size is already sufficient,
      // do not set it so that we can have a second reference to an
      // immutable Buffer.
      values_->setSize(byteSize);
    }

    BaseVector::inMemoryBytes_ += values_->capacity();
    for (const auto& stringBuffer : stringBuffers_) {
      BaseVector::inMemoryBytes_ += stringBuffer->capacity();
    }
  }

  virtual ~FlatVector() override = default;

  T valueAtFast(vector_size_t idx) const;

  const T valueAt(vector_size_t idx) const override {
    return valueAtFast(idx);
  }

  std::unique_ptr<SimpleVector<uint64_t>> hashAll() const override;

  /**
   * Loads a SIMD vector of data at the virtual byteOffset given
   * Note this method is implemented on each vector type, but is intentionally
   * not virtual for performance reasons
   *
   * @param byteOffset - the byte offset to load from
   */
  xsimd::batch<T> loadSIMDValueBufferAt(size_t index) const;

  // dictionary vector makes internal usehere for SIMD functions
  template <typename X>
  friend class DictionaryVector;

  // Sequence vector needs to get shared_ptr to value array
  template <typename X>
  friend class SequenceVector;

  /**
   * @return a smart pointer holding the values for
   * this vector. This is used during execution to process over the subset of
   * values when possible.
   */
  const BufferPtr& values() const override {
    return values_;
  }

  BufferPtr mutableValues(vector_size_t size) {
    if (values_ && values_->isMutable() &&
        values_->capacity() >= BaseVector::byteSize<T>(size)) {
      return values_;
    }

    values_ = AlignedBuffer::allocate<T>(size, BaseVector::pool_);
    rawValues_ = values_->asMutable<T>();
    return values_;
  }

  /**
   * @return true if this number of comparison values on this vector should use
   * simd for equality constraint filtering, false to use standard set
   * examination filtering.
   */
  bool useSimdEquality(size_t numCmpVals) const;

  /**
   * @return the raw values of this vector as a continuous array.
   */
  const T* rawValues() const;

  const void* valuesAsVoid() const override {
    return rawValues_;
  }

  template <typename As>
  const As* rawValues() const {
    return reinterpret_cast<const As*>(rawValues_);
  }

  // Bool uses compact representation, use mutableRawValues<uint64_t> and
  // bits::setBit instead.
  T* mutableRawValues() {
    if (!(values_ && values_->unique() && values_->isMutable())) {
      BufferPtr newValues =
          AlignedBuffer::allocate<T>(BaseVector::length_, BaseVector::pool());
      if (values_) {
        // This codepath is not yet enabled for OPAQUE types (asMutable will
        // fail below)
        int32_t numBytes = BaseVector::byteSize<T>(BaseVector::length_);
        memcpy(newValues->asMutable<uint8_t>(), rawValues_, numBytes);
      }
      values_ = newValues;
      rawValues_ = values_->asMutable<T>();
    }
    return rawValues_;
  }

  template <typename As>
  As* mutableRawValues() {
    return reinterpret_cast<As*>(mutableRawValues());
  }

  Range<T> asRange() const;

  void set(vector_size_t idx, T value) {
    VELOX_DCHECK(idx < BaseVector::length_);
    VELOX_DCHECK(values_->isMutable());
    rawValues_[idx] = value;
    if (BaseVector::nulls_) {
      BaseVector::setNull(idx, false);
    }
  }

  void setNoCopy(const vector_size_t /* unused */, const T& /* unused */) {
    VELOX_UNREACHABLE();
  }

  void copy(
      const BaseVector* source,
      const SelectivityVector& rows,
      const vector_size_t* toSourceRow) override {
    if (!rows.hasSelections()) {
      return;
    }
    copyValuesAndNulls(source, rows, toSourceRow);
  }

  void copy(
      const BaseVector* source,
      vector_size_t targetIndex,
      vector_size_t sourceIndex,
      vector_size_t count) override {
    if (count == 0) {
      return;
    }
    copyValuesAndNulls(source, targetIndex, sourceIndex, count);
  }

  void copyRanges(
      const BaseVector* source,
      const folly::Range<const BaseVector::CopyRange*>& ranges) override {
    for (auto& range : ranges) {
      copy(source, range.targetIndex, range.sourceIndex, range.count);
    }
  }

  void resize(vector_size_t newSize, bool setNotNull = true) override;

  VectorPtr slice(vector_size_t offset, vector_size_t length) const override;

  std::optional<int32_t> compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const override {
    if (other->isFlatEncoding()) {
      auto otherFlat = other->asUnchecked<FlatVector<T>>();
      return compareFlat<true>(otherFlat, index, otherIndex, flags);
    }

    return SimpleVector<T>::compare(other, index, otherIndex, flags);
  }

  template <bool compareNulls>
  std::optional<int32_t> compareFlat(
      const FlatVector<T>* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const {
    if constexpr (compareNulls) {
      bool otherNull = other->isNullAt(otherIndex);
      bool isNull = BaseVector::isNullAt(index);
      if (isNull || otherNull) {
        return BaseVector::compareNulls(isNull, otherNull, flags);
      }
    }

    auto thisValue = valueAtFast(index);
    auto otherValue = other->valueAtFast(otherIndex);
    auto result = SimpleVector<T>::comparePrimitiveAsc(thisValue, otherValue);
    return flags.ascending ? result : result * -1;
  }

  void sortIndices(std::vector<vector_size_t>& indices, CompareFlags flags)
      const override {
    auto compareNonNull = [&](vector_size_t left, vector_size_t right) {
      auto leftValue = valueAtFast(left);
      auto rightValue = valueAtFast(right);
      auto result = SimpleVector<T>::comparePrimitiveAsc(leftValue, rightValue);
      return (flags.ascending ? result : result * -1) < 0;
    };

    if (BaseVector::rawNulls_) {
      std::sort(
          indices.begin(),
          indices.end(),
          [&](vector_size_t left, vector_size_t right) {
            bool leftNull = BaseVector::isNullAt(left);
            bool rightNull = BaseVector::isNullAt(right);
            if (leftNull || rightNull) {
              return BaseVector::compareNulls(leftNull, rightNull, flags)
                         .value() < 0;
            }

            return compareNonNull(left, right);
          });
    } else {
      std::sort(indices.begin(), indices.end(), compareNonNull);
    }
  }

  void sortIndices(
      std::vector<vector_size_t>& indices,
      const vector_size_t* mapping,
      CompareFlags flags) const override {
    auto compareNonNull = [&](vector_size_t left, vector_size_t right) {
      auto leftValue = valueAtFast(mapping[left]);
      auto rightValue = valueAtFast(mapping[right]);
      auto result = SimpleVector<T>::comparePrimitiveAsc(leftValue, rightValue);
      return (flags.ascending ? result : result * -1) < 0;
    };

    if (BaseVector::rawNulls_) {
      std::sort(
          indices.begin(),
          indices.end(),
          [&](vector_size_t left, vector_size_t right) {
            bool leftNull = BaseVector::isNullAt(mapping[left]);
            bool rightNull = BaseVector::isNullAt(mapping[right]);
            if (leftNull || rightNull) {
              return BaseVector::compareNulls(leftNull, rightNull, flags)
                         .value() < 0;
            }

            return compareNonNull(left, right);
          });
    } else {
      std::sort(indices.begin(), indices.end(), compareNonNull);
    }
  }

  bool isScalar() const override {
    return this->typeKind() != TypeKind::UNKNOWN;
  }

  uint64_t retainedSize() const override {
    auto size =
        BaseVector::retainedSize() + (values_ ? values_->capacity() : 0);
    for (auto& buffer : stringBuffers_) {
      size += buffer->capacity();
    }
    return size;
  }

  /**
   * Used for vectors of type VARCHAR and VARBINARY to hold data referenced by
   * StringView's. It is safe to share these among multiple vectors. These
   * buffers are append only. It is allowed to append data, but it is prohibited
   * to modify already written data.
   */
  const std::vector<BufferPtr>& stringBuffers() const {
    return stringBuffers_;
  }

  /// Used for vectors of type VARCHAR and VARBINARY to replace the old data
  /// buffers with 'buffers' which are referenced by StringView's.
  void setStringBuffers(std::vector<BufferPtr> buffers) {
    VELOX_DCHECK_GE(stringBuffers_.size(), stringBufferSet_.size());

    stringBuffers_ = std::move(buffers);
    stringBufferSet_.clear();
    stringBufferSet_.reserve(stringBuffers_.size());
    for (const auto& bufferPtr : stringBuffers_) {
      stringBufferSet_.insert(bufferPtr.get());
    }
  }

  /// Used for vectors of type VARCHAR and VARBINARY to release the data buffers
  /// referenced by StringView's.
  void clearStringBuffers() {
    VELOX_DCHECK_GE(stringBuffers_.size(), stringBufferSet_.size());

    stringBuffers_.clear();
    stringBufferSet_.clear();
  }

  /// Used for vectors of type VARCHAR and VARBINARY to hold a reference on
  /// 'buffer'. The function returns false if 'buffer' has already been
  /// referenced by this vector.
  bool addStringBuffer(const BufferPtr& buffer) {
    VELOX_DCHECK_GE(stringBuffers_.size(), stringBufferSet_.size());

    if (FOLLY_UNLIKELY(!stringBufferSet_.insert(buffer.get()).second)) {
      return false;
    }
    stringBuffers_.push_back(buffer);
    return true;
  }

  void acquireSharedStringBuffers(const BaseVector* source);

  Buffer* getBufferWithSpace(vector_size_t /* unused */) {
    return nullptr;
  }

  void ensureWritable(const SelectivityVector& rows) override;

  bool isWritable() const override {
    return this->isNullsWritable() &&
        (!values_ || (values_->unique() && values_->isMutable()));
  }

  /// Calls BaseVector::prapareForReuse() to check and reset nulls buffer if
  /// needed, checks and resets values buffer. Resets all strings buffers except
  /// the first one. Keeps the first string buffer if singly-referenced and
  /// mutable. Resizes the buffer to zero to allow for reuse instead of append.
  void prepareForReuse() override;

 private:
  void copyValuesAndNulls(
      const BaseVector* source,
      const SelectivityVector& rows,
      const vector_size_t* toSourceRow);

  void copyValuesAndNulls(
      const BaseVector* source,
      vector_size_t targetIndex,
      vector_size_t sourceIndex,
      vector_size_t count);

  // Ensures that the values buffer has space for 'newSize' elements and is
  // mutable. Sets elements between the old and new sizes to 'initialValue' if
  // the new size > old size.
  void resizeValues(
      vector_size_t newSize,
      const std::optional<T>& initialValue);

  // Contiguous values.
  // If strings, these are velox::StringViews into memory held by
  // 'stringBuffers_'
  BufferPtr values_;

  // Caches 'values->as<T>()'
  T* rawValues_;

  // If T is velox::StringView, the referenced is held by
  // one of these.
  std::vector<BufferPtr> stringBuffers_;

  // Used by 'acquireSharedStringBuffers()' to fast check if a buffer to share
  // has already been referenced by 'stringBuffers_'.
  //
  // NOTE: we need to ensure 'stringBuffers_' and 'stringBufferSet_' are always
  // consistent.
  folly::F14FastSet<const Buffer*> stringBufferSet_;
};

template <>
bool FlatVector<bool>::valueAtFast(vector_size_t idx) const;

template <>
const bool* FlatVector<bool>::rawValues() const;

template <>
Range<bool> FlatVector<bool>::asRange() const;

template <>
void FlatVector<StringView>::set(vector_size_t idx, StringView value);

template <>
void FlatVector<StringView>::setNoCopy(
    const vector_size_t idx,
    const StringView& value);

template <>
void FlatVector<bool>::set(vector_size_t idx, bool value);

template <>
void FlatVector<StringView>::copy(
    const BaseVector* source,
    const SelectivityVector& rows,
    const vector_size_t* toSourceRow);

template <>
void FlatVector<StringView>::copy(
    const BaseVector* source,
    vector_size_t targetIndex,
    vector_size_t sourceIndex,
    vector_size_t count);

template <>
void FlatVector<StringView>::copyRanges(
    const BaseVector* source,
    const folly::Range<const CopyRange*>& ranges);

template <>
void FlatVector<bool>::copyValuesAndNulls(
    const BaseVector* source,
    const SelectivityVector& rows,
    const vector_size_t* toSourceRow);

template <>
void FlatVector<bool>::copyValuesAndNulls(
    const BaseVector* source,
    vector_size_t targetIndex,
    vector_size_t sourceIndex,
    vector_size_t count);

template <>
Buffer* FlatVector<StringView>::getBufferWithSpace(vector_size_t size);

template <>
void FlatVector<StringView>::prepareForReuse();

template <typename T>
using FlatVectorPtr = std::shared_ptr<FlatVector<T>>;

// Error vector uses an opaque flat vector to store std::exception_ptr.
// Since opaque types are stored as shared_ptr<void>, this ends up being a
// double pointer in the form of std::shared_ptr<std::exception_ptr>. This is
// fine since we only need to actually follow the pointer in failure cases.
using ErrorVector = FlatVector<std::shared_ptr<void>>;
using ErrorVectorPtr = std::shared_ptr<ErrorVector>;

} // namespace facebook::velox

#include "velox/vector/FlatVector-inl.h"

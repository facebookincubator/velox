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

#include <type_traits>

#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>
#include <glog/logging.h>

#include <velox/vector/BaseVector.h>
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox {

using column_index_t = uint32_t;

constexpr column_index_t kConstantChannel =
    std::numeric_limits<column_index_t>::max();

class RowVector : public BaseVector {
 public:
  RowVector(
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const Type> type,
      BufferPtr nulls,
      size_t length,
      std::vector<VectorPtr> children,
      std::optional<vector_size_t> nullCount = std::nullopt)
      : BaseVector(
            pool,
            type,
            VectorEncoding::Simple::ROW,
            nulls,
            length,
            std::nullopt,
            nullCount,
            1),
        childrenSize_(children.size()),
        children_(std::move(children)) {
    // Some columns may not be projected out
    VELOX_CHECK_LE(children_.size(), type->size());
    [[maybe_unused]] const auto* rowType =
        dynamic_cast<const RowType*>(type.get());

    // Check child vector types.
    // This can be an expensive operation, so it's only done at debug time.
    for (auto i = 0; i < children_.size(); i++) {
      const auto& child = children_[i];
      if (child) {
        VELOX_DCHECK(
            child->type()->kindEquals(type->childAt(i)),
            "Got type {} for field `{}` at position {}, but expected {}.",
            child->type()->toString(),
            rowType->nameOf(i),
            i,
            type->childAt(i)->toString());
      }
    }
  }

  static std::shared_ptr<RowVector> createEmpty(
      std::shared_ptr<const Type> type,
      velox::memory::MemoryPool* pool);

  virtual ~RowVector() override {}

  std::optional<int32_t> compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const override;

  uint64_t hashValueAt(vector_size_t index) const override;

  std::unique_ptr<SimpleVector<uint64_t>> hashAll() const override;

  /// Return the number of child vectors.
  /// This will exactly match the number of fields.
  size_t childrenSize() const {
    return childrenSize_;
  }

  /// Get the child vector at a given offset.
  VectorPtr& childAt(column_index_t index) {
    VELOX_USER_CHECK_LT(index, childrenSize_);
    return children_[index];
  }

  const VectorPtr& childAt(column_index_t index) const {
    VELOX_USER_CHECK_LT(index, childrenSize_);
    return children_[index];
  }

  std::vector<VectorPtr>& children() {
    return children_;
  }

  const std::vector<VectorPtr>& children() const {
    return children_;
  }

  void copy(
      const BaseVector* source,
      vector_size_t targetIndex,
      vector_size_t sourceIndex,
      vector_size_t count) override;

  void copy(
      const BaseVector* source,
      const SelectivityVector& rows,
      const vector_size_t* toSourceRow) override;

  void copyRanges(
      const BaseVector* source,
      const folly::Range<const CopyRange*>& ranges) override;

  uint64_t retainedSize() const override {
    auto size = BaseVector::retainedSize();
    for (auto& child : children_) {
      if (child) {
        size += child->retainedSize();
      }
    }
    return size;
  }

  uint64_t estimateFlatSize() const override;

  using BaseVector::toString;

  std::string toString(vector_size_t index) const override;

  void ensureWritable(const SelectivityVector& rows) override;

  bool isWritable() const override;

  /// Calls BaseVector::prepareForReuse() to check and reset nulls buffer if
  /// needed, then calls BaseVector::prepareForReuse(child, 0) for all children.
  void prepareForReuse() override;

  bool mayHaveNullsRecursive() const override {
    if (BaseVector::mayHaveNullsRecursive()) {
      return true;
    }

    for (const auto& child : children_) {
      if (child->mayHaveNullsRecursive()) {
        return true;
      }
    }

    return false;
  }

  VectorPtr slice(vector_size_t offset, vector_size_t length) const override;

 private:
  vector_size_t childSize() const {
    bool allConstant = false;
    for (auto& child : children_) {
      if (child) {
        if (!child->isConstantEncoding()) {
          return child->size();
        }
        allConstant = true;
      }
    }
    if (!allConstant) {
      // All are nullptr.
      return 0;
    }
    // If all children are constants which do not have a meaningful
    // size, the size is one past the last referenced child index.
    return BaseVector::length_;
  }

  void appendToChildren(
      const RowVector* source,
      vector_size_t sourceIndex,
      vector_size_t count,
      vector_size_t childSize);

  const size_t childrenSize_;
  mutable std::vector<VectorPtr> children_;
};

// Common parent class for ARRAY and MAP vectors.  Contains 'offsets' and
// 'sizes' data and provide manipulations on them.
struct ArrayVectorBase : BaseVector {
  const BufferPtr& offsets() const {
    return offsets_;
  }

  const BufferPtr& sizes() const {
    return sizes_;
  }

  const vector_size_t* rawOffsets() const {
    return rawOffsets_;
  }

  const vector_size_t* rawSizes() const {
    return rawSizes_;
  }

  vector_size_t offsetAt(vector_size_t index) const {
    return rawOffsets_[index];
  }

  vector_size_t sizeAt(vector_size_t index) const {
    return rawSizes_[index];
  }

  BufferPtr mutableOffsets(size_t size) {
    return ensureIndices(offsets_, rawOffsets_, size);
  }

  BufferPtr mutableSizes(size_t size) {
    return ensureIndices(sizes_, rawSizes_, size);
  }

  void resize(vector_size_t size, bool setNotNull = true) override {
    if (BaseVector::length_ < size) {
      resizeIndices(size, 0, &offsets_, &rawOffsets_);
      resizeIndices(size, 0, &sizes_, &rawSizes_);
    }
    BaseVector::resize(size, setNotNull);
  }

  void
  setOffsetAndSize(vector_size_t i, vector_size_t offset, vector_size_t size) {
    offsets_->asMutable<vector_size_t>()[i] = offset;
    sizes_->asMutable<vector_size_t>()[i] = size;
  }

  /// Verify that an ArrayVector/MapVector does not contain overlapping [offset,
  /// size] ranges. Throws in case overlaps are found.
  void checkRanges() const;

 protected:
  ArrayVectorBase(
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const Type> type,
      VectorEncoding::Simple encoding,
      BufferPtr nulls,
      size_t length,
      std::optional<vector_size_t> nullCount,
      BufferPtr offsets,
      BufferPtr lengths)
      : BaseVector(
            pool,
            type,
            encoding,
            std::move(nulls),
            length,
            std::nullopt /*distinctValueCount*/,
            nullCount),
        offsets_(std::move(offsets)),
        rawOffsets_(offsets_->as<vector_size_t>()),
        sizes_(std::move(lengths)),
        rawSizes_(sizes_->as<vector_size_t>()) {
    VELOX_CHECK_GE(
        offsets_->capacity(), BaseVector::length_ * sizeof(vector_size_t));
    VELOX_CHECK_GE(
        sizes_->capacity(), BaseVector::length_ * sizeof(vector_size_t));
  }

  void copyRangesImpl(
      const BaseVector* source,
      const folly::Range<const CopyRange*>& ranges,
      VectorPtr* targetValues,
      const BaseVector* sourceValues,
      VectorPtr* targetKeys,
      const BaseVector* sourceKeys);

 private:
  BufferPtr
  ensureIndices(BufferPtr& buf, const vector_size_t*& raw, vector_size_t size) {
    if (buf && buf->isMutable() &&
        buf->capacity() >= size * sizeof(vector_size_t)) {
      return buf;
    }
    resizeIndices(size, 0, &buf, &raw);
    return buf;
  }

 protected:
  BufferPtr offsets_;
  const vector_size_t* rawOffsets_;
  BufferPtr sizes_;
  const vector_size_t* rawSizes_;
};

class ArrayVector : public ArrayVectorBase {
 public:
  ArrayVector(
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const Type> type,
      BufferPtr nulls,
      size_t length,
      BufferPtr offsets,
      BufferPtr lengths,
      VectorPtr elements,
      std::optional<vector_size_t> nullCount = std::nullopt)
      : ArrayVectorBase(
            pool,
            type,
            VectorEncoding::Simple::ARRAY,
            std::move(nulls),
            length,
            nullCount,
            std::move(offsets),
            std::move(lengths)),
        elements_(BaseVector::getOrCreateEmpty(
            std::move(elements),
            type->childAt(0),
            pool)) {
    VELOX_CHECK_EQ(type->kind(), TypeKind::ARRAY);
    VELOX_CHECK(
        elements_->type()->kindEquals(type->childAt(0)),
        "Unexpected element type: {}. Expected: {}",
        elements_->type()->toString(),
        type->childAt(0)->toString());
  }

  std::optional<int32_t> compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const override;

  uint64_t hashValueAt(vector_size_t index) const override;

  std::unique_ptr<SimpleVector<uint64_t>> hashAll() const override;

  const VectorPtr& elements() const {
    return elements_;
  }

  VectorPtr& elements() {
    return elements_;
  }

  void setElements(VectorPtr elements) {
    elements_ = BaseVector::getOrCreateEmpty(
        std::move(elements), type()->childAt(0), pool_);
  }

  void copyRanges(
      const BaseVector* source,
      const folly::Range<const CopyRange*>& ranges) override {
    const ArrayVector* sourceArray{};
    if (auto wrapped = source->wrappedVector();
        !wrapped->isConstantEncoding()) {
      sourceArray = wrapped->asUnchecked<ArrayVector>();
    }
    copyRangesImpl(
        source,
        ranges,
        &elements_,
        sourceArray ? sourceArray->elements_.get() : nullptr,
        nullptr,
        nullptr);
  }

  uint64_t retainedSize() const override {
    return BaseVector::retainedSize() + offsets_->capacity() +
        sizes_->capacity() + elements_->retainedSize();
  }

  uint64_t estimateFlatSize() const override;

  using BaseVector::toString;

  std::string toString(vector_size_t index) const override;

  void ensureWritable(const SelectivityVector& rows) override;

  bool isWritable() const override;

  /// Calls BaseVector::prepareForReuse() to check and reset nulls buffer if
  /// needed, checks and resets offsets and sizes buffers, zeros out offsets and
  /// sizes if reusable, calls BaseVector::prepareForReuse(elements, 0) for the
  /// elements vector.
  void prepareForReuse() override;

  bool mayHaveNullsRecursive() const override {
    return BaseVector::mayHaveNullsRecursive() ||
        elements_->mayHaveNullsRecursive();
  }

  VectorPtr slice(vector_size_t offset, vector_size_t length) const override;

 private:
  VectorPtr elements_;
};

class MapVector : public ArrayVectorBase {
 public:
  MapVector(
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const Type> type,
      BufferPtr nulls,
      size_t length,
      BufferPtr offsets,
      BufferPtr sizes,
      VectorPtr keys,
      VectorPtr values,
      std::optional<vector_size_t> nullCount = std::nullopt,
      bool sortedKeys = false)
      : ArrayVectorBase(
            pool,
            type,
            VectorEncoding::Simple::MAP,
            std::move(nulls),
            length,
            nullCount,
            std::move(offsets),
            std::move(sizes)),
        keys_(BaseVector::getOrCreateEmpty(
            std::move(keys),
            type->childAt(0),
            pool)),
        values_(BaseVector::getOrCreateEmpty(
            std::move(values),
            type->childAt(1),
            pool)),
        sortedKeys_{sortedKeys} {
    VELOX_CHECK_EQ(type->kind(), TypeKind::MAP);

    VELOX_CHECK(
        keys_->type()->kindEquals(type->childAt(0)),
        "Unexpected key type: {}. Expected: {}",
        keys_->type()->toString(),
        type->childAt(0)->toString());
    VELOX_CHECK(
        values_->type()->kindEquals(type->childAt(1)),
        "Unexpected value type: {}. Expected: {}",
        values_->type()->toString(),
        type->childAt(1)->toString());
  }

  virtual ~MapVector() override {}

  std::optional<int32_t> compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const override;

  uint64_t hashValueAt(vector_size_t index) const override;

  std::unique_ptr<SimpleVector<uint64_t>> hashAll() const override;

  void resize(vector_size_t size, bool setNotNull = true) override {
    if (BaseVector::length_ < size) {
      resizeIndices(size, 0, &offsets_, &rawOffsets_);
      resizeIndices(size, 0, &sizes_, &rawSizes_);
    }
    BaseVector::resize(size, setNotNull);
  }

  const VectorPtr& mapKeys() const {
    return keys_;
  }

  VectorPtr& mapKeys() {
    return keys_;
  }

  const VectorPtr& mapValues() const {
    return values_;
  }

  VectorPtr& mapValues() {
    return values_;
  }

  bool hasSortedKeys() const {
    return sortedKeys_;
  }

  void setKeysAndValues(VectorPtr keys, VectorPtr values) {
    keys_ = BaseVector::getOrCreateEmpty(
        std::move(keys), type()->childAt(0), pool_);
    values_ = BaseVector::getOrCreateEmpty(
        std::move(values), type()->childAt(1), pool_);
  }

  void copyRanges(
      const BaseVector* source,
      const folly::Range<const CopyRange*>& ranges) override {
    const MapVector* sourceMap{};
    if (auto wrapped = source->wrappedVector();
        !wrapped->isConstantEncoding()) {
      sourceMap = wrapped->asUnchecked<MapVector>();
    }
    copyRangesImpl(
        source,
        ranges,
        &values_,
        sourceMap ? sourceMap->values_.get() : nullptr,
        &keys_,
        sourceMap ? sourceMap->keys_.get() : nullptr);
  }

  uint64_t retainedSize() const override {
    return BaseVector::retainedSize() + offsets_->capacity() +
        sizes_->capacity() + keys_->retainedSize() + values_->retainedSize();
  }

  uint64_t estimateFlatSize() const override;

  using BaseVector::toString;

  std::string toString(vector_size_t index) const override;

  // Sorts all maps smallest key first. This enables linear time
  // comparison and log time lookup.  This may only be done if there
  // are no other references to 'map'. Checks that 'map' is uniquely
  // referenced. This is guaranteed after construction or when
  // retrieving values from aggregation or join row containers.
  static void canonicalize(
      const std::shared_ptr<MapVector>& map,
      bool useStableSort = false);

  // Returns indices into the map at 'index' such
  // that keys[indices[i]] < keys[indices[i + 1]].
  std::vector<vector_size_t> sortedKeyIndices(vector_size_t index) const;

  void ensureWritable(const SelectivityVector& rows) override;

  bool isWritable() const override;

  /// Calls BaseVector::prepareForReuse() to check and reset nulls buffer if
  /// needed, checks and resets offsets and sizes buffers, zeros out offsets and
  /// sizes if reusable, calls BaseVector::prepareForReuse(keys|values, 0) for
  /// the keys and values vectors.
  void prepareForReuse() override;

  bool mayHaveNullsRecursive() const override {
    return BaseVector::mayHaveNullsRecursive() ||
        keys_->mayHaveNullsRecursive() || values_->mayHaveNullsRecursive();
  }

  VectorPtr slice(vector_size_t offset, vector_size_t length) const override;

 protected:
  virtual void resetDataDependentFlags(const SelectivityVector* rows) override {
    BaseVector::resetDataDependentFlags(rows);
    sortedKeys_ = false;
  }

 private:
  // Returns true if the keys for map at 'index' are sorted from first
  // to last in the type's collation order.
  bool isSorted(vector_size_t index) const;

  // makes a Buffer with 0, 1, 2,... size-1. This is later sorted to
  // get elements in key order in each map.
  BufferPtr elementIndices() const;

  VectorPtr keys_;
  VectorPtr values_;
  bool sortedKeys_;
};

using RowVectorPtr = std::shared_ptr<RowVector>;
using ArrayVectorPtr = std::shared_ptr<ArrayVector>;
using MapVectorPtr = std::shared_ptr<MapVector>;

// Allocates a buffer to fit at least 'size' offsets and initializes them to
// zero.
inline BufferPtr allocateOffsets(vector_size_t size, memory::MemoryPool* pool) {
  return AlignedBuffer::allocate<vector_size_t>(size, pool, 0);
}

// Allocates a buffer to fit at least 'size' sizes and initializes them to
// zero.
inline BufferPtr allocateSizes(vector_size_t size, memory::MemoryPool* pool) {
  return AlignedBuffer::allocate<vector_size_t>(size, pool, 0);
}
} // namespace facebook::velox

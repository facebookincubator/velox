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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <fmt/format.h>
#include <folly/Format.h>
#include <folly/Range.h>
#include <folly/container/F14Map.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/CompareFlags.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Nulls.h"
#include "velox/type/Type.h"
#include "velox/type/Variant.h"
#include "velox/vector/BuilderTypeUtils.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/TypeAliases.h"
#include "velox/vector/VectorEncoding.h"
#include "velox/vector/VectorUtil.h"

namespace facebook {
namespace velox {

template <typename T>
class SimpleVector;

template <typename T>
class FlatVector;

class VectorPool;

/**
 * Base class for all columnar-based vectors of any type.
 */
class BaseVector {
 public:
  static constexpr uint64_t kNullHash = 1;

  BaseVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      VectorEncoding::Simple encoding,
      BufferPtr nulls,
      size_t length,
      std::optional<vector_size_t> distinctValueCount = std::nullopt,
      std::optional<vector_size_t> nullCount = std::nullopt,
      std::optional<ByteCount> representedByteCount = std::nullopt,
      std::optional<ByteCount> storageByteCount = std::nullopt);

  virtual ~BaseVector() = default;

  VectorEncoding::Simple encoding() const {
    return encoding_;
  }

  inline bool isLazy() const {
    return encoding() == VectorEncoding::Simple::LAZY;
  }

  // Returns false if vector has no nulls. Return true if vector may have nulls.
  virtual bool mayHaveNulls() const {
    return rawNulls_;
  }

  // Returns false if this vector and all of its children have no nulls. Returns
  // true if this vector or any of its children may have nulls.
  virtual bool mayHaveNullsRecursive() const {
    return mayHaveNulls();
  }

  inline bool isIndexInRange(vector_size_t index) const {
    // This compiles better than index >= 0 && index < length_.
    return static_cast<uint32_t>(index) < length_;
  }

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<BaseVector, T>);
    return dynamic_cast<T*>(this);
  }

  template <typename T>
  const T* as() const {
    static_assert(std::is_base_of_v<BaseVector, T>);
    return dynamic_cast<const T*>(this);
  }

  // Use when the type of 'this' is already known. dynamic_cast() is slow.
  template <typename T>
  T* asUnchecked() {
    static_assert(std::is_base_of_v<BaseVector, T>);
    DCHECK(dynamic_cast<const T*>(this) != nullptr);
    return static_cast<T*>(this);
  }

  template <typename T>
  const T* asUnchecked() const {
    static_assert(std::is_base_of_v<BaseVector, T>);
    DCHECK(dynamic_cast<const T*>(this) != nullptr);
    return static_cast<const T*>(this);
  }

  template <typename T>
  const FlatVector<T>* asFlatVector() const {
    return dynamic_cast<const FlatVector<T>*>(this);
  }

  template <typename T>
  FlatVector<T>* asFlatVector() {
    return dynamic_cast<FlatVector<T>*>(this);
  }

  velox::memory::MemoryPool* pool() const {
    return pool_;
  }

  virtual bool isNullAt(vector_size_t idx) const {
    VELOX_DCHECK_GE(idx, 0, "Index must not be negative");
    VELOX_DCHECK_LT(idx, length_, "Index is too large");
    return rawNulls_ ? bits::isBitNull(rawNulls_, idx) : false;
  }

  std::optional<vector_size_t> getNullCount() const {
    return nullCount_;
  }

  void setNullCount(vector_size_t newNullCount) {
    nullCount_ = newNullCount;
  }

  const TypePtr& type() const {
    return type_;
  }

  TypeKind typeKind() const {
    return typeKind_;
  }

  /**
   * Returns a smart pointer to the null bitmap data for this
   * vector. May hold nullptr if there are no nulls. Not const because
   * some vectors may generate this on first access.
   */
  const BufferPtr& nulls() const {
    return nulls_;
  }

  const uint64_t* rawNulls() const {
    return rawNulls_;
  }

  uint64_t* mutableRawNulls() {
    ensureNulls();
    return const_cast<uint64_t*>(rawNulls_);
  }

  virtual BufferPtr mutableNulls(vector_size_t size) {
    ensureNullsCapacity(size);
    return nulls_;
  }

  std::optional<vector_size_t> getDistinctValueCount() const {
    return distinctValueCount_;
  }

  /**
   * @return the number of rows of data in this vector
   */
  vector_size_t size() const {
    return length_;
  }

  virtual void append(const BaseVector* other) {
    auto totalSize = BaseVector::length_ + other->size();
    auto previousSize = BaseVector::size();
    resize(totalSize);
    copy(other, previousSize, 0, other->size());
  }

  /**
   * @return the number of bytes this vector takes on disk when in a compressed
   * and serialized format
   */
  std::optional<ByteCount> storageBytes() const {
    return storageByteCount_;
  }

  /**
   * @return the number of bytes required to naively represent all of the data
   * in this vector - the raw data size if not in a compressed or otherwise
   * optimized format
   */
  std::optional<ByteCount> representedBytes() const {
    return representedByteCount_;
  }

  /**
   * @return the number of bytes required to hold this vector in memory
   */
  ByteCount inMemoryBytes() const {
    return inMemoryBytes_;
  }

  /**
   * @return true if this vector has the same value at the given index as the
   * other vector at the other vector's index (including if both are null),
   * false otherwise
   * @throws if the type_ of other doesn't match the type_ of this
   */
  virtual bool equalValueAt(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex) const {
    static constexpr CompareFlags kEqualValueAtFlags = {
        false, false, true /*equalOnly*/, false /*stopAtNull**/};
    // Will always have value because stopAtNull is false.
    return compare(other, index, otherIndex, kEqualValueAtFlags).value() == 0;
  }

  int32_t compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex) const {
    // Default compare flags always generate value.
    return compare(other, index, otherIndex, CompareFlags()).value();
  }

  // Returns < 0 if 'this' at 'index' is less than 'other' at
  // 'otherIndex', 0 if equal and > 0 otherwise.
  // If flags.stopAtNull is set, returns std::nullopt if null encountered
  // whether it's top-level null or inside the data of complex type.
  virtual std::optional<int32_t> compare(
      const BaseVector* other,
      vector_size_t index,
      vector_size_t otherIndex,
      CompareFlags flags) const = 0;

  /// Sort values at specified 'indices'. Used to sort map keys.
  virtual void sortIndices(
      std::vector<vector_size_t>& indices,
      CompareFlags flags) const {
    std::sort(
        indices.begin(),
        indices.end(),
        [&](vector_size_t left, vector_size_t right) {
          return compare(this, left, right, flags) < 0;
        });
  }

  /// Sort values at specified 'indices' after applying the 'mapping'. Used to
  /// sort map keys.
  virtual void sortIndices(
      std::vector<vector_size_t>& indices,
      const vector_size_t* mapping,
      CompareFlags flags) const {
    std::sort(
        indices.begin(),
        indices.end(),
        [&](vector_size_t left, vector_size_t right) {
          return compare(this, mapping[left], mapping[right], flags) < 0;
        });
  }

  /**
   * @return the hash of the value at the given index in this vector
   */
  virtual uint64_t hashValueAt(vector_size_t index) const = 0;

  /**
   * @return a new vector that contains the hashes for all entries
   */
  virtual std::unique_ptr<SimpleVector<uint64_t>> hashAll() const = 0;

  /// Returns true if this vector is encoded as flat (FlatVector).
  bool isFlatEncoding() const {
    return encoding_ == VectorEncoding::Simple::FLAT;
  }

  /// Returns true if this vector is encoded as constant (ConstantVector).
  bool isConstantEncoding() const {
    return encoding_ == VectorEncoding::Simple::CONSTANT;
  }

  // Returns true if this vector has a scalar type. If so, values are
  // accessed by valueAt after casting the vector to a type()
  // dependent instantiation of SimpleVector<T>.
  virtual bool isScalar() const {
    return false;
  }

  // Returns the scalar or complex vector wrapped inside any nesting of
  // dictionary, sequence or constant vectors.
  virtual const BaseVector* wrappedVector() const {
    return this;
  }

  // Returns the index to apply for 'index' in the vector returned by
  // wrappedVector(). Translates the index over any nesting of
  // dictionaries, sequences and constants.
  virtual vector_size_t wrappedIndex(vector_size_t index) const {
    return index;
  }

  // Sets the null indicator at 'idx'. 'true' means null.
  FOLLY_ALWAYS_INLINE virtual void setNull(vector_size_t idx, bool value) {
    VELOX_DCHECK(idx >= 0 && idx < length_);
    if (!nulls_ && !value) {
      return;
    }
    ensureNulls();
    bits::setBit(
        nulls_->asMutable<uint64_t>(), idx, bits::kNull ? value : !value);
  }

  static int32_t
  countNulls(const BufferPtr& nulls, vector_size_t begin, vector_size_t end) {
    return nulls ? bits::countNulls(nulls->as<uint64_t>(), begin, end) : 0;
  }

  static int32_t countNulls(const BufferPtr& nulls, vector_size_t size) {
    return countNulls(nulls, 0, size);
  }

  // Returns whether or not the nulls buffer can be modified.
  // This does not guarantee the existence of the nulls buffer, if using this
  // within BaseVector you still may need to call ensureNulls.
  virtual bool isNullsWritable() const {
    return !nulls_ || (nulls_->unique() && nulls_->isMutable());
  }

  // Sets null when 'nulls' has null value for a row in 'rows'
  virtual void addNulls(const uint64_t* bits, const SelectivityVector& rows);

  // Clears null when 'nulls' has non-null value for a row in 'rows'
  virtual void clearNulls(const SelectivityVector& rows);

  virtual void clearNulls(vector_size_t begin, vector_size_t end);

  void clearAllNulls() {
    clearNulls(0, size());
  }

  // Sets the size to 'newSize' and ensures there is space for the
  // indicated number of nulls and top level values (eg. values for Flat,
  // indices for Dictionary, etc). Any immutable buffers that need to be resized
  // are copied. 'setNotNull' indicates if nulls in range [oldSize, newSize]
  // should be set to not null.
  // Note: caller must ensure that the vector is singly referenced.
  virtual void resize(vector_size_t newSize, bool setNotNull = true);

  // Sets the rows of 'this' given by 'rows' to
  // 'source.valueAt(toSourceRow ? toSourceRow[row] : row)', where
  // 'row' iterates over 'rows'.
  virtual void copy(
      const BaseVector* source,
      const SelectivityVector& rows,
      const vector_size_t* toSourceRow) {
    rows.applyToSelected([&](vector_size_t row) {
      auto sourceRow = toSourceRow ? toSourceRow[row] : row;
      if (sourceRow >= source->size()) {
        return;
      }
      if (source->isNullAt(sourceRow)) {
        setNull(row, true);
      } else {
        copy(source, row, sourceRow, 1);
      }
    });
  }

  // Utility for making a deep copy of a whole vector.
  static std::shared_ptr<BaseVector> copy(const BaseVector& vector) {
    auto result =
        BaseVector::create(vector.type(), vector.size(), vector.pool());
    result->copy(&vector, 0, 0, vector.size());
    return result;
  }

  virtual void copy(
      const BaseVector* source,
      vector_size_t targetIndex,
      vector_size_t sourceIndex,
      vector_size_t count) {
    if (count == 0) {
      return;
    }
    CopyRange range{sourceIndex, targetIndex, count};
    copyRanges(source, folly::Range(&range, 1));
  }

  struct CopyRange {
    vector_size_t sourceIndex;
    vector_size_t targetIndex;
    vector_size_t count;
  };

  // Copy multiple ranges at once.  This is more efficient than calling `copy`
  // multiple times, especially for ARRAY, MAP, and VARCHAR.
  virtual void copyRanges(
      const BaseVector* /*source*/,
      const folly::Range<const CopyRange*>& /*ranges*/) {
    VELOX_UNSUPPORTED("Can only copy into flat or complex vectors");
  }

  // Construct a zero-copy slice of the vector with the indicated offset and
  // length.
  virtual std::shared_ptr<BaseVector> slice(
      vector_size_t offset,
      vector_size_t length) const = 0;

  // Returns a vector of the type of 'source' where 'indices' contains
  // an index into 'source' for each element of 'source'. The
  // resulting vector has position i set to source[i]. This is
  // equivalent to wrapping 'source' in a dictionary with 'indices'
  // but this may reuse structure if said structure is uniquely owned
  // or if a copy is more efficient than dictionary wrapping.
  static std::shared_ptr<BaseVector> transpose(
      BufferPtr indices,
      std::shared_ptr<BaseVector>&& source);

  static std::shared_ptr<BaseVector> createConstant(
      const TypePtr& type,
      variant value,
      vector_size_t size,
      velox::memory::MemoryPool* pool);

  static std::shared_ptr<BaseVector> createNullConstant(
      const TypePtr& type,
      vector_size_t size,
      velox::memory::MemoryPool* pool);

  static std::shared_ptr<BaseVector> wrapInDictionary(
      BufferPtr nulls,
      BufferPtr indices,
      vector_size_t size,
      std::shared_ptr<BaseVector> vector);

  static std::shared_ptr<BaseVector> wrapInSequence(
      BufferPtr lengths,
      vector_size_t size,
      std::shared_ptr<BaseVector> vector);

  // Creates a ConstantVector of specified length and value coming from the
  // 'index' element of the 'vector'. Peels off any encodings of the 'vector'
  // before making a new ConstantVector. The result vector is either a
  // ConstantVector holding a scalar value or a ConstantVector wrapping flat or
  // lazy vector. The result cannot be a wrapping over another constant or
  // dictionary vector.
  static std::shared_ptr<BaseVector> wrapInConstant(
      vector_size_t length,
      vector_size_t index,
      std::shared_ptr<BaseVector> vector);

  // Makes 'result' writable for 'rows'. A wrapper (e.g. dictionary, constant,
  // sequence) is flattened and a multiply referenced flat vector is copied.
  // The content of 'rows' is not copied, as these values are intended to be
  // overwritten.
  //
  // After invoking this function, the 'result' is guaranteed to be a flat
  // uniquely-referenced vector with all data-dependent flags reset.
  //
  // Use SelectivityVector::empty() to make the 'result' writable and preserve
  // all current values.
  static void ensureWritable(
      const SelectivityVector& rows,
      const TypePtr& type,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<BaseVector>& result,
      VectorPool* vectorPool = nullptr);

  virtual void ensureWritable(const SelectivityVector& rows);

  // Returns true if the following conditions hold:
  //  * The vector is singly referenced.
  //  * The vector has a Flat-like encoding (Flat, Array, Map, Row).
  //  * Any child Buffers are mutable  and singly referenced.
  //  * All of these conditions hold for child Vectors recursively.
  // This function is templated rather than taking a std::shared_ptr<BaseVector>
  // because if we were to do that the compiler would allocate a new shared_ptr
  // when this function is called making it not unique.
  template <typename T>
  static bool isVectorWritable(const std::shared_ptr<T>& vector) {
    if (!vector.unique()) {
      return false;
    }

    return vector->isWritable();
  }

  virtual bool isWritable() const {
    return false;
  }

  // Flattens the input vector.
  //
  // TODO: This method reuses ensureWritable(), which ensures that both:
  //  (a) the vector is flattened, and
  //  (b) it's singly-referenced
  //
  // We don't necessarily need (b) if we only want to flatten vectors.
  static void flattenVector(
      std::shared_ptr<BaseVector>& vector,
      size_t vectorSize) {
    BaseVector::ensureWritable(
        SelectivityVector::empty(vectorSize),
        vector->type(),
        vector->pool(),
        vector);
  }

  template <typename T>
  static inline uint64_t byteSize(vector_size_t count) {
    return sizeof(T) * count;
  }

  // If 'vector' is a wrapper, returns the underlying values vector. This is
  // virtual and defined here because we must be able to access this in type
  // agnostic code without a switch on all data types.
  virtual std::shared_ptr<BaseVector> valueVector() const {
    VELOX_UNSUPPORTED("Vector is not a wrapper");
  }

  virtual BaseVector* loadedVector() {
    return this;
  }

  virtual const BaseVector* loadedVector() const {
    return this;
  }

  static std::shared_ptr<BaseVector> loadedVectorShared(
      std::shared_ptr<BaseVector>);

  virtual const BufferPtr& values() const {
    VELOX_UNSUPPORTED("Only flat vectors have a values buffer");
  }

  virtual const void* valuesAsVoid() const {
    VELOX_UNSUPPORTED("Only flat vectors have a values buffer");
  }

  // If 'this' is a wrapper, returns the wrap info, interpretation depends on
  // encoding.
  virtual BufferPtr wrapInfo() const {
    throw std::runtime_error("Vector is not a wrapper");
  }

  template <typename T = BaseVector>
  static std::shared_ptr<T> create(
      const TypePtr& type,
      vector_size_t size,
      velox::memory::MemoryPool* pool) {
    return std::static_pointer_cast<T>(createInternal(type, size, pool));
  }

  static std::shared_ptr<BaseVector> getOrCreateEmpty(
      std::shared_ptr<BaseVector> vector,
      const TypePtr& type,
      velox::memory::MemoryPool* pool) {
    return vector ? vector : create(type, 0, pool);
  }

  void setNulls(const BufferPtr& nulls);

  void resetNulls() {
    setNulls(nullptr);
  }

  // Ensures that '*indices' has space for 'size' elements. Sets
  // elements between the old and new sizes to 'initialValue' if the
  // new size > old size. If memory is moved, '*raw' is maintained to
  // point to element 0 of (*indices)->as<vector_size_t>().
  void resizeIndices(
      vector_size_t size,
      vector_size_t initialValue,
      BufferPtr* indices,
      const vector_size_t** raw) {
    resizeIndices(size, initialValue, this->pool(), indices, raw);
  }

  static void resizeIndices(
      vector_size_t size,
      vector_size_t initialValue,
      velox::memory::MemoryPool* pool,
      BufferPtr* indices,
      const vector_size_t** raw);

  // Makes sure '*buffer' has space for 'size' items of T and is writable. Sets
  // 'raw' to point to the writable contents of '*buffer'.
  template <typename T, typename RawT>
  static void ensureBuffer(
      vector_size_t size,
      velox::memory::MemoryPool* pool,
      BufferPtr* buffer,
      RawT** raw) {
    vector_size_t minBytes = byteSize<T>(size);
    if (*buffer && (*buffer)->capacity() >= minBytes && (*buffer)->unique()) {
      (*buffer)->setSize(minBytes);
      if (raw) {
        *raw = (*buffer)->asMutable<RawT>();
      }
      return;
    }
    if (*buffer) {
      AlignedBuffer::reallocate<T>(buffer, size);
    } else {
      *buffer = AlignedBuffer::allocate<T>(size, pool);
    }
    if (raw) {
      *raw = (*buffer)->asMutable<RawT>();
    }
    (*buffer)->setSize(minBytes);
  }

  // Returns the byte size of memory that is kept live through 'this'.
  virtual uint64_t retainedSize() const {
    return nulls_ ? nulls_->capacity() : 0;
  }

  /// Returns an estimate of the 'retainedSize' of a flat representation of the
  /// data stored in this vector. Returns zero if this is a lazy vector that
  /// hasn't been loaded yet.
  virtual uint64_t estimateFlatSize() const;

  /// To safely reuse a vector one needs to (1) ensure that the vector as well
  /// as all its buffers and child vectors are singly-referenced and mutable
  /// (for buffers); (2) clear append-only string buffers and child vectors
  /// (elements of arrays, keys and values of maps, fields of structs); (3)
  /// reset all data-dependent flags.
  ///
  /// This method takes a non-const reference to a 'vector' and updates it to
  /// possibly a new flat vector of the specified size that is safe to reuse.
  /// If input 'vector' is not singly-referenced or not flat, replaces 'vector'
  /// with a new vector of the same type and specified size. If some of the
  /// buffers cannot be reused, these buffers are reset. Child vectors are
  /// updated by calling this method recursively with size zero. Data-dependent
  /// flags are reset after this call.
  static void prepareForReuse(
      std::shared_ptr<BaseVector>& vector,
      vector_size_t size);

  /// Resets non-reusable buffers and updates child vectors by calling
  /// BaseVector::prepareForReuse.
  /// Base implementation checks and resets nulls buffer if needed. Keeps the
  /// nulls buffer if singly-referenced, mutable and has at least one null bit
  /// set.
  virtual void prepareForReuse();

  // True if left and right are the same or if right is
  // TypeKind::UNKNOWN.  ArrayVector copying may come across unknown
  // type data for null-only content. Nulls can be transferred between
  // two unknowns but values cannot be assigned into an unknown 'left'
  // from a not-unknown 'right'.
  static bool compatibleKind(TypeKind left, TypeKind right) {
    return left == right || right == TypeKind::UNKNOWN;
  }

  /// Returns a brief summary of the vector. If 'recursive' is true, includes a
  /// summary of all the layers of encodings starting with the top layer.
  ///
  /// For example,
  ///     with recursive 'false':
  ///
  ///         [DICTIONARY INTEGER: 5 elements, no nulls]
  ///
  ///     with recursive 'true':
  ///
  ///         [DICTIONARY INTEGER: 5 elements, no nulls], [FLAT INTEGER: 10
  ///             elements, no nulls]
  std::string toString(bool recursive) const;

  /// Same as toString(false). Provided to allow for easy invocation from LLDB.
  std::string toString() const {
    return toString(false);
  }

  /// Returns string representation of the value in the specified row.
  virtual std::string toString(vector_size_t index) const;

  /// Returns a list of values in rows [from, to).
  ///
  /// Automatically adjusts 'from' and 'to' to a range of valid indices. Returns
  /// empty string if 'from' is greater than or equal to vector size or 'to' is
  /// less than or equal to zero. Returns values up to the end of the vector if
  /// 'to' is greater than vector size. Returns values from the start of the
  /// vector if 'from' is negative.
  ///
  /// The type of the 'delimiter' is a const char* and not an std::string to
  /// allow for invoking this method from LLDB.
  std::string toString(
      vector_size_t from,
      vector_size_t to,
      const char* delimiter,
      bool includeRowNumbers = true) const;

  /// Returns a list of values in rows [from, to). Values are separated by a new
  /// line and prefixed with a row number.
  ///
  /// This method is provided to allow to easy invocation from LLDB.
  std::string toString(vector_size_t from, vector_size_t to) const {
    return toString(from, to, "\n");
  }

  void setCodegenOutput() {
    isCodegenOutput_ = true;
  }

  bool isCodegenOutput() const {
    return isCodegenOutput_;
  }

  /// Marks the vector as containing or being a lazy vector and being wrapped.
  /// Should only be used if 'this' is lazy or has a nested lazy vector.
  /// Returns true if this is the first time it was wrapped, else returns false.
  bool markAsContainingLazyAndWrapped() {
    if (containsLazyAndIsWrapped_) {
      return false;
    }
    containsLazyAndIsWrapped_ = true;
    return true;
  }

 protected:
  /// Returns a brief summary of the vector. The default implementation includes
  /// encoding, type, number of rows and number of nulls.
  ///
  /// For example,
  ///     [FLAT INTEGER: 3 elements, no nulls]
  ///     [DICTIONARY INTEGER: 5 elements, 1 nulls]
  virtual std::string toSummaryString() const;

  /*
   * Allocates or reallocates nulls_ with at least the given size if nulls_
   * hasn't been allocated yet or has been allocated with a smaller capacity.
   */
  void ensureNullsCapacity(vector_size_t minimumSize, bool setNotNull = false);

  FOLLY_ALWAYS_INLINE static std::optional<int32_t>
  compareNulls(bool thisNull, bool otherNull, CompareFlags flags) {
    DCHECK(thisNull || otherNull);
    // Null handling.
    if (flags.stopAtNull) {
      return std::nullopt;
    }

    if (thisNull) {
      if (otherNull) {
        return 0;
      }
      return flags.nullsFirst ? -1 : 1;
    }
    if (otherNull) {
      return flags.nullsFirst ? 1 : -1;
    }

    VELOX_UNREACHABLE(
        "The function should be called only if one of the inputs is null");
  }

  void ensureNulls() {
    ensureNullsCapacity(length_, true);
  }

  // Slice a buffer with specific type.
  //
  // For boolean type and if the offset is not multiple of 8, return a shifted
  // copy; otherwise return a BufferView into the original buffer (with shared
  // ownership of original buffer).
  static BufferPtr sliceBuffer(
      const Type&,
      const BufferPtr&,
      vector_size_t offset,
      vector_size_t length,
      memory::MemoryPool*);

  BufferPtr sliceNulls(vector_size_t offset, vector_size_t length) const {
    return sliceBuffer(*BOOLEAN(), nulls_, offset, length, pool_);
  }

  // Reset data-dependent flags to the "unknown" status. This is needed whenever
  // a vector is mutated because the modification may invalidate these flags.
  // Currently, we call this function in BaseVector::ensureWritable() and
  // BaseVector::prepareForReuse() that are expected to be called before any
  // vector mutation.
  //
  // Per-vector flags are reset to default values. Per-row flags are reset only
  // at the selected rows. If rows is a nullptr, per-row flags are reset at all
  // rows.
  virtual void resetDataDependentFlags(const SelectivityVector* /*rows*/) {
    nullCount_ = std::nullopt;
    distinctValueCount_ = std::nullopt;
    representedByteCount_ = std::nullopt;
    storageByteCount_ = std::nullopt;
  }

  const TypePtr type_;
  const TypeKind typeKind_;
  const VectorEncoding::Simple encoding_;
  BufferPtr nulls_;
  // Caches raw pointer to 'nulls->as<uint64_t>().
  const uint64_t* rawNulls_ = nullptr;
  velox::memory::MemoryPool* pool_;
  vector_size_t length_ = 0;

  /**
   * Holds the number of nulls in the vector. If the number of nulls
   * is not available, it is set to std::nullopt. Setting the value to
   * zero does have implications (SIMD operations need null count to be
   * zero) and is not the same as std::nullopt.
   */
  std::optional<vector_size_t> nullCount_;
  std::optional<vector_size_t> distinctValueCount_;
  std::optional<ByteCount> representedByteCount_;
  std::optional<ByteCount> storageByteCount_;
  ByteCount inMemoryBytes_ = 0;

 private:
  static std::shared_ptr<BaseVector> createInternal(
      const TypePtr& type,
      vector_size_t size,
      velox::memory::MemoryPool* pool);

  bool isCodegenOutput_ = false;

  friend class LazyVector;

  /// Is true if this vector is a lazy vector or contains one and is being
  /// wrapped. Keeping track of this helps to enforce the invariant that an
  /// unloaded lazy vector should not be wrapped by two separate top level
  /// vectors. This would ensure we avoid it being loaded for two separate set
  /// of rows.
  bool containsLazyAndIsWrapped_{false};
};

template <>
uint64_t BaseVector::byteSize<bool>(vector_size_t count);

template <>
inline uint64_t BaseVector::byteSize<UnknownValue>(vector_size_t) {
  return 0;
}

using VectorPtr = std::shared_ptr<BaseVector>;

// Returns true if vector is a Lazy vector, possibly wrapped, that hasn't
// been loaded yet.
bool isLazyNotLoaded(const BaseVector& vector);

// Allocates a buffer to fit at least 'size' indices and initializes them to
// zero.
inline BufferPtr allocateIndices(vector_size_t size, memory::MemoryPool* pool) {
  return AlignedBuffer::allocate<vector_size_t>(size, pool, 0);
}

// Allocates a buffer to fit at least 'size' null bits and initializes them to
// the provided 'initValue' which has a default value of non-null.
inline BufferPtr allocateNulls(
    vector_size_t size,
    memory::MemoryPool* pool,
    bool initValue = bits::kNotNull) {
  return AlignedBuffer::allocate<bool>(size, pool, initValue);
}

// Returns a summary of the null bits in the specified buffer and prints out
// first 'maxBitsToPrint' bits. Automatically adjusts if 'maxBitsToPrint' is
// greater than total number of bits available.
// For example: 3 out of 8 rows are null: .nn.n...
std::string printNulls(
    const BufferPtr& nulls,
    vector_size_t maxBitsToPrint = 30);

// Returns a summary of the indices buffer and prints out first
// 'maxIndicesToPrint' indices. Automatically adjusts if 'maxIndicesToPrint' is
// greater than total number of indices available.
// For example: 5 unique indices out of 6: 34, 79, 11, 0, 0, 33.
std::string printIndices(
    const BufferPtr& indices,
    vector_size_t maxIndicesToPrint = 10);

} // namespace velox
} // namespace facebook

namespace folly {

// Allow VectorEncoding::Simple to be transparently used by folly::sformat.
// e.g: folly::sformat("type: {}", encodingType);
template <>
class FormatValue<facebook::velox::VectorEncoding::Simple> {
 public:
  explicit FormatValue(const facebook::velox::VectorEncoding::Simple& type)
      : type_(type) {}

  template <typename FormatCallback>
  void format(FormatArg& arg, FormatCallback& cb) const {
    return format_value::formatString(
        facebook::velox::VectorEncoding::mapSimpleToName(type_), arg, cb);
  }

 private:
  facebook::velox::VectorEncoding::Simple type_;
};

} // namespace folly

template <>
struct fmt::formatter<facebook::velox::VectorEncoding::Simple> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(
      const facebook::velox::VectorEncoding::Simple& x,
      FormatContext& ctx) {
    return format_to(
        ctx.out(), "{}", facebook::velox::VectorEncoding::mapSimpleToName(x));
  }
};

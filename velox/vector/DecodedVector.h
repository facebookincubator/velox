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
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox {

/// Takes a flat, constant or dictionary vector with possibly many layers of
/// dictionary wrappings and converts it into a flat or constant base vector +
/// at most one wrapping. Combines multiple layers of indices and nulls into
/// one.
///
/// Decoding a vector is straightforward if it is flat. However, if it is not,
/// the following steps are taken:
/// 1. It first traverses the top dictionary layers (if they exist) and
///    combines their indices and nulls
/// 2. Next, if it encounters a constant layer, it does the following:
///    ** If the dictionary layers over it were adding additional nulls, then it
///    replaces all non-null indices with the constant index.
///    ** Else if the dictionary layers did not add any additional nulls then it
///    converts the resultant wrap into a single constant wrap with the same
///    index as this constant layer.
///    ** However, if the constant layer is a null constant, then regardless of
///    the dictionary layers, it converts the resultant wrap into a constant
///    wrap representing a null constant
///    ** Finally, if the constant is a scalar, the base is set to the
///    constantVector itself, otherwise the base points to the complex vector
///    wrapped underneath the constant layer
/// 3. Next, If it encounters a non-constant base layer:
///    ** It combines the nulls from that base layer into the set of nulls that
///    it is tracking
///    ** Additionally, it will flatten the base layer if its not already flat.
///    Currently, such a transformation is only supported for bias encoding.
///
/// Having access to a flat base’s data buffer and a single level of indices and
/// nulls (or a constant index) means that we can read all values in constant
/// time.
///
/// Memory Allocation of internal state:
///
/// All memory required is directly allocated from the system’s memory allocator
/// and is not managed by any memory pool. This means that the indices and nulls
/// buffers either point directly to the corresponding buffers of the input
/// vector which is decoded, or, are directly malloced from the system
/// allocator. This allows objects of DecodedVector to be cached and re-used
/// when such operations need to be performed frequently. This saves time wasted
/// on memory allocations (see LocalDecodedVector class).
///
/// NOTE:
/// DecodedVector is only designed to read a potentially encoded vector. If the
/// primary use-case is to peel and eventually wrap a peeled result then
/// checkout the PeeledEncodings class. There are certain extensions of the
/// DecodedVector API which are necessary to implement functionality in
/// PeeledEncodings and should only be used there. Please refrain from using
/// those APIs as their behaviour can diverge significantly from others, namely,
/// makeIndices(), wrap(), dictionaryWrapping().
class DecodedVector {
 public:
  /// Default constructor. The caller must call decode() or makeIndices() next.
  DecodedVector() = default;

  /// Disable copy constructor and assignment.
  DecodedVector(const DecodedVector& other) = delete;
  DecodedVector& operator=(const DecodedVector& other) = delete;

  /// Allow std::move.
  DecodedVector(DecodedVector&& other) = default;

  /// Decodes 'vector' for 'rows'.
  ///
  /// Decoding is trivial if vector is flat, constant or single-level
  /// dictionary. If a vector is a multi-level dictionary, the indices from all
  /// the dictionaries are combined. The result of decoding is a flat or
  /// constant base vector and an optional array of indices.
  ///
  /// Loads the underlying lazy vector if not already loaded before decoding
  /// unless loadLazy is false.
  ///
  /// loadLazy = false is used in HashAggregation to implement pushdown of
  /// aggregation into table scan. In this case, DecodedVector is used to
  /// combine possibly multiple levels of wrappings into just one and then load
  /// LazyVector only for the necessary rows. This uses ValueHook which adds
  /// values to aggregation accumulators without intermediate materialization.
  ///
  /// Limitations: Decoding a constant vector wrapping a lazy vector that has
  /// not been loaded yet with is not supported loadLazy = false.
  /// if `rows` is not passed then the vector is decoded for its size.
  DecodedVector(
      const BaseVector& vector,
      const SelectivityVector& rows,
      bool loadLazy = true) {
    decode(vector, &rows, loadLazy);
  }

  DecodedVector(const BaseVector& vector, bool loadLazy = true) {
    decode(vector, nullptr, loadLazy);
  }

  /// Resets the internal state and decodes 'vector' for 'rows'. See
  /// constructor.
  void decode(
      const BaseVector& vector,
      const SelectivityVector& rows,
      bool loadLazy = true) {
    decode(vector, &rows, loadLazy);
  }

  void decode(const BaseVector& vector, bool loadLazy = true) {
    decode(vector, nullptr, loadLazy);
  }

  /// Returns the values buffer for the base vector. Assumes the vector is of
  /// scalar type and has been already decoded. Use indices() to access
  /// individual values, i.e. data()[indices[i]] returns the value at the
  /// top-level row 'i' given that 'i' is one of the rows specified for
  /// decoding.
  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_);
  }

  /// Returns the raw nulls buffer for the base vector combined with nulls found
  /// in dictionary wrappings. May return nullptr if there are no nulls. Use
  /// top-level row numbers to access individual null flags, e.g.
  ///
  ///  nulls() ? bits::isBitNull(nulls(), i) : false
  ///
  /// returns the null flag for top-level row 'i' given that 'i' is one of the
  /// rows specified for decoding.
  const uint64_t* nulls();

  /// Returns true if wrappings may have added nulls.
  bool hasExtraNulls() const {
    return hasExtraNulls_;
  }

  /// Returns the mapping from top-level rows to rows in the base vector or
  /// data() buffer.
  const vector_size_t* indices() {
    if (!indices_) {
      fillInIndices();
    }
    return &indices_[0];
  }

  /// Given a top-level row returns corresponding index in the base vector or
  /// data().
  vector_size_t index(vector_size_t idx) const {
    if (isIdentityMapping_) {
      return idx;
    }
    if (isConstantMapping_) {
      return constantIndex_;
    }
    VELOX_DCHECK(indices_);
    return indices_[idx];
  }

  /// Returns a scalar value for the top-level row 'idx'.
  template <typename T>
  T valueAt(vector_size_t idx) const {
    return reinterpret_cast<const T*>(data_)[index(idx)];
  }

  /// If false, there are no nulls. Otherwise, there is a possibility that there
  /// are some nulls, but no certainty.
  bool mayHaveNulls() const {
    return mayHaveNulls_;
  }

  bool mayHaveNullsRecursive() const {
    return mayHaveNulls_ || baseVector_->mayHaveNullsRecursive();
  }

  /// Return null flag for the top-level row.
  bool isNullAt(vector_size_t idx) const {
    if (!nulls_) {
      return false;
    }

    if (isIdentityMapping_ || hasExtraNulls_) {
      return bits::isBitNull(nulls_, idx);
    }

    if (isConstantMapping_) {
      return bits::isBitNull(nulls_, 0);
    }

    VELOX_DCHECK(indices_);
    return bits::isBitNull(nulls_, indices_[idx]);
  }

  /// Returns the largest decoded row number + 1, i.e. rows.end().
  vector_size_t size() const {
    return size_;
  }

  /// Returns the flat or constant base vector.
  const BaseVector* base() const {
    return baseVector_;
  }

  /// Returns true if the decoded vector was flat.
  bool isIdentityMapping() const {
    return isIdentityMapping_;
  }

  /// Returns true if the decoded vector was constant.
  bool isConstantMapping() const {
    return isConstantMapping_;
  }

  /////////////////////////////////////////////////////////////////
  /// BEGIN: Members that must only be used by PeeledEncoding class.
  /// See class comment for more details.

  /// Given a dictionary vector with at least 'numLevel' levels of dictionary
  /// wrapping, combines 'numLevel' wrappings into one.
  /// Example usage: <InputVector> : Dictionary2(Dictionary1(Complex)) If we
  /// call makeIndices(<Input Vector>, Level = 2), only Dictionary2 and
  /// Dictionary1 will have indices and nulls merged and the nulls from the
  /// Complex flat vector will not be merged. If instead Level is set to 3, then
  /// additionally nulls from the complex base vector will also be merged.
  void makeIndices(
      const BaseVector& vector,
      const SelectivityVector& rows,
      int32_t numLevels) {
    return makeIndices(vector, &rows, numLevels);
  }

  void makeIndices(const BaseVector& vector, int32_t numLevels) {
    return makeIndices(vector, nullptr, numLevels);
  }

  /// Wraps a vector with the same wrapping as another. 'wrapper' must
  /// have been previously decoded by 'this'. This is used when 'data'
  /// is a component of the base vector of 'wrapper' and must be used
  /// in the same context, thus with the same indirections.
  VectorPtr wrap(VectorPtr data, const BaseVector& wrapper, vector_size_t size);

  VectorPtr wrap(
      VectorPtr data,
      const BaseVector& wrapper,
      const SelectivityVector& rows) {
    return wrap(std::move(data), wrapper, rows.end());
  }

  struct DictionaryWrapping {
    BufferPtr indices;
    BufferPtr nulls;
  };

  /// Returns 'indices' and 'nulls' buffers that represent the combined
  /// dictionary wrapping of the decoded vector. Requires
  /// isIdentityMapping() == false and isConstantMapping() == false.
  /// NOTE: The nulls buffer returned will also have nulls from the base()
  /// combined into it. To control which levels are combined, please make sure
  /// to use makeIndices() instead of decoded() when initializing the
  /// DecodedVector.
  DictionaryWrapping dictionaryWrapping(
      const BaseVector& wrapper,
      vector_size_t size) const;

  DictionaryWrapping dictionaryWrapping(
      const BaseVector& wrapper,
      const SelectivityVector& rows) const {
    return dictionaryWrapping(wrapper, rows.end());
  }

  /// END: Members that must only be used by PeeledEncoding
  /////////////////////////////////////////////////////////

  /// Pre-allocated vector of 0, 1, 2,..
  static const std::vector<vector_size_t>& consecutiveIndices();

 private:
  DecodedVector(
      const BaseVector& vector,
      const SelectivityVector* rows,
      bool loadLazy = true) {
    decode(vector, rows, loadLazy);
  }

  void decode(
      const BaseVector& vector,
      const SelectivityVector* rows,
      bool loadLazy = true);

  void makeIndices(
      const BaseVector& vector,
      const SelectivityVector* rows,
      int32_t numLevels);

  /// Pre-allocated vector of all zeros.
  static const std::vector<vector_size_t>& zeroIndices();

  bool indicesNotCopied() const {
    return copiedIndices_.empty() || indices_ < copiedIndices_.data() ||
        indices_ >= &copiedIndices_.back();
  }

  bool nullsNotCopied() const {
    return copiedNulls_.empty() || nulls_ != copiedNulls_.data();
  }

  void setFlatNulls(const BaseVector& vector, const SelectivityVector* rows);

  template <TypeKind kind>
  void decodeBiased(const BaseVector& vector, const SelectivityVector* rows);

  void makeIndicesMutable();

  void combineWrappers(
      const BaseVector* vector,
      const SelectivityVector* rows,
      int numLevels = -1);

  void applyDictionaryWrapper(
      const BaseVector& dictionaryVector,
      const SelectivityVector* rows);

  void applySequenceWrapper(
      const BaseVector& sequenceVector,
      const SelectivityVector* rows);

  void copyNulls(vector_size_t size);

  void fillInIndices();

  void setBaseData(const BaseVector& vector, const SelectivityVector* rows);

  void setBaseDataForConstant(
      const BaseVector& vector,
      const SelectivityVector* rows);

  void setBaseDataForBias(
      const BaseVector& vector,
      const SelectivityVector* rows);

  void reset(vector_size_t size);

  // If `rows` is null applies the `func` to all rows in [0, size_)
  // otherwise, applies it to selected rows only.
  template <typename Func>
  void applyToRows(const SelectivityVector* rows, Func&& func) const;

  // If `rows` is null returns 'size_', otherwise returns rows->end().
  inline vector_size_t end(const SelectivityVector* rows) const {
    return rows ? rows->end() : size_;
  }

  // If `rows` is null returns 'size', otherwise returns rows->end().
  inline vector_size_t end(vector_size_t size, const SelectivityVector* rows)
      const {
    return rows ? rows->end() : size;
  }

  // Last valid index into 'indices_' + 1.
  vector_size_t size_ = 0;

  // The indices into 'data_' or 'baseVector_' for the rows in
  // 'rows' given to decode(). Only positions that are in
  // 'selection' are guaranteed to have valid values.
  const vector_size_t* indices_ = nullptr;

  // The base array of 'vector' given to decode(), nullptr if vector is of
  // complex type.
  const void* data_ = nullptr;

  // The first bit holds the value when a constant bool is decoded, and data_ in
  // that case will refer to the address of this variable.
  uint64_t constantBoolDataHolder_;

  // Null bitmask of the base vector if wrappings didn't add nulls
  // (hasExtraNulls_ is false). Otherwise, null bitmask of the base vector
  // combined with null bitmasks in all the wrappings (hasExtraNulls_ is true).
  // May be nullptr if there are no nulls.
  const uint64_t* nulls_ = nullptr;

  // Nulls bitmask indexed using top-level row numbers and containing null bits
  // of the base vector combined with null bits in all the wrappings. May be
  // nullptr if there are no nulls or allNullsInitialized_ is false. Initialized
  // on first access.
  std::optional<const uint64_t*> allNulls_ = nullptr;

  // The base vector of 'vector' given to decode(). This is the data
  // after sequence, constant and dictionary vectors have been peeled
  // off.
  const BaseVector* baseVector_ = nullptr;

  // True if either the leaf vector has nulls or if nulls were added
  // by a dictionary wrapper.
  bool mayHaveNulls_ = false;

  // True if nulls added by a dictionary wrapper.
  bool hasExtraNulls_ = false;

  bool isIdentityMapping_ = false;

  bool isConstantMapping_ = false;

  bool loadLazy_ = false;

  // Index of an element of the baseVector_ that points to a constant value of
  // complex type. Applies only when isConstantMapping_ is true and baseVector_
  // is of complex type (array, map, row).
  vector_size_t constantIndex_{0};

  // Holds data that needs to be copied out from the base vector,
  // e.g. exploded BiasVector values.
  std::vector<uint64_t> tempSpace_;

  // Holds indices if an array of indices needs to be materialized,
  // e.g. when combining nested dictionaries.
  std::vector<vector_size_t> copiedIndices_;

  // Used as backing for 'nulls_' when null-ness is combined from
  // dictionary and base values.
  std::vector<uint64_t> copiedNulls_;

  // Used as 'nulls_' for a null constant vector.
  static uint64_t constantNullMask_;
};

template <>
inline bool DecodedVector::valueAt(vector_size_t idx) const {
  return bits::isBitSet(reinterpret_cast<const uint64_t*>(data_), index(idx));
}

template <>
inline UnscaledLongDecimal DecodedVector::valueAt(vector_size_t idx) const {
  auto valuePosition = reinterpret_cast<const char*>(data_) +
      sizeof(UnscaledLongDecimal) * index(idx);
  return UnscaledLongDecimal::deserialize(valuePosition);
}

} // namespace facebook::velox

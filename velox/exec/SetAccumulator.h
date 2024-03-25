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

#include <folly/container/F14Set.h>

#include "velox/common/base/IOUtils.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/AddressableNonNullValueList.h"
#include "velox/exec/Strings.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace detail {

/// Maintains a set of unique values. Non-null values are stored in F14FastSet.
/// A separate flag tracks presence of the null value.
/// The SetAccumulator also tracks the order in which the values are added to
/// the accumulator (for ordered aggregations). So each value is associated with
/// an index of its position.

/// SetAccumulator supports serialization/deserialization to/from an array of
/// byte streams.
/// These are used in the spilling logic of operators using SetAccumulator.

/// The serialization format is :
/// i) The first element of the array is the index of the null value
/// (or -1 if no null value).
/// ii) The list of values (and optionally some metadata) in the order of
/// their positions. The null position (if one) is skipped.
/// iii) For scalar and string types, only the value is serialized in the
/// order of their indexes in the accumulator. For a complex type, a tuple
/// of (hash, value) is serialized.
template <
    typename T,
    typename Hash = std::hash<T>,
    typename EqualTo = std::equal_to<T>>
struct SetAccumulator {
  std::optional<vector_size_t> nullIndex;

  folly::F14FastMap<
      T,
      int32_t,
      Hash,
      EqualTo,
      AlignedStlAllocator<std::pair<const T, vector_size_t>, 16>>
      uniqueValues;

  SetAccumulator(const TypePtr& /*type*/, HashStringAllocator* allocator)
      : uniqueValues{AlignedStlAllocator<std::pair<const T, vector_size_t>, 16>(
            allocator)} {}

  SetAccumulator(Hash hash, EqualTo equalTo, HashStringAllocator* allocator)
      : uniqueValues{
            0,
            hash,
            equalTo,
            AlignedStlAllocator<std::pair<const T, vector_size_t>, 16>(
                allocator)} {}

  /// Adds value if new. No-op if the value was added before.
  void addValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* /*allocator*/) {
    const auto cnt = uniqueValues.size();
    if (decoded.isNullAt(index)) {
      if (!nullIndex.has_value()) {
        nullIndex = cnt;
      }
    } else {
      uniqueValues.insert(
          {decoded.valueAt<T>(index), nullIndex.has_value() ? cnt + 1 : cnt});
    }
  }

  /// Adds new values from an array.
  void addValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
    }
  }

  /// Adds non-null value if new. No-op if the value is NULL or was added
  /// before.
  void addNonNullValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* /*allocator*/) {
    const auto cnt = uniqueValues.size();
    if (!decoded.isNullAt(index)) {
      uniqueValues.insert({decoded.valueAt<T>(index), cnt});
    }
  }

  /// Adds new non-null values from an array.
  void addNonNullValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addNonNullValue(values, offset + i, allocator);
    }
  }

  /// Deserializes accumulator from previously serialized value.
  void deserialize(
      const FlatVector<StringView>& vector,
      vector_size_t index,
      vector_size_t size,
      HashStringAllocator* /*allocator*/) {
    // The serialized value is the nullIndex (kNoNullIndex if no null is
    // present) followed by the unique values ordered by index.
    deserializeNullIndex(vector.valueAt(index).data());

    // Mark the nullPosition beyond values to correctly offset when reading the
    // stream.
    const auto nullPosition = nullIndex.has_value() ? nullIndex.value() : size;
    for (auto i = 1, j = 0; i < size; i++, j++) {
      T value = *reinterpret_cast<const T*>(vector.valueAt(index + i).data());
      auto pos = (j < nullPosition) ? j : i;
      uniqueValues.insert({value, pos});
    }
  }

  /// Returns number of unique values including null.
  size_t size() const {
    return uniqueValues.size() + (nullIndex.has_value() ? 1 : 0);
  }

  /// Copies the unique values and null into the specified vector starting at
  /// the specified offset.
  vector_size_t extractValues(FlatVector<T>& values, vector_size_t offset) {
    for (auto value : uniqueValues) {
      values.set(offset + value.second, value.first);
    }

    if (nullIndex.has_value()) {
      values.setNull(offset + nullIndex.value(), true);
    }

    return nullIndex.has_value() ? uniqueValues.size() + 1
                                 : uniqueValues.size();
  }

  /// Serializes a sequence of VARBINARY values starting at result[index].
  /// This is used for the spill of this accumulator.
  void serialize(const VectorPtr& result, vector_size_t index) {
    auto* flatResult = result->as<FlatVector<StringView>>();
    VELOX_CHECK_LE(uniqueValues.size() + 1, flatResult->size());

    auto nullIndexValue = nullIndexSerializationValue();
    flatResult->set(
        index,
        StringView(
            reinterpret_cast<const char*>(&nullIndexValue),
            sizeof(vector_size_t)));

    // The null position is skipped when serializing values, so setting an out
    // of bound value for no null position.
    const auto nullPosition =
        nullIndex.has_value() ? nullIndex.value() : uniqueValues.size();
    const auto sizeOfT = sizeof(T);
    for (const auto& value : uniqueValues) {
      auto pos = value.second;
      auto offset = (pos < nullPosition ? pos : pos - 1) + index + 1;
      flatResult->set(
          offset,
          StringView(reinterpret_cast<const char*>(&value.first), sizeOfT));
    }
  }

  void free(HashStringAllocator& allocator) {
    using UT = decltype(uniqueValues);
    uniqueValues.~UT();
  }

  void deserializeNullIndex(const char* input) {
    VELOX_CHECK(!nullIndex.has_value());
    auto streamNullIndex = *reinterpret_cast<const vector_size_t*>(input);
    if (streamNullIndex != kNoNullIndex) {
      nullIndex = streamNullIndex;
    }
  }

  vector_size_t nullIndexSerializationValue() {
    return nullIndex.has_value() ? nullIndex.value() : kNoNullIndex;
  }

  static const vector_size_t kNoNullIndex = -1;
};

/// Maintains a set of unique strings.
struct StringViewSetAccumulator {
  /// A set of unique StringViews pointing to storage managed by 'strings'.
  SetAccumulator<StringView> base;

  /// Stores unique non-null non-inline strings.
  Strings strings;

  StringViewSetAccumulator(const TypePtr& type, HashStringAllocator* allocator)
      : base{type, allocator} {}

  void addValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* allocator) {
    const auto cnt = base.uniqueValues.size();
    if (decoded.isNullAt(index)) {
      if (!base.nullIndex.has_value()) {
        base.nullIndex = cnt;
      }
    } else {
      auto value = decoded.valueAt<StringView>(index);
      addValue(value, base.nullIndex.has_value() ? cnt + 1 : cnt, allocator);
    }
  }

  void addValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
    }
  }

  void addNonNullValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* allocator) {
    const auto cnt = base.uniqueValues.size();
    if (!decoded.isNullAt(index)) {
      auto value = decoded.valueAt<StringView>(index);
      if (!value.isInline()) {
        if (base.uniqueValues.contains(value)) {
          return;
        }
        value = strings.append(value, *allocator);
      }
      base.uniqueValues.insert({value, cnt});
    }
  }

  void addNonNullValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addNonNullValue(values, offset + i, allocator);
    }
  }

  void deserialize(
      const FlatVector<StringView>& vector,
      vector_size_t index,
      vector_size_t size,
      HashStringAllocator* allocator) {
    base.deserializeNullIndex(vector.valueAt(index).data());

    // Mark the nullPosition beyond values to correctly offset when reading the
    // stream.
    const auto nullPosition =
        base.nullIndex.has_value() ? base.nullIndex.value() : size;
    for (auto i = 1; i < size; i++) {
      auto pos = i - 1 < nullPosition ? i - 1 : i;
      addUniqueValue(vector.valueAt(index + i), pos, allocator);
    }
  }

  size_t size() const {
    return base.size();
  }

  vector_size_t extractValues(
      FlatVector<StringView>& values,
      vector_size_t offset) {
    return base.extractValues(values, offset);
  }

  /// Serialize an array of VARBINARY representation starting from
  /// result[index]. This is used for the spill of this accumulator.
  void serialize(const VectorPtr& result, vector_size_t index) {
    auto* flatResult = result->as<FlatVector<StringView>>();
    VELOX_CHECK_LE(base.uniqueValues.size() + 1, flatResult->size());

    auto nullIndexValue = base.nullIndexSerializationValue();
    flatResult->set(
        index, StringView((const char*)&nullIndexValue, sizeof(vector_size_t)));

    // The null position is skipped when serializing values, so setting an out
    // of bound value for no null position.
    const auto nullPosition = base.nullIndex.has_value()
        ? base.nullIndex.value()
        : base.uniqueValues.size();
    for (const auto& value : base.uniqueValues) {
      auto pos = value.second;
      auto offset = (pos < nullPosition ? pos : pos - 1) + index + 1;
      flatResult->set(offset, value.first);
    }
  }

  void free(HashStringAllocator& allocator) {
    strings.free(allocator);
    using Base = decltype(base);
    base.~Base();
  }

 private:
  void addValue(
      const StringView& value,
      vector_size_t index,
      HashStringAllocator* allocator) {
    if (base.uniqueValues.contains(value)) {
      return;
    }

    addUniqueValue(value, index, allocator);
  }

  void addUniqueValue(
      const StringView& value,
      vector_size_t index,
      HashStringAllocator* allocator) {
    VELOX_CHECK(!base.uniqueValues.contains(value));
    StringView valueCopy = value;
    if (!valueCopy.isInline()) {
      valueCopy = strings.append(value, *allocator);
    }
    base.uniqueValues.insert({valueCopy, index});
  }
};

/// Maintains a set of unique arrays, maps or structs.
struct ComplexTypeSetAccumulator {
  /// A set of pointers to values stored in AddressableNonNullValueList.
  SetAccumulator<
      AddressableNonNullValueList::Entry,
      AddressableNonNullValueList::Hash,
      AddressableNonNullValueList::EqualTo>
      base;

  /// Stores unique non-null values.
  AddressableNonNullValueList values;

  // Tracks the size of the biggest ComplexType in the set. This is used for
  // allocating a temporary buffer during serialization.
  size_t maxSize = 0;

  static constexpr size_t kSizeOfHash = sizeof(uint64_t);

  ComplexTypeSetAccumulator(const TypePtr& type, HashStringAllocator* allocator)
      : base{
            AddressableNonNullValueList::Hash{},
            AddressableNonNullValueList::EqualTo{type},
            allocator} {}

  void addValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* allocator) {
    const auto cnt = base.uniqueValues.size();
    if (decoded.isNullAt(index)) {
      if (!base.nullIndex.has_value()) {
        base.nullIndex = cnt;
      }
    } else {
      const auto entry = values.append(decoded, index, allocator);
      const auto position = base.nullIndex.has_value() ? cnt + 1 : cnt;
      addEntry(entry, position);
    }
  }

  void addValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
    }
  }

  void addNonNullValue(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* allocator) {
    const auto cnt = base.uniqueValues.size();
    if (!decoded.isNullAt(index)) {
      auto entry = values.append(decoded, index, allocator);

      if (!base.uniqueValues.insert({entry, cnt}).second) {
        values.removeLast(entry);
      }
    }
  }

  void addNonNullValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    VELOX_DCHECK(!arrayVector.isNullAt(index));
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addNonNullValue(values, offset + i, allocator);
    }
  }

  void deserialize(
      const FlatVector<StringView>& vector,
      vector_size_t index,
      vector_size_t size,
      HashStringAllocator* allocator) {
    base.deserializeNullIndex(vector.valueAt(index).data());

    // Mark the nullPosition beyond values to correctly offset when reading the
    // stream.
    const auto nullPosition =
        base.nullIndex.has_value() ? base.nullIndex.value() : size;
    for (auto i = 1; i < size; i++) {
      auto value = vector.valueAt(index + i);
      auto stream = common::InputByteStream(value.data());
      auto hash = stream.read<uint64_t>();
      auto length = value.size() - kSizeOfHash;
      auto contents = StringView(stream.read<char>(length), length);
      auto position = values.appendSerialized(contents, allocator);

      auto pos = (i - 1 < nullPosition) ? i - 1 : i;
      addEntry({position, contents.size(), hash}, pos);
    }
  }

  size_t size() const {
    return base.size();
  }

  vector_size_t extractValues(BaseVector& values, vector_size_t offset) {
    for (const auto& position : base.uniqueValues) {
      AddressableNonNullValueList::read(
          position.first, values, offset + position.second);
    }

    if (base.nullIndex.has_value()) {
      values.setNull(offset + base.nullIndex.value(), true);
    }

    return base.uniqueValues.size() + (base.nullIndex.has_value() ? 1 : 0);
  }

  /// Starting from result[index] append a sequence of VARBINARY values for
  /// serialization for spilling..
  void serialize(const VectorPtr& result, vector_size_t index) {
    auto* flatResult = result->as<FlatVector<StringView>>();
    VELOX_CHECK_LE(base.uniqueValues.size() + 1, flatResult->size());

    auto nullIndexValue = base.nullIndexSerializationValue();
    flatResult->set(
        index, StringView((const char*)&nullIndexValue, sizeof(vector_size_t)));

    // Temporary buffer used during serialization.
    auto* tempBuffer =
        (char*)(flatResult->pool()->allocate(maxSize + kSizeOfHash));

    // The null position is skipped when serializing values, so setting an out
    // of bound value for no null position.
    const auto nullPosition = base.nullIndex.has_value()
        ? base.nullIndex.value()
        : base.uniqueValues.size();
    for (const auto& value : base.uniqueValues) {
      auto pos = value.second;
      auto offset = (pos < nullPosition ? pos : pos - 1) + index + 1;
      SerializationStream stream(tempBuffer, kSizeOfHash + value.first.size);
      // Complex type hash.
      stream.append(&value.first.hash, kSizeOfHash);
      // Complex type value.
      stream.append(value.first);
      flatResult->set(
          offset, StringView(tempBuffer, kSizeOfHash + value.first.size));
    }

    flatResult->pool()->free(tempBuffer, maxSize + kSizeOfHash);
  }

  void free(HashStringAllocator& allocator) {
    values.free(allocator);
    using Base = decltype(base);
    base.~Base();
  }

 private:
  // Simple stream abstraction for serialization logic. 'append' calls to concat
  // values to the stream (for the input buffer) are exposed to the user.
  struct SerializationStream {
    char* rawBuffer;
    const vector_size_t totalSize;
    vector_size_t offset = 0;

    SerializationStream(char* buffer, vector_size_t totalSize)
        : rawBuffer(buffer), totalSize(totalSize) {}

    void append(const void* value, vector_size_t size) {
      VELOX_CHECK_LE(offset + size, totalSize);
      memcpy(rawBuffer + offset, value, size);
      offset += size;
    }

    void append(const AddressableNonNullValueList::Entry& entry) {
      VELOX_CHECK_LE(offset + entry.size, totalSize);
      AddressableNonNullValueList::readSerialized(entry, rawBuffer + offset);
      offset += entry.size;
    }
  };

  void addEntry(
      const AddressableNonNullValueList::Entry& entry,
      vector_size_t index) {
    if (!base.uniqueValues.insert({entry, index}).second) {
      values.removeLast(entry);
    } else {
      maxSize = maxSize < entry.size ? entry.size : maxSize;
    }
  }
};

template <typename T>
struct SetAccumulatorTypeTraits {
  using AccumulatorType = SetAccumulator<T>;
};

template <>
struct SetAccumulatorTypeTraits<StringView> {
  using AccumulatorType = StringViewSetAccumulator;
};

template <>
struct SetAccumulatorTypeTraits<ComplexType> {
  using AccumulatorType = ComplexTypeSetAccumulator;
};
} // namespace detail

template <typename T>
using SetAccumulator =
    typename detail::SetAccumulatorTypeTraits<T>::AccumulatorType;

} // namespace facebook::velox::aggregate::prestosql

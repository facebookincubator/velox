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
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
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

  size_t maxSpillSize() const {
    return sizeof(char) + // hasNull
        2 * sizeof(vector_size_t) + // nullIndex and size
        ((sizeof(vector_size_t) + sizeof(T)) *
         uniqueValues.size()); // index + sizeof(T) * all entries in uniqueSize
  }

  void addFromSpill(
      FlatVector<StringView>& flatVector,
      HashStringAllocator* allocator) {
    size_t offset{0};
    const char* rawBuffer = flatVector.valueAt(0).data();

    {
      char hasNull{0};
      memcpy(&hasNull, rawBuffer + offset, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (hasNull != 0) {
        vector_size_t nullIndexValue{0};
        memcpy(&nullIndexValue, rawBuffer + offset, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);

        nullIndex = nullIndexValue;
      }
    }

    vector_size_t size{0};
    memcpy(&size, rawBuffer + offset, sizeof(size));

    offset += sizeof(size);

    for (size_t i = 0; i < size; ++i) {
      T value{};
      memcpy(&value, rawBuffer + offset, sizeof(T));

      offset += sizeof(T);

      vector_size_t index{0};
      memcpy(&index, rawBuffer + offset, sizeof(index));

      offset += sizeof(index);

      uniqueValues.emplace(std::pair<const T, vector_size_t>{value, index});
    }
  }

  size_t extractForSpill(char* data, size_t size) const {
    VELOX_CHECK_LE(maxSpillSize(), size, "Spill buffer too small");

    size_t offset{0};

    {
      char hasNull = nullIndex.has_value() ? 1 : 0;
      memcpy(data + offset, &hasNull, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (nullIndex.has_value()) {
        vector_size_t nullIndexValue = nullIndex.value();
        memcpy(data + offset, &nullIndexValue, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);
      }
    }

    {
      vector_size_t countEntries = uniqueValues.size();
      memcpy(data + offset, &countEntries, sizeof(countEntries));

      offset += sizeof(countEntries);
    }

    for (auto value : uniqueValues) {
      T innerValue = value.first;
      memcpy(data + offset, &innerValue, sizeof(innerValue));

      offset += sizeof(innerValue);

      vector_size_t index = value.second;
      memcpy(data + offset, &index, sizeof(index));

      offset += sizeof(index);
    }

    return offset;
  }

  void free(HashStringAllocator& allocator) {
    clear(allocator);
    using UT = decltype(uniqueValues);
    uniqueValues.~UT();
  }

  /// Clear all data, assuming future reuse of SetAllocator.
  void clear(HashStringAllocator&) {
    uniqueValues.clear();
    nullIndex = std::nullopt;
  }
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
      if (!value.isInline()) {
        if (base.uniqueValues.contains(value)) {
          return;
        }
        value = strings.append(value, *allocator);
      }
      base.uniqueValues.insert(
          {value, base.nullIndex.has_value() ? cnt + 1 : cnt});
    }
  }

  void addValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
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

  size_t maxSpillSize() const {
    size_t maxSpillSize =
        sizeof(char) + 2 * sizeof(vector_size_t); // nullIndex and size

    for (const auto& value : base.uniqueValues) {
      maxSpillSize +=
          (sizeof(int32_t) + sizeof(vector_size_t) + value.first.size());
    }

    return maxSpillSize;
  }

  size_t extractForSpill(char* data, size_t size) const {
    if (size < maxSpillSize()) {
      VELOX_FAIL("Spill buffer too small");
    }

    size_t offset{0};

    {
      char hasNull = base.nullIndex.has_value() ? 1 : 0;
      memcpy(data + offset, &hasNull, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (base.nullIndex.has_value()) {
        vector_size_t nullIndexValue = base.nullIndex.value();
        memcpy(data + offset, &nullIndexValue, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);
      }
    }

    {
      vector_size_t countEntries = base.uniqueValues.size();
      memcpy(data + offset, &countEntries, sizeof(countEntries));

      offset += sizeof(countEntries);
    }

    for (auto value : base.uniqueValues) {
      const StringView& innerValue = value.first;
      int32_t stringSize = innerValue.size();
      memcpy(data + offset, &stringSize, sizeof(stringSize));

      offset += sizeof(stringSize);

      memcpy(data + offset, innerValue.data(), innerValue.size());

      offset += innerValue.size();

      vector_size_t index = value.second;
      memcpy(data + offset, &index, sizeof(index));

      offset += sizeof(index);
    }

    return offset;
  }

  void addFromSpill(
      FlatVector<StringView>& flatVector,
      HashStringAllocator* allocator) {
    size_t offset{0};
    const char* rawBuffer = flatVector.valueAt(0).data();

    {
      char hasNull{0};
      memcpy(&hasNull, rawBuffer + offset, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (hasNull != 0) {
        vector_size_t nullIndexValue{0};
        memcpy(&nullIndexValue, rawBuffer + offset, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);

        base.nullIndex = nullIndexValue;
      }
    }

    vector_size_t size{0};
    memcpy(&size, rawBuffer + offset, sizeof(size));

    offset += sizeof(size);

    for (size_t i = 0; i < size; ++i) {
      int32_t stringSize{0};
      memcpy(&stringSize, rawBuffer + offset, sizeof(stringSize));

      offset += sizeof(stringSize);
      StringView stringView{rawBuffer + offset, stringSize};

      if (!stringView.isInline()) {
        stringView = strings.append(stringView, *allocator);
      }

      offset += stringSize;

      vector_size_t index{0};
      memcpy(&index, rawBuffer + offset, sizeof(index));

      offset += sizeof(index);

      base.uniqueValues.emplace(
          std::pair<const StringView, vector_size_t>{stringView, index});
    }
  }

  void free(HashStringAllocator& allocator) {
    clear(allocator);
    using Base = decltype(base);
    base.~Base();
  }

  void clear(HashStringAllocator& allocator) {
    strings.free(allocator);
    base.clear(allocator);
  }
};

/// Maintains a set of unique arrays, maps or structs.
struct ComplexTypeSetAccumulator {
  /// A set of pointers to values stored in AddressableNonNullValueList.
  SetAccumulator<
      HashStringAllocator::Position,
      AddressableNonNullValueList::Hash,
      AddressableNonNullValueList::EqualTo>
      base;

  /// Stores unique non-null values.
  AddressableNonNullValueList values;

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
      auto position = values.append(decoded, index, allocator);

      if (!base.uniqueValues
               .insert({position, base.nullIndex.has_value() ? cnt + 1 : cnt})
               .second) {
        values.removeLast(position);
      }
    }
  }

  void addValues(
      const ArrayVector& arrayVector,
      vector_size_t index,
      const DecodedVector& values,
      HashStringAllocator* allocator) {
    const auto size = arrayVector.sizeAt(index);
    const auto offset = arrayVector.offsetAt(index);

    for (auto i = 0; i < size; ++i) {
      addValue(values, offset + i, allocator);
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

  size_t maxSpillSize() const {
    size_t maxSpillSize =
        sizeof(char) + 2 * sizeof(vector_size_t); // nullIndex and size

    for (const auto& position : base.uniqueValues) {
      maxSpillSize +=
          (sizeof(size_t) + sizeof(vector_size_t) +
           AddressableNonNullValueList::getSerializedSize(position.first));
    }

    return maxSpillSize;
  }

  size_t extractForSpill(char* data, size_t size) const {
    if (size < maxSpillSize()) {
      VELOX_FAIL("Spill buffer too small");
    }

    size_t offset{0};

    {
      char hasNull = base.nullIndex.has_value() ? 1 : 0;
      memcpy(data + offset, &hasNull, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (base.nullIndex.has_value()) {
        vector_size_t nullIndexValue = base.nullIndex.value();
        memcpy(data + offset, &nullIndexValue, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);
      }
    }

    {
      vector_size_t countEntries = base.uniqueValues.size();
      memcpy(data + offset, &countEntries, sizeof(countEntries));

      offset += sizeof(countEntries);
    }

    for (const auto& position : base.uniqueValues) {
      const auto streamSize =
          AddressableNonNullValueList::getSerializedSize(position.first);
      memcpy(data + offset, &streamSize, sizeof(streamSize));

      offset += sizeof(streamSize);

      AddressableNonNullValueList::copySerializedTo(
          position.first, data + offset, streamSize);

      offset += streamSize;

      vector_size_t index = position.second;
      memcpy(data + offset, &index, sizeof(index));

      offset += sizeof(index);
    }

    return offset;
  }

  void addFromSpill(
      FlatVector<StringView>& flatVector,
      HashStringAllocator* allocator) {
    size_t offset{0};
    const char* rawBuffer = flatVector.valueAt(0).data();

    {
      char hasNull{0};
      memcpy(&hasNull, rawBuffer + offset, sizeof(hasNull));

      offset += sizeof(hasNull);

      if (hasNull != 0) {
        vector_size_t nullIndexValue{0};
        memcpy(&nullIndexValue, rawBuffer + offset, sizeof(nullIndexValue));

        offset += sizeof(nullIndexValue);

        base.nullIndex = nullIndexValue;
      }
    }

    vector_size_t size{0};
    memcpy(&size, rawBuffer + offset, sizeof(size));

    offset += sizeof(size);

    for (size_t i = 0; i < size; ++i) {
      size_t streamSize{0};
      memcpy(&streamSize, rawBuffer + offset, sizeof(streamSize));

      offset += sizeof(streamSize);

      auto position =
          values.appendSerialized(allocator, rawBuffer + offset, streamSize);

      offset += streamSize;

      vector_size_t index{0};
      memcpy(&index, rawBuffer + offset, sizeof(index));

      offset += sizeof(index);

      base.uniqueValues.emplace(
          std::pair<const HashStringAllocator::Position, vector_size_t>{
              position, index});
    }
  }

  void free(HashStringAllocator& allocator) {
    clear(allocator);
    using Base = decltype(base);
    base.~Base();
  }

  void clear(HashStringAllocator& allocator) {
    values.free(allocator);
    base.clear(allocator);
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

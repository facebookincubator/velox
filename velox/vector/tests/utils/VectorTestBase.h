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

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <gtest/gtest.h>

namespace facebook::velox::test {

/// Returns indices buffer with sequential values going from size - 1 to 0.
BufferPtr makeIndicesInReverse(vector_size_t size, memory::MemoryPool* pool);

BufferPtr makeIndices(
    vector_size_t size,
    std::function<vector_size_t(vector_size_t)> indexAt,
    memory::MemoryPool* pool);

// TODO: enable ASSERT_EQ for vectors.
void assertEqualVectors(const VectorPtr& expected, const VectorPtr& actual);

// Verify that the values in both 'expected' and 'actual' vectors is the same
// but only for the rows marked valid in 'rowsToCompare'.
void assertEqualVectors(
    const VectorPtr& expected,
    const VectorPtr& actual,
    const SelectivityVector& rowsToCompare);

/// Verify that 'vector' is copyable, by copying all rows.
void assertCopyableVector(const VectorPtr& vector);

class VectorTestBase {
 protected:
  VectorTestBase() = default;

  ~VectorTestBase();

  template <typename T>
  using EvalType = typename CppToType<T>::NativeType;

  static std::shared_ptr<const RowType> makeRowType(
      std::vector<std::shared_ptr<const Type>>&& types) {
    return velox::test::VectorMaker::rowType(
        std::forward<std::vector<std::shared_ptr<const Type>>&&>(types));
  }

  void setNulls(
      const VectorPtr& vector,
      std::function<bool(vector_size_t /*row*/)> isNullAt) {
    for (vector_size_t i = 0; i < vector->size(); i++) {
      if (isNullAt(i)) {
        vector->setNull(i, true);
      }
    }
  }

  static std::function<bool(vector_size_t /*row*/)> nullEvery(
      int n,
      int startingFrom = 0) {
    return velox::test::VectorMaker::nullEvery(n, startingFrom);
  }

  // Helper function for comparing vector results
  template <typename T1, typename T2>
  bool
  compareValues(const T1& a, const std::optional<T2>& b, std::string& error) {
    bool result = (a == b.value());
    if (!result) {
      error = " " + std::to_string(a) + " vs " + std::to_string(b.value());
    } else {
      error = "";
    }
    return result;
  }

  bool compareValues(
      const StringView& a,
      const std::optional<std::string>& b,
      std::string& error) {
    bool result = (a.getString() == b.value());
    if (!result) {
      error = " " + a.getString() + " vs " + b.value();
    } else {
      error = "";
    }
    return result;
  }

  RowVectorPtr makeRowVector(
      const std::vector<std::string>& childNames,
      const std::vector<VectorPtr>& children,
      std::function<bool(vector_size_t /*row*/)> isNullAt = nullptr) {
    auto rowVector = vectorMaker_.rowVector(childNames, children);
    if (isNullAt) {
      setNulls(rowVector, isNullAt);
    }
    return rowVector;
  }

  RowVectorPtr makeRowVector(
      const std::vector<VectorPtr>& children,
      std::function<bool(vector_size_t /*row*/)> isNullAt = nullptr) {
    auto rowVector = vectorMaker_.rowVector(children);
    if (isNullAt) {
      setNulls(rowVector, isNullAt);
    }
    return rowVector;
  }

  RowVectorPtr makeRowVector(
      const std::shared_ptr<const RowType>& rowType,
      vector_size_t size) {
    return vectorMaker_.rowVector(rowType, size);
  }

  // Returns a one element ArrayVector with 'vector' as elements of array at 0.
  VectorPtr asArray(VectorPtr vector) {
    BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(
        1, vector->pool(), vector->size());
    BufferPtr offsets =
        AlignedBuffer::allocate<vector_size_t>(1, vector->pool(), 0);
    return std::make_shared<ArrayVector>(
        vector->pool(),
        ARRAY(vector->type()),
        BufferPtr(nullptr),
        1,
        offsets,
        sizes,
        vector,
        0);
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatVector(
      vector_size_t size,
      std::function<T(vector_size_t /*row*/)> valueAt,
      std::function<bool(vector_size_t /*row*/)> isNullAt = nullptr,
      const TypePtr& type = CppToType<T>::create()) {
    return vectorMaker_.flatVector<T>(size, valueAt, isNullAt, type);
  }

  /// Decimal Vector type cannot be inferred from the cpp type alone as the cpp
  /// type does not contain the precision and scale.
  template <typename T>
  FlatVectorPtr<EvalType<T>> makeFlatVector(
      const std::vector<T>& data,
      const TypePtr& type = CppToType<T>::create()) {
    return vectorMaker_.flatVector<T>(data, type);
  }

  template <typename T>
  FlatVectorPtr<EvalType<T>> makeNullableFlatVector(
      const std::vector<std::optional<T>>& data,
      const TypePtr& type = CppToType<T>::create()) {
    return vectorMaker_.flatVectorNullable(data, type);
  }

  template <typename T, int TupleIndex, typename TupleType>
  FlatVectorPtr<T> makeFlatVector(
      const std::vector<TupleType>& data,
      const TypePtr& type = CppToType<T>::create()) {
    return vectorMaker_.flatVector<T, TupleIndex, TupleType>(data, type);
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatVector(size_t size) {
    return vectorMaker_.flatVector<T>(size);
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatVector(size_t size, const TypePtr& type) {
    return vectorMaker_.flatVector<T>(size, type);
  }

  template <typename T>
  FlatVectorPtr<T> makeAllNullFlatVector(vector_size_t size) {
    return vectorMaker_.allNullFlatVector<T>(size);
  }

  FlatVectorPtr<UnscaledShortDecimal> makeShortDecimalFlatVector(
      const std::vector<int64_t>& unscaledValues,
      const TypePtr& type) {
    return vectorMaker_.shortDecimalFlatVector(unscaledValues, type);
  }

  FlatVectorPtr<UnscaledLongDecimal> makeLongDecimalFlatVector(
      const std::vector<int128_t>& unscaledValues,
      const TypePtr& type) {
    return vectorMaker_.longDecimalFlatVector(unscaledValues, type);
  }

  FlatVectorPtr<UnscaledShortDecimal> makeNullableShortDecimalFlatVector(
      const std::vector<std::optional<int64_t>>& unscaledValues,
      const TypePtr& type) {
    return vectorMaker_.shortDecimalFlatVectorNullable(unscaledValues, type);
  }

  FlatVectorPtr<UnscaledLongDecimal> makeNullableLongDecimalFlatVector(
      const std::vector<std::optional<int128_t>>& unscaledValues,
      const TypePtr& type) {
    return vectorMaker_.longDecimalFlatVectorNullable(unscaledValues, type);
  }

  // Convenience function to create arrayVectors (vector of arrays) based on
  // input values from nested std::vectors. The underlying elements are
  // non-nullable.
  //
  // Example:
  //   auto arrayVector = makeArrayVector<int64_t>({
  //       {1, 2, 3, 4, 5},
  //       {},
  //       {1, 2, 3},
  //   });
  //   EXPECT_EQ(3, arrayVector->size());
  template <typename T>
  ArrayVectorPtr makeArrayVector(
      const std::vector<std::vector<T>>& data,
      const TypePtr& elementType = CppToType<T>::create()) {
    return vectorMaker_.arrayVector<T>(data, elementType);
  }

  ArrayVectorPtr makeAllNullArrayVector(
      vector_size_t size,
      const TypePtr& elementType) {
    return vectorMaker_.allNullArrayVector(size, elementType);
  }

  // Create an ArrayVector<ROW> from nested std::vectors of variants.
  // Example:
  //   auto arrayVector = makeArrayOfRowVector(
  //     ROW({INTEGER(), VARCHAR()}),
  //     {
  //       {variant::row({1, "red"}), variant::row({1, "blue"})},
  //       {},
  //       {variant::row({3, "green"})},
  //   });
  //   EXPECT_EQ(3, arrayVector->size());
  //
  // Use variant(TypeKind::ROW) to specify a null array element.
  ArrayVectorPtr makeArrayOfRowVector(
      const RowTypePtr& rowType,
      const std::vector<std::vector<variant>>& data) {
    return vectorMaker_.arrayOfRowVector(rowType, data);
  }

  // Create an ArrayVector<ArrayVector<T>> from nested std::vectors of values.
  // Example:
  //   using innerArrayType = std::vector<std::optional<int64_t>>;
  //   using outerArrayType =
  //       std::vector<std::optional<std::vector<std::optional<int64_t>>>>;
  //
  //   innerArrayType a{1, 2, 3};
  //   innerArrayType b{4, 5};
  //   innerArrayType c{6, 7, 8};
  //   outerArrayType row1{{a}, {b}};
  //   outerArrayType row2{{a}, {c}};
  //   outerArrayType row3{{{}}};
  //   outerArrayType row4{{{std::nullopt}}};
  //   auto arrayVector = makeNullableNestedArrayVector<int64_t>(
  //       {{row1}, {row2}, {row3}, {row4}, std::nullopt});
  //
  //   EXPECT_EQ(5, arrayVector->size());
  template <typename T>
  ArrayVectorPtr makeNullableNestedArrayVector(
      const std::vector<std::optional<
          std::vector<std::optional<std::vector<std::optional<T>>>>>>& data) {
    vector_size_t size = data.size();
    BufferPtr offsets = allocateOffsets(size, pool());
    BufferPtr sizes = allocateSizes(size, pool());
    BufferPtr nulls = allocateNulls(size, pool());

    auto rawOffsets = offsets->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();
    auto rawNulls = nulls->asMutable<uint64_t>();

    // Flatten the outermost layer of std::vector of the input, and populate
    // the sizes and offsets for the top-level array vector.
    std::vector<std::optional<std::vector<std::optional<T>>>> flattenedData;
    vector_size_t i = 0;
    for (const auto& vector : data) {
      if (!vector.has_value()) {
        bits::setNull(rawNulls, i, true);
        rawSizes[i] = 0;
        rawOffsets[i] = 0;
      } else {
        flattenedData.insert(
            flattenedData.end(), vector->begin(), vector->end());

        rawSizes[i] = vector->size();
        rawOffsets[i] = (i == 0) ? 0 : rawOffsets[i - 1] + rawSizes[i - 1];
      }
      ++i;
    }

    // Create the underlying vector.
    auto baseArray = makeNullableArrayVector<T>(flattenedData);

    // Build and return a second-level of ArrayVector on top of baseArray.
    return std::make_shared<ArrayVector>(
        pool(),
        ARRAY(ARRAY(CppToType<T>::create())),
        nulls,
        size,
        offsets,
        sizes,
        baseArray,
        0);
  }

  // Create an ArrayVector<MapVector<TKey, TValue>> from nested std::vectors of
  // pairs. Example:
  //   using S = StringView;
  //   using P = std::pair<int64_t, std::optional<S>>;
  //   std::vector<P> a {P{1, S{"red"}}, P{2, S{"blue"}}, P{3, S{"green"}}};
  //   std::vector<P> b {P{1, S{"yellow"}}, P{2, S{"orange"}}};
  //   std::vector<P> c {P{1, S{"red"}}, P{2, S{"yellow"}}, P{3, S{"purple"}}};
  //   std::vector<std::vector<std::vector<P>>> data = {{a, b, b}, {b, c}, {c,
  //   a, c}};
  //   auto arrayVector = makeArrayOfMapVector<int64_t, S>(data);
  //
  //   EXPECT_EQ(3, arrayVector->size());
  template <typename TKey, typename TValue>
  ArrayVectorPtr makeArrayOfMapVector(
      const std::vector<
          std::vector<std::vector<std::pair<TKey, std::optional<TValue>>>>>&
          data) {
    vector_size_t size = data.size();
    BufferPtr offsets = AlignedBuffer::allocate<vector_size_t>(size, pool());
    BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(size, pool());

    auto rawOffsets = offsets->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();

    // Flatten the outermost std::vector of the input and populate the sizes and
    // offsets for the top-level array vector.
    std::vector<std::vector<std::pair<TKey, std::optional<TValue>>>>
        flattenedData;
    vector_size_t i = 0;
    for (const auto& vector : data) {
      flattenedData.insert(flattenedData.end(), vector.begin(), vector.end());

      rawSizes[i] = vector.size();
      rawOffsets[i] = (i == 0) ? 0 : rawOffsets[i - 1] + rawSizes[i - 1];

      ++i;
    }

    // Create the underlying map vector.
    auto baseVector = makeMapVector<TKey, TValue>(flattenedData);

    // Build and return a second-level of array vector on top of baseVector.
    return std::make_shared<ArrayVector>(
        pool(),
        ARRAY(MAP(CppToType<TKey>::create(), CppToType<TValue>::create())),
        BufferPtr(nullptr),
        size,
        offsets,
        sizes,
        baseVector,
        0);
  }

  template <typename TKey, typename TValue>
  ArrayVectorPtr makeArrayOfMapVector(
      const std::vector<std::vector<
          std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>>&
          data) {
    vector_size_t size = data.size();
    BufferPtr offsets = AlignedBuffer::allocate<vector_size_t>(size, pool());
    BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(size, pool());

    auto rawOffsets = offsets->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();

    // Flatten the outermost std::vector of the input and populate the sizes and
    // offsets for the top-level array vector.
    std::vector<
        std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>
        flattenedData;
    vector_size_t i = 0;
    for (const auto& vector : data) {
      flattenedData.insert(flattenedData.end(), vector.begin(), vector.end());
      rawSizes[i] = vector.size();
      rawOffsets[i] = (i == 0) ? 0 : rawOffsets[i - 1] + rawSizes[i - 1];
      ++i;
    }

    // Create the underlying map vector.
    auto baseVector = makeNullableMapVector<TKey, TValue>(flattenedData);

    // Build and return a second-level of array vector on top of baseVector.
    return std::make_shared<ArrayVector>(
        pool(),
        ARRAY(MAP(CppToType<TKey>::create(), CppToType<TValue>::create())),
        BufferPtr(nullptr),
        size,
        offsets,
        sizes,
        baseVector,
        0);
  }

  // Convenience function to create arrayVectors (vector of arrays) based on
  // input values from nested std::vectors. The underlying array elements are
  // nullable.
  //
  // Example:
  //   auto arrayVector = makeNullableArrayVector<int64_t>({
  //       {1, 2, std::nullopt, 4},
  //       {},
  //       {std::nullopt},
  //   });
  //   EXPECT_EQ(3, arrayVector->size());
  template <typename T>
  ArrayVectorPtr makeNullableArrayVector(
      const std::vector<std::vector<std::optional<T>>>& data,
      const TypePtr& type = ARRAY(CppToType<T>::create())) {
    std::vector<std::optional<std::vector<std::optional<T>>>> convData;
    convData.reserve(data.size());
    for (auto& array : data) {
      convData.push_back(array);
    }
    return vectorMaker_.arrayVectorNullable<T>(convData, type);
  }

  // Just like makeNullableArrayVector above, but allows specifying null arrays.
  //
  //  Example:
  //   auto arrayVector = makeNullableArrayVector<int64_t>({
  //       {{1, 2, std::nullopt, 4}},
  //       {{}},
  //       std::nullopt,
  //   });
  template <typename T>
  ArrayVectorPtr makeNullableArrayVector(
      const std::vector<std::optional<std::vector<std::optional<T>>>>& data) {
    return vectorMaker_.arrayVectorNullable<T>(data);
  }

  template <typename T>
  ArrayVectorPtr makeArrayVector(
      vector_size_t size,
      std::function<vector_size_t(vector_size_t /* row */)> sizeAt,
      std::function<T(vector_size_t /* idx */)> valueAt,
      std::function<bool(vector_size_t /* row */)> isNullAt = nullptr,
      std::function<bool(vector_size_t /* idx */)> valueIsNullAt = nullptr) {
    return vectorMaker_.arrayVector<T>(
        size, sizeAt, valueAt, isNullAt, valueIsNullAt);
  }

  template <typename T>
  ArrayVectorPtr makeArrayVector(
      vector_size_t size,
      std::function<vector_size_t(vector_size_t /* row */)> sizeAt,
      std::function<T(vector_size_t /* row */, vector_size_t /* idx */)>
          valueAt,
      std::function<bool(vector_size_t /*row */)> isNullAt = nullptr) {
    return vectorMaker_.arrayVector<T>(size, sizeAt, valueAt, isNullAt);
  }

  // Convenience function to create vector from a base vector.
  // The size of the arrays is computed from the difference of offsets.
  // An optional vector of nulls can be passed to specify null rows.
  // The offset for a null value must match previous offset
  // i.e size computed should be zero.
  // Example:
  //   auto arrayVector = makeArrayVector({0, 2 ,2}, elementVector, {1});
  //
  //   creates an array vector with array at index 1 as null.
  // You can make higher order ArrayVectors (i.e array of arrays etc), by
  // repeatedly calling this function and passing in resultant ArrayVector
  // and appropriate offsets.
  ArrayVectorPtr makeArrayVector(
      const std::vector<vector_size_t>& offsets,
      const VectorPtr& elementVector,
      const std::vector<vector_size_t>& nulls = {}) {
    return vectorMaker_.arrayVector(offsets, elementVector, nulls);
  }

  template <typename TKey, typename TValue>
  MapVectorPtr makeMapVector(
      vector_size_t size,
      std::function<vector_size_t(vector_size_t /* row */)> sizeAt,
      std::function<TKey(vector_size_t /* idx */)> keyAt,
      std::function<TValue(vector_size_t /* idx */)> valueAt,
      std::function<bool(vector_size_t /*row */)> isNullAt = nullptr,
      std::function<bool(vector_size_t /*row */)> valueIsNullAt = nullptr,
      const TypePtr& type =
          MAP(CppToType<TKey>::create(), CppToType<TValue>::create())) {
    return vectorMaker_.mapVector<TKey, TValue>(
        size, sizeAt, keyAt, valueAt, isNullAt, valueIsNullAt, type);
  }

  // Create map vector from nested std::vector representation.
  template <typename TKey, typename TValue>
  MapVectorPtr makeMapVector(
      const std::vector<std::vector<std::pair<TKey, std::optional<TValue>>>>&
          maps,
      const TypePtr& type =
          MAP(CppToType<TKey>::create(), CppToType<TValue>::create())) {
    std::vector<vector_size_t> lengths;
    std::vector<TKey> keys;
    std::vector<TValue> values;
    std::vector<bool> nullValues;
    auto undefined = TValue();

    for (const auto& map : maps) {
      lengths.push_back(map.size());
      for (const auto& [key, value] : map) {
        keys.push_back(key);
        values.push_back(value.value_or(undefined));
        nullValues.push_back(!value.has_value());
      }
    }

    return makeMapVector<TKey, TValue>(
        maps.size(),
        [&](vector_size_t row) { return lengths[row]; },
        [&](vector_size_t idx) { return keys[idx]; },
        [&](vector_size_t idx) { return values[idx]; },
        nullptr,
        [&](vector_size_t idx) { return nullValues[idx]; },
        type);
  }

  // Create nullabe map vector from nested std::vector representation.
  template <typename TKey, typename TValue>
  MapVectorPtr makeNullableMapVector(
      const std::vector<
          std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>&
          maps) {
    std::vector<vector_size_t> lengths;
    std::vector<TKey> keys;
    std::vector<TValue> values;
    std::vector<bool> nullValues;
    std::vector<bool> nullRow;

    auto undefined = TValue();

    for (const auto& map : maps) {
      if (!map.has_value()) {
        nullRow.push_back(true);
        lengths.push_back(0);
        continue;
      }
      nullRow.push_back(false);
      lengths.push_back(map->size());
      for (const auto& [key, value] : map.value()) {
        keys.push_back(key);
        values.push_back(value.value_or(undefined));
        nullValues.push_back(!value.has_value());
      }
    }

    return makeMapVector<TKey, TValue>(
        maps.size(),
        [&](vector_size_t row) { return lengths[row]; },
        [&](vector_size_t idx) { return keys[idx]; },
        [&](vector_size_t idx) { return values[idx]; },
        [&](vector_size_t row) { return nullRow[row]; },
        [&](vector_size_t idx) { return nullValues[idx]; });
  }

  // Convenience function to create vector from vectors of keys and values.
  // The size of the maps is computed from the difference of offsets.
  // The sizes of the key and value vectors must be equal.
  // An optional vector of nulls can be passed to specify null rows.
  // The offset for a null value must match previous offset
  // i.e size computed should be zero.
  // Example:
  //   auto mapVector = makeMapVector({0, 2 ,2}, keyVector, valueVector, {1});
  //
  //   creates a map vector with map at index 1 as null.
  // You can make higher order MapVectors (i.e maps with maps as values etc),
  // by repeatedly calling this function and passing in resultant MapVector
  // and appropriate offsets.
  MapVectorPtr makeMapVector(
      const std::vector<vector_size_t>& offsets,
      const VectorPtr& keyVector,
      const VectorPtr& valueVector,
      const std::vector<vector_size_t>& nulls = {}) {
    return vectorMaker_.mapVector(offsets, keyVector, valueVector, nulls);
  }

  MapVectorPtr makeAllNullMapVector(
      vector_size_t size,
      const TypePtr& keyType,
      const TypePtr& valueType) {
    return vectorMaker_.allNullMapVector(size, keyType, valueType);
  }

  VectorPtr makeConstant(const variant& value, vector_size_t size) {
    return BaseVector::createConstant(value.inferType(), value, size, pool());
  }

  template <typename T>
  VectorPtr makeConstant(
      T value,
      vector_size_t size,
      const TypePtr& type = CppToType<EvalType<T>>::create()) {
    return std::make_shared<ConstantVector<EvalType<T>>>(
        pool(), size, false, type, std::move(value));
  }

  template <typename T>
  VectorPtr makeConstant(
      const std::optional<T>& value,
      vector_size_t size,
      const TypePtr& type = CppToType<EvalType<T>>::create()) {
    return std::make_shared<ConstantVector<EvalType<T>>>(
        pool(),
        size,
        /*isNull=*/!value.has_value(),
        type,
        value ? EvalType<T>(*value) : EvalType<T>(),
        SimpleVectorStats<EvalType<T>>{},
        sizeof(EvalType<T>));
  }

  /// Create constant vector of type ROW from a Variant.
  VectorPtr makeConstantRow(
      const RowTypePtr& rowType,
      variant value,
      vector_size_t size) {
    return vectorMaker_.constantRow(rowType, value, size);
  }

  /// Create constant vector of type ARRAY from a std::vector.
  template <typename T>
  VectorPtr makeConstantArray(vector_size_t size, const std::vector<T>& data) {
    return BaseVector::wrapInConstant(
        size, 0, vectorMaker_.arrayVector<T>({data}));
  }

  VectorPtr makeNullConstant(TypeKind typeKind, vector_size_t size) {
    return BaseVector::createNullConstant(
        createType(typeKind, {}), size, pool());
  }

  BufferPtr makeIndices(
      vector_size_t size,
      std::function<vector_size_t(vector_size_t)> indexAt) const;

  BufferPtr makeIndices(const std::vector<vector_size_t>& indices) const;

  BufferPtr makeOddIndices(vector_size_t size);

  BufferPtr makeEvenIndices(vector_size_t size);

  BufferPtr makeIndicesInReverse(vector_size_t size) {
    return ::facebook::velox::test::makeIndicesInReverse(size, pool());
  }

  BufferPtr makeNulls(
      vector_size_t size,
      std::function<bool(vector_size_t /*row*/)> isNullAt);

  /// Creates a null buffer from a vector of booleans.
  BufferPtr makeNulls(const std::vector<bool>& values);

  static VectorPtr
  wrapInDictionary(BufferPtr indices, vector_size_t size, VectorPtr vector);

  static VectorPtr wrapInDictionary(BufferPtr indices, VectorPtr vector);

  template <typename T = BaseVector>
  static std::shared_ptr<T> flatten(const VectorPtr& vector) {
    return velox::test::VectorMaker::flatten<T>(vector);
  }

  // Convenience function to create a vector of type Map(K, ARRAY(K)).
  // Supports null keys, and values and even null elements.
  // Example:
  //    createMapOfArraysVector<int64_t>(
  //      {{{1, std::nullopt}},
  //       {{2, {{4, 5, std::nullopt}}}},
  //       {{std::nullopt, {{7, 8, 9}}}}});
  template <typename K, typename V>
  VectorPtr createMapOfArraysVector(
      std::vector<std::map<
          std::optional<K>,
          std::optional<std::vector<std::optional<V>>>>> maps) {
    std::vector<std::optional<K>> keys;
    std::vector<std::optional<std::vector<std::optional<V>>>> values;
    for (auto& map : maps) {
      for (const auto& [key, value] : map) {
        keys.push_back(key);
        values.push_back(value);
      }
    }

    auto mapValues = makeNullableArrayVector(values);
    auto mapKeys = makeNullableFlatVector<K>(keys);
    auto size = maps.size();

    auto offsets = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
    auto sizes = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());

    auto rawOffsets = offsets->template asMutable<vector_size_t>();
    auto rawSizes = sizes->template asMutable<vector_size_t>();

    vector_size_t offset = 0;
    for (vector_size_t i = 0; i < size; i++) {
      rawSizes[i] = maps[i].size();
      rawOffsets[i] = offset;
      offset += maps[i].size();
    }

    return std::make_shared<MapVector>(
        pool_.get(),
        MAP(CppToType<K>::create(), ARRAY(CppToType<V>::create())),
        nullptr,
        size,
        offsets,
        sizes,
        mapKeys,
        mapValues);
  }

  memory::MemoryPool* pool() const {
    return pool_.get();
  }

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::getProcessDefaultMemoryManager().getPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addChild("leaf")};
  velox::test::VectorMaker vectorMaker_{pool_.get()};
  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
};

} // namespace facebook::velox::test

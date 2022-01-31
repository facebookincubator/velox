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

#include <optional>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "velox/common/base/Exceptions.h"
#include "velox/expression/VectorUdfTypeSystem.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace {
using namespace facebook::velox;

DecodedVector* decode(DecodedVector& decoder, const BaseVector& vector) {
  SelectivityVector rows(vector.size());
  decoder.decode(vector, rows);
  return &decoder;
}

template <bool returnsOptionalValues>
class MapViewTest : public functions::test::FunctionBaseTest {
  using ViewType = exec::MapView<returnsOptionalValues, int64_t, int64_t>;
  using ReadFunction = std::function<
      ViewType(exec::VectorReader<Map<int64_t, int64_t>>&, size_t)>;

 protected:
  using map_type = std::vector<std::pair<int64_t, std::optional<int64_t>>>;

  // What value to use for NULL in the test data.  If the view type is
  // not returnsOptionalValues, we use 0 as an arbitrary value.
  std::optional<int64_t> nullValue =
      returnsOptionalValues ? std::nullopt : std::make_optional(0);
  map_type map1 = {};
  map_type map2 = {{1, 4}, {3, 3}, {4, nullValue}};
  map_type map3 = {
      {10, 10},
      {4, nullValue},
      {1, 4},
      {10, 4},
      {10, nullValue},
  };

  std::vector<map_type> mapsData = {map1, map2, map3};

  MapVectorPtr createTestMapVector() {
    return makeMapVector<int64_t, int64_t>(mapsData);
  }

  ViewType read(
      exec::VectorReader<Map<int64_t, int64_t>>& reader,
      size_t offset) {
    if constexpr (returnsOptionalValues) {
      return reader[offset];
    } else {
      return reader.readNullFree(offset);
    }
  }

  bool mapValueHasValue(typename ViewType::ValueAccessor value) {
    if constexpr (returnsOptionalValues) {
      return value.has_value();
    } else {
      return true;
    }
  }

  int64_t mapValueValue(typename ViewType::ValueAccessor value) {
    if constexpr (returnsOptionalValues) {
      return value.value();
    } else {
      return value;
    }
  }

  void readingRangeLoopTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    for (auto i = 0; i < mapsData.size(); i++) {
      auto mapView = read(reader, i);
      auto it = mapsData[i].begin();
      int count = 0;
      ASSERT_EQ(mapsData[i].size(), mapView.size());
      for (const auto& entry : mapView) {
        ASSERT_EQ(entry.first, it->first);
        ASSERT_EQ(mapValueHasValue(entry.second), it->second.has_value());
        if (it->second.has_value()) {
          ASSERT_EQ(mapValueValue(entry.second), it->second.value());
        }
        ASSERT_EQ(entry.second, it->second);
        it++;
        count++;
      }
      ASSERT_EQ(count, mapsData[i].size());
    }
  }

  void readingIteratorLoopTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    for (auto i = 0; i < mapsData.size(); ++i) {
      auto mapView = read(reader, i);
      auto it = mapsData[i].begin();
      int count = 0;
      ASSERT_EQ(mapsData[i].size(), mapView.size());
      for (auto itView = mapView.begin(); itView != mapView.end(); ++itView) {
        ASSERT_EQ(itView->first, it->first);
        ASSERT_EQ(mapValueHasValue(itView->second), it->second.has_value());
        if (it->second.has_value()) {
          ASSERT_EQ(mapValueValue(itView->second), it->second.value());
        }
        ASSERT_EQ(itView->second, it->second);
        it++;
        count++;
      }
      ASSERT_EQ(count, mapsData[i].size());
    }
  }

  // MapView can be seen as std::vector<pair<key, value>>.
  void indexedLoopTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    for (auto i = 0; i < mapsData.size(); ++i) {
      auto mapView = read(reader, i);
      auto it = mapsData[i].begin();
      int count = 0;
      ASSERT_EQ(mapsData[i].size(), mapView.size());
      for (int j = 0; j < mapView.size(); j++) {
        ASSERT_EQ(mapView[j].first, it->first);
        ASSERT_EQ(mapValueHasValue(mapView[j].second), it->second.has_value());
        if (it->second.has_value()) {
          ASSERT_EQ(mapValueValue(mapView[j].second), it->second.value());
        }
        ASSERT_EQ(mapView[j].second, it->second);
        it++;
        count++;
      }
      ASSERT_EQ(count, mapsData[i].size());
    }
  }

  void encodedTest() {
    VectorPtr mapVector = createTestMapVector();
    // Wrap in dictionary.
    auto vectorSize = mapVector->size();
    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(vectorSize, pool_.get());
    auto rawIndices = indices->asMutable<vector_size_t>();
    // Assign indices such that array is reversed.
    for (size_t i = 0; i < vectorSize; ++i) {
      rawIndices[i] = vectorSize - 1 - i;
    }
    mapVector = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices, vectorSize, mapVector);

    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector));

    ASSERT_EQ(read(reader, 0).size(), 5);
    ASSERT_EQ(read(reader, 1).size(), 3);
    ASSERT_EQ(read(reader, 2).size(), 0);
  }

  void compareLazyValueAccessTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    // Compare LazyValueAccess with constant.
    ASSERT_EQ(read(reader, 1)[0].first, 1);
    ASSERT_NE(read(reader, 1)[0].first, 10);
    ASSERT_EQ(1, read(reader, 1)[0].first);
    ASSERT_NE(10, read(reader, 1)[0].first);

    // Compare LazyValueAccess with LazyValueAccess.
    ASSERT_EQ(read(reader, 2)[2].first, read(reader, 1)[0].first);
    ASSERT_NE(read(reader, 2)[2].first, read(reader, 1)[1].first);

    // Compare LazyValueAccess with VectorOptionalValueAccessor value.
    ASSERT_EQ(
        read(reader, 2)[1].first, mapValueValue(read(reader, 1)[0].second));
    ASSERT_NE(
        read(reader, 2)[2].first, mapValueValue(read(reader, 1)[1].second));
    ASSERT_EQ(
        mapValueValue(read(reader, 1)[0].second), read(reader, 2)[1].first);
    ASSERT_NE(
        mapValueValue(read(reader, 1)[1].second), read(reader, 2)[2].first);

    // Compare null VectorOptionalValueAccessor with LazyValueAccess.
    ASSERT_NE(
        mapValueValue(read(reader, 1)[1].second), read(reader, 1)[2].first);
  }

  void compareVectorOptionalValueAccessorTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    // Compare VectorOptionalValueAccessor with std::optional.
    ASSERT_EQ(read(reader, 2)[2].second, std::optional(4));
    ASSERT_EQ(read(reader, 2)[2].second, std::optional(4l));
    ASSERT_EQ(read(reader, 2)[2].second, std::optional(4ll));
    ASSERT_EQ(read(reader, 2)[2].second, std::optional(4.0F));

    ASSERT_NE(read(reader, 2)[2].second, std::optional(4.01F));
    ASSERT_NE(read(reader, 2)[2].second, std::optional(8));

    ASSERT_EQ(std::optional(4), read(reader, 2)[2].second);
    ASSERT_EQ(std::optional(4l), read(reader, 2)[2].second);
    ASSERT_EQ(std::optional(4ll), read(reader, 2)[2].second);

    ASSERT_NE(std::optional(4.01F), read(reader, 2)[2].second);

    if constexpr (returnsOptionalValues) {
      ASSERT_EQ(std::nullopt, read(reader, 1)[2].second);
      ASSERT_NE(std::nullopt, read(reader, 1)[1].second);

      std::optional<int64_t> nullOpt;
      ASSERT_EQ(read(reader, 1)[2].second, std::nullopt);
      ASSERT_NE(read(reader, 1)[1].second, std::nullopt);

      ASSERT_EQ(read(reader, 1)[2].second, nullOpt);
      ASSERT_NE(read(reader, 1)[1].second, nullOpt);
    }

    // Compare VectorOptionalValueAccessor<T> with T::exec_t.
    ASSERT_EQ(read(reader, 2)[2].second, 4);
    ASSERT_EQ(read(reader, 2)[2].second, 4l);
    ASSERT_EQ(read(reader, 2)[2].second, 4ll);
    ASSERT_EQ(read(reader, 2)[2].second, 4.0F);

    ASSERT_NE(read(reader, 2)[2].second, 4.01F);
    ASSERT_NE(read(reader, 2)[2].second, 8);

    ASSERT_EQ(4, read(reader, 2)[2].second);
    ASSERT_EQ(4l, read(reader, 2)[2].second);
    ASSERT_EQ(4ll, read(reader, 2)[2].second);
    ASSERT_NE(4.01F, read(reader, 2)[2].second);

    // VectorOptionalValueAccessor is null here.
    ASSERT_NE(4.01F, read(reader, 1)[2].second);
    ASSERT_NE(read(reader, 1)[2].second, 4);

    // Compare VectorOptionalValueAccessor with VectorOptionalValueAccessor.
    ASSERT_EQ(read(reader, 2)[2].second, read(reader, 2)[3].second);
    ASSERT_NE(read(reader, 2)[2].second, read(reader, 2)[0].second);

    // Compare with empty VectorOptionalValueAccessor.
    // One null and one not null.
    ASSERT_NE(read(reader, 1)[1].second, read(reader, 1)[2].second);
    ASSERT_NE(read(reader, 1)[2].second, read(reader, 1)[1].second);
    // Both are null.
    ASSERT_EQ(read(reader, 2)[1].second, read(reader, 1)[2].second);
  }

  void compareMapViewElementTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    // Compare VectorOptionalValueAccessor with constant.
    ASSERT_NE(read(reader, 2)[2], read(reader, 2)[1]);
    ASSERT_EQ(read(reader, 1)[0], read(reader, 2)[2]);
  }

  void assignToOptionalTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    std::optional<int64_t> element = read(reader, 2)[2].second;
    std::optional<int64_t> element2 = read(reader, 2)[1].second;
    ASSERT_EQ(element, read(reader, 2)[2].second);
    ASSERT_EQ(element2, read(reader, 2)[1].second);
    ASSERT_NE(element2, element);
  }

  void findTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    ASSERT_EQ(read(reader, 1).find(5), read(reader, 1).end());
    ASSERT_NE(read(reader, 1).find(4), read(reader, 1).end());
    ASSERT_EQ(read(reader, 1).find(4)->first, 4);

    ASSERT_EQ(read(reader, 1).find(4)->second, nullValue);
  }

  void atTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    ASSERT_THROW(read(reader, 1).at(5), VeloxException);

    ASSERT_EQ(read(reader, 1).at(4), nullValue);

    ASSERT_EQ(read(reader, 1).at(3), 3);
  }

  void readingStructureBindingLoopTest() {
    auto mapVector = createTestMapVector();
    DecodedVector decoded;
    exec::VectorReader<Map<int64_t, int64_t>> reader(
        decode(decoded, *mapVector.get()));

    for (auto i = 0; i < mapsData.size(); i++) {
      auto mapView = read(reader, i);
      auto it = mapsData[i].begin();
      int count = 0;
      ASSERT_EQ(mapsData[i].size(), mapView.size());
      for (const auto& [key, value] : mapView) {
        ASSERT_EQ(key, it->first);
        ASSERT_EQ(mapValueHasValue(value), it->second.has_value());
        if (it->second.has_value()) {
          ASSERT_EQ(mapValueValue(value), it->second.value());
        }
        ASSERT_EQ(value, it->second);
        it++;
        count++;
      }
      ASSERT_EQ(count, mapsData[i].size());
    }
  }
};

class NullableMapViewTest : public MapViewTest<true> {};

class NullFreeMapViewTest : public MapViewTest<false> {};

TEST_F(NullableMapViewTest, testReadingRangeLoop) {
  readingRangeLoopTest();
}

TEST_F(NullableMapViewTest, testReadingIteratorLoop) {
  readingIteratorLoopTest();
}

TEST_F(NullableMapViewTest, testIndexedLoop) {
  indexedLoopTest();
}

TEST_F(NullableMapViewTest, encoded) {
  encodedTest();
}

TEST_F(NullableMapViewTest, testCompareLazyValueAccess) {
  compareLazyValueAccessTest();
}

TEST_F(NullableMapViewTest, testCompareVectorOptionalValueAccessor) {
  compareVectorOptionalValueAccessorTest();
}

TEST_F(NullableMapViewTest, testCompareMapViewElement) {
  compareMapViewElementTest();
}

TEST_F(NullableMapViewTest, testAssignToOptional) {
  assignToOptionalTest();
}

TEST_F(NullableMapViewTest, testFind) {
  findTest();
}

TEST_F(NullableMapViewTest, testAt) {
  atTest();
}

TEST_F(NullableMapViewTest, testValueOr) {
  auto mapVector = createTestMapVector();
  DecodedVector decoded;
  exec::VectorReader<Map<int64_t, int64_t>> reader(
      decode(decoded, *mapVector.get()));

  ASSERT_EQ(reader[1].at(4).value_or(10), 10);
  ASSERT_EQ(reader[1].at(3).value_or(10), 3);
}

// Function that takes a map from array of doubles to integer as input.
template <typename T>
struct MapComplexKeyF {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      double& out,
      const arg_type<Map<Array<double>, double>>& input) {
    out = 0;
    for (const auto& entry : input) {
      for (auto v : entry.first) {
        out += v.value();
      }
    }

    double outTest = 0;
    // Test operator-> on MapView::Iterator.
    for (auto it = input.begin(); it != input.end(); it++) {
      auto keyArray = it->first;
      for (auto v : keyArray) {
        outTest += v.value();
      }
    }

    EXPECT_EQ(out, outTest);
    return true;
  }
};

TEST_F(NullableMapViewTest, mapCoplexKey) {
  registerFunction<MapComplexKeyF, double, Map<Array<double>, double>>(
      {"func"});

  const vector_size_t size = 10;
  auto values1 = makeArrayVector<double>(
      size,
      [](auto /*row*/) { return 10; },
      [](auto row, auto /*index*/) { return row; });

  auto values2 = makeArrayVector<double>(
      size,
      [](auto /*row*/) { return 1; },
      [](auto /*row*/, auto index) { return 1.2 * index; });

  auto result = evaluate<FlatVector<double>>(
      "func(map(array_constructor(c0), c1))",
      makeRowVector({values1, values2}));

  auto expected =
      makeFlatVector<double>(size, [](auto row) { return row * 10; });

  ASSERT_EQ(size, result->size());
  for (auto i = 0; i < size; i++) {
    EXPECT_NEAR(expected->valueAt(i), result->valueAt(i), 0.0000001);
  }
}

TEST_F(NullableMapViewTest, testReadingStructureBindingLoop) {
  readingStructureBindingLoopTest();
}

TEST_F(NullFreeMapViewTest, testReadingRangeLoop) {
  readingRangeLoopTest();
}

TEST_F(NullFreeMapViewTest, testReadingIteratorLoop) {
  readingIteratorLoopTest();
}

TEST_F(NullFreeMapViewTest, testIndexedLoop) {
  indexedLoopTest();
}

TEST_F(NullFreeMapViewTest, encoded) {
  encodedTest();
}

TEST_F(NullFreeMapViewTest, testCompareLazyValueAccess) {
  compareLazyValueAccessTest();
}

TEST_F(NullFreeMapViewTest, testCompareVectorOptionalValueAccessor) {
  compareVectorOptionalValueAccessorTest();
}

TEST_F(NullFreeMapViewTest, testCompareMapViewElement) {
  compareMapViewElementTest();
}

TEST_F(NullFreeMapViewTest, testAssignToOptional) {
  assignToOptionalTest();
}

TEST_F(NullFreeMapViewTest, testFind) {
  findTest();
}

TEST_F(NullFreeMapViewTest, testAt) {
  atTest();
}

TEST_F(NullFreeMapViewTest, testReadingStructureBindingLoop) {
  readingStructureBindingLoopTest();
}

} // namespace

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
#include "velox/functions/prestosql/aggregates/PrestoHasher.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"
namespace facebook::velox::aggregate::test {
template <typename T>
using limits = std::numeric_limits<T>;

class PrestoHasherTest : public AggregationTestBase {
 protected:
  template <typename T>
  void assertHash(
      const std::vector<std::optional<T>>& data,
      const std::vector<int64_t>& expected) {
    auto vector = makeNullableFlatVector<T>(data);
    assertHash(vector, expected);
  }

  void assertHash(
      const VectorPtr& vector,
      const std::vector<int64_t>& expected) {
    auto vectorSize = vector->size();

    auto checkHashes = [&](const VectorPtr& vector,
                           const std::vector<int64_t>& expected) {
      auto vectorSize = vector->size();
      SelectivityVector rows(vectorSize);
      PrestoHasher hasher(*vector, rows);

      BufferPtr hashes =
          AlignedBuffer::allocate<int64_t>(vectorSize, pool_.get());
      hasher.hash(rows, hashes);

      auto rawHashes = hashes->as<int64_t>();
      for (int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], rawHashes[i]) << "at " << i;
      }
    };

    checkHashes(vector, expected);

    // Wrap in Dictionary
    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(vectorSize, pool_.get());
    auto rawIndices = indices->asMutable<vector_size_t>();
    // Just 1:1 mapping.
    for (size_t i = 0; i < vectorSize; ++i) {
      rawIndices[i] = i;
    }
    auto dictionaryVector = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices, vectorSize, vector);

    checkHashes(dictionaryVector, expected);

    // Wrap in constant.
    std::vector<int64_t> constantExpected{0};
    for (int i = 0; i < vectorSize; i++) {
      constantExpected[0] = expected[i];
      auto constantVector = BaseVector::wrapInConstant(1, i, vector);
      checkHashes(constantVector, constantExpected);
    }
  }

  template <typename T>
  void testIntegral() {
    assertHash<T>(
        {1, 0, 100, std::nullopt},
        {9155312661752487122, 0, -367469560765523433, 0});
  }

  VectorPtr makeMapOfArraysVector(
      std::map<
          std::optional<int64_t>,
          std::optional<std::vector<std::optional<int64_t>>>> map) {
    std::vector<std::optional<int64_t>> keys;
    std::vector<std::optional<std::vector<std::optional<int64_t>>>> values;
    for (const auto& [key, vals] : map) {
      keys.push_back(key);
      values.push_back(vals);
    }

    auto arrayValues = vectorMaker_.arrayVectorNullable(values);

    auto mapKeys = makeNullableFlatVector<int64_t>(keys);
    auto size = mapKeys->size();
    auto offsets = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
    auto sizes = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());

    auto rawOffsets = offsets->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();

    // Create a vector of maps with 1 key , 1 value
    vector_size_t offset = 0;
    for (vector_size_t i = 0; i < size; i++) {
      rawSizes[i] = 1;
      rawOffsets[i] = offset;
      offset += 1;
    }

    return std::make_shared<MapVector>(
        pool_.get(),
        MAP(BIGINT(), ARRAY(BIGINT())),
        nullptr,
        size,
        offsets,
        sizes,
        mapKeys,
        arrayValues);
  }

  std::optional<std::vector<std::optional<int64_t>>> O(
      std::vector<std::optional<int64_t>> data) {
    return std::make_optional(data);
  };
};

TEST_F(PrestoHasherTest, ints) {
  testIntegral<int8_t>();
  testIntegral<int16_t>();
  testIntegral<int32_t>();
  testIntegral<int64_t>();
  assertHash<int32_t>({99999, std::nullopt}, {-6111805494244633133, 0});
  assertHash<int64_t>({99999, std::nullopt}, {-6111805494244633133, 0});
  assertHash<int64_t>(
      {limits<int64_t>::max(), limits<int64_t>::min()},
      {516552589260593319, 7024193345362591744});
  assertHash<int32_t>(
      {limits<int32_t>::max(), limits<int32_t>::min()},
      {-7833837855883860365, -3072160283202506188});
  assertHash<int16_t>(
      {limits<int16_t>::max(), limits<int16_t>::min()},
      {670463525973519066, -4530432376425028794});
  assertHash<int8_t>(
      {limits<int8_t>::max(), limits<int8_t>::min()},
      {695722463566662829, 6845023471056522234});
}

TEST_F(PrestoHasherTest, timestamp) {
  assertHash<Timestamp>(
      {Timestamp(1, 100),
       Timestamp(10, 10),
       Timestamp(100, 1000),
       Timestamp(0, 0),
       std::nullopt},
      {2343331593029422743, -3897102175227929705, 3043507167507853989, 0, 0});
}

TEST_F(PrestoHasherTest, date) {
  assertHash<Date>(
      {Date(0), Date(1000), std::nullopt}, {0, 2343331593029422743, 0});
}

TEST_F(PrestoHasherTest, doubles) {
  assertHash<double>(
      {1.0,
       0.0,
       std::nan("0"),
       99999.99,
       std::nullopt,
       limits<double>::max(),
       limits<double>::min()},
      {2156309669339463680L,
       0,
       -6389168865844396032,
       9140591727513491554,
       0,
       -7863427759443830617,
       8379980348704423936});
}

TEST_F(PrestoHasherTest, floats) {
  assertHash<float>(
      {1.0f,
       0.0f,
       std::nanf("0"),
       99999.99f,
       std::nullopt,
       limits<float>::max(),
       limits<float>::min()},
      {-6641611864725600567L,
       0,
       6018425597037803642,
       -6781949276808836923,
       0,
       -4588979133863754154,
       3801170566614750614});
}

TEST_F(PrestoHasherTest, varcharType) {
  assertHash<StringView>(
      {StringView("abcd"),
       StringView(""),
       std::nullopt,
       StringView(u8"Thanks \u0020\u007F")},
      {-2449070131962342708, -1205034819632174695, 0, 2911531567394159200});
}

TEST_F(PrestoHasherTest, bools) {
  assertHash<bool>({true, false, std::nullopt}, {1231, 1237, 0});
}

TEST_F(PrestoHasherTest, arrays) {
  auto baseArrayVector = vectorMaker_.arrayVectorNullable<int64_t>(
      {O({1, 2}),
       O({3, 4}),
       O({4, 5}),
       O({6, 7}),
       O({8, 9}),
       O({10, 11}),
       O({12, std::nullopt}),
       std::nullopt,
       O({})});

  assertHash(
      baseArrayVector,
      {4329740752828760473,
       655643799837771513,
       8633635089947142034,
       9138382565482297209,
       1065928229506940121,
       8616704993676952121,
       1942639070761256766,
       0,
       0});

  // Array of arrays.
  auto arrayOfArrayVector =
      vectorMaker_.arrayVector({0, 2, 4}, baseArrayVector);

  assertHash(
      arrayOfArrayVector,
      {5750398621562484864, 79909248200426023, -4270118586511114434});

  // Array with nulls
  auto arrayWithNulls = vectorMaker_.arrayVectorNullable<int64_t>({
      O({std::nullopt}),
      O({1, 2, 3}),
      O({1024, std::nullopt, -99, -999}),
      O({}),
      O({std::nullopt, -1}),
  });

  assertHash(
      arrayWithNulls,
      {0, -2582109863103049084, 2242047241851842487, 0, -6507640756101998425});
}

TEST_F(PrestoHasherTest, maps) {
  auto mapVector = makeMapVector<int64_t, int64_t>(
      3,
      [](auto /**/) { return 1; },
      [](auto row) { return row; },
      [](auto row) { return row + 1; });

  assertHash(
      mapVector,
      {9155312661752487122, -6461599496541202183, 5488675304642487510});

  auto mapOfArrays = makeMapOfArraysVector(
      {{1, O({1, 2, 3})}, {2, O({4, 5, 6})}, {3, O({7, 8, 9})}});
  assertHash(
      mapOfArrays,
      {-6691024575166067114, -7912800814947937532, -5636922976001735986});

  // map with nulls
  auto mapWithNullArrays = makeMapOfArraysVector(
      {{1, std::nullopt},
       {2, O({4, 5, std::nullopt})},
       {std::nullopt, O({7, 8, 9})}});
  assertHash(
      mapWithNullArrays,
      {2644717257979355699, 9155312661752487122, 6562918552317873797});
}

TEST_F(PrestoHasherTest, rows) {
  auto row = makeRowVector(
      {makeFlatVector<int64_t>({1, 3}), makeFlatVector<int64_t>({2, 4})});

  assertHash(row, {4329740752828761434, 655643799837772474});

  row = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt}),
       makeNullableFlatVector<int64_t>({std::nullopt, 4})});

  assertHash(row, {7113531408683827503, -1169223928725763049});
}

TEST_F(PrestoHasherTest, dictionary) {
  auto baseArrayVector =
      vectorMaker_.arrayVector<int64_t>({{1, 2}, {3, 4}, {4, 5}});

  auto baseSize = baseArrayVector->size();
  auto size = baseSize * 2;

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (int i = 0; i < size; i++) {
    rawIndices[i] = i % baseSize;
  }

  auto arrayDictionary = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, size, baseArrayVector);

  assertHash(
      arrayDictionary,
      {4329740752828760473,
       655643799837771513,
       8633635089947142034,
       4329740752828760473,
       655643799837771513,
       8633635089947142034});

  // Create dictionary of maps.
  auto baseMapVector = makeMapOfArraysVector(
      {{1, O({1, 2, 3})}, {2, O({4, 5, 6})}, {3, O({7, 8, 9})}});

  auto mapDictionary = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, size, baseMapVector);

  assertHash(
      mapDictionary,
      {-6691024575166067114,
       -7912800814947937532,
       -5636922976001735986,
       -6691024575166067114,
       -7912800814947937532,
       -5636922976001735986});
}

} // namespace facebook::velox::aggregate::test
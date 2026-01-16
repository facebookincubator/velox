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

#include "velox/type/Variant.h"
#include <gtest/gtest.h>
#include <numeric>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"

namespace facebook::velox::test {
namespace {

// Comprehensive tests for Variant::create() method.

template <TypeKind KIND, typename T>
void testCreate(T value) {
  T maxVal = std::numeric_limits<T>::max();
  T minVal = std::numeric_limits<T>::lowest(); // since we also test floats.

  auto maxVariant = Variant::template create<KIND>(maxVal);
  EXPECT_FALSE(maxVariant.isNull());
  EXPECT_EQ(maxVariant.kind(), KIND);
  EXPECT_EQ(maxVariant.template value<KIND>(), maxVal);

  auto minVariant = Variant::template create<KIND>(minVal);
  EXPECT_EQ(minVariant.template value<KIND>(), minVal);
  EXPECT_EQ(minVariant.template value<KIND>(), minVal);

  auto testVariant = Variant::template create<KIND>(value);
  EXPECT_EQ(testVariant.template value<KIND>(), value);
};

TEST(VariantCreateTest, primitiveTypes) {
  // Different integer precisions.
  testCreate<TypeKind::TINYINT>((int8_t)-5);
  testCreate<TypeKind::SMALLINT>((int16_t)12345);
  testCreate<TypeKind::INTEGER>((int32_t)123456789);
  testCreate<TypeKind::BIGINT>((int64_t)9876543210LL);
  testCreate<TypeKind::HUGEINT>((int128_t)1234567890123456789LL);
  testCreate<TypeKind::HUGEINT>((int128_t)0);

  // Boolean.
  testCreate<TypeKind::BOOLEAN>(true);
  testCreate<TypeKind::BOOLEAN>(false);

  // Float/double values.
  testCreate<TypeKind::REAL>((float)0.0);
  testCreate<TypeKind::REAL>((float)3.14159f);
  testCreate<TypeKind::REAL>((float)-123.456f);

  testCreate<TypeKind::DOUBLE>((double)0.0);
  testCreate<TypeKind::DOUBLE>((double)3.141592653589793);
  testCreate<TypeKind::DOUBLE>((double)-987654.321);

  // Test special float values.
  auto infVariant =
      Variant::create<TypeKind::REAL>(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isinf(infVariant.value<TypeKind::REAL>()));

  auto nanVariant =
      Variant::create<TypeKind::REAL>(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(std::isnan(nanVariant.value<TypeKind::REAL>()));

  infVariant = Variant::create<TypeKind::DOUBLE>(
      std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isinf(infVariant.value<TypeKind::DOUBLE>()));

  nanVariant = Variant::create<TypeKind::DOUBLE>(
      std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(std::isnan(nanVariant.value<TypeKind::DOUBLE>()));

  // Timestamps.
  testCreate<TypeKind::TIMESTAMP>(Timestamp(0, 0));
  testCreate<TypeKind::TIMESTAMP>(Timestamp(1234567890, 123456789));
  testCreate<TypeKind::TIMESTAMP>(Timestamp(999999, 888888));
}

TEST(VariantCreateTest, varcharType) {
  // Test VARCHAR type.
  std::string emptyStr = "";
  std::string simpleStr = "Hello, Velox!";
  std::string longStr(10000, 'a');
  std::string specialChars = "Special chars: \n\t\r !@#$%^&*()";
  std::string unicodeStr = "Unicode: 你好世界 🌍";

  auto emptyVariant = Variant::create<TypeKind::VARCHAR>(emptyStr);
  EXPECT_FALSE(emptyVariant.isNull());
  EXPECT_EQ(emptyVariant.kind(), TypeKind::VARCHAR);
  EXPECT_EQ(emptyVariant.value<TypeKind::VARCHAR>(), emptyStr);

  auto simpleVariant = Variant::create<TypeKind::VARCHAR>(simpleStr);
  EXPECT_EQ(simpleVariant.value<TypeKind::VARCHAR>(), simpleStr);

  auto longVariant = Variant::create<TypeKind::VARCHAR>(longStr);
  EXPECT_EQ(longVariant.value<TypeKind::VARCHAR>(), longStr);

  auto specialVariant = Variant::create<TypeKind::VARCHAR>(specialChars);
  EXPECT_EQ(specialVariant.value<TypeKind::VARCHAR>(), specialChars);

  auto unicodeVariant = Variant::create<TypeKind::VARCHAR>(unicodeStr);
  EXPECT_EQ(unicodeVariant.value<TypeKind::VARCHAR>(), unicodeStr);

  // Other string types.
  auto stdStringViewVariant =
      Variant::create<TypeKind::VARCHAR>(std::string_view(longStr));
  EXPECT_EQ(stdStringViewVariant.value<TypeKind::VARCHAR>(), longStr);

  auto stringViewVariant =
      Variant::create<TypeKind::VARCHAR>(StringView(longStr));
  EXPECT_EQ(stringViewVariant.value<TypeKind::VARCHAR>(), longStr);

  // Test move semantics
  std::string moveStr = "Move me!";
  auto moveVariant = Variant::create<TypeKind::VARCHAR>(std::move(moveStr));
  EXPECT_EQ(moveVariant.value<TypeKind::VARCHAR>(), "Move me!");
}

TEST(VariantCreateTest, varbinaryType) {
  // Test VARBINARY type.
  std::string emptyBinary = "";
  std::string simpleBinary = "Binary data \0 with nulls \x00\x01\x02";
  std::string largeBinary(5000, '\xFF');

  auto emptyVariant = Variant::create<TypeKind::VARBINARY>(emptyBinary);
  EXPECT_FALSE(emptyVariant.isNull());
  EXPECT_EQ(emptyVariant.kind(), TypeKind::VARBINARY);
  EXPECT_EQ(emptyVariant.value<TypeKind::VARBINARY>(), emptyBinary);

  auto simpleVariant = Variant::create<TypeKind::VARBINARY>(simpleBinary);
  EXPECT_EQ(simpleVariant.value<TypeKind::VARBINARY>(), simpleBinary);

  auto largeVariant = Variant::create<TypeKind::VARBINARY>(largeBinary);
  EXPECT_EQ(largeVariant.value<TypeKind::VARBINARY>(), largeBinary);

  // Other string types.
  auto stdStringViewVariant =
      Variant::create<TypeKind::VARBINARY>(std::string_view(simpleBinary));
  EXPECT_EQ(stdStringViewVariant.value<TypeKind::VARBINARY>(), simpleBinary);

  auto stringViewVariant =
      Variant::create<TypeKind::VARBINARY>(StringView(simpleBinary));
  EXPECT_EQ(stringViewVariant.value<TypeKind::VARBINARY>(), simpleBinary);

  // Test move semantics.
  std::string moveBinary = "Move binary!";
  auto moveVariant =
      Variant::create<TypeKind::VARBINARY>(std::move(moveBinary));
  EXPECT_EQ(moveVariant.value<TypeKind::VARBINARY>(), "Move binary!");
}

TEST(VariantCreateTest, arrayType) {
  // Empty array.
  std::vector<Variant> emptyArray;
  auto emptyVariant = Variant::create<TypeKind::ARRAY>(emptyArray);
  EXPECT_FALSE(emptyVariant.isNull());
  EXPECT_EQ(emptyVariant.kind(), TypeKind::ARRAY);
  EXPECT_TRUE(emptyVariant.value<TypeKind::ARRAY>().empty());

  // Array of integers.
  std::vector<Variant> intArray = {
      Variant(1), Variant(2), Variant(3), Variant(4)};
  auto intVariant = Variant::create<TypeKind::ARRAY>(intArray);
  const auto& resultIntArray = intVariant.value<TypeKind::ARRAY>();
  EXPECT_EQ(resultIntArray.size(), 4);
  EXPECT_EQ(resultIntArray[0].value<TypeKind::INTEGER>(), 1);
  EXPECT_EQ(resultIntArray[3].value<TypeKind::INTEGER>(), 4);

  // Array of strings.
  std::vector<Variant> strArray = {
      Variant("apple"), Variant("banana"), Variant("cherry")};
  auto strVariant = Variant::create<TypeKind::ARRAY>(strArray);
  const auto& resultStrArray = strVariant.value<TypeKind::ARRAY>();
  EXPECT_EQ(resultStrArray.size(), 3);
  EXPECT_EQ(resultStrArray[0].value<TypeKind::VARCHAR>(), "apple");

  // Array with null elements.
  std::vector<Variant> nullArray = {
      Variant(1), Variant::null(TypeKind::INTEGER), Variant(3)};
  auto nullVariant = Variant::create<TypeKind::ARRAY>(nullArray);
  const auto& resultNullArray = nullVariant.value<TypeKind::ARRAY>();
  EXPECT_EQ(resultNullArray.size(), 3);
  EXPECT_FALSE(resultNullArray[0].isNull());
  EXPECT_TRUE(resultNullArray[1].isNull());
  EXPECT_FALSE(resultNullArray[2].isNull());

  // Nested arrays.
  std::vector<Variant> nestedArray = {
      Variant::array({Variant(1), Variant(2)}),
      Variant::array({Variant(3), Variant(4), Variant(5)})};
  auto nestedVariant = Variant::create<TypeKind::ARRAY>(nestedArray);
  const auto& resultNestedArray = nestedVariant.value<TypeKind::ARRAY>();
  EXPECT_EQ(resultNestedArray.size(), 2);
  EXPECT_EQ(resultNestedArray[0].value<TypeKind::ARRAY>().size(), 2);
  EXPECT_EQ(resultNestedArray[1].value<TypeKind::ARRAY>().size(), 3);

  // Test move semantics.
  std::vector<Variant> moveArray = {Variant(100), Variant(200)};
  auto moveVariant = Variant::create<TypeKind::ARRAY>(std::move(moveArray));
  EXPECT_EQ(moveVariant.value<TypeKind::ARRAY>().size(), 2);
}

TEST(VariantCreateTest, mapType) {
  // Empty map.
  std::map<Variant, Variant> emptyMap;
  auto emptyVariant = Variant::create<TypeKind::MAP>(emptyMap);
  EXPECT_FALSE(emptyVariant.isNull());
  EXPECT_EQ(emptyVariant.kind(), TypeKind::MAP);
  EXPECT_TRUE(emptyVariant.value<TypeKind::MAP>().empty());

  // Simple map.
  std::map<Variant, Variant> simpleMap = {
      {Variant(1), Variant("one")},
      {Variant(2), Variant("two")},
      {Variant(3), Variant("three")}};
  auto simpleVariant = Variant::create<TypeKind::MAP>(simpleMap);
  const auto& resultMap = simpleVariant.value<TypeKind::MAP>();
  EXPECT_EQ(resultMap.size(), 3);
  EXPECT_EQ(
      resultMap.at(Variant(1)).value<TypeKind::VARCHAR>(), std::string("one"));
  EXPECT_EQ(
      resultMap.at(Variant(2)).value<TypeKind::VARCHAR>(), std::string("two"));

  // Map with various key types.
  std::map<Variant, Variant> stringKeyMap = {
      {Variant("key1"), Variant(100)}, {Variant("key2"), Variant(200)}};
  auto stringKeyVariant = Variant::create<TypeKind::MAP>(stringKeyMap);
  const auto& resultStringKeyMap = stringKeyVariant.value<TypeKind::MAP>();
  EXPECT_EQ(resultStringKeyMap.size(), 2);

  // Map with null values.
  std::map<Variant, Variant> nullValueMap = {
      {Variant(1), Variant(10)},
      {Variant(2), Variant::null(TypeKind::INTEGER)},
      {Variant(3), Variant(30)}};
  auto nullValueVariant = Variant::create<TypeKind::MAP>(nullValueMap);
  const auto& resultNullValueMap = nullValueVariant.value<TypeKind::MAP>();
  EXPECT_EQ(resultNullValueMap.size(), 3);
  EXPECT_FALSE(resultNullValueMap.at(Variant(1)).isNull());
  EXPECT_TRUE(resultNullValueMap.at(Variant(2)).isNull());

  // Nested map.
  std::map<Variant, Variant> nestedMap = {
      {Variant(1), Variant::map({{Variant("a"), Variant(1)}})},
      {Variant(2), Variant::map({{Variant("b"), Variant(2)}})}};
  auto nestedVariant = Variant::create<TypeKind::MAP>(nestedMap);
  const auto& resultNestedMap = nestedVariant.value<TypeKind::MAP>();
  EXPECT_EQ(resultNestedMap.size(), 2);

  // Test move semantics.
  std::map<Variant, Variant> moveMap = {{Variant(99), Variant("ninety-nine")}};
  auto moveVariant = Variant::create<TypeKind::MAP>(std::move(moveMap));
  EXPECT_EQ(moveVariant.value<TypeKind::MAP>().size(), 1);
}

TEST(VariantCreateTest, rowType) {
  // Empty row.
  std::vector<Variant> emptyRow;
  auto emptyVariant = Variant::create<TypeKind::ROW>(emptyRow);
  EXPECT_FALSE(emptyVariant.isNull());
  EXPECT_EQ(emptyVariant.kind(), TypeKind::ROW);
  EXPECT_TRUE(emptyVariant.value<TypeKind::ROW>().empty());

  // Simple row.
  std::vector<Variant> simpleRow = {Variant(1), Variant("test"), Variant(3.14)};
  auto simpleVariant = Variant::create<TypeKind::ROW>(simpleRow);
  const auto& resultRow = simpleVariant.value<TypeKind::ROW>();
  EXPECT_EQ(resultRow.size(), 3);
  EXPECT_EQ(resultRow[0].value<TypeKind::INTEGER>(), 1);
  EXPECT_EQ(resultRow[1].value<TypeKind::VARCHAR>(), "test");
  EXPECT_DOUBLE_EQ(resultRow[2].value<TypeKind::DOUBLE>(), 3.14);

  // Row with null fields
  std::vector<Variant> nullRow = {
      Variant(100),
      Variant::null(TypeKind::VARCHAR),
      Variant(200),
      Variant::null(TypeKind::DOUBLE)};
  auto nullVariant = Variant::create<TypeKind::ROW>(nullRow);
  const auto& resultNullRow = nullVariant.value<TypeKind::ROW>();
  EXPECT_EQ(resultNullRow.size(), 4);
  EXPECT_FALSE(resultNullRow[0].isNull());
  EXPECT_TRUE(resultNullRow[1].isNull());
  EXPECT_FALSE(resultNullRow[2].isNull());
  EXPECT_TRUE(resultNullRow[3].isNull());

  // Nested row.
  std::vector<Variant> nestedRow = {
      Variant(1),
      Variant::row({Variant("nested"), Variant(42)}),
      Variant(true)};
  auto nestedVariant = Variant::create<TypeKind::ROW>(nestedRow);
  const auto& resultNestedRow = nestedVariant.value<TypeKind::ROW>();
  EXPECT_EQ(resultNestedRow.size(), 3);
  EXPECT_EQ(resultNestedRow[1].value<TypeKind::ROW>().size(), 2);

  // Complex row with mixed types.
  std::vector<Variant> complexRow = {
      Variant(static_cast<int8_t>(127)),
      Variant(static_cast<int16_t>(32000)),
      Variant(static_cast<int32_t>(2000000)),
      Variant(static_cast<int64_t>(9000000000LL)),
      Variant(static_cast<int128_t>(12345)),
      Variant(1.23f),
      Variant(4.56789),
      Variant("complex string"),
      Variant(Timestamp(1000, 2000)),
      Variant::array({Variant(1), Variant(2), Variant(3)}),
      Variant::map({{Variant("key"), Variant("value")}})};
  auto complexVariant = Variant::create<TypeKind::ROW>(complexRow);
  const auto& resultComplexRow = complexVariant.value<TypeKind::ROW>();
  EXPECT_EQ(resultComplexRow.size(), 11);

  // Test move semantics.
  std::vector<Variant> moveRow = {Variant(999), Variant("move")};
  auto moveVariant = Variant::create<TypeKind::ROW>(std::move(moveRow));
  EXPECT_EQ(moveVariant.value<TypeKind::ROW>().size(), 2);
}

// Test the template <typename T> create() version. This version uses
// CppToType<T>::typeKind to deduce the TypeKind.
TEST(VariantCreateTest, templateTypeVersion) {
  // Test with bool.
  bool boolVal = true;
  auto boolVariant = Variant::create<bool>(boolVal);
  EXPECT_EQ(boolVariant.kind(), TypeKind::BOOLEAN);
  EXPECT_TRUE(boolVariant.value<TypeKind::BOOLEAN>());

  // Test with int32_t.
  int32_t intVal = 42;
  auto intVariant = Variant::create<int32_t>(intVal);
  EXPECT_EQ(intVariant.kind(), TypeKind::INTEGER);
  EXPECT_EQ(intVariant.value<TypeKind::INTEGER>(), 42);

  // Test with int64_t.
  int64_t bigintVal = 9876543210LL;
  auto bigintVariant = Variant::create<int64_t>(bigintVal);
  EXPECT_EQ(bigintVariant.kind(), TypeKind::BIGINT);
  EXPECT_EQ(bigintVariant.value<TypeKind::BIGINT>(), 9876543210LL);

  // Test with float.
  float floatVal = 3.14f;
  auto floatVariant = Variant::create<float>(floatVal);
  EXPECT_EQ(floatVariant.kind(), TypeKind::REAL);
  EXPECT_FLOAT_EQ(floatVariant.value<TypeKind::REAL>(), 3.14f);

  // Test with double.
  double doubleVal = 2.71828;
  auto doubleVariant = Variant::create<double>(doubleVal);
  EXPECT_EQ(doubleVariant.kind(), TypeKind::DOUBLE);
  EXPECT_DOUBLE_EQ(doubleVariant.value<TypeKind::DOUBLE>(), 2.71828);

  // Test with std::string (VARCHAR).
  std::string strVal = "test string";
  auto strVariant = Variant::create<std::string>(strVal);
  EXPECT_EQ(strVariant.kind(), TypeKind::VARCHAR);
  EXPECT_EQ(strVariant.value<TypeKind::VARCHAR>(), "test string");
}

TEST(VariantCreateTest, moveSemantics) {
  // Test move semantics explicitly for large objects

  // Large string.
  std::string largeString(10000, 'x');
  auto moveStringVariant =
      Variant::create<TypeKind::VARCHAR>(std::move(largeString));
  EXPECT_EQ(moveStringVariant.value<TypeKind::VARCHAR>().size(), 10000);
  // Original string should be moved-from (typically empty, but not guaranteed)

  // Large array.
  std::vector<Variant> largeArray;
  for (int i = 0; i < 1000; ++i) {
    largeArray.push_back(Variant(i));
  }
  auto moveArrayVariant =
      Variant::create<TypeKind::ARRAY>(std::move(largeArray));
  EXPECT_EQ(moveArrayVariant.value<TypeKind::ARRAY>().size(), 1000);

  // Large map.
  std::map<Variant, Variant> largeMap;
  for (int i = 0; i < 1000; ++i) {
    largeMap.emplace(Variant(i), Variant(i * 2));
  }
  auto moveMapVariant = Variant::create<TypeKind::MAP>(std::move(largeMap));
  EXPECT_EQ(moveMapVariant.value<TypeKind::MAP>().size(), 1000);

  // Large row.
  std::vector<Variant> largeRow;
  for (int i = 0; i < 5000; ++i) {
    largeRow.push_back(Variant(static_cast<int64_t>(i)));
  }
  auto moveRowVariant = Variant::create<TypeKind::ROW>(std::move(largeRow));
  EXPECT_EQ(moveRowVariant.value<TypeKind::ROW>().size(), 5000);
}

TEST(VariantCreateTest, copySemantics) {
  // Test copy semantics to ensure values are properly copied

  // Test with string
  std::string originalString = "original";
  auto copiedStringVariant = Variant::create<TypeKind::VARCHAR>(originalString);
  EXPECT_EQ(copiedStringVariant.value<TypeKind::VARCHAR>(), "original");
  EXPECT_EQ(originalString, "original"); // Original should be unchanged

  // Test with array
  std::vector<Variant> originalArray = {Variant(1), Variant(2), Variant(3)};
  auto copiedArrayVariant = Variant::create<TypeKind::ARRAY>(originalArray);
  EXPECT_EQ(copiedArrayVariant.value<TypeKind::ARRAY>().size(), 3);
  EXPECT_EQ(originalArray.size(), 3); // Original should be unchanged

  // Test with map
  std::map<Variant, Variant> originalMap = {
      {Variant(1), Variant("one")}, {Variant(2), Variant("two")}};
  auto copiedMapVariant = Variant::create<TypeKind::MAP>(originalMap);
  EXPECT_EQ(copiedMapVariant.value<TypeKind::MAP>().size(), 2);
  EXPECT_EQ(originalMap.size(), 2); // Original should be unchanged

  // Test with row
  std::vector<Variant> originalRow = {
      Variant(100), Variant("test"), Variant(3.14)};
  auto copiedRowVariant = Variant::create<TypeKind::ROW>(originalRow);
  EXPECT_EQ(copiedRowVariant.value<TypeKind::ROW>().size(), 3);
  EXPECT_EQ(originalRow.size(), 3); // Original should be unchanged
}

TEST(VariantCreateTest, equalityAfterCreate) {
  // Test that variants created via create() are equal to those created via
  // constructors/factory methods

  // BOOLEAN
  auto boolCreate = Variant::create<TypeKind::BOOLEAN>(true);
  auto boolConstructor = Variant(true);
  EXPECT_EQ(boolCreate, boolConstructor);

  // INTEGER
  auto intCreate = Variant::create<TypeKind::INTEGER>(42);
  auto intConstructor = Variant(42);
  EXPECT_EQ(intCreate, intConstructor);

  // BIGINT
  auto bigintCreate = Variant::create<TypeKind::BIGINT>(9876543210LL);
  auto bigintConstructor = Variant(9876543210LL);
  EXPECT_EQ(bigintCreate, bigintConstructor);

  // REAL
  auto realCreate = Variant::create<TypeKind::REAL>(3.14f);
  auto realConstructor = Variant(3.14f);
  EXPECT_TRUE(realCreate.equalsWithEpsilon(realConstructor));

  // DOUBLE
  auto doubleCreate = Variant::create<TypeKind::DOUBLE>(2.71828);
  auto doubleConstructor = Variant(2.71828);
  EXPECT_TRUE(doubleCreate.equalsWithEpsilon(doubleConstructor));

  // VARCHAR
  auto varcharCreate = Variant::create<TypeKind::VARCHAR>(std::string("test"));
  auto varcharConstructor = Variant("test");
  EXPECT_EQ(varcharCreate, varcharConstructor);

  // TIMESTAMP
  Timestamp ts(12345, 67890);
  auto timestampCreate = Variant::create<TypeKind::TIMESTAMP>(ts);
  auto timestampFactory = Variant::timestamp(ts);
  EXPECT_EQ(timestampCreate, timestampFactory);

  // ARRAY
  std::vector<Variant> arr = {Variant(1), Variant(2), Variant(3)};
  auto arrayCreate = Variant::create<TypeKind::ARRAY>(arr);
  auto arrayFactory = Variant::array(arr);
  EXPECT_EQ(arrayCreate, arrayFactory);

  // MAP
  std::map<Variant, Variant> mp = {
      {Variant(1), Variant("one")}, {Variant(2), Variant("two")}};
  auto mapCreate = Variant::create<TypeKind::MAP>(mp);
  auto mapFactory = Variant::map(mp);
  EXPECT_EQ(mapCreate, mapFactory);

  // ROW
  std::vector<Variant> r = {Variant(1), Variant("test"), Variant(3.14)};
  auto rowCreate = Variant::create<TypeKind::ROW>(r);
  auto rowFactory = Variant::row(r);
  EXPECT_EQ(rowCreate, rowFactory);
}

TEST(VariantCreateTest, hashConsistency) {
  // Test that variants created via create() have consistent hashes

  // Same values should have same hash
  auto v1 = Variant::create<TypeKind::INTEGER>(42);
  auto v2 = Variant::create<TypeKind::INTEGER>(42);
  EXPECT_EQ(v1.hash(), v2.hash());

  // Different values should (usually) have different hashes
  auto v3 = Variant::create<TypeKind::INTEGER>(43);
  EXPECT_NE(v1.hash(), v3.hash());

  // Test with strings
  auto s1 = Variant::create<TypeKind::VARCHAR>(std::string("hello"));
  auto s2 = Variant::create<TypeKind::VARCHAR>(std::string("hello"));
  auto s3 = Variant::create<TypeKind::VARCHAR>(std::string("world"));
  EXPECT_EQ(s1.hash(), s2.hash());
  EXPECT_NE(s1.hash(), s3.hash());

  // Test with arrays
  std::vector<Variant> arr1 = {Variant(1), Variant(2)};
  std::vector<Variant> arr2 = {Variant(1), Variant(2)};
  std::vector<Variant> arr3 = {Variant(1), Variant(3)};
  auto a1 = Variant::create<TypeKind::ARRAY>(arr1);
  auto a2 = Variant::create<TypeKind::ARRAY>(arr2);
  auto a3 = Variant::create<TypeKind::ARRAY>(arr3);
  EXPECT_EQ(a1.hash(), a2.hash());
  EXPECT_NE(a1.hash(), a3.hash());
}

TEST(VariantCreateTest, typeInference) {
  // Test that variants created via create() correctly infer their types

  auto boolVar = Variant::create<TypeKind::BOOLEAN>(true);
  EXPECT_EQ(*boolVar.inferType(), *BOOLEAN());

  auto intVar = Variant::create<TypeKind::INTEGER>(42);
  EXPECT_EQ(*intVar.inferType(), *INTEGER());

  auto bigintVar = Variant::create<TypeKind::BIGINT>(9876543210LL);
  EXPECT_EQ(*bigintVar.inferType(), *BIGINT());

  auto realVar = Variant::create<TypeKind::REAL>(3.14f);
  EXPECT_EQ(*realVar.inferType(), *REAL());

  auto doubleVar = Variant::create<TypeKind::DOUBLE>(2.71828);
  EXPECT_EQ(*doubleVar.inferType(), *DOUBLE());

  auto varcharVar = Variant::create<TypeKind::VARCHAR>(std::string("test"));
  EXPECT_EQ(*varcharVar.inferType(), *VARCHAR());

  auto timestampVar =
      Variant::create<TypeKind::TIMESTAMP>(Timestamp(12345, 67890));
  EXPECT_EQ(*timestampVar.inferType(), *TIMESTAMP());

  auto arrayVar =
      Variant::create<TypeKind::ARRAY>(std::vector<Variant>{Variant(1)});
  EXPECT_EQ(*arrayVar.inferType(), *ARRAY(INTEGER()));

  auto mapVar = Variant::create<TypeKind::MAP>(
      std::map<Variant, Variant>{{Variant(1), Variant("one")}});
  EXPECT_EQ(*mapVar.inferType(), *MAP(INTEGER(), VARCHAR()));

  auto rowVar = Variant::create<TypeKind::ROW>(
      std::vector<Variant>{Variant(1), Variant("test")});
  EXPECT_EQ(*rowVar.inferType(), *ROW({INTEGER(), VARCHAR()}));
}

TEST(VariantTest, arrayInferType) {
  EXPECT_EQ(*ARRAY(UNKNOWN()), *Variant(TypeKind::ARRAY).inferType());
  EXPECT_EQ(*ARRAY(UNKNOWN()), *Variant::array({}).inferType());
  EXPECT_EQ(*ARRAY(UNKNOWN()), *Variant::null(TypeKind::ARRAY).inferType());
  EXPECT_EQ(
      *ARRAY(BIGINT()),
      *Variant::array({Variant(TypeKind::BIGINT)}).inferType());
  EXPECT_EQ(
      *ARRAY(VARCHAR()),
      *Variant::array({Variant(TypeKind::VARCHAR)}).inferType());
  EXPECT_EQ(
      *ARRAY(ARRAY(DOUBLE())),
      *Variant::array({Variant::array({Variant(TypeKind::DOUBLE)})})
           .inferType());
  VELOX_ASSERT_THROW(
      Variant::array({Variant(123456789), Variant("velox")}),
      "All array elements must be of the same kind");
}

TEST(VariantTest, mapInferType) {
  EXPECT_EQ(*Variant::map({{1LL, 1LL}}).inferType(), *MAP(BIGINT(), BIGINT()));
  EXPECT_EQ(*Variant::map({}).inferType(), *MAP(UNKNOWN(), UNKNOWN()));
  EXPECT_EQ(
      *MAP(UNKNOWN(), UNKNOWN()), *Variant::null(TypeKind::MAP).inferType());

  const Variant nullBigint = Variant::null(TypeKind::BIGINT);
  const Variant nullReal = Variant::null(TypeKind::REAL);
  EXPECT_EQ(
      *Variant::map({{nullBigint, nullReal}, {1LL, 1.0f}}).inferType(),
      *MAP(BIGINT(), REAL()));
  EXPECT_EQ(
      *Variant::map({{nullBigint, 1.0f}, {1LL, nullReal}}).inferType(),
      *MAP(BIGINT(), REAL()));
  EXPECT_EQ(
      *Variant::map({{nullBigint, 1.0f}}).inferType(), *MAP(UNKNOWN(), REAL()));
}

TEST(VariantTest, rowInferType) {
  EXPECT_EQ(
      *ROW({BIGINT(), VARCHAR()}),
      *Variant::row({Variant(1LL), Variant("velox")}).inferType());
  EXPECT_EQ(*ROW({}), *Variant::null(TypeKind::ROW).inferType());
}

TEST(VariantTest, arrayTypeCompatibility) {
  const auto empty = Variant::array({});

  EXPECT_TRUE(empty.isTypeCompatible(ARRAY(UNKNOWN())));
  EXPECT_TRUE(empty.isTypeCompatible(ARRAY(BIGINT())));

  EXPECT_FALSE(empty.isTypeCompatible(UNKNOWN()));
  EXPECT_FALSE(empty.isTypeCompatible(BIGINT()));
  EXPECT_FALSE(empty.isTypeCompatible(MAP(INTEGER(), REAL())));

  const auto null = Variant::null(TypeKind::ARRAY);

  EXPECT_TRUE(null.isTypeCompatible(ARRAY(UNKNOWN())));
  EXPECT_TRUE(null.isTypeCompatible(ARRAY(BIGINT())));

  EXPECT_FALSE(null.isTypeCompatible(UNKNOWN()));
  EXPECT_FALSE(null.isTypeCompatible(BIGINT()));
  EXPECT_FALSE(null.isTypeCompatible(MAP(INTEGER(), REAL())));

  const auto array = Variant::array({1, 2, 3});

  EXPECT_TRUE(array.isTypeCompatible(ARRAY(INTEGER())));

  EXPECT_FALSE(array.isTypeCompatible(INTEGER()));
  EXPECT_FALSE(array.isTypeCompatible(ARRAY(REAL())));
}

TEST(VariantTest, mapTypeCompatibility) {
  const auto empty = Variant::map({});

  EXPECT_TRUE(empty.isTypeCompatible(MAP(UNKNOWN(), UNKNOWN())));
  EXPECT_TRUE(empty.isTypeCompatible(MAP(BIGINT(), BIGINT())));
  EXPECT_TRUE(empty.isTypeCompatible(MAP(REAL(), UNKNOWN())));
  EXPECT_TRUE(empty.isTypeCompatible(MAP(INTEGER(), DOUBLE())));

  EXPECT_FALSE(empty.isTypeCompatible(UNKNOWN()));
  EXPECT_FALSE(empty.isTypeCompatible(BIGINT()));
  EXPECT_FALSE(empty.isTypeCompatible(ARRAY(INTEGER())));

  const auto null = Variant::null(TypeKind::MAP);

  EXPECT_TRUE(null.isTypeCompatible(MAP(UNKNOWN(), UNKNOWN())));
  EXPECT_TRUE(null.isTypeCompatible(MAP(BIGINT(), BIGINT())));
  EXPECT_TRUE(null.isTypeCompatible(MAP(REAL(), UNKNOWN())));
  EXPECT_TRUE(null.isTypeCompatible(MAP(INTEGER(), DOUBLE())));

  EXPECT_FALSE(null.isTypeCompatible(UNKNOWN()));
  EXPECT_FALSE(null.isTypeCompatible(BIGINT()));
  EXPECT_FALSE(null.isTypeCompatible(ARRAY(INTEGER())));

  const auto map = Variant::map({{1, 1.0f}, {2, 2.0f}});

  EXPECT_TRUE(map.isTypeCompatible(MAP(INTEGER(), REAL())));

  EXPECT_FALSE(map.isTypeCompatible(MAP(INTEGER(), DOUBLE())));
  EXPECT_FALSE(map.isTypeCompatible(UNKNOWN()));
  EXPECT_FALSE(map.isTypeCompatible(ARRAY(BIGINT())));
}

TEST(VariantTest, rowTypeCompatibility) {
  const auto empty = Variant::row({});

  EXPECT_TRUE(empty.isTypeCompatible(ROW({})));

  EXPECT_FALSE(empty.isTypeCompatible(BIGINT()));
  EXPECT_FALSE(empty.isTypeCompatible(ROW({INTEGER(), REAL()})));

  const auto null = Variant::null(TypeKind::ROW);

  EXPECT_TRUE(null.isTypeCompatible(ROW({INTEGER(), REAL()})));
  EXPECT_TRUE(null.isTypeCompatible(ROW({"a", "b"}, {INTEGER(), REAL()})));

  EXPECT_FALSE(null.isTypeCompatible(BIGINT()));

  const auto row = Variant::row({1, 2.0f});

  EXPECT_TRUE(row.isTypeCompatible(ROW({INTEGER(), REAL()})));
  EXPECT_TRUE(row.isTypeCompatible(ROW({"a", "b"}, {INTEGER(), REAL()})));

  EXPECT_FALSE(row.isTypeCompatible(ROW({INTEGER()})));
  EXPECT_FALSE(row.isTypeCompatible(ROW({INTEGER(), DOUBLE()})));
  EXPECT_FALSE(row.isTypeCompatible(BIGINT()));
}

TEST(VariantTest, nullVariant) {
  // Type of null variant is UNKNOWN and should be compatible with all types.
  auto nullVariant = Variant{};
  ASSERT_TRUE(nullVariant.isTypeCompatible(VARCHAR()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(INTEGER()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(BIGINT()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(DOUBLE()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(DATE()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(TIMESTAMP()));
  ASSERT_TRUE(nullVariant.isTypeCompatible(MAP(BIGINT(), DOUBLE())));
  ASSERT_TRUE(nullVariant.isTypeCompatible(ARRAY(DATE())));
  ASSERT_TRUE(
      nullVariant.isTypeCompatible(ROW({DATE(), BIGINT(), TIMESTAMP()})));
}

/// Test Variant::equalsWithEpsilon by summing up large 64-bit integers (> 15
/// digits long) into double in different order to get slightly different
/// results due to loss of precision.
TEST(VariantTest, equalsWithEpsilonDouble) {
  std::vector<int64_t> data = {
      -6524373357247204968,
      -1459602477200235160,
      -5427507077629018454,
      -6362318851342815124,
      -6567761115475435067,
      9193194088128540374,
      -7862838580565801772,
      -7650459730033994045,
      327870505158904254,
  };

  double sum1 = std::accumulate(data.begin(), data.end(), 0.0);

  double sumEven = 0;
  double sumOdd = 0;
  for (auto i = 0; i < data.size(); i++) {
    if (i % 2 == 0) {
      sumEven += data[i];
    } else {
      sumOdd += data[i];
    }
  }

  double sum2 = sumOdd + sumEven;

  ASSERT_NE(sum1, sum2);
  ASSERT_DOUBLE_EQ(sum1, sum2);
  ASSERT_TRUE(Variant(sum1).equalsWithEpsilon(Variant(sum2)));

  // Add up all numbers but one. Make sure the result is not equal to sum1.
  double sum3 = 0;
  for (auto i = 0; i < data.size(); i++) {
    if (i != 5) {
      sum3 += data[i];
    }
  }

  ASSERT_NE(sum1, sum3);
  ASSERT_FALSE(Variant(sum1).equalsWithEpsilon(Variant(sum3)));
}

/// Similar to equalsWithEpsilonDouble, test Variant::equalsWithEpsilon by
/// summing up large 32-bit integers into float in different order to get
/// slightly different results due to loss of precision.
TEST(VariantTest, equalsWithEpsilonFloat) {
  std::vector<int32_t> data{
      -795755684,
      581869302,
      -404620562,
      -708632711,
      545404204,
      -133711905,
      -372047867,
      949333985,
      -1579004998,
      1323567403,
  };

  float sum1 = std::accumulate(data.begin(), data.end(), 0.0f);

  float sumEven = 0;
  float sumOdd = 0;
  for (auto i = 0; i < data.size(); i++) {
    if (i % 2 == 0) {
      sumEven += data[i];
    } else {
      sumOdd += data[i];
    }
  }

  float sum2 = sumOdd + sumEven;

  ASSERT_NE(sum1, sum2);
  ASSERT_FLOAT_EQ(sum1, sum2);
  ASSERT_TRUE(Variant(sum1).equalsWithEpsilon(Variant(sum2)));

  // Add up all numbers but one. Make sure the result is not equal to sum1.
  float sum3 = 0;
  for (auto i = 0; i < data.size(); i++) {
    if (i != 5) {
      sum3 += data[i];
    }
  }

  ASSERT_NE(sum1, sum3);
  ASSERT_FALSE(Variant(sum1).equalsWithEpsilon(Variant(sum3)));
}

TEST(VariantTest, mapWithNaNKey) {
  // Verify that map variants treat all NaN keys as equivalent and comparable
  // (consider them the largest) with other values.
  static const double KNan = std::numeric_limits<double>::quiet_NaN();
  auto mapType = MAP(DOUBLE(), INTEGER());
  {
    // NaN added at the start of insertions.
    std::map<Variant, Variant> mapVariant;
    mapVariant.insert({Variant(KNan), Variant(1)});
    mapVariant.insert({Variant(1.2), Variant(2)});
    mapVariant.insert({Variant(12.4), Variant(3)});
    EXPECT_EQ(
        R"([{"key":1.2,"value":2},{"key":12.4,"value":3},{"key":"NaN","value":1}])",
        Variant::map(mapVariant).toJson(mapType));
  }

  {
    // NaN added in the middle of insertions.
    std::map<Variant, Variant> mapVariant;
    mapVariant.insert({Variant(1.2), Variant(2)});
    mapVariant.insert({Variant(KNan), Variant(1)});
    mapVariant.insert({Variant(12.4), Variant(3)});
    EXPECT_EQ(
        R"([{"key":1.2,"value":2},{"key":12.4,"value":3},{"key":"NaN","value":1}])",
        Variant::map(mapVariant).toJson(mapType));
  }
}

struct Foo {};
struct Bar {};

TEST(VariantOpaqueTest, opaque) {
  auto foo = std::make_shared<Foo>();
  auto foo2 = std::make_shared<Foo>();
  auto bar = std::make_shared<Bar>();
  {
    Variant v = Variant::opaque(foo);
    EXPECT_TRUE(v.hasValue());
    EXPECT_EQ(TypeKind::OPAQUE, v.kind());
    EXPECT_EQ(foo, v.opaque<Foo>());
    VELOX_ASSERT_THROW(
        v.opaque<Bar>(),
        "Requested OPAQUE<facebook::velox::test::(anonymous namespace)::Bar> but contains OPAQUE<facebook::velox::test::(anonymous namespace)::Foo>");
    EXPECT_EQ(*v.inferType(), *OPAQUE<Foo>());
  }

  // Check that the expected shared ptrs are acquired.
  {
    EXPECT_EQ(1, foo.use_count());
    Variant v = Variant::opaque(foo);
    EXPECT_EQ(2, foo.use_count());
    Variant vv = v;
    EXPECT_EQ(3, foo.use_count());
    {
      Variant tmp = std::move(vv);
    }
    EXPECT_EQ(2, foo.use_count());
    v = 0;
    EXPECT_EQ(1, foo.use_count());
  }

  // Test opaque equality.
  {
    Variant v1 = Variant::opaque(foo);
    Variant vv1 = Variant::opaque(foo);
    Variant v2 = Variant::opaque(foo2);
    Variant v3 = Variant::opaque(bar);
    Variant vint = 123;
    EXPECT_EQ(v1, vv1);
    EXPECT_NE(v1, v2);
    EXPECT_NE(v1, v3);
    EXPECT_NE(v1, vint);
  }

  // Test hashes. The semantic of the hash follows the object it points to
  // (it hashes the pointer).
  {
    Variant v1 = Variant::opaque(foo);
    Variant vv1 = Variant::opaque(foo);

    Variant v2 = Variant::opaque(foo2);
    Variant v3 = Variant::opaque(bar);

    EXPECT_EQ(v1.hash(), vv1.hash());
    EXPECT_NE(v1.hash(), v2.hash());
    EXPECT_NE(vv1.hash(), v2.hash());

    EXPECT_NE(v1.hash(), v3.hash());
    EXPECT_NE(v2.hash(), v3.hash());
  }

  // Test opaque casting.
  {
    Variant fooOpaque = Variant::opaque(foo);
    Variant barOpaque = Variant::opaque(bar);
    Variant int1 = Variant((int64_t)123);

    auto castFoo1 = fooOpaque.tryOpaque<Foo>();
    auto castBar1 = fooOpaque.tryOpaque<Bar>();
    auto castBar2 = barOpaque.tryOpaque<Bar>();

    EXPECT_EQ(castFoo1, foo);
    EXPECT_EQ(castBar1, nullptr);
    EXPECT_EQ(castBar2, bar);
    VELOX_ASSERT_THROW(int1.tryOpaque<Foo>(), "wrong kind! BIGINT != OPAQUE");
  }
}

void testSerDe(const Variant& value) {
  auto serialized = value.serialize();
  auto copy = Variant::create(serialized);

  ASSERT_EQ(value, copy);
}

TEST(VariantSerializationTest, serialize) {
  // Null values.
  testSerDe(Variant(TypeKind::BOOLEAN));
  testSerDe(Variant(TypeKind::TINYINT));
  testSerDe(Variant(TypeKind::SMALLINT));
  testSerDe(Variant(TypeKind::INTEGER));
  testSerDe(Variant(TypeKind::BIGINT));
  testSerDe(Variant(TypeKind::REAL));
  testSerDe(Variant(TypeKind::DOUBLE));
  testSerDe(Variant(TypeKind::VARCHAR));
  testSerDe(Variant(TypeKind::VARBINARY));
  testSerDe(Variant(TypeKind::TIMESTAMP));
  testSerDe(Variant(TypeKind::ARRAY));
  testSerDe(Variant(TypeKind::MAP));
  testSerDe(Variant(TypeKind::ROW));
  testSerDe(Variant(TypeKind::UNKNOWN));

  // Non-null values.
  testSerDe(Variant(true));
  testSerDe(Variant(static_cast<int8_t>(12)));
  testSerDe(Variant(static_cast<int16_t>(1234)));
  testSerDe(Variant(static_cast<int32_t>(12345)));
  testSerDe(Variant(static_cast<int64_t>(1234567)));
  testSerDe(Variant(static_cast<float>(1.2f)));
  testSerDe(Variant(static_cast<double>(1.234)));
  testSerDe(Variant("This is a test."));
  testSerDe(Variant::binary("This is a test."));
  testSerDe(Variant(Timestamp(1, 2)));
}

TEST(VariantSerializationTest, serializeArrayTypes) {
  // Empty array.
  testSerDe(Variant::array({}));

  // Simple array with primitive elements.
  testSerDe(Variant::array({Variant(1), Variant(2), Variant(3)}));

  // Array with integers of same type.
  testSerDe(
      Variant::array({
          Variant(static_cast<int32_t>(1)),
          Variant(static_cast<int32_t>(100)),
          Variant(static_cast<int32_t>(10000)),
      }));

  // Array with strings.
  testSerDe(
      Variant::array({
          Variant("hello"),
          Variant("world"),
          Variant("test"),
      }));

  // Array with mixed null and non-null elements.
  testSerDe(
      Variant::array({
          Variant(1),
          Variant::null(TypeKind::INTEGER),
          Variant(3),
          Variant::null(TypeKind::INTEGER),
      }));

  // Array with floating point values.
  testSerDe(
      Variant::array({
          Variant(1.5),
          Variant(2.7),
          Variant(-3.14),
      }));

  // Array with boolean values.
  testSerDe(
      Variant::array({
          Variant(true),
          Variant(false),
          Variant(true),
      }));

  // Array with timestamps.
  testSerDe(
      Variant::array({
          Variant(Timestamp(1000, 500)),
          Variant(Timestamp(2000, 1000)),
      }));

  // Nested array (array of arrays).
  testSerDe(
      Variant::array({
          Variant::array({Variant(1), Variant(2)}),
          Variant::array({Variant(3), Variant(4), Variant(5)}),
          Variant::array({}),
      }));

  // Deeply nested arrays.
  testSerDe(
      Variant::array({Variant::array(
          {Variant::array({Variant(1), Variant(2)}),
           Variant::array({Variant(3)})})}));

  // Array with null array elements.
  testSerDe(
      Variant::array({
          Variant::array({Variant(1)}),
          Variant::null(TypeKind::ARRAY),
          Variant::array({Variant(2)}),
      }));

  // Large array.
  std::vector<Variant> largeArray;
  for (int i = 0; i < 100; ++i) {
    largeArray.push_back(Variant(i));
  }
  testSerDe(Variant::array(largeArray));
}

TEST(VariantSerializationTest, serializeMapTypes) {
  // Empty map.
  testSerDe(Variant::map({}));

  // Simple map with integer keys and string values.
  testSerDe(
      Variant::map({
          {Variant(1), Variant("one")},
          {Variant(2), Variant("two")},
          {Variant(3), Variant("three")},
      }));

  // Map with string keys and integer values.
  testSerDe(
      Variant::map({
          {Variant("a"), Variant(1)},
          {Variant("b"), Variant(2)},
          {Variant("c"), Variant(3)},
      }));

  // Map with null values.
  testSerDe(
      Variant::map({
          {Variant(1), Variant(100)},
          {Variant(2), Variant::null(TypeKind::INTEGER)},
          {Variant(3), Variant(300)},
      }));

  // Map with floating point keys and values.
  testSerDe(
      Variant::map({
          {Variant(1.5), Variant(10.5)},
          {Variant(2.5), Variant(20.5)},
      }));

  // Map with boolean keys.
  testSerDe(
      Variant::map({
          {Variant(true), Variant("yes")},
          {Variant(false), Variant("no")},
      }));

  // Nested map (map of maps).
  testSerDe(
      Variant::map({
          {Variant(1),
           Variant::map({
               {Variant("a"), Variant(10)},
               {Variant("b"), Variant(20)},
           })},
          {Variant(2),
           Variant::map({
               {Variant("c"), Variant(30)},
           })},
      }));

  // Map with array values.
  testSerDe(
      Variant::map({
          {Variant(1), Variant::array({Variant(10), Variant(20)})},
          {Variant(2), Variant::array({Variant(30), Variant(40), Variant(50)})},
      }));

  // Map with complex mixed types.
  testSerDe(
      Variant::map({
          {Variant(1), Variant("string")},
          {Variant(2), Variant(42)},
          {Variant(3), Variant(3.14)},
          {Variant(4), Variant(true)},
      }));

  // Map with timestamp keys and values.
  testSerDe(
      Variant::map({
          {Variant(Timestamp(1000, 0)), Variant(Timestamp(2000, 0))},
          {Variant(Timestamp(3000, 500)), Variant(Timestamp(4000, 1000))},
      }));

  // Large map.
  std::map<Variant, Variant> largeMap;
  for (int i = 0; i < 100; ++i) {
    largeMap.emplace(Variant(i), Variant(i * 10));
  }
  testSerDe(Variant::map(largeMap));
}

TEST(VariantSerializationTest, serializeRowTypes) {
  // Empty row.
  testSerDe(Variant::row({}));

  // Simple row with primitive fields.
  testSerDe(Variant::row({Variant(1), Variant("test"), Variant(3.14)}));

  // Row with different integer types.
  testSerDe(
      Variant::row({
          Variant(static_cast<int8_t>(1)),
          Variant(static_cast<int16_t>(100)),
          Variant(static_cast<int32_t>(10000)),
          Variant(static_cast<int64_t>(1000000LL)),
      }));

  // Row with null fields.
  testSerDe(
      Variant::row({
          Variant(1),
          Variant::null(TypeKind::VARCHAR),
          Variant(3.14),
          Variant::null(TypeKind::INTEGER),
      }));

  // Row with all null fields.
  testSerDe(
      Variant::row({
          Variant::null(TypeKind::INTEGER),
          Variant::null(TypeKind::VARCHAR),
          Variant::null(TypeKind::DOUBLE),
      }));

  // Row with boolean fields.
  testSerDe(Variant::row({Variant(true), Variant(false), Variant(true)}));

  // Row with floating point fields.
  testSerDe(
      Variant::row({Variant(1.5f), Variant(2.7), Variant(-3.14159265358979)}));

  // Row with timestamp fields.
  testSerDe(
      Variant::row({
          Variant(Timestamp(1000, 500)),
          Variant(Timestamp(2000, 1000)),
          Variant(Timestamp(3000, 1500)),
      }));

  // Nested row (row containing rows).
  testSerDe(
      Variant::row({
          Variant(1),
          Variant::row({Variant("inner"), Variant(42)}),
          Variant(3.14),
      }));

  // Row with array fields.
  testSerDe(
      Variant::row({
          Variant::array({Variant(1), Variant(2), Variant(3)}),
          Variant::array({Variant("a"), Variant("b")}),
      }));

  // Row with map fields.
  testSerDe(
      Variant::row({
          Variant::map({
              {Variant(1), Variant("one")},
              {Variant(2), Variant("two")},
          }),
          Variant::map({
              {Variant("x"), Variant(10)},
              {Variant("y"), Variant(20)},
          }),
      }));

  // Complex nested row with mixed types.
  testSerDe(
      Variant::row({
          Variant(1),
          Variant::array({Variant(10), Variant(20)}),
          Variant::map({
              {Variant("key"), Variant("value")},
          }),
          Variant::row({Variant(true), Variant(false)}),
          Variant("string field"),
      }));

  // Deeply nested row structure.
  testSerDe(
      Variant::row({
          Variant::row({
              Variant::row({Variant(1), Variant(2)}),
              Variant::row({Variant(3), Variant(4)}),
          }),
          Variant(100),
      }));

  // Large row.
  std::vector<Variant> largeRow;
  for (int i = 0; i < 100; ++i) {
    largeRow.push_back(Variant(i));
  }
  testSerDe(Variant::row(largeRow));
}

TEST(VariantSerializationTest, serializeComplexNestedStructures) {
  // Array of maps (all elements must be MAP type).
  testSerDe(
      Variant::array({
          Variant::map({
              {Variant(1), Variant("a")},
              {Variant(2), Variant("b")},
          }),
          Variant::map({
              {Variant(3), Variant("c")},
          }),
      }));

  // Map with row values.
  testSerDe(
      Variant::map({
          {Variant(1), Variant::row({Variant(10), Variant("ten")})},
          {Variant(2), Variant::row({Variant(20), Variant("twenty")})},
      }));

  // Row of arrays of maps.
  testSerDe(
      Variant::row({
          Variant::array({
              Variant::map({{Variant(1), Variant(100)}}),
              Variant::map({{Variant(2), Variant(200)}}),
          }),
          Variant::array({
              Variant::map({{Variant(3), Variant(300)}}),
          }),
      }));

  // Map with array keys and row values (complex structure).
  testSerDe(
      Variant::map({
          {Variant::array({Variant(1), Variant(2)}),
           Variant::row({Variant("a"), Variant(true)})},
          {Variant::array({Variant(3)}),
           Variant::row({Variant("b"), Variant(false)})},
      }));

  // Array of arrays (homogeneous).
  testSerDe(
      Variant::array({
          Variant::array({Variant(1), Variant(2)}),
          Variant::array({Variant(3), Variant(4)}),
          Variant::null(TypeKind::ARRAY),
      }));

  // Array of rows (homogeneous).
  testSerDe(
      Variant::array({
          Variant::row({Variant(true), Variant(3.14)}),
          Variant::row({Variant(false), Variant(2.71)}),
      }));
}

struct SerializableClass {
  const std::string name;
  const bool value;
  SerializableClass(std::string name, bool value)
      : name(std::move(name)), value(value) {}
};

class VariantOpaqueSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static folly::once_flag once;
    folly::call_once(once, []() {
      OpaqueType::registerSerialization<SerializableClass>(
          "SerializableClass",
          [](const std::shared_ptr<SerializableClass>& obj) -> std::string {
            return folly::json::serialize(
                folly::dynamic::object("name", obj->name)("value", obj->value),
                getSerializationOptions());
          },
          [](const std::string& json) -> std::shared_ptr<SerializableClass> {
            folly::dynamic obj = folly::parseJson(json);
            return std::make_shared<SerializableClass>(
                obj["name"].asString(), obj["value"].asBool());
          });
    });

    value_ = Variant::opaque<SerializableClass>(
        std::make_shared<SerializableClass>("test_class", false));
  }

  Variant value_;
};

TEST_F(VariantOpaqueSerializationTest, serializeOpaque) {
  auto serialized = value_.serialize();
  auto deserialized = Variant::create(serialized);
  auto opaque = deserialized.value<TypeKind::OPAQUE>().obj;
  auto original = std::static_pointer_cast<SerializableClass>(opaque);
  EXPECT_EQ(original->name, "test_class");
  EXPECT_EQ(original->value, false);
}

TEST_F(VariantOpaqueSerializationTest, opaqueToJson) {
  const auto type = value_.inferType();

  const auto expected =
      R"(Opaque<type:OPAQUE<facebook::velox::test::(anonymous namespace)::SerializableClass>,value:"{"name":"test_class","value":false}">)";
  EXPECT_EQ(value_.toJson(type), expected);
  EXPECT_EQ(value_.toString(type), expected);
}

TEST(VariantFloatingToJsonTest, normalTest) {
  // Zero
  EXPECT_EQ(Variant::create<float>(0).toJson(REAL()), "0");
  EXPECT_EQ(Variant::create<double>(0).toJson(DOUBLE()), "0");

  // Infinite
  EXPECT_EQ(
      Variant::create<float>(std::numeric_limits<float>::infinity())
          .toJson(REAL()),
      R"("Infinity")");
  EXPECT_EQ(
      Variant::create<double>(std::numeric_limits<double>::infinity())
          .toJson(DOUBLE()),
      R"("Infinity")");

  // NaN
  EXPECT_EQ(Variant::create<float>(0.0 / 0.0).toJson(REAL()), R"("NaN")");
  EXPECT_EQ(Variant::create<double>(0.0 / 0.0).toJson(DOUBLE()), R"("NaN")");
}

TEST(VariantTest, opaqueSerializationNotRegistered) {
  struct opaqueSerializationTestStruct {};
  auto opaqueBeforeRegistration =
      Variant::opaque<opaqueSerializationTestStruct>(
          std::make_shared<opaqueSerializationTestStruct>());
  EXPECT_THROW(opaqueBeforeRegistration.serialize(), VeloxException);

  OpaqueType::registerSerialization<opaqueSerializationTestStruct>(
      "opaqueSerializationStruct");

  auto opaqueAfterRegistration = Variant::opaque<opaqueSerializationTestStruct>(
      std::make_shared<opaqueSerializationTestStruct>());
  EXPECT_THROW(opaqueAfterRegistration.serialize(), VeloxException);
}

TEST(VariantTest, toJsonRow) {
  auto rowType = ROW({{"c0", DECIMAL(20, 3)}});
  EXPECT_EQ(
      "[123456.789]",
      Variant::row({static_cast<int128_t>(123456789)}).toJson(rowType));

  rowType = ROW({{"c0", DECIMAL(10, 2)}});
  EXPECT_EQ("[12345.67]", Variant::row({1234567LL}).toJson(rowType));

  rowType = ROW({{"c0", DECIMAL(20, 1)}, {"c1", BOOLEAN()}, {"c3", VARCHAR()}});
  EXPECT_EQ(
      R"([1234567890.1,true,"test works fine"])",
      Variant::row({static_cast<int128_t>(12345678901),
                    Variant((bool)true),
                    Variant((std::string) "test works fine")})
          .toJson(rowType));

  // Row Variant tests with wrong type passed to Variant::toJson()
  rowType = ROW({{"c0", DECIMAL(10, 3)}});
  VELOX_ASSERT_THROW(
      Variant::row({static_cast<int128_t>(123456789)}).toJson(rowType),
      "(HUGEINT vs. BIGINT) Wrong type in Variant::toJson");

  rowType = ROW({{"c0", DECIMAL(20, 3)}});
  VELOX_ASSERT_THROW(
      Variant::row({123456789LL}).toJson(rowType),
      "(BIGINT vs. HUGEINT) Wrong type in Variant::toJson");
  VELOX_ASSERT_THROW(
      Variant::row(
          {static_cast<int128_t>(123456789),
           Variant(
               "test confirms Variant child count is greater than expected"),
           Variant(false)})
          .toJson(rowType),
      "(3 vs. 1) Wrong number of fields in a struct in Variant::toJson");

  rowType =
      ROW({{"c0", DECIMAL(19, 4)}, {"c1", VARCHAR()}, {"c2", DECIMAL(10, 3)}});
  VELOX_ASSERT_THROW(
      Variant::row(
          {static_cast<int128_t>(12345678912),
           Variant(
               "test confirms Variant child count is lesser than expected")})
          .toJson(rowType),
      "(2 vs. 3) Wrong number of fields in a struct in Variant::toJson");

  // Row Variant tests that contains NULL variants.
  EXPECT_EQ(
      "[null,null,null]",
      Variant::row({Variant::null(TypeKind::HUGEINT),
                    Variant::null(TypeKind::VARCHAR),
                    Variant::null(TypeKind::BIGINT)})
          .toJson(rowType));
}

TEST(VariantTest, toJsonArray) {
  auto arrayType = ARRAY(DECIMAL(9, 2));
  EXPECT_EQ(
      "[1234567.89,6345654.64,2345452.78]",
      Variant::array({123456789LL, 634565464LL, 234545278LL})
          .toJson(arrayType));

  arrayType = ARRAY(DECIMAL(20, 3));
  EXPECT_EQ(
      "[123456.789,634565.464,234545.278]",
      Variant::array({static_cast<int128_t>(123456789),
                      static_cast<int128_t>(634565464),
                      static_cast<int128_t>(234545278)})
          .toJson(arrayType));

  // Array is empty.
  EXPECT_EQ("[]", Variant::array({}).toJson(arrayType));

  // Array Variant tests that contains NULL variants.
  EXPECT_EQ(
      "[null,null,null]",
      Variant::array({Variant::null(TypeKind::HUGEINT),
                      Variant::null(TypeKind::HUGEINT),
                      Variant::null(TypeKind::HUGEINT)})
          .toJson(arrayType));
}

TEST(VariantTest, toJsonMap) {
  auto mapType = MAP(VARCHAR(), DECIMAL(6, 3));
  std::map<Variant, Variant> mapValue = {
      {"key1", 235499LL}, {"key2", 123456LL}};
  EXPECT_EQ(
      R"([{"key":"key1","value":235.499},{"key":"key2","value":123.456}])",
      Variant::map(mapValue).toJson(mapType));

  mapType = MAP(VARCHAR(), DECIMAL(20, 3));
  mapValue = {
      {"key1", static_cast<int128_t>(45464562323423)},
      {"key2", static_cast<int128_t>(12334581232456)}};
  EXPECT_EQ(
      R"([{"key":"key1","value":45464562323.423},{"key":"key2","value":12334581232.456}])",
      Variant::map(mapValue).toJson(mapType));

  // Map Variant tests that contains NULL variants.
  mapValue = {
      {Variant::null(TypeKind::VARCHAR), Variant::null(TypeKind::HUGEINT)}};
  EXPECT_EQ(
      R"([{"key":null,"value":null}])", Variant::map(mapValue).toJson(mapType));
}

TEST(VariantTest, typeWithCustomComparison) {
  auto zero = Variant::typeWithCustomComparison<TypeKind::BIGINT>(
      0, test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON());
  auto one = Variant::typeWithCustomComparison<TypeKind::BIGINT>(
      1, test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON());
  auto zeroEquivalent = Variant::typeWithCustomComparison<TypeKind::BIGINT>(
      256, test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON());
  auto oneEquivalent = Variant::typeWithCustomComparison<TypeKind::BIGINT>(
      257, test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON());
  auto null = Variant::null(TypeKind::BIGINT);

  ASSERT_TRUE(zero.equals(zeroEquivalent));
  ASSERT_TRUE(zero.equalsWithEpsilon(zeroEquivalent));

  ASSERT_TRUE(one.equals(oneEquivalent));
  ASSERT_TRUE(one.equalsWithEpsilon(oneEquivalent));

  ASSERT_FALSE(zero.equals(one));
  ASSERT_FALSE(zero.equalsWithEpsilon(one));

  ASSERT_FALSE(one.equals(zeroEquivalent));
  ASSERT_FALSE(one.equalsWithEpsilon(zeroEquivalent));

  ASSERT_FALSE(zero.equals(null));
  ASSERT_FALSE(zero.equalsWithEpsilon(null));

  ASSERT_FALSE(null.equals(one));
  ASSERT_FALSE(null.equalsWithEpsilon(one));

  ASSERT_FALSE(zero < zeroEquivalent);
  ASSERT_FALSE(zero.lessThanWithEpsilon(zeroEquivalent));

  ASSERT_FALSE(one < oneEquivalent);
  ASSERT_FALSE(one.lessThanWithEpsilon(oneEquivalent));

  ASSERT_TRUE(zero < one);
  ASSERT_TRUE(zero.lessThanWithEpsilon(one));

  ASSERT_FALSE(one < zeroEquivalent);
  ASSERT_FALSE(one.lessThanWithEpsilon(zeroEquivalent));

  ASSERT_FALSE(zero < null);
  ASSERT_FALSE(zero.lessThanWithEpsilon(null));

  ASSERT_TRUE(null < one);
  ASSERT_TRUE(null.lessThanWithEpsilon(one));

  ASSERT_EQ(zero.hash(), zeroEquivalent.hash());
  ASSERT_EQ(one.hash(), oneEquivalent.hash());
  ASSERT_NE(zero.hash(), one.hash());
  ASSERT_NE(zero.hash(), null.hash());
}

TEST(VariantTest, hashMap) {
  auto a = Variant::map({{1, 10}, {2, 20}});
  auto b = Variant::map({{1, 20}, {2, 10}});
  auto c = Variant::map({{2, 20}, {1, 10}});
  auto d = Variant::map({{1, 10}, {2, 20}, {3, 30}});

  ASSERT_NE(a.hash(), b.hash());
  ASSERT_EQ(a.hash(), c.hash());
  ASSERT_NE(a.hash(), d.hash());
}

TEST(VariantTest, toString) {
  EXPECT_EQ(Variant::array({1, 2, 3}).toString(ARRAY(INTEGER())), "[1,2,3]");
  EXPECT_EQ(
      Variant::map({{1, 2}, {3, 4}}).toString(MAP(INTEGER(), INTEGER())),
      R"([{"key":1,"value":2},{"key":3,"value":4}])");
  EXPECT_EQ(
      Variant::row({1, 2, 3}).toString(ROW({INTEGER(), INTEGER(), INTEGER()})),
      "[1,2,3]");
}

template <typename T>
void testPrimitiveGetter(T v) {
  auto value = Variant(v);
  EXPECT_FALSE(value.isNull());
  EXPECT_EQ(value.value<T>(), value);
}

TEST(VariantTest, primitiveGetters) {
  testPrimitiveGetter<bool>(true);
  testPrimitiveGetter<int32_t>(10);
  testPrimitiveGetter<int64_t>(10);
  testPrimitiveGetter<float>(1.2);
  testPrimitiveGetter<double>(1.2);
}

template <typename T>
void testArrayGetter(const std::vector<T>& inputs) {
  std::vector<Variant> variants;
  variants.reserve(inputs.size());
  for (const auto& v : inputs) {
    variants.emplace_back(v);
  }

  auto value = Variant::array(variants);

  EXPECT_FALSE(value.isNull());

  {
    auto variantItems = value.array();
    EXPECT_EQ(variantItems.size(), inputs.size());
    for (auto i = 0; i < inputs.size(); ++i) {
      EXPECT_FALSE(variantItems.at(i).isNull());
      EXPECT_EQ(variantItems.at(i).template value<T>(), inputs.at(i));
    }
  }

  {
    auto primitiveItems = value.template array<T>();
    EXPECT_EQ(primitiveItems, inputs);
  }

  {
    auto primitiveItems = value.template nullableArray<T>();
    EXPECT_EQ(primitiveItems.size(), inputs.size());
    for (auto i = 0; i < inputs.size(); ++i) {
      EXPECT_TRUE(primitiveItems.at(i).has_value());
      EXPECT_EQ(primitiveItems.at(i).value(), inputs.at(i));
    }
  }
}

template <typename T>
void testNullableArrayGetter(const std::vector<std::optional<T>>& inputs) {
  std::vector<Variant> variants;
  variants.reserve(inputs.size());
  for (const auto& v : inputs) {
    if (v.has_value()) {
      variants.emplace_back(v.value());
    } else {
      variants.emplace_back(Variant::null(CppToType<T>::typeKind));
    }
  }

  auto value = Variant::array(variants);
  EXPECT_FALSE(value.isNull());

  auto primitiveItems = value.template nullableArray<T>();
  EXPECT_EQ(primitiveItems, inputs);
}

TEST(VariantTest, arrayGetter) {
  testArrayGetter<bool>({true, false, true});
  testArrayGetter<int32_t>({1, 2, 3});
  testArrayGetter<int64_t>({1, 2, 3});
  testArrayGetter<float>({1.2, 2.3, 3.4});
  testArrayGetter<double>({1.2, 2.3, 3.4});

  testNullableArrayGetter<bool>({true, false, std::nullopt});
  testNullableArrayGetter<int32_t>({1, 2, std::nullopt, 4});
  testNullableArrayGetter<int64_t>({1, 2, std::nullopt, 4});
  testNullableArrayGetter<float>({1.1, 2.2, std::nullopt});
  testNullableArrayGetter<double>({1.1, 2.2, std::nullopt});
}

template <typename K, typename V>
void testMapGetter(const std::map<K, V>& inputs) {
  std::map<Variant, Variant> variants;
  for (const auto& [k, v] : inputs) {
    variants.emplace(k, v);
  }

  auto value = Variant::map(variants);

  EXPECT_FALSE(value.isNull());

  {
    auto variantItems = value.map();
    EXPECT_EQ(variantItems.size(), inputs.size());

    auto expectedIt = inputs.begin();
    for (auto it = variantItems.begin(); it != variantItems.end(); it++) {
      auto [k, v] = *it;

      EXPECT_FALSE(k.isNull());
      EXPECT_FALSE(v.isNull());
      EXPECT_EQ(k.template value<K>(), (*expectedIt).first);
      EXPECT_EQ(v.template value<V>(), (*expectedIt).second);
      expectedIt++;
    }
  }

  {
    auto primitiveItems = value.template map<K, V>();
    EXPECT_EQ(primitiveItems, inputs);
  }

  {
    auto primitiveItems = value.template nullableMap<K, V>();
    EXPECT_EQ(primitiveItems.size(), inputs.size());

    auto expectedIt = inputs.begin();
    for (auto it = primitiveItems.begin(); it != primitiveItems.end(); it++) {
      auto [k, v] = *it;

      EXPECT_TRUE(v.has_value());
      EXPECT_EQ(k, (*expectedIt).first);
      EXPECT_EQ(v.value(), (*expectedIt).second);
      expectedIt++;
    }
  }
}

template <typename K, typename V>
void testNullableMapGetter(const std::map<K, std::optional<V>>& inputs) {
  std::map<Variant, Variant> variants;
  for (const auto& [k, v] : inputs) {
    if (v.has_value()) {
      variants.emplace(k, v.value());
    } else {
      variants.emplace(k, Variant::null(CppToType<V>::typeKind));
    }
  }

  auto value = Variant::map(variants);
  EXPECT_FALSE(value.isNull());

  auto primitiveItems = value.template nullableMap<K, V>();
  EXPECT_EQ(primitiveItems, inputs);
}

TEST(VariantTest, mapGetter) {
  testMapGetter<int32_t, float>({{1, 1.2}, {2, 2.3}, {3, 3.4}});
  testMapGetter<int8_t, double>({{1, 1.2}, {2, 2.3}, {3, 3.4}});

  testNullableMapGetter<int32_t, float>(
      {{1, 1.2}, {2, std::nullopt}, {3, 3.4}});
  testNullableMapGetter<int8_t, double>(
      {{1, 1.2}, {2, 2.3}, {3, std::nullopt}});
}

template <typename T>
void testArrayOfArraysGetter(const std::vector<std::vector<T>>& inputs) {
  std::vector<Variant> variants;
  variants.reserve(inputs.size());

  for (const auto& input : inputs) {
    std::vector<Variant> innerVariants;
    innerVariants.reserve(input.size());
    for (const auto& v : input) {
      innerVariants.emplace_back(v);
    }
    variants.emplace_back(Variant::array(innerVariants));
  }

  auto value = Variant::array(variants);
  EXPECT_FALSE(value.isNull());

  {
    auto primitiveItems = value.template arrayOfArrays<T>();
    EXPECT_EQ(primitiveItems, inputs);
  }
}

TEST(VariantTest, arrayOfArraysGetter) {
  testArrayOfArraysGetter<int32_t>({{1, 2}, {3, 4, 5}, {}});
  testArrayOfArraysGetter<double>({{1.1, 2.2}, {}, {3.3}});
}

template <typename K, typename V>
void testMapOfArraysGetter(const std::map<K, std::vector<V>>& inputs) {
  std::map<Variant, Variant> variants;

  for (const auto& [k, input] : inputs) {
    std::vector<Variant> innerVariants;
    innerVariants.reserve(input.size());
    for (const auto& v : input) {
      innerVariants.emplace_back(v);
    }
    variants.emplace(k, Variant::array(innerVariants));
  }

  auto value = Variant::map(variants);
  EXPECT_FALSE(value.isNull());

  {
    auto primitiveItems = value.template mapOfArrays<K, V>();
    EXPECT_EQ(primitiveItems, inputs);
  }
}

TEST(VariantTest, mapOfArraysGetter) {
  testMapOfArraysGetter<int32_t, float>(
      {{1, {1.1f, 2.2f}}, {2, {3.3f}}, {3, {}}});
  testMapOfArraysGetter<std::string, int64_t>(
      {{"a", {1, 2}}, {"b", {}}, {"c", {3}}});
}

template <typename T>
void testNullableArrayOfArraysGetter(
    const std::vector<std::optional<std::vector<std::optional<T>>>>& inputs) {
  std::vector<Variant> variants;
  variants.reserve(inputs.size());
  for (const auto& arrOpt : inputs) {
    if (!arrOpt.has_value()) {
      variants.emplace_back(Variant::null(TypeKind::ARRAY));
    } else {
      std::vector<Variant> innerVariants;
      innerVariants.reserve(arrOpt->size());
      for (const auto& vOpt : *arrOpt) {
        if (vOpt.has_value()) {
          innerVariants.emplace_back(*vOpt);
        } else {
          innerVariants.emplace_back(Variant::null(CppToType<T>::typeKind));
        }
      }
      variants.emplace_back(Variant::array(innerVariants));
    }
  }
  auto value = Variant::array(variants);
  EXPECT_FALSE(value.isNull());
  auto primitiveItems = value.template nullableArrayOfArrays<T>();
  EXPECT_EQ(primitiveItems, inputs);
}

TEST(VariantTest, nullableArrayOfArraysGetter) {
  testNullableArrayOfArraysGetter<int32_t>(
      {std::nullopt,
       std::vector<std::optional<int32_t>>{1, std::nullopt, 3},
       std::vector<std::optional<int32_t>>{},
       std::vector<std::optional<int32_t>>{std::nullopt, 2}});
  testNullableArrayOfArraysGetter<double>(
      {std::vector<std::optional<double>>{1.1, std::nullopt},
       std::nullopt,
       std::vector<std::optional<double>>{3.3}});
}

template <typename K, typename V>
void testNullableMapOfArraysGetter(
    const std::map<K, std::optional<std::vector<std::optional<V>>>>& inputs) {
  std::map<Variant, Variant> variants;
  for (const auto& [k, arrOpt] : inputs) {
    if (!arrOpt.has_value()) {
      variants.emplace(k, Variant::null(TypeKind::ARRAY));
    } else {
      std::vector<Variant> innerVariants;
      innerVariants.reserve(arrOpt->size());
      for (const auto& vOpt : *arrOpt) {
        if (vOpt.has_value()) {
          innerVariants.emplace_back(*vOpt);
        } else {
          innerVariants.emplace_back(Variant::null(CppToType<V>::typeKind));
        }
      }
      variants.emplace(k, Variant::array(innerVariants));
    }
  }
  auto value = Variant::map(variants);
  EXPECT_FALSE(value.isNull());
  auto primitiveItems = value.template nullableMapOfArrays<K, V>();
  EXPECT_EQ(primitiveItems, inputs);
}

TEST(VariantTest, nullableMapOfArraysGetter) {
  testNullableMapOfArraysGetter<int32_t, float>(
      {{1, std::nullopt},
       {2, std::vector<std::optional<float>>{1.1f, std::nullopt}},
       {3, std::vector<std::optional<float>>{}}});
  testNullableMapOfArraysGetter<std::string, int64_t>(
      {{"a", std::vector<std::optional<int64_t>>{1, std::nullopt}},
       {"b", std::nullopt},
       {"c", std::vector<std::optional<int64_t>>{3}}});
}

TEST(VariantTest, accessNullVariantValue) {
  Variant nullVar = Variant::null(TypeKind::INTEGER);
  VELOX_ASSERT_THROW(
      nullVar.value<TypeKind::INTEGER>(), "missing Variant value");
}

TEST(VariantTest, accessWrongTypeVariantValue) {
  Variant intVar = Variant::create<int32_t>(1);
  VELOX_ASSERT_THROW(
      intVar.value<TypeKind::VARCHAR>(), "wrong kind! INTEGER != VARCHAR");
}

} // namespace
} // namespace facebook::velox::test

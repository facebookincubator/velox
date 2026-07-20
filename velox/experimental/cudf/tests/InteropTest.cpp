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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::cudf_velox;

namespace {

/// Tests the direct toCudfTable conversion (bypassing Arrow) by round-tripping
/// Velox RowVector -> cudf table (direct) -> Velox RowVector (via Arrow) and
/// verifying the result matches the original.
class InteropTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  /// Round-trips a RowVector through the direct toCudfTable and back via
  /// with_arrow::toVeloxColumn, then asserts equality with the original.
  void roundTrip(const RowVectorPtr& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Convert Velox -> cudf using the new direct path.
    // auto cudfTable = cudf_velox::toCudfTable(input, pool_.get(), stream, mr);
    auto cudfTable =
        cudf_velox::with_arrow::toCudfTable(input, pool_.get(), stream, mr);
    ASSERT_NE(cudfTable, nullptr);
    ASSERT_EQ(cudfTable->num_rows(), input->size());
    ASSERT_EQ(
        cudfTable->num_columns(),
        static_cast<cudf::size_type>(input->childrenSize()));

    // Convert cudf -> Velox using the existing Arrow path.
    auto result = cudf_velox::with_arrow::toVeloxColumn(
        cudfTable->view(), pool_.get(), input->type(), stream, mr);
    stream.synchronize();

    // Verify.
    test::assertEqualVectors(input, result);
  }
};

// ========== Integer types ==========

TEST_F(InteropTest, tinyint) {
  auto input =
      makeRowVector({"c0"}, {makeFlatVector<int8_t>({1, -2, 3, 0, 127, -128})});
  roundTrip(input);
}

TEST_F(InteropTest, smallint) {
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<int16_t>({1, -2, 300, 0, 32767, -32768})});
  roundTrip(input);
}

TEST_F(InteropTest, integer) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int32_t>({1, -2, 300, 0, 2147483647, -2147483648})});
  roundTrip(input);
}

TEST_F(InteropTest, bigint) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>({1, -2, 300, 0, 9223372036854775807LL})});
  roundTrip(input);
}

// ========== Floating point types ==========

TEST_F(InteropTest, real) {
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<float>({1.5f, -2.5f, 0.0f, 3.14f})});
  roundTrip(input);
}

TEST_F(InteropTest, double_type) {
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<double>({1.5, -2.5, 0.0, 3.14159265358979})});
  roundTrip(input);
}

// ========== Boolean ==========

TEST_F(InteropTest, boolean) {
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<bool>({true, false, true, true, false})});
  roundTrip(input);
}

// ========== String types ==========

TEST_F(InteropTest, varchar) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<StringView>(
          {"hello", "world", "", "a longer string value", "x"})});
  roundTrip(input);
}

TEST_F(InteropTest, varbinary) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<StringView>({"binary\x00data"_sv, "more"_sv, ""_sv})});
  roundTrip(input);
}

// ========== With nulls ==========

TEST_F(InteropTest, integerWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3, std::nullopt, 5})});
  roundTrip(input);
}

TEST_F(InteropTest, booleanWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<bool>(
          {true, std::nullopt, false, std::nullopt, true})});
  roundTrip(input);
}

TEST_F(InteropTest, varcharWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<StringView>(
          {"hello"_sv, std::nullopt, "world"_sv, std::nullopt})});
  roundTrip(input);
}

TEST_F(InteropTest, doubleWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<double>(
          {1.1, std::nullopt, 3.3, std::nullopt, 5.5})});
  roundTrip(input);
}

// ========== Timestamp ==========

TEST_F(InteropTest, timestamp) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<Timestamp>(
          {Timestamp(0, 0),
           Timestamp(1, 500000000),
           Timestamp(100, 0),
           Timestamp(1000000, 999999999)})});
  roundTrip(input);
}

TEST_F(InteropTest, timestampWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<Timestamp>(
          {Timestamp(0, 0), std::nullopt, Timestamp(100, 0), std::nullopt})});
  roundTrip(input);
}

// ========== Multiple columns ==========

TEST_F(InteropTest, multipleColumns) {
  auto input = makeRowVector(
      {"c0", "c1", "c2", "c3"},
      {makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
       makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
       makeFlatVector<bool>({true, false, true, false, true}),
       makeFlatVector<StringView>({"a", "bb", "ccc", "dddd", "eeeee"})});
  roundTrip(input);
}

TEST_F(InteropTest, multipleColumnsWithNulls) {
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt, 5}),
       makeNullableFlatVector<StringView>(
           {"a"_sv, "b"_sv, std::nullopt, "d"_sv, std::nullopt}),
       makeNullableFlatVector<bool>(
           {true, std::nullopt, false, std::nullopt, true})});
  roundTrip(input);
}

// ========== Empty input ==========

TEST_F(InteropTest, emptyInput) {
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>({}), makeFlatVector<StringView>({})});
  roundTrip(input);
}

TEST_F(InteropTest, emptyBoolean) {
  auto input = makeRowVector({"c0"}, {makeFlatVector<bool>({})});
  roundTrip(input);
}

TEST_F(InteropTest, emptyDouble) {
  auto input = makeRowVector({"c0"}, {makeFlatVector<double>({})});
  roundTrip(input);
}

TEST_F(InteropTest, emptyTimestamp) {
  auto input = makeRowVector({"c0"}, {makeFlatVector<Timestamp>({})});
  roundTrip(input);
}

TEST_F(InteropTest, emptyArray) {
  auto input = makeRowVector({"c0"}, {makeArrayVector<int32_t>({})});
  roundTrip(input);
}

TEST_F(InteropTest, emptyStruct) {
  auto structVector = makeRowVector(
      {makeFlatVector<int32_t>({}), makeFlatVector<StringView>({})});
  auto input = makeRowVector({"c0"}, {structVector});
  roundTrip(input);
}

TEST_F(InteropTest, emptyMultipleColumns) {
  auto input = makeRowVector(
      {"c0", "c1", "c2", "c3"},
      {makeFlatVector<int64_t>({}),
       makeFlatVector<bool>({}),
       makeFlatVector<StringView>({}),
       makeArrayVector<int32_t>({})});
  roundTrip(input);
}

// ========== All nulls ==========

TEST_F(InteropTest, allNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt})});
  roundTrip(input);
}

TEST_F(InteropTest, allNullsString) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<StringView>(
          {std::nullopt, std::nullopt, std::nullopt})});
  roundTrip(input);
}

// ========== ARRAY (LIST) types ==========

TEST_F(InteropTest, arrayOfIntegers) {
  auto input = makeRowVector(
      {"c0"},
      {makeArrayVector<int32_t>({
          {1, 2, 3},
          {4, 5},
          {6},
          {},
          {7, 8, 9, 10},
      })});
  roundTrip(input);
}

TEST_F(InteropTest, arrayOfStrings) {
  auto input = makeRowVector(
      {"c0"},
      {makeArrayVector<StringView>({
          {"hello"_sv, "world"_sv},
          {},
          {"foo"_sv, "bar"_sv, "baz"_sv},
      })});
  roundTrip(input);
}

TEST_F(InteropTest, arrayOfDoubles) {
  auto input = makeRowVector(
      {"c0"},
      {makeArrayVector<double>({
          {1.1, 2.2, 3.3},
          {},
          {4.4},
      })});
  roundTrip(input);
}

TEST_F(InteropTest, arrayWithNulls) {
  // Array column with some null array entries.
  auto elements =
      makeNullableFlatVector<int32_t>({1, 2, std::nullopt, 4, 5, std::nullopt});
  auto offsets = makeIndices({0, 3, 3, 3});
  auto sizes = makeIndices({3, 0, 3, 0});
  auto arrayVector = std::make_shared<ArrayVector>(
      pool_.get(),
      ARRAY(INTEGER()),
      makeNulls({false, true, false, true}),
      4,
      offsets,
      sizes,
      elements);

  auto input = makeRowVector({"c0"}, {arrayVector});
  roundTrip(input);
}

// ========== ROW (STRUCT) types ==========

TEST_F(InteropTest, structSimple) {
  auto structVector = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<StringView>({"a"_sv, "b"_sv, "c"_sv, "d"_sv})});
  auto input = makeRowVector({"c0"}, {structVector});
  roundTrip(input);
}

TEST_F(InteropTest, structWithNulls) {
  auto structVector = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3, std::nullopt}),
       makeNullableFlatVector<double>({1.1, 2.2, std::nullopt, 4.4})},
      // Null the struct itself at row 1.
      [](auto row) { return row == 1; });
  auto input = makeRowVector({"c0"}, {structVector});
  roundTrip(input);
}

TEST_F(InteropTest, structMultipleFields) {
  auto structVector = makeRowVector(
      {makeFlatVector<int64_t>({10, 20, 30}),
       makeFlatVector<bool>({true, false, true}),
       makeFlatVector<StringView>({"x"_sv, "y"_sv, "z"_sv})});
  auto input = makeRowVector({"c0"}, {structVector});
  roundTrip(input);
}

// ========== Nested complex types ==========

TEST_F(InteropTest, arrayOfArrays) {
  // Build ARRAY(ARRAY(INTEGER)).
  auto innerArray = makeArrayVector<int32_t>({
      {1, 2},
      {3},
      {4, 5, 6},
      {},
      {7},
  });
  auto outerOffsets = makeIndices({0, 2, 2, 3});
  auto outerSizes = makeIndices({2, 0, 2, 1});
  auto outerArray = std::make_shared<ArrayVector>(
      pool_.get(),
      ARRAY(ARRAY(INTEGER())),
      nullptr,
      4,
      outerOffsets,
      outerSizes,
      innerArray);

  auto input = makeRowVector({"c0"}, {outerArray});
  roundTrip(input);
}

TEST_F(InteropTest, structWithArray) {
  auto arrayChild = makeArrayVector<int32_t>({
      {1, 2, 3},
      {4, 5},
      {6},
  });
  auto intChild = makeFlatVector<int64_t>({10, 20, 30});
  auto structVector = makeRowVector({intChild, arrayChild});
  auto input = makeRowVector({"c0"}, {structVector});
  roundTrip(input);
}

TEST_F(InteropTest, arrayOfStructs) {
  auto structElements = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
       makeFlatVector<StringView>({"a"_sv, "b"_sv, "c"_sv, "d"_sv, "e"_sv})});
  auto offsets = makeIndices({0, 2, 2});
  auto sizes = makeIndices({2, 0, 3});
  auto arrayVector = std::make_shared<ArrayVector>(
      pool_.get(),
      ARRAY(ROW({INTEGER(), VARCHAR()})),
      nullptr,
      3,
      offsets,
      sizes,
      structElements);

  auto input = makeRowVector({"c0"}, {arrayVector});
  roundTrip(input);
}

// ========== Mixed complex and scalar columns ==========

TEST_F(InteropTest, mixedScalarAndArray) {
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeArrayVector<int64_t>({
           {10, 20},
           {30},
           {40, 50, 60},
       }),
       makeFlatVector<StringView>({"a"_sv, "b"_sv, "c"_sv})});
  roundTrip(input);
}

TEST_F(InteropTest, mixedScalarAndStruct) {
  auto structChild = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<double>({1.1, 2.2, 3.3})});
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<StringView>({"x"_sv, "y"_sv, "z"_sv}), structChild});
  roundTrip(input);
}

TEST_F(InteropTest, dictionary) {
  constexpr int kSize = 10;
  auto baseValues = makeFlatVector<int64_t>({10, 20, 30, 40, 50});
  // Create indices that map to the base values
  auto indices = makeIndices(kSize, [](auto i) { return i % 5; });
  auto dictionaryChild =
      BaseVector::wrapInDictionary(nullptr, indices, kSize, baseValues);
  auto input = makeRowVector({dictionaryChild});
  roundTrip(input);
}

TEST_F(InteropTest, constant) {
  auto child = makeConstant<int64_t>(10, 5);
  auto input = makeRowVector({child});
  roundTrip(input);
}

} // namespace

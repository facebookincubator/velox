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

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::exec::test;

namespace {

/// Tests the direct toCudfTable conversion (bypassing Arrow) by round-tripping
/// Velox RowVector -> cudf table (direct) -> Velox RowVector (via Arrow) and
/// verifying the result matches the original.
class VeloxCudfInteropTest : public virtual testing::Test,
                         public velox::test::VectorTestBase {
 protected:

  /// Round-trips a RowVector through the direct toCudfTable and back via
  /// with_arrow::toVeloxColumn, then asserts equality with the original.
  void roundTrip(const RowVectorPtr& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Convert Velox -> cudf using the new direct path.
    auto cudfTable = cudf_velox::toCudfTable(input, pool_.get(), stream, mr);
    ASSERT_NE(cudfTable, nullptr);
    ASSERT_EQ(cudfTable->num_rows(), input->size());
    ASSERT_EQ(
        cudfTable->num_columns(),
        static_cast<cudf::size_type>(input->childrenSize()));

    // Convert cudf -> Velox using the existing Arrow path.
    std::vector<std::string> columnNames;
    auto rowType = asRowType(input->type());
    for (auto i = 0; i < rowType->size(); ++i) {
      columnNames.push_back(rowType->nameOf(i));
    }
    auto result = cudf_velox::with_arrow::toVeloxColumn(
        cudfTable->view(), pool_.get(), columnNames, stream, mr);
    stream.synchronize();

    // Verify.
    velox::test::assertEqualVectors(input, result);
  }
};

// ========== Integer types ==========

TEST_F(VeloxCudfInteropTest, tinyint) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int8_t>({1, -2, 3, 0, 127, -128})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, smallint) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int16_t>({1, -2, 300, 0, 32767, -32768})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, integer) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int32_t>({1, -2, 300, 0, 2147483647, -2147483648})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, bigint) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>({1, -2, 300, 0, 9223372036854775807LL})});
  roundTrip(input);
}

// ========== Floating point types ==========

TEST_F(VeloxCudfInteropTest, real) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<float>({1.5f, -2.5f, 0.0f, 3.14f})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, double_type) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<double>({1.5, -2.5, 0.0, 3.14159265358979})});
  roundTrip(input);
}

// ========== Boolean ==========

TEST_F(VeloxCudfInteropTest, boolean) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<bool>({true, false, true, true, false})});
  roundTrip(input);
}

// ========== String types ==========

TEST_F(VeloxCudfInteropTest, varchar) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<StringView>(
          {"hello", "world", "", "a longer string value", "x"})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, varbinary) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<StringView>(
          {"binary\x00data"_sv, "more"_sv, ""_sv})});
  roundTrip(input);
}

// ========== With nulls ==========

TEST_F(VeloxCudfInteropTest, integerWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {1, std::nullopt, 3, std::nullopt, 5})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, booleanWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<bool>(
          {true, std::nullopt, false, std::nullopt, true})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, varcharWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<StringView>(
          {"hello"_sv, std::nullopt, "world"_sv, std::nullopt})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, doubleWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<double>(
          {1.1, std::nullopt, 3.3, std::nullopt, 5.5})});
  roundTrip(input);
}

// ========== Timestamp ==========

TEST_F(VeloxCudfInteropTest, timestamp) {
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<Timestamp>(
          {Timestamp(0, 0),
           Timestamp(1, 500000000),
           Timestamp(100, 0),
           Timestamp(1000000, 999999999)})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, timestampWithNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<Timestamp>(
          {Timestamp(0, 0),
           std::nullopt,
           Timestamp(100, 0),
           std::nullopt})});
  roundTrip(input);
}

// ========== Multiple columns ==========

TEST_F(VeloxCudfInteropTest, multipleColumns) {
  auto input = makeRowVector(
      {"c0", "c1", "c2", "c3"},
      {makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
       makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
       makeFlatVector<bool>({true, false, true, false, true}),
       makeFlatVector<StringView>(
           {"a", "bb", "ccc", "dddd", "eeeee"})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, multipleColumnsWithNulls) {
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {makeNullableFlatVector<int64_t>(
           {1, std::nullopt, 3, std::nullopt, 5}),
       makeNullableFlatVector<StringView>(
           {"a"_sv, "b"_sv, std::nullopt, "d"_sv, std::nullopt}),
       makeNullableFlatVector<bool>(
           {true, std::nullopt, false, std::nullopt, true})});
  roundTrip(input);
}

// ========== Empty input ==========

TEST_F(VeloxCudfInteropTest, emptyInput) {
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>({}),
       makeFlatVector<StringView>({})});
  roundTrip(input);
}

// ========== All nulls ==========

TEST_F(VeloxCudfInteropTest, allNulls) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt})});
  roundTrip(input);
}

TEST_F(VeloxCudfInteropTest, allNullsString) {
  auto input = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<StringView>(
          {std::nullopt, std::nullopt, std::nullopt})});
  roundTrip(input);
}

} // namespace

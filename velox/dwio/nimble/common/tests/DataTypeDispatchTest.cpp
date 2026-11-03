/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/DataTypeDispatch.h"

#include <string>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"

using namespace facebook;

namespace {

struct TypeName {
  template <typename T>
  std::string operator()() const {
    if constexpr (std::is_same_v<T, int8_t>) {
      return "int8";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return "uint8";
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return "int16";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return "uint16";
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return "int32";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return "uint32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return "int64";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return "uint64";
    } else if constexpr (std::is_same_v<T, float>) {
      return "float";
    } else if constexpr (std::is_same_v<T, double>) {
      return "double";
    } else if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else if constexpr (std::is_same_v<T, std::string_view>) {
      return "string_view";
    }
  }
};

std::string dispatchDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

std::string tryDispatchDataType(nimble::DataType dataType) {
  NIMBLE_TRY_RETURN_BY_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

std::string dispatchVarintDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_VARINT_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

std::string dispatchNonBoolDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

std::string dispatchNumericDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

std::string dispatchFloatingPointDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE(
      dataType, T, TypeName{}.operator()<T>());
}

std::string dispatchIntegerDataType(nimble::DataType dataType) {
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE(dataType, T, TypeName{}.operator()<T>());
}

} // namespace

TEST(DataTypeDispatchTest, dispatchesAllDataTypes) {
  EXPECT_EQ(dispatchDataType(nimble::DataType::Int8), "int8");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Uint8), "uint8");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Int16), "int16");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Uint16), "uint16");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Int32), "int32");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Uint32), "uint32");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Int64), "int64");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Uint64), "uint64");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Float), "float");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Double), "double");
  EXPECT_EQ(dispatchDataType(nimble::DataType::Bool), "bool");
  EXPECT_EQ(dispatchDataType(nimble::DataType::String), "string_view");
}

TEST(DataTypeDispatchTest, tryDispatchReturnsDefaultForUndefined) {
  EXPECT_EQ(tryDispatchDataType(nimble::DataType::Undefined), "");
}

TEST(DataTypeDispatchTest, dispatchRejectsUndefined) {
  NIMBLE_ASSERT_THROW(
      dispatchDataType(nimble::DataType::Undefined), "Unsupported data type");
}

TEST(DataTypeDispatchTest, dispatchesVarintDataTypes) {
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Int32), "int32");
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Uint32), "uint32");
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Int64), "int64");
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Uint64), "uint64");
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Float), "float");
  EXPECT_EQ(dispatchVarintDataType(nimble::DataType::Double), "double");

  NIMBLE_ASSERT_THROW(
      dispatchVarintDataType(nimble::DataType::Int16),
      "Unsupported varint data type");
  NIMBLE_ASSERT_THROW(
      dispatchVarintDataType(nimble::DataType::String),
      "Unsupported varint data type");
}

TEST(DataTypeDispatchTest, dispatchesNonBoolDataTypes) {
  EXPECT_EQ(dispatchNonBoolDataType(nimble::DataType::Int8), "int8");
  EXPECT_EQ(dispatchNonBoolDataType(nimble::DataType::String), "string_view");

  NIMBLE_ASSERT_THROW(
      dispatchNonBoolDataType(nimble::DataType::Bool),
      "Unsupported non-bool data type");
  NIMBLE_ASSERT_THROW(
      dispatchNonBoolDataType(nimble::DataType::Undefined),
      "Unsupported non-bool data type");
}

TEST(DataTypeDispatchTest, dispatchesNumericDataTypes) {
  EXPECT_EQ(dispatchNumericDataType(nimble::DataType::Int8), "int8");
  EXPECT_EQ(dispatchNumericDataType(nimble::DataType::Double), "double");

  NIMBLE_ASSERT_THROW(
      dispatchNumericDataType(nimble::DataType::Bool),
      "Unsupported numeric data type");
  NIMBLE_ASSERT_THROW(
      dispatchNumericDataType(nimble::DataType::String),
      "Unsupported numeric data type");
}

TEST(DataTypeDispatchTest, dispatchesFloatingPointDataTypes) {
  EXPECT_EQ(dispatchFloatingPointDataType(nimble::DataType::Float), "float");
  EXPECT_EQ(dispatchFloatingPointDataType(nimble::DataType::Double), "double");

  NIMBLE_ASSERT_THROW(
      dispatchFloatingPointDataType(nimble::DataType::Int32),
      "Unsupported floating point data type");
  NIMBLE_ASSERT_THROW(
      dispatchFloatingPointDataType(nimble::DataType::String),
      "Unsupported floating point data type");
}

TEST(DataTypeDispatchTest, dispatchesIntegerDataTypes) {
  EXPECT_EQ(dispatchIntegerDataType(nimble::DataType::Int8), "int8");
  EXPECT_EQ(dispatchIntegerDataType(nimble::DataType::Uint64), "uint64");

  NIMBLE_ASSERT_THROW(
      dispatchIntegerDataType(nimble::DataType::Float),
      "Unsupported integer data type");
  NIMBLE_ASSERT_THROW(
      dispatchIntegerDataType(nimble::DataType::Bool),
      "Unsupported integer data type");
}

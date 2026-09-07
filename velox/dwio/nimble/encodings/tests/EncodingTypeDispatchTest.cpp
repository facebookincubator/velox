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
#include "velox/dwio/nimble/encodings/common/EncodingTypeDispatch.h"

#include <memory>
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

struct ConstructedEncodingBase {
  virtual ~ConstructedEncodingBase() = default;
  virtual std::string typeName() const = 0;
};

template <typename T>
struct ConstructedEncoding : ConstructedEncodingBase {
  ConstructedEncoding(
      int& /*pool*/,
      std::string_view /*data*/,
      int /*stringBufferFactory*/,
      int /*options*/) {}

  std::string typeName() const override {
    return TypeName{}.operator()<T>();
  }
};

std::unique_ptr<ConstructedEncodingBase> makeEncodingByDataType(
    nimble::DataType dataType) {
  int pool{0};
  std::string_view data;
  int stringBufferFactory{0};
  int options{0};
  RETURN_ENCODING_BY_DATA_TYPE(ConstructedEncoding, dataType);
}

std::unique_ptr<ConstructedEncodingBase> makeEncodingByVarintType(
    nimble::DataType dataType) {
  int pool{0};
  std::string_view data;
  int stringBufferFactory{0};
  int options{0};
  RETURN_ENCODING_BY_VARINT_TYPE(ConstructedEncoding, dataType);
}

std::unique_ptr<ConstructedEncodingBase> makeEncodingByNonBoolType(
    nimble::DataType dataType) {
  int pool{0};
  std::string_view data;
  int stringBufferFactory{0};
  int options{0};
  RETURN_ENCODING_BY_NON_BOOL_TYPE(ConstructedEncoding, dataType);
}

std::unique_ptr<ConstructedEncodingBase> makeEncodingByNumericType(
    nimble::DataType dataType) {
  int pool{0};
  std::string_view data;
  int stringBufferFactory{0};
  int options{0};
  RETURN_ENCODING_BY_NUMERIC_TYPE(ConstructedEncoding, dataType);
}

std::unique_ptr<ConstructedEncodingBase> makeEncodingByIntegerType(
    nimble::DataType dataType) {
  int pool{0};
  std::string_view data;
  int stringBufferFactory{0};
  int options{0};
  RETURN_ENCODING_BY_INTEGER_TYPE(ConstructedEncoding, dataType);
}

} // namespace

TEST(EncodingTypeDispatchTest, encodingConstructionMacrosDispatchDataTypes) {
  EXPECT_EQ(makeEncodingByDataType(nimble::DataType::Bool)->typeName(), "bool");
  EXPECT_EQ(
      makeEncodingByDataType(nimble::DataType::String)->typeName(),
      "string_view");

  NIMBLE_ASSERT_THROW(
      makeEncodingByDataType(nimble::DataType::Undefined),
      "Unknown encoding type");
}

TEST(
    EncodingTypeDispatchTest,
    encodingConstructionMacrosDispatchSupportedSubsets) {
  EXPECT_EQ(
      makeEncodingByVarintType(nimble::DataType::Uint32)->typeName(), "uint32");
  EXPECT_EQ(
      makeEncodingByNonBoolType(nimble::DataType::String)->typeName(),
      "string_view");
  EXPECT_EQ(
      makeEncodingByNumericType(nimble::DataType::Double)->typeName(),
      "double");
  EXPECT_EQ(
      makeEncodingByIntegerType(nimble::DataType::Int16)->typeName(), "int16");

  NIMBLE_ASSERT_THROW(
      makeEncodingByVarintType(nimble::DataType::Int16),
      "Trying to deserialize a varint stream");
  NIMBLE_ASSERT_THROW(
      makeEncodingByNonBoolType(nimble::DataType::Bool),
      "Trying to deserialize a non-bool stream");
  NIMBLE_ASSERT_THROW(
      makeEncodingByNumericType(nimble::DataType::String),
      "Trying to deserialize a non-numeric stream");
  NIMBLE_ASSERT_THROW(
      makeEncodingByIntegerType(nimble::DataType::Double),
      "Trying to deserialize an integer stream");
}

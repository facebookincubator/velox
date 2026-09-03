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

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <sstream>

#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble::test {

// --- toString for EncodingType ---

TEST(TypesTest, encodingTypeStringConversion) {
  const std::vector<std::pair<EncodingType, std::string_view>> testCases{
      {EncodingType::Trivial, "Trivial"},
      {EncodingType::RLE, "RLE"},
      {EncodingType::Dictionary, "Dictionary"},
      {EncodingType::FixedBitWidth, "FixedBitWidth"},
      {EncodingType::Sentinel, "Sentinel"},
      {EncodingType::Nullable, "Nullable"},
      {EncodingType::SparseBool, "SparseBool"},
      {EncodingType::Varint, "Varint"},
      {EncodingType::Delta, "Delta"},
      {EncodingType::Constant, "Constant"},
      {EncodingType::MainlyConstant, "MainlyConstant"},
      {EncodingType::Prefix, "Prefix"},
      {EncodingType::ALP, "ALP"},
      {EncodingType::PFOR, "PFOR"},
      {EncodingType::SimdForBitpack, "SimdForBitpack"},
      {EncodingType::BlockBitPacking, "BlockBitPacking"},
      {EncodingType::SubIntSplit, "SubIntSplit"},
      {EncodingType::FrequencyPartition, "FrequencyPartition"},
      {EncodingType::FOR, "FOR"},
      {EncodingType::Fsst, "Fsst"},
      {EncodingType::Huffman, "Huffman"},
      {EncodingType::DeltaBlock, "DeltaBlock"},
  };
  for (const auto& [type, name] : testCases) {
    SCOPED_TRACE(name);
    EXPECT_EQ(toString(type), name);
    EXPECT_EQ(toEncodingType(name), type);
  }
}

TEST(TypesTest, encodingTypeToStringUnknown) {
  auto result = toString(static_cast<EncodingType>(255));
  EXPECT_NE(result.find("Unknown"), std::string::npos);
}

TEST(TypesTest, toEncodingTypeUnknown) {
  EXPECT_ANY_THROW(toEncodingType("unknown"));
}

TEST(TypesTest, readOnlyEncoding) {
  EXPECT_FALSE(isReadOnlyEncoding(EncodingType::PFOR));
  EXPECT_FALSE(isReadOnlyEncoding("PFOR"));
  EXPECT_TRUE(isReadOnlyEncoding(EncodingType::FOR));
  EXPECT_TRUE(isReadOnlyEncoding("FOR"));
  EXPECT_FALSE(isReadOnlyEncoding(EncodingType::Trivial));
  EXPECT_FALSE(isReadOnlyEncoding("Trivial"));
  EXPECT_FALSE(isReadOnlyEncoding("unknown"));
}

TEST(TypesTest, encodingTypeStreamOperator) {
  std::ostringstream ss;
  ss << EncodingType::Trivial;
  EXPECT_EQ(ss.str(), "Trivial");
}

// --- toString for DataType ---

TEST(TypesTest, dataTypeToString) {
  EXPECT_EQ(toString(DataType::Int8), "Int8");
  EXPECT_EQ(toString(DataType::Uint8), "Uint8");
  EXPECT_EQ(toString(DataType::Int16), "Int16");
  EXPECT_EQ(toString(DataType::Uint16), "Uint16");
  EXPECT_EQ(toString(DataType::Int32), "Int32");
  EXPECT_EQ(toString(DataType::Uint32), "Uint32");
  EXPECT_EQ(toString(DataType::Int64), "Int64");
  EXPECT_EQ(toString(DataType::Uint64), "Uint64");
  EXPECT_EQ(toString(DataType::Float), "Float");
  EXPECT_EQ(toString(DataType::Double), "Double");
  EXPECT_EQ(toString(DataType::Bool), "Bool");
  EXPECT_EQ(toString(DataType::String), "String");
}

TEST(TypesTest, dataTypeToStringUnknown) {
  auto result = toString(static_cast<DataType>(200));
  EXPECT_NE(result.find("Unknown"), std::string::npos);
}

TEST(TypesTest, dataTypeStreamOperator) {
  std::ostringstream ss;
  ss << DataType::Int32;
  EXPECT_EQ(ss.str(), "Int32");
}

TEST(TypesTest, dataTypeFmtFormat) {
  EXPECT_EQ(fmt::format("{}", DataType::Int32), "Int32");
  EXPECT_EQ(fmt::format("{}", DataType::String), "String");
}

// --- toString for CompressionType ---

TEST(TypesTest, compressionTypeStringConversion) {
  const std::vector<std::pair<CompressionType, std::string_view>> testCases{
      {CompressionType::Uncompressed, "Uncompressed"},
      {CompressionType::Zstd, "Zstd"},
      {CompressionType::MetaInternal, "MetaInternal"},
      {CompressionType::Lz4, "Lz4"},
      {CompressionType::OpenZL, "OpenZL"},
  };
  for (const auto& [type, name] : testCases) {
    SCOPED_TRACE(name);
    EXPECT_EQ(toString(type), name);
    EXPECT_EQ(toCompressionType(name), type);
  }
}

TEST(TypesTest, compressionTypeToStringUnknown) {
  auto result = toString(static_cast<CompressionType>(200));
  EXPECT_NE(result.find("Unknown"), std::string::npos);
}

TEST(TypesTest, toCompressionTypeUnknown) {
  EXPECT_ANY_THROW(toCompressionType("unknown"));
}

TEST(TypesTest, compressionTypeStreamOperator) {
  std::ostringstream ss;
  ss << CompressionType::Zstd;
  EXPECT_EQ(ss.str(), "Zstd");
}

TEST(TypesTest, compressionTypeFmtFormatter) {
  auto str = fmt::format("{}", CompressionType::Zstd);
  EXPECT_EQ(str, "Zstd");
}

TEST(TypesTest, encodingTypeFmtFormatter) {
  auto str = fmt::format("{}", EncodingType::Dictionary);
  EXPECT_EQ(str, "Dictionary");
}

// --- toString for ChecksumType ---

TEST(TypesTest, checksumTypeToString) {
  EXPECT_EQ(toString(ChecksumType::XXH3_64), "XXH3_64");
}

TEST(TypesTest, checksumTypeToStringUnknown) {
  auto result = toString(static_cast<ChecksumType>(200));
  EXPECT_NE(result.find("Unknown"), std::string::npos);
}

// --- Variant ---

TEST(TypesTest, variantSetGetInt) {
  VariantType v;
  Variant<int32_t>::set(v, 42);
  EXPECT_EQ(Variant<int32_t>::get(v), 42);
}

TEST(TypesTest, variantSetGetNegativeInt) {
  VariantType v;
  Variant<int64_t>::set(v, -12345);
  EXPECT_EQ(Variant<int64_t>::get(v), -12345);
}

TEST(TypesTest, variantSetGetDouble) {
  VariantType v;
  Variant<double>::set(v, 3.14);
  EXPECT_DOUBLE_EQ(Variant<double>::get(v), 3.14);
}

TEST(TypesTest, variantSetGetFloat) {
  VariantType v;
  Variant<float>::set(v, 2.5f);
  EXPECT_FLOAT_EQ(Variant<float>::get(v), 2.5f);
}

TEST(TypesTest, variantSetGetBool) {
  VariantType v;
  Variant<bool>::set(v, true);
  EXPECT_TRUE(Variant<bool>::get(v));

  Variant<bool>::set(v, false);
  EXPECT_FALSE(Variant<bool>::get(v));
}

TEST(TypesTest, variantSetGetString) {
  VariantType v;
  Variant<std::string>::set(v, std::string("hello"));
  EXPECT_EQ(Variant<std::string>::get(v), "hello");
}

TEST(TypesTest, variantSetGetStringView) {
  VariantType v;
  // string_view specialization stores as std::string internally
  Variant<std::string_view>::set(v, std::string_view("world"));
  auto result = Variant<std::string_view>::get(v);
  EXPECT_EQ(result, "world");
}

TEST(TypesTest, variantSetGetUint8) {
  VariantType v;
  Variant<uint8_t>::set(v, 200);
  EXPECT_EQ(Variant<uint8_t>::get(v), 200);
}

TEST(TypesTest, variantSetGetInt16) {
  VariantType v;
  Variant<int16_t>::set(v, -300);
  EXPECT_EQ(Variant<int16_t>::get(v), -300);
}

// --- TypeTraits ---

TEST(TypesTest, typeTraitsDataType) {
  EXPECT_EQ(TypeTraits<int8_t>::dataType, DataType::Int8);
  EXPECT_EQ(TypeTraits<uint8_t>::dataType, DataType::Uint8);
  EXPECT_EQ(TypeTraits<int16_t>::dataType, DataType::Int16);
  EXPECT_EQ(TypeTraits<uint16_t>::dataType, DataType::Uint16);
  EXPECT_EQ(TypeTraits<int32_t>::dataType, DataType::Int32);
  EXPECT_EQ(TypeTraits<uint32_t>::dataType, DataType::Uint32);
  EXPECT_EQ(TypeTraits<int64_t>::dataType, DataType::Int64);
  EXPECT_EQ(TypeTraits<uint64_t>::dataType, DataType::Uint64);
  EXPECT_EQ(TypeTraits<float>::dataType, DataType::Float);
  EXPECT_EQ(TypeTraits<double>::dataType, DataType::Double);
  EXPECT_EQ(TypeTraits<bool>::dataType, DataType::Bool);
  EXPECT_EQ(TypeTraits<std::string>::dataType, DataType::String);
  EXPECT_EQ(TypeTraits<std::string_view>::dataType, DataType::String);
}

TEST(TypesTest, typeTraitsPhysicalType) {
  static_assert(std::is_same_v<TypeTraits<int8_t>::physicalType, uint8_t>);
  static_assert(std::is_same_v<TypeTraits<uint8_t>::physicalType, uint8_t>);
  static_assert(std::is_same_v<TypeTraits<int16_t>::physicalType, uint16_t>);
  static_assert(std::is_same_v<TypeTraits<uint16_t>::physicalType, uint16_t>);
  static_assert(std::is_same_v<TypeTraits<int32_t>::physicalType, uint32_t>);
  static_assert(std::is_same_v<TypeTraits<uint32_t>::physicalType, uint32_t>);
  static_assert(std::is_same_v<TypeTraits<int64_t>::physicalType, uint64_t>);
  static_assert(std::is_same_v<TypeTraits<uint64_t>::physicalType, uint64_t>);
  static_assert(std::is_same_v<TypeTraits<float>::physicalType, uint32_t>);
  static_assert(std::is_same_v<TypeTraits<double>::physicalType, uint64_t>);
  static_assert(std::is_same_v<TypeTraits<bool>::physicalType, bool>);
  static_assert(
      std::is_same_v<TypeTraits<std::string>::physicalType, std::string>);
  static_assert(std::is_same_v<
                TypeTraits<std::string_view>::physicalType,
                std::string_view>);
}

TEST(TypesTest, decodedValueWidth) {
  EXPECT_EQ(decodedValueWidth(DataType::Bool), sizeof(bool));
  EXPECT_EQ(decodedValueWidth(DataType::Int8), sizeof(int8_t));
  EXPECT_EQ(decodedValueWidth(DataType::Uint8), sizeof(uint8_t));
  EXPECT_EQ(decodedValueWidth(DataType::Int16), sizeof(int16_t));
  EXPECT_EQ(decodedValueWidth(DataType::Uint16), sizeof(uint16_t));
  EXPECT_EQ(decodedValueWidth(DataType::Int32), sizeof(int32_t));
  EXPECT_EQ(decodedValueWidth(DataType::Uint32), sizeof(uint32_t));
  EXPECT_EQ(decodedValueWidth(DataType::Int64), sizeof(int64_t));
  EXPECT_EQ(decodedValueWidth(DataType::Uint64), sizeof(uint64_t));
  EXPECT_EQ(decodedValueWidth(DataType::Float), sizeof(float));
  EXPECT_EQ(decodedValueWidth(DataType::Double), sizeof(double));
  EXPECT_EQ(decodedValueWidth(DataType::String), sizeof(std::string_view));
  NIMBLE_ASSERT_THROW(
      decodedValueWidth(DataType::Undefined), "Unsupported data type");
}

// --- Type predicates ---

TEST(TypesTest, isIntegralType) {
  EXPECT_TRUE(isIntegralType<int8_t>());
  EXPECT_TRUE(isIntegralType<uint8_t>());
  EXPECT_TRUE(isIntegralType<int16_t>());
  EXPECT_TRUE(isIntegralType<uint16_t>());
  EXPECT_TRUE(isIntegralType<int32_t>());
  EXPECT_TRUE(isIntegralType<uint32_t>());
  EXPECT_TRUE(isIntegralType<int64_t>());
  EXPECT_TRUE(isIntegralType<uint64_t>());
  EXPECT_FALSE(isIntegralType<float>());
  EXPECT_FALSE(isIntegralType<double>());
  EXPECT_FALSE(isIntegralType<bool>());
  EXPECT_FALSE(isIntegralType<std::string>());
}

TEST(TypesTest, isSignedIntegralType) {
  EXPECT_TRUE(isSignedIntegralType<int8_t>());
  EXPECT_TRUE(isSignedIntegralType<int16_t>());
  EXPECT_TRUE(isSignedIntegralType<int32_t>());
  EXPECT_TRUE(isSignedIntegralType<int64_t>());
  EXPECT_FALSE(isSignedIntegralType<uint32_t>());
  EXPECT_FALSE(isSignedIntegralType<float>());
}

TEST(TypesTest, isUnsignedIntegralType) {
  EXPECT_TRUE(isUnsignedIntegralType<uint8_t>());
  EXPECT_TRUE(isUnsignedIntegralType<uint16_t>());
  EXPECT_TRUE(isUnsignedIntegralType<uint32_t>());
  EXPECT_TRUE(isUnsignedIntegralType<uint64_t>());
  EXPECT_FALSE(isUnsignedIntegralType<int32_t>());
  EXPECT_FALSE(isUnsignedIntegralType<float>());
}

TEST(TypesTest, isOneByteIntegralType) {
  EXPECT_TRUE(isOneByteIntegralType<int8_t>());
  EXPECT_TRUE(isOneByteIntegralType<uint8_t>());
  EXPECT_FALSE(isOneByteIntegralType<int16_t>());
  EXPECT_FALSE(isOneByteIntegralType<uint16_t>());
  EXPECT_FALSE(isOneByteIntegralType<float>());
}

TEST(TypesTest, isTwoByteIntegralType) {
  EXPECT_TRUE(isTwoByteIntegralType<int16_t>());
  EXPECT_TRUE(isTwoByteIntegralType<uint16_t>());
  EXPECT_FALSE(isTwoByteIntegralType<int8_t>());
  EXPECT_FALSE(isTwoByteIntegralType<uint8_t>());
  EXPECT_FALSE(isTwoByteIntegralType<float>());
}

TEST(TypesTest, isFourByteIntegralType) {
  EXPECT_TRUE(isFourByteIntegralType<int32_t>());
  EXPECT_TRUE(isFourByteIntegralType<uint32_t>());
  EXPECT_FALSE(isFourByteIntegralType<int64_t>());
  EXPECT_FALSE(isFourByteIntegralType<int16_t>());
  EXPECT_FALSE(isFourByteIntegralType<float>());
}

TEST(TypesTest, isEightByteIntegralType) {
  EXPECT_TRUE(isEightByteIntegralType<int64_t>());
  EXPECT_TRUE(isEightByteIntegralType<uint64_t>());
  EXPECT_FALSE(isEightByteIntegralType<int32_t>());
  EXPECT_FALSE(isEightByteIntegralType<uint32_t>());
  EXPECT_FALSE(isEightByteIntegralType<int16_t>());
  EXPECT_FALSE(isEightByteIntegralType<double>());
}

TEST(TypesTest, isFloatingPointType) {
  EXPECT_TRUE(isFloatingPointType<float>());
  EXPECT_TRUE(isFloatingPointType<double>());
  EXPECT_FALSE(isFloatingPointType<int32_t>());
  EXPECT_FALSE(isFloatingPointType<bool>());
}

TEST(TypesTest, isNumericType) {
  EXPECT_TRUE(isNumericType<int32_t>());
  EXPECT_TRUE(isNumericType<uint64_t>());
  EXPECT_TRUE(isNumericType<float>());
  EXPECT_TRUE(isNumericType<double>());
  EXPECT_FALSE(isNumericType<bool>());
  EXPECT_FALSE(isNumericType<std::string>());
}

TEST(TypesTest, isStringType) {
  EXPECT_TRUE(isStringType<std::string>());
  EXPECT_TRUE(isStringType<std::string_view>());
  EXPECT_FALSE(isStringType<int32_t>());
  EXPECT_FALSE(isStringType<bool>());
}

TEST(TypesTest, isBoolType) {
  EXPECT_TRUE(isBoolType<bool>());
  EXPECT_FALSE(isBoolType<int32_t>());
  EXPECT_FALSE(isBoolType<std::string>());
}

} // namespace facebook::nimble::test

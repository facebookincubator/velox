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

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "fmt/core.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, supportsEncodingViewMatchesViewableEncodingSet) {
  const std::vector<nimble::EncodingType> supportedEncodings{
      nimble::EncodingType::Constant,
      nimble::EncodingType::Trivial,
      nimble::EncodingType::MainlyConstant,
      nimble::EncodingType::ALP,
      nimble::EncodingType::FixedBitWidth,
      nimble::EncodingType::Dictionary,
      nimble::EncodingType::SparseBool,
      nimble::EncodingType::RLE,
      nimble::EncodingType::FOR,
      nimble::EncodingType::DeltaBlock,
      nimble::EncodingType::Huffman,
      nimble::EncodingType::PFOR,
      nimble::EncodingType::SimdForBitpack,
      nimble::EncodingType::BlockBitPacking};
  for (const auto encodingType : supportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_TRUE(nimble::supportsEncodingView(encodingType));
  }

  const std::vector<nimble::EncodingType> unsupportedEncodings{
      nimble::EncodingType::Sentinel,
      nimble::EncodingType::Nullable,
      nimble::EncodingType::Varint,
      nimble::EncodingType::Delta,
      nimble::EncodingType::Prefix,
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingType::FrequencyPartition,
      nimble::EncodingType::Fsst,
      nimble::EncodingType::MainlyConstantV2,
      nimble::EncodingType::SharedDictionary};
  for (const auto encodingType : unsupportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_FALSE(nimble::supportsEncodingView(encodingType));
  }
}

TEST_F(EncodingViewTest, rejectsCompressedTrivialEncoding) {
  const nimble::Encoding::Options options;
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 4096; ++i) {
    values.push_back(i % 17);
  }
  auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd, options);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "EncodingView does not support compressed Trivial streams");
}

TEST_F(EncodingViewTest, rejectsVarintEncoding) {
  const nimble::Encoding::Options options;
  const auto values = makeVector({10, 11, 12, 1000, 1001});
  auto serialized =
      nimble::test::Encoder<nimble::VarintEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "Varint does not support EncodingView");
}

TEST_F(EncodingViewTest, rejectsUnsupportedEncodingTypes) {
  const std::vector<std::pair<nimble::EncodingType, nimble::DataType>>
      unsupportedEncodings{
          {nimble::EncodingType::Sentinel, nimble::DataType::Int32},
          {nimble::EncodingType::Nullable, nimble::DataType::Int32},
          {nimble::EncodingType::Delta, nimble::DataType::Int32},
          {nimble::EncodingType::Prefix, nimble::DataType::String},
          {nimble::EncodingType::SubIntSplit, nimble::DataType::Uint32},
          {nimble::EncodingType::FrequencyPartition, nimble::DataType::Uint32},
          {nimble::EncodingType::Fsst, nimble::DataType::String},
      };

  const nimble::Encoding::Options options;
  for (const auto& [encodingType, dataType] : unsupportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
    char* pos = serialized.data();
    nimble::encoding::writeChar(static_cast<char>(encodingType), pos);
    nimble::encoding::writeChar(static_cast<char>(dataType), pos);
    nimble::encoding::writeUint32(/*value=*/1, pos);

    NIMBLE_ASSERT_THROW(
        nimble::createEncodingView(serialized, pool_.get(), options),
        "does not support EncodingView");
  }
}

TEST_F(EncodingViewTest, rejectsUnsupportedDataType) {
  const nimble::Encoding::Options options;
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::Trivial);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::Undefined);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "does not support EncodingView");
}

TEST_F(EncodingViewTest, rejectsFixedBitWidthWithNonNumericDataType) {
  const nimble::Encoding::Options options;
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::FixedBitWidth);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::String);

  NIMBLE_ASSERT_USER_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "FixedBitWidth encoding should not be selected for non-numeric data types");
}

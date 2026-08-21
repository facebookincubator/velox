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
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

#include <string>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

using namespace facebook;

namespace {

std::string serializePrefix(
    nimble::EncodingType encodingType,
    nimble::DataType dataType,
    uint32_t rowCount,
    bool useVarint) {
  std::string data(
      nimble::EncodingPrefix::serializedSize(rowCount, useVarint), '\0');
  char* pos = data.data();
  nimble::EncodingPrefix::serialize(
      encodingType, dataType, rowCount, useVarint, pos);
  EXPECT_EQ(pos, data.data() + data.size());
  return data;
}

} // namespace

TEST(EncodingPrefixTest, readsEncodingTypeAndDataType) {
  const auto data = serializePrefix(
      nimble::EncodingType::Delta,
      nimble::DataType::Int64,
      /*rowCount=*/123,
      /*useVarint=*/false);

  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(data), nimble::EncodingType::Delta);
  EXPECT_EQ(nimble::EncodingPrefix::dataType(data), nimble::DataType::Int64);
  EXPECT_EQ(
      nimble::EncodingPrefix::readDataType(data), nimble::DataType::Int64);
}

TEST(EncodingPrefixTest, readsFixedRowCountPrefix) {
  const auto data = serializePrefix(
      nimble::EncodingType::Trivial,
      nimble::DataType::Uint32,
      /*rowCount=*/123,
      /*useVarint=*/false);

  EXPECT_EQ(
      nimble::EncodingPrefix::readRowCount(data, /*useVarint=*/false), 123);
  EXPECT_EQ(
      nimble::EncodingPrefix::prefixSize(data, /*useVarint=*/false),
      nimble::EncodingPrefix::kFixedPrefixSize);
}

TEST(EncodingPrefixTest, readsVarintRowCountPrefix) {
  constexpr uint32_t rowCount = 1'000'000;
  const auto data = serializePrefix(
      nimble::EncodingType::RLE,
      nimble::DataType::String,
      rowCount,
      /*useVarint=*/true);

  EXPECT_EQ(
      nimble::EncodingPrefix::readRowCount(data, /*useVarint=*/true), rowCount);
  EXPECT_EQ(
      nimble::EncodingPrefix::prefixSize(data, /*useVarint=*/true),
      data.size());
}

TEST(EncodingPrefixTest, factoryRejectsTruncatedTypePrefix) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const std::string oneByte{static_cast<char>(nimble::EncodingType::Trivial)};

  for (const std::string_view data :
       {std::string_view{}, std::string_view{oneByte}}) {
    try {
      nimble::EncodingFactory{}.create(*pool, data, nullptr);
      FAIL() << "Expected a truncated prefix to fail";
    } catch (const nimble::NimbleUserError& error) {
      EXPECT_EQ(nimble::error_code::CorruptedFile, error.errorCode());
      EXPECT_EQ("Truncated encoding prefix.", error.errorMessage());
    }
  }
}

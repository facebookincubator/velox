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
#include <sstream>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Options.h"

using namespace facebook::nimble;

TEST(OptionsTest, serializationVersionEnumValues) {
  // Verify enum underlying values match expected wire format versions.
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kLegacyCompact), 2);
  EXPECT_EQ(
      static_cast<uint8_t>(SerializationVersion::kLegacySerialization), 3);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kSerialization), 4);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kProjection), 5);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kTablet), 6);
}

TEST(OptionsTest, toStringVersion) {
  EXPECT_EQ(toString(SerializationVersion::kLegacyCompact), "kLegacyCompact");
  EXPECT_EQ(
      toString(SerializationVersion::kLegacySerialization),
      "kLegacySerialization");
  EXPECT_EQ(toString(SerializationVersion::kSerialization), "kSerialization");
  EXPECT_EQ(toString(SerializationVersion::kProjection), "kProjection");
  EXPECT_EQ(toString(SerializationVersion::kTablet), "kTablet");
}

TEST(OptionsTest, streamOperator) {
  std::ostringstream os;
  os << SerializationVersion::kTablet;
  EXPECT_EQ(os.str(), "kTablet");
}

TEST(OptionsTest, fmtFormatter) {
  EXPECT_EQ(
      fmt::format("{}", SerializationVersion::kLegacyCompact),
      "kLegacyCompact");
  EXPECT_EQ(fmt::format("{}", SerializationVersion::kTablet), "kTablet");
}

TEST(OptionsTest, serializerOptionsDefaults) {
  SerializerOptions options{};

  // Verify default values.
  EXPECT_EQ(options.version, SerializationVersion::kSerialization);
  EXPECT_TRUE(options.flatMapColumns.empty());

  // Verify encoding layout tree and compression options defaults.
  EXPECT_FALSE(options.encodingLayoutTree.has_value());
  EXPECT_FALSE(options.compressionOptions.has_value());

  // Verify default encoding selection policy factory creates a valid policy.
  auto policy = options.encodingSelectionPolicyCreator(DataType::Int32);
  EXPECT_NE(policy, nullptr);
}

TEST(OptionsTest, serializerOptionsWithFlatMapColumns) {
  SerializerOptions options{
      .version = SerializationVersion::kLegacyCompact,
      .flatMapColumns = {{"col1", {}}, {"col2", {}}},
  };

  EXPECT_EQ(options.version, SerializationVersion::kLegacyCompact);
  EXPECT_EQ(options.flatMapColumns.size(), 2);
  EXPECT_TRUE(options.flatMapColumns.contains("col1"));
  EXPECT_TRUE(options.flatMapColumns.contains("col2"));
}

TEST(OptionsTest, deserializerOptionsDefaults) {
  DeserializerOptions options{};

  EXPECT_EQ(options.decodeExecutor, nullptr);
  EXPECT_EQ(options.maxDecodeParallelism, 0u);
  EXPECT_TRUE(options.decodePools.empty());
}

TEST(OptionsTest, serializerOptionsWithTabletVersion) {
  SerializerOptions options{.version = SerializationVersion::kTablet};

  EXPECT_EQ(options.version, SerializationVersion::kTablet);
}

TEST(OptionsTest, serializationHeaderFlags) {
  struct TestCase {
    SerializationVersion version;
    bool expectedHasSerializationHeaderFlags;
    bool expectedUsesCompactHeaderFlags;
  };

  const TestCase testCases[]{
      {
          .version = SerializationVersion::kLegacyCompact,
          .expectedHasSerializationHeaderFlags = false,
          .expectedUsesCompactHeaderFlags = false,
      },
      {
          .version = SerializationVersion::kLegacySerialization,
          .expectedHasSerializationHeaderFlags = false,
          .expectedUsesCompactHeaderFlags = false,
      },
      {
          .version = SerializationVersion::kSerialization,
          .expectedHasSerializationHeaderFlags = true,
          .expectedUsesCompactHeaderFlags = true,
      },
      {
          .version = SerializationVersion::kProjection,
          .expectedHasSerializationHeaderFlags = true,
          .expectedUsesCompactHeaderFlags = true,
      },
      {
          .version = SerializationVersion::kTablet,
          .expectedHasSerializationHeaderFlags = true,
          .expectedUsesCompactHeaderFlags = false,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(toString(testCase.version));
    EXPECT_EQ(
        hasSerializationHeaderFlags(testCase.version),
        testCase.expectedHasSerializationHeaderFlags);
    EXPECT_EQ(
        usesCompactHeaderFlags(testCase.version),
        testCase.expectedUsesCompactHeaderFlags);
  }
}

TEST(OptionsTest, isTabletVersion) {
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kLegacyCompact));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kLegacySerialization));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kSerialization));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kProjection));
  EXPECT_TRUE(isTabletVersion(SerializationVersion::kTablet));
}

TEST(OptionsTest, getTrailerEncodingTypeBasic) {
  EXPECT_EQ(
      getTrailerEncodingType(EncodingType::Trivial), EncodingType::Trivial);
  EXPECT_EQ(getTrailerEncodingType(EncodingType::Varint), EncodingType::Varint);
  EXPECT_EQ(getTrailerEncodingType(EncodingType::Delta), EncodingType::Delta);
  EXPECT_EQ(
      getTrailerEncodingType(EncodingType::FixedBitWidth),
      EncodingType::FixedBitWidth);
}

TEST(OptionsTest, getTrailerEncodingTypeError) {
  NIMBLE_ASSERT_THROW(
      getTrailerEncodingType(EncodingType::RLE),
      "Unsupported EncodingType for stream sizes trailer");
  NIMBLE_ASSERT_THROW(
      getTrailerEncodingType(EncodingType::Dictionary),
      "Unsupported EncodingType for stream sizes trailer");
  NIMBLE_ASSERT_THROW(
      getTrailerEncodingType(EncodingType::MainlyConstant),
      "Unsupported EncodingType for stream sizes trailer");
}

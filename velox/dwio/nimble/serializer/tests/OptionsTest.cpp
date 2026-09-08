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
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kLegacy), 0);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kLegacyCompact), 2);
  EXPECT_EQ(
      static_cast<uint8_t>(SerializationVersion::kLegacySerialization), 3);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kSerialization), 4);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kProjection), 5);
  EXPECT_EQ(static_cast<uint8_t>(SerializationVersion::kTablet), 6);
}

TEST(OptionsTest, toStringVersion) {
  EXPECT_EQ(toString(SerializationVersion::kLegacy), "kLegacy");
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
  EXPECT_EQ(options.compressionType, CompressionType::Uncompressed);
  EXPECT_EQ(options.compressionThreshold, 0);
  EXPECT_EQ(options.compressionLevel, 0);
  EXPECT_EQ(options.version, std::nullopt);
  EXPECT_TRUE(options.flatMapColumns.empty());

  // Verify helper methods with defaults.
  EXPECT_FALSE(options.hasVersionHeader());
  EXPECT_EQ(options.serializationVersion(), SerializationVersion::kLegacy);
  EXPECT_FALSE(options.enableEncoding());

  // Verify encoding layout tree and compression options defaults.
  EXPECT_FALSE(options.encodingLayoutTree.has_value());
  EXPECT_FALSE(options.compressionOptions.has_value());

  // Verify default encoding selection policy factory creates a valid policy.
  auto policy = options.encodingSelectionPolicyCreator(DataType::Int32);
  EXPECT_NE(policy, nullptr);
}

TEST(OptionsTest, serializerOptionsWithLegacyVersion) {
  // Raw options preserve explicit kLegacy; Serializer normalizes it before
  // writing.
  SerializerOptions options{.version = SerializationVersion::kLegacy};

  EXPECT_TRUE(options.version.has_value());
  EXPECT_EQ(*options.version, SerializationVersion::kLegacy);
  EXPECT_TRUE(options.hasVersionHeader());
  EXPECT_EQ(options.serializationVersion(), SerializationVersion::kLegacy);
  EXPECT_FALSE(options.enableEncoding());
}

TEST(OptionsTest, serializerOptionsWithFlatMapColumns) {
  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionLevel = 3,
      .version = SerializationVersion::kLegacyCompact,
      .flatMapColumns = {{"col1", {}}, {"col2", {}}},
  };

  EXPECT_EQ(options.compressionType, CompressionType::Zstd);
  EXPECT_EQ(options.compressionLevel, 3);
  EXPECT_TRUE(options.hasVersionHeader());
  EXPECT_EQ(options.flatMapColumns.size(), 2);
  EXPECT_TRUE(options.flatMapColumns.contains("col1"));
  EXPECT_TRUE(options.flatMapColumns.contains("col2"));
}

TEST(OptionsTest, deserializerOptionsDefaults) {
  DeserializerOptions options{};

  EXPECT_FALSE(options.hasHeader);
  EXPECT_EQ(options.decodeExecutor, nullptr);
  EXPECT_EQ(options.maxDecodeParallelism, 0u);
  EXPECT_TRUE(options.decodePools.empty());
}

TEST(OptionsTest, deserializerOptionsWithVersion) {
  DeserializerOptions options{.hasHeader = true};

  EXPECT_TRUE(options.hasHeader);
}

TEST(OptionsTest, serializerOptionsWithCompactRawVersion) {
  SerializerOptions options{.version = SerializationVersion::kLegacyCompact};

  EXPECT_TRUE(options.hasVersionHeader());
  EXPECT_EQ(
      options.serializationVersion(), SerializationVersion::kLegacyCompact);
  EXPECT_TRUE(options.enableEncoding());
}

TEST(OptionsTest, serializerOptionsWithTabletVersion) {
  SerializerOptions options{.version = SerializationVersion::kTablet};

  EXPECT_TRUE(options.hasVersionHeader());
  EXPECT_EQ(options.serializationVersion(), SerializationVersion::kTablet);
  EXPECT_TRUE(options.enableEncoding());
}

TEST(OptionsTest, nonLegacyFormat) {
  EXPECT_FALSE(nonLegacyFormat(SerializationVersion::kLegacy));
  EXPECT_TRUE(nonLegacyFormat(SerializationVersion::kLegacyCompact));
  EXPECT_TRUE(nonLegacyFormat(SerializationVersion::kLegacySerialization));
  EXPECT_TRUE(nonLegacyFormat(SerializationVersion::kSerialization));
  EXPECT_TRUE(nonLegacyFormat(SerializationVersion::kProjection));
  EXPECT_TRUE(nonLegacyFormat(SerializationVersion::kTablet));
}

TEST(OptionsTest, serializationHeaderFlags) {
  struct TestCase {
    SerializationVersion version;
    bool expectedHasSerializationHeaderFlags;
    bool expectedUsesCompactHeaderFlags;
  };

  const TestCase testCases[]{
      {
          .version = SerializationVersion::kLegacy,
          .expectedHasSerializationHeaderFlags = false,
          .expectedUsesCompactHeaderFlags = false,
      },
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

  EXPECT_FALSE(hasSerializationHeaderFlags(std::nullopt));
  EXPECT_FALSE(usesCompactHeaderFlags(std::nullopt));
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(toString(testCase.version));
    EXPECT_EQ(
        hasSerializationHeaderFlags(testCase.version),
        testCase.expectedHasSerializationHeaderFlags);
    EXPECT_EQ(
        hasSerializationHeaderFlags(std::optional{testCase.version}),
        testCase.expectedHasSerializationHeaderFlags);
    EXPECT_EQ(
        usesCompactHeaderFlags(testCase.version),
        testCase.expectedUsesCompactHeaderFlags);
    EXPECT_EQ(
        usesCompactHeaderFlags(std::optional{testCase.version}),
        testCase.expectedUsesCompactHeaderFlags);
  }
}

TEST(OptionsTest, isTabletVersion) {
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kLegacy));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kLegacyCompact));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kLegacySerialization));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kSerialization));
  EXPECT_FALSE(isTabletVersion(SerializationVersion::kProjection));
  EXPECT_TRUE(isTabletVersion(SerializationVersion::kTablet));
}

TEST(OptionsTest, isTabletVersionOptional) {
  EXPECT_FALSE(isTabletVersion(std::nullopt));
  EXPECT_FALSE(isTabletVersion(std::optional{SerializationVersion::kLegacy}));
  EXPECT_FALSE(
      isTabletVersion(std::optional{SerializationVersion::kLegacyCompact}));
  EXPECT_FALSE(isTabletVersion(
      std::optional{SerializationVersion::kLegacySerialization}));
  EXPECT_FALSE(
      isTabletVersion(std::optional{SerializationVersion::kSerialization}));
  EXPECT_FALSE(
      isTabletVersion(std::optional{SerializationVersion::kProjection}));
  EXPECT_TRUE(isTabletVersion(std::optional{SerializationVersion::kTablet}));
}

TEST(OptionsTest, nonLegacyNonTabletSelectsSequentialStreamLayout) {
  struct TestCase {
    SerializationVersion version;
    bool expected;
  };

  const TestCase testCases[] = {
      {SerializationVersion::kLegacy, false},
      {SerializationVersion::kLegacyCompact, true},
      {SerializationVersion::kLegacySerialization, true},
      {SerializationVersion::kSerialization, true},
      {SerializationVersion::kProjection, true},
      {SerializationVersion::kTablet, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(toString(testCase.version));
    EXPECT_EQ(
        nonLegacyFormat(testCase.version) && !isTabletVersion(testCase.version),
        testCase.expected);
  }
}

TEST(OptionsTest, usesVarintRowCount) {
  EXPECT_FALSE(usesVarintRowCount(SerializationVersion::kLegacy));
  EXPECT_TRUE(usesVarintRowCount(SerializationVersion::kLegacyCompact));
  EXPECT_TRUE(usesVarintRowCount(SerializationVersion::kLegacySerialization));
  EXPECT_TRUE(usesVarintRowCount(SerializationVersion::kSerialization));
  EXPECT_TRUE(usesVarintRowCount(SerializationVersion::kProjection));
  EXPECT_TRUE(usesVarintRowCount(SerializationVersion::kTablet));
}

TEST(OptionsTest, usesVarintRowCountOptional) {
  EXPECT_FALSE(usesVarintRowCount(std::nullopt));
  EXPECT_FALSE(
      usesVarintRowCount(std::optional{SerializationVersion::kLegacy}));
  EXPECT_TRUE(
      usesVarintRowCount(std::optional{SerializationVersion::kLegacyCompact}));
  EXPECT_TRUE(usesVarintRowCount(
      std::optional{SerializationVersion::kLegacySerialization}));
  EXPECT_TRUE(
      usesVarintRowCount(std::optional{SerializationVersion::kSerialization}));
  EXPECT_TRUE(
      usesVarintRowCount(std::optional{SerializationVersion::kProjection}));
  EXPECT_TRUE(usesVarintRowCount(std::optional{SerializationVersion::kTablet}));
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

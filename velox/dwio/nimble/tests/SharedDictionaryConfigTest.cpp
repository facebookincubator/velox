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

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble {
namespace {

TEST(SharedDictionaryConfigTest, defaultValues) {
  SharedDictionaryConfig dictionary;
  EXPECT_EQ(dictionary.scope, SharedDictionaryScope::Stripe);
  EXPECT_EQ(dictionary.dictionaryId, 0);
  EXPECT_FALSE(dictionary.useExternalAlphabet);
  EXPECT_TRUE(dictionary.alphabetEncodings.empty());

  SharedDictionaryEncodingConfig config;
  EXPECT_TRUE(config.empty());
  EXPECT_TRUE(config.columns.empty());
  EXPECT_TRUE(config.flatMaps.empty());
  EXPECT_EQ(config.externalResolver, nullptr);
}

TEST(SharedDictionaryConfigTest, config) {
  const auto config =
      SharedDictionaryEncodingConfig::builder()
          .addColumnDictionary(
              "value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::Stripe, .dictionaryId = 5})
          .addColumnDictionary(
              "nested.value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::External, .dictionaryId = 9})
          .addColumnDictionary(
              "nested.items[*].value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::File, .dictionaryId = 11})
          .addFlatmapValueDictionary(
              "features",
              /*key=*/42,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::File,
                  .dictionaryId = 17,
                  .useExternalAlphabet = true,
                  .alphabetEncodings = {EncodingType::FixedBitWidth}},
              "items[*].score")
          .addFlatmapValueDictionary(
              "features",
              /*key=*/43,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::File, .dictionaryId = 18})
          .build();

  EXPECT_FALSE(config.empty());
  ASSERT_EQ(config.columns.size(), 3);
  ASSERT_EQ(config.flatMaps.size(), 1);
  EXPECT_EQ(config.externalResolver, nullptr);

  EXPECT_EQ(config.columns[0].fieldPath, "value");
  EXPECT_EQ(config.columns[0].dictionary.scope, SharedDictionaryScope::Stripe);
  EXPECT_EQ(config.columns[0].dictionary.dictionaryId, 5);

  EXPECT_EQ(config.columns[1].fieldPath, "nested.value");
  EXPECT_EQ(
      config.columns[1].dictionary.scope, SharedDictionaryScope::External);
  EXPECT_EQ(config.columns[1].dictionary.dictionaryId, 9);

  EXPECT_EQ(config.columns[2].fieldPath, "nested.items[*].value");
  EXPECT_EQ(config.columns[2].dictionary.scope, SharedDictionaryScope::File);
  EXPECT_EQ(config.columns[2].dictionary.dictionaryId, 11);

  const auto& columnDictionary = config.flatMaps[0];
  EXPECT_EQ(columnDictionary.fieldPath, "features");
  ASSERT_EQ(columnDictionary.keys.size(), 2);
  const auto& keyDictionary = columnDictionary.keys[0];
  EXPECT_EQ(keyDictionary.key, 42);
  EXPECT_EQ(keyDictionary.valueSubfield, "items[*].score");
  EXPECT_EQ(keyDictionary.dictionary.scope, SharedDictionaryScope::File);
  EXPECT_EQ(keyDictionary.dictionary.dictionaryId, 17);
  EXPECT_TRUE(keyDictionary.dictionary.useExternalAlphabet);
  EXPECT_EQ(
      keyDictionary.dictionary.alphabetEncodings,
      std::vector<EncodingType>{EncodingType::FixedBitWidth});

  const auto& arrayKeyDictionary = columnDictionary.keys[1];
  EXPECT_EQ(arrayKeyDictionary.key, 43);
  EXPECT_EQ(arrayKeyDictionary.valueSubfield, "");
  EXPECT_EQ(arrayKeyDictionary.dictionary.scope, SharedDictionaryScope::File);
  EXPECT_EQ(arrayKeyDictionary.dictionary.dictionaryId, 18);
}

TEST(SharedDictionaryConfigTest, builderSeedsExistingConfig) {
  const auto seedConfig =
      SharedDictionaryEncodingConfig::builder()
          .addColumnDictionary(
              "value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::Stripe, .dictionaryId = 5})
          .build();
  auto configCopy = seedConfig;

  const auto config =
      SharedDictionaryEncodingConfig::builder(std::move(configCopy))
          .addFlatmapValueDictionary(
              "features",
              /*key=*/10,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::External, .dictionaryId = 9})
          .build();

  ASSERT_EQ(config.columns.size(), seedConfig.columns.size());
  ASSERT_EQ(config.flatMaps.size(), 1);
  EXPECT_EQ(config.columns[0].fieldPath, seedConfig.columns[0].fieldPath);
  EXPECT_EQ(
      config.columns[0].dictionary.scope,
      seedConfig.columns[0].dictionary.scope);
  EXPECT_EQ(
      config.columns[0].dictionary.dictionaryId,
      seedConfig.columns[0].dictionary.dictionaryId);
  EXPECT_TRUE(seedConfig.flatMaps.empty());
  EXPECT_EQ(config.flatMaps[0].fieldPath, "features");
  ASSERT_EQ(config.flatMaps[0].keys.size(), 1);
  EXPECT_EQ(config.flatMaps[0].keys[0].key, 10);
}

TEST(SharedDictionaryConfigTest, builderRejectsDuplicateTargets) {
  auto config =
      SharedDictionaryEncodingConfig::builder()
          .addColumnDictionary("nested.value", SharedDictionaryConfig{})
          .addFlatmapValueDictionary(
              "features", /*key=*/42, SharedDictionaryConfig{})
          .addFlatmapValueDictionary(
              "features",
              /*key=*/42,
              SharedDictionaryConfig{},
              "items[*].value")
          .build();
  auto builder = SharedDictionaryEncodingConfig::builder(std::move(config));

  NIMBLE_ASSERT_USER_THROW(
      builder.addColumnDictionary("nested.value", SharedDictionaryConfig{}),
      "Duplicate shared dictionary column configuration for path "
      "'nested.value'.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addFlatmapValueDictionary(
          "features", /*key=*/42, SharedDictionaryConfig{}),
      "Duplicate shared dictionary flat-map value configuration for path "
      "'features', key 42, and value subfield ''.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addFlatmapValueDictionary(
          "features", /*key=*/42, SharedDictionaryConfig{}, "items[*].value"),
      "Duplicate shared dictionary flat-map value configuration for path "
      "'features', key 42, and value subfield 'items[*].value'.");
}

TEST(SharedDictionaryConfigTest, builderRejectsInvalidPaths) {
  struct TestCase {
    std::string_view name;
    std::string_view fieldPath;
    bool flatMap;
    std::string_view expectedMessage;
  };

  const std::vector<TestCase> testCases{
      {
          .name = "empty column path",
          .fieldPath = "",
          .flatMap = false,
          .expectedMessage = "Shared dictionary path must not be empty.",
      },
      {
          .name = "column path with subscript",
          .fieldPath = "features[10]",
          .flatMap = false,
          .expectedMessage =
              "Shared dictionary column path 'features[10]' only supports "
              "nested row fields and all-subscript array/map elements.",
      },
      {
          .name = "empty flat-map path",
          .fieldPath = "",
          .flatMap = true,
          .expectedMessage = "Shared dictionary path must not be empty.",
      },
      {
          .name = "nested flat-map path",
          .fieldPath = "nested.features",
          .flatMap = true,
          .expectedMessage =
              "Shared dictionary flat-map column path 'nested.features' must "
              "be a top-level writer input column.",
      },
      {
          .name = "flat-map path with subscript",
          .fieldPath = "features[10]",
          .flatMap = true,
          .expectedMessage =
              "Shared dictionary flat-map column path 'features[10]' must be "
              "a top-level writer input column.",
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    auto builder = SharedDictionaryEncodingConfig::builder();
    if (testCase.flatMap) {
      NIMBLE_ASSERT_USER_THROW(
          builder.addFlatmapValueDictionary(
              std::string{testCase.fieldPath},
              /*key=*/10,
              SharedDictionaryConfig{}),
          testCase.expectedMessage);
    } else {
      NIMBLE_ASSERT_USER_THROW(
          builder.addColumnDictionary(
              std::string{testCase.fieldPath}, SharedDictionaryConfig{}),
          testCase.expectedMessage);
    }
  }
}

TEST(SharedDictionaryConfigTest, builderRejectsInvalidValueSubfields) {
  struct TestCase {
    std::string_view name;
    std::string_view valueSubfield;
    std::string_view expectedMessage;
  };

  const std::vector<TestCase> testCases{
      {
          .name = "map key subscript",
          .valueSubfield = "items[10].value",
          .expectedMessage =
              "Shared dictionary flat-map value subfield 'items[10].value' "
              "only supports nested row fields and all-subscript array/map "
              "elements.",
      },
      {
          .name = "leading key subscript",
          .valueSubfield = "[10].value",
          .expectedMessage =
              "Shared dictionary path '[10].value' must start with a field "
              "name.",
      },
      {
          .name = "leading all subscript",
          .valueSubfield = "[*]",
          .expectedMessage =
              "Shared dictionary path '[*]' must start with a field name.",
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    auto builder = SharedDictionaryEncodingConfig::builder();
    NIMBLE_ASSERT_USER_THROW(
        builder.addFlatmapValueDictionary(
            "features",
            /*key=*/10,
            SharedDictionaryConfig{},
            std::string{testCase.valueSubfield}),
        testCase.expectedMessage);
  }
}

} // namespace
} // namespace facebook::nimble

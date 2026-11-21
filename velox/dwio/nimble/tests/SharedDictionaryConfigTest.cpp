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

#include <gmock/gmock.h>
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
  EXPECT_EQ(dictionary.resolverKey, 0);
  EXPECT_FALSE(dictionary.useExternalAlphabet);
  EXPECT_TRUE(dictionary.alphabetEncodings.empty());

  SharedDictionaryEncodingConfig config;
  EXPECT_TRUE(config.empty());
  EXPECT_FALSE(config.hasStripeDictionary());
  EXPECT_TRUE(config.alphabetEncodings.empty());
  EXPECT_TRUE(config.columns.empty());
  EXPECT_TRUE(config.provided.empty());
  EXPECT_TRUE(config.providedFlatMaps.empty());
  EXPECT_EQ(config.externalResolver, nullptr);
}

TEST(SharedDictionaryConfigTest, enabledColumns) {
  const auto config = SharedDictionaryEncodingConfig::builder()
                          .setAlphabetEncodings({EncodingType::FixedBitWidth})
                          .addColumn("value", SharedDictionaryScope::File)
                          .addColumn("features", SharedDictionaryScope::File)
                          .build();

  EXPECT_FALSE(config.empty());
  // File scope is writer-assigned but not stripe scoped.
  EXPECT_FALSE(config.hasStripeDictionary());
  EXPECT_EQ(
      config.alphabetEncodings,
      std::vector<EncodingType>{EncodingType::FixedBitWidth});
  EXPECT_THAT(
      config.columns,
      ::testing::UnorderedElementsAre(
          ::testing::Pair("value", SharedDictionaryScope::File),
          ::testing::Pair("features", SharedDictionaryScope::File)));
  EXPECT_TRUE(config.provided.empty());
  EXPECT_TRUE(config.providedFlatMaps.empty());
}

TEST(SharedDictionaryConfigTest, stripeScopeIsTheDefaultForEnabledColumns) {
  const auto config =
      SharedDictionaryEncodingConfig::builder().addColumn("value").build();
  EXPECT_THAT(
      config.columns,
      ::testing::ElementsAre(
          ::testing::Pair("value", SharedDictionaryScope::Stripe)));
  EXPECT_TRUE(config.hasStripeDictionary());
}

TEST(SharedDictionaryConfigTest, enabledColumnsMayMixWriterAssignedScopes) {
  // One file can hold both, which the per-value-stream configuration allowed
  // and a single file-wide scope would not.
  const auto config =
      SharedDictionaryEncodingConfig::builder()
          .addColumn("stripeColumn", SharedDictionaryScope::Stripe)
          .addColumn("fileColumn", SharedDictionaryScope::File)
          .build();
  EXPECT_THAT(
      config.columns,
      ::testing::UnorderedElementsAre(
          ::testing::Pair("stripeColumn", SharedDictionaryScope::Stripe),
          ::testing::Pair("fileColumn", SharedDictionaryScope::File)));
  EXPECT_TRUE(config.hasStripeDictionary());
}

TEST(SharedDictionaryConfigTest, providedDictionaries) {
  const auto config =
      SharedDictionaryEncodingConfig::builder()
          .addProvidedDictionary(
              "nested.value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::External, .dictionaryId = 9})
          .addProvidedDictionary(
              "nested.items[*].value",
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::File,
                  .useExternalAlphabet = true,
                  .resolverKey = 11})
          .addProvidedFlatmapValueDictionary(
              "features",
              /*key=*/42,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::File,
                  .useExternalAlphabet = true,
                  .resolverKey = 17,
                  .alphabetEncodings = {EncodingType::FixedBitWidth}},
              "items[*].score")
          .addProvidedFlatmapValueDictionary(
              "features",
              /*key=*/43,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::External, .dictionaryId = 18})
          .build();

  EXPECT_FALSE(config.empty());
  EXPECT_FALSE(config.hasStripeDictionary());
  EXPECT_TRUE(config.columns.empty());
  ASSERT_EQ(config.provided.size(), 2);
  ASSERT_EQ(config.providedFlatMaps.size(), 1);

  // Struct members stay reachable here: a provided dictionary addresses one
  // value stream, so a path is meaningful.
  EXPECT_EQ(config.provided[0].fieldPath, "nested.value");
  EXPECT_EQ(
      config.provided[0].dictionary.scope, SharedDictionaryScope::External);
  EXPECT_EQ(config.provided[0].dictionary.dictionaryId, 9);

  EXPECT_EQ(config.provided[1].fieldPath, "nested.items[*].value");
  EXPECT_EQ(config.provided[1].dictionary.scope, SharedDictionaryScope::File);
  EXPECT_TRUE(config.provided[1].dictionary.useExternalAlphabet);
  EXPECT_EQ(config.provided[1].dictionary.resolverKey, 11);

  const auto& flatMap = config.providedFlatMaps[0];
  EXPECT_EQ(flatMap.fieldPath, "features");
  ASSERT_EQ(flatMap.keys.size(), 2);
  EXPECT_EQ(flatMap.keys[0].key, 42);
  EXPECT_EQ(flatMap.keys[0].valueSubfield, "items[*].score");
  EXPECT_EQ(flatMap.keys[0].dictionary.resolverKey, 17);
  EXPECT_EQ(
      flatMap.keys[0].dictionary.alphabetEncodings,
      std::vector<EncodingType>{EncodingType::FixedBitWidth});
  EXPECT_EQ(flatMap.keys[1].key, 43);
  EXPECT_EQ(flatMap.keys[1].valueSubfield, "");
  EXPECT_EQ(flatMap.keys[1].dictionary.dictionaryId, 18);
}

TEST(SharedDictionaryConfigTest, builderSeedsExistingConfig) {
  const auto seedConfig =
      SharedDictionaryEncodingConfig::builder().addColumn("value").build();
  auto configCopy = seedConfig;

  const auto config =
      SharedDictionaryEncodingConfig::builder(std::move(configCopy))
          .addProvidedFlatmapValueDictionary(
              "features",
              /*key=*/10,
              SharedDictionaryConfig{
                  .scope = SharedDictionaryScope::External, .dictionaryId = 9})
          .build();

  EXPECT_EQ(config.columns, seedConfig.columns);
  EXPECT_TRUE(seedConfig.providedFlatMaps.empty());
  ASSERT_EQ(config.providedFlatMaps.size(), 1);
  EXPECT_EQ(config.providedFlatMaps[0].fieldPath, "features");
  ASSERT_EQ(config.providedFlatMaps[0].keys.size(), 1);
  EXPECT_EQ(config.providedFlatMaps[0].keys[0].key, 10);
}

TEST(SharedDictionaryConfigTest, builderRejectsDuplicateTargets) {
  const SharedDictionaryConfig external{
      .scope = SharedDictionaryScope::External};
  auto config =
      SharedDictionaryEncodingConfig::builder()
          .addColumn("value")
          .addProvidedDictionary("nested.value", external)
          .addProvidedFlatmapValueDictionary("features", /*key=*/42, external)
          .addProvidedFlatmapValueDictionary(
              "features", /*key=*/42, external, "items[*].value")
          .build();
  auto builder = SharedDictionaryEncodingConfig::builder(std::move(config));

  NIMBLE_ASSERT_USER_THROW(
      builder.addColumn("value"),
      "Duplicate shared dictionary column 'value'.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addProvidedDictionary("nested.value", external),
      "Duplicate provided shared dictionary configuration for path "
      "'nested.value'.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addProvidedFlatmapValueDictionary(
          "features", /*key=*/42, external),
      "Duplicate provided shared dictionary flat-map value configuration for "
      "path 'features', key 42, and value subfield ''.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addProvidedFlatmapValueDictionary(
          "features", /*key=*/42, external, "items[*].value"),
      "Duplicate provided shared dictionary flat-map value configuration for "
      "path 'features', key 42, and value subfield 'items[*].value'.");
}

TEST(SharedDictionaryConfigTest, builderRejectsColumnPaths) {
  auto builder = SharedDictionaryEncodingConfig::builder();
  NIMBLE_ASSERT_USER_THROW(
      builder.addColumn(""),
      "Shared dictionary column name must not be empty.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addColumn("nested.value"),
      "Shared dictionary column 'nested.value' must be a top-level writer "
      "input column name, not a path.");
  NIMBLE_ASSERT_USER_THROW(
      builder.addColumn("features[*]"),
      "Shared dictionary column 'features[*]' must be a top-level writer "
      "input column name, not a path.");
}

TEST(SharedDictionaryConfigTest, builderRejectsWriterAssignedProvidedScopes) {
  auto builder = SharedDictionaryEncodingConfig::builder();
  NIMBLE_ASSERT_USER_THROW(
      builder.addProvidedDictionary(
          "value",
          SharedDictionaryConfig{.scope = SharedDictionaryScope::Stripe}),
      "Provided shared dictionary 'value' must use external scope, or file "
      "scope with useExternalAlphabet.");
  // File scope without a provided alphabet has no resolver key to carry.
  NIMBLE_ASSERT_USER_THROW(
      builder.addProvidedDictionary(
          "value",
          SharedDictionaryConfig{.scope = SharedDictionaryScope::File}),
      "Provided shared dictionary 'value' must use external scope, or file "
      "scope with useExternalAlphabet.");
}

TEST(SharedDictionaryConfigTest, builderRejectsExternalColumnScope) {
  auto builder = SharedDictionaryEncodingConfig::builder();
  NIMBLE_ASSERT_USER_THROW(
      builder.addColumn("value", SharedDictionaryScope::External),
      "Enabled column 'value' cannot use external scope");
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
          builder.addProvidedFlatmapValueDictionary(
              std::string{testCase.fieldPath},
              /*key=*/10,
              SharedDictionaryConfig{.scope = SharedDictionaryScope::External}),
          testCase.expectedMessage);
    } else {
      NIMBLE_ASSERT_USER_THROW(
          builder.addProvidedDictionary(
              std::string{testCase.fieldPath},
              SharedDictionaryConfig{.scope = SharedDictionaryScope::External}),
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
        builder.addProvidedFlatmapValueDictionary(
            "features",
            /*key=*/10,
            SharedDictionaryConfig{.scope = SharedDictionaryScope::External},
            std::string{testCase.valueSubfield}),
        testCase.expectedMessage);
  }
}

} // namespace
} // namespace facebook::nimble

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

#include <array>
#include <cstdint>
#include <span>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"

namespace facebook::nimble {
namespace {

class SharedDictionaryCatalogTest : public testing::Test {
 protected:
  static void expectReference(
      const SharedDictionaryReference& reference,
      uint32_t valueStreamId,
      uint32_t dictionaryId,
      DataType dataType) {
    EXPECT_EQ(reference.valueStreamId, valueStreamId);
    EXPECT_EQ(reference.dictionaryId, dictionaryId);
    EXPECT_EQ(reference.dataType, dataType);
  }

  static void expectFileDictionary(
      const FileDictionary& fileDictionary,
      uint32_t dictionaryId,
      DataType dataType,
      uint64_t offset,
      uint32_t length) {
    EXPECT_EQ(fileDictionary.dictionaryId, dictionaryId);
    EXPECT_EQ(fileDictionary.dataType, dataType);
    EXPECT_EQ(fileDictionary.offset, offset);
    EXPECT_EQ(fileDictionary.length, length);
  }
};

TEST_F(SharedDictionaryCatalogTest, roundTripsReferencesAndFileDictionaries) {
  const std::array stripeReferences{SharedDictionaryReference{
      .valueStreamId = 10, .dictionaryId = 100, .dataType = DataType::Int32}};
  const std::array fileReferences{SharedDictionaryReference{
      .valueStreamId = 20, .dictionaryId = 7, .dataType = DataType::Int32}};
  const std::array externalReferences{SharedDictionaryReference{
      .valueStreamId = 30, .dictionaryId = 11, .dataType = DataType::Int64}};
  const std::array fileDictionaries{FileDictionary{
      .dictionaryId = 7,
      .dataType = DataType::Int32,
      .offset = 123,
      .length = 45}};

  struct TestCase {
    const char* name;
    bool hasStripeReference;
    bool hasFileReference;
    bool hasExternalReference;
  };
  const std::array testCases{
      TestCase{"stripeOnly", true, false, false},
      TestCase{"fileOnly", false, true, false},
      TestCase{"externalOnly", false, false, true},
      TestCase{"stripeAndFile", true, true, false},
      TestCase{"stripeAndExternal", true, false, true},
      TestCase{"fileAndExternal", false, true, true},
      TestCase{"allScopes", true, true, true}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    const auto catalog = SharedDictionaryCatalog::deserialize(
        SharedDictionaryCatalog::serialize(
            testCase.hasStripeReference
                ? std::span<const SharedDictionaryReference>{stripeReferences}
                : std::span<const SharedDictionaryReference>{},
            testCase.hasFileReference
                ? std::span<const SharedDictionaryReference>{fileReferences}
                : std::span<const SharedDictionaryReference>{},
            testCase.hasExternalReference
                ? std::span<const SharedDictionaryReference>{externalReferences}
                : std::span<const SharedDictionaryReference>{},
            testCase.hasFileReference
                ? std::span<const FileDictionary>{fileDictionaries}
                : std::span<const FileDictionary>{}));

    ASSERT_EQ(
        catalog.stripeDictionaryReferences().size(),
        testCase.hasStripeReference ? 1 : 0);
    ASSERT_EQ(
        catalog.fileDictionaryReferences().size(),
        testCase.hasFileReference ? 1 : 0);
    ASSERT_EQ(
        catalog.externalDictionaryReferences().size(),
        testCase.hasExternalReference ? 1 : 0);
    ASSERT_EQ(
        catalog.fileDictionaries().size(), testCase.hasFileReference ? 1 : 0);

    const auto* stripeReference = catalog.findStripeDictionaryReference(10);
    if (testCase.hasStripeReference) {
      ASSERT_NE(stripeReference, nullptr);
      expectReference(*stripeReference, 10, 100, DataType::Int32);
    } else {
      EXPECT_EQ(stripeReference, nullptr);
    }

    const auto* fileReference = catalog.findFileDictionaryReference(20);
    const auto* fileDictionary = catalog.findFileDictionary(7);
    if (testCase.hasFileReference) {
      ASSERT_NE(fileReference, nullptr);
      expectReference(*fileReference, 20, 7, DataType::Int32);
      ASSERT_NE(fileDictionary, nullptr);
      expectFileDictionary(*fileDictionary, 7, DataType::Int32, 123, 45);
    } else {
      EXPECT_EQ(fileReference, nullptr);
      EXPECT_EQ(fileDictionary, nullptr);
    }

    const auto* externalReference = catalog.findExternalDictionaryReference(30);
    if (testCase.hasExternalReference) {
      ASSERT_NE(externalReference, nullptr);
      expectReference(*externalReference, 30, 11, DataType::Int64);
    } else {
      EXPECT_EQ(externalReference, nullptr);
    }

    EXPECT_EQ(catalog.findStripeDictionaryReference(999), nullptr);
    EXPECT_EQ(catalog.findFileDictionaryReference(999), nullptr);
    EXPECT_EQ(catalog.findExternalDictionaryReference(999), nullptr);
    EXPECT_EQ(catalog.findFileDictionary(999), nullptr);
  }
}

TEST_F(SharedDictionaryCatalogTest, rejectsEmptyCatalog) {
  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize({}, {}, {}, {}),
      "Shared dictionary catalog must contain at least one entry.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsDuplicateValueStreamAcrossScopes) {
  const std::array stripeReferences{SharedDictionaryReference{
      .valueStreamId = 10, .dictionaryId = 100, .dataType = DataType::Int32}};
  const std::array fileReferences{SharedDictionaryReference{
      .valueStreamId = 10, .dictionaryId = 7, .dataType = DataType::Int32}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize(
          stripeReferences, fileReferences, {}, {}),
      "Duplicate shared dictionary value stream 10.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsInvalidReferenceDictionaryId) {
  const std::array references{SharedDictionaryReference{
      .valueStreamId = 10,
      .dictionaryId = kInvalidSharedDictionaryId,
      .dataType = DataType::Int32}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize(references, {}, {}, {}),
      "Shared dictionary reference requires a valid dictionary id.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsFileDictionaryWithoutAlphabet) {
  const std::array fileDictionaries{FileDictionary{
      .dictionaryId = 7,
      .dataType = DataType::Int32,
      .offset = 123,
      .length = 0}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize({}, {}, {}, fileDictionaries),
      "File dictionary must contain a non-empty alphabet.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsMissingFileDictionary) {
  const std::array fileReferences{SharedDictionaryReference{
      .valueStreamId = 20, .dictionaryId = 7, .dataType = DataType::Int32}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize({}, fileReferences, {}, {}),
      "File shared dictionary 7 does not exist.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsFileDictionaryTypeMismatch) {
  const std::array fileReferences{SharedDictionaryReference{
      .valueStreamId = 20, .dictionaryId = 7, .dataType = DataType::Int32}};
  const std::array fileDictionaries{FileDictionary{
      .dictionaryId = 7,
      .dataType = DataType::Int64,
      .offset = 123,
      .length = 45}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize(
          {}, fileReferences, {}, fileDictionaries),
      "File shared dictionary 7 has an inconsistent type.");
}

TEST_F(SharedDictionaryCatalogTest, rejectsDuplicateFileDictionaryId) {
  const std::array fileDictionaries{
      FileDictionary{
          .dictionaryId = 7,
          .dataType = DataType::Int32,
          .offset = 123,
          .length = 45},
      FileDictionary{
          .dictionaryId = 7,
          .dataType = DataType::Int32,
          .offset = 168,
          .length = 9}};

  NIMBLE_ASSERT_THROW(
      SharedDictionaryCatalog::serialize({}, {}, {}, fileDictionaries),
      "Duplicate file shared dictionary ID 7.");
}

} // namespace
} // namespace facebook::nimble

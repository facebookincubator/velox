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

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/tools/ExternalDictionaryBuilder.h"

namespace facebook::nimble {
namespace {

class ExternalDictionaryBuilderTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::initialize({});
    }
    velox::filesystems::registerLocalFileSystem();
  }

  void SetUp() final {
    auto* memoryManager = velox::memory::memoryManager();
    VELOX_CHECK_NOT_NULL(memoryManager);
    pool_ = memoryManager->addLeafPool("external_builder_test");
  }

  template <typename T>
  std::vector<T> materializeAlphabet(ExternalDictionary encodedAlphabet) {
    auto encodedAlphabetOwner = std::make_shared<const std::string>(
        std::move(encodedAlphabet.encodedAlphabet));
    const std::string_view encodedAlphabetView{*encodedAlphabetOwner};
    const auto alphabet = SharedDictionaryAlphabet::create(
        encodedAlphabetView, std::move(encodedAlphabetOwner), pool_.get());
    Vector<T> values{pool_.get()};
    alphabet->materializeAll<T>(values);
    return {values.data(), values.data() + values.size()};
  }

  template <typename T>
  std::vector<T> materializeAlphabet(const SharedDictionaryAlphabet& alphabet) {
    Vector<T> values{pool_.get()};
    alphabet.materializeAll<T>(values);
    return {values.data(), values.data() + values.size()};
  }

  ExternalDictionaryBuilder dictionaryBuilder() const {
    return ExternalDictionaryBuilder{pool_.get()};
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(ExternalDictionaryBuilderTest, buildsSortedUniqueAlphabet) {
  const Vector<int32_t> source{pool_.get(), {30, 10, 20, 10}};
  const ExternalDictionaryBuilder::Options options;
  const auto builder = dictionaryBuilder();

  auto encodedAlphabet = builder.build(source, options);

  EXPECT_EQ(encodedAlphabet.dataType, DataType::Int32);
  EXPECT_EQ(
      encodedAlphabet.alphabetEncodingType,
      EncodingPrefix::encodingType(encodedAlphabet.encodedAlphabet));
  EXPECT_TRUE(encodedAlphabet.sortValues);
  EXPECT_EQ(encodedAlphabet.valueCount, 3);
  EXPECT_EQ(
      materializeAlphabet<int32_t>(std::move(encodedAlphabet)),
      std::vector<int32_t>({10, 20, 30}));
}

TEST_F(
    ExternalDictionaryBuilderTest,
    preservesFirstSeenOrderWhenSortingDisabled) {
  const Vector<int32_t> source{pool_.get(), {30, 10, 20, 10}};
  const ExternalDictionaryBuilder::Options options{
      .sortValues = false,
  };
  const auto builder = dictionaryBuilder();

  auto encodedAlphabet = builder.build(source, options);

  EXPECT_FALSE(encodedAlphabet.sortValues);
  EXPECT_EQ(encodedAlphabet.valueCount, 3);
  EXPECT_EQ(
      materializeAlphabet<int32_t>(std::move(encodedAlphabet)),
      std::vector<int32_t>({30, 10, 20}));
}

TEST_F(ExternalDictionaryBuilderTest, usesForcedAlphabetEncoding) {
  const Vector<uint32_t> source{pool_.get(), {1, 2, 3}};
  const ExternalDictionaryBuilder::Options options{
      .alphabetEncoding = EncodingType::FixedBitWidth,
  };
  const auto builder = dictionaryBuilder();

  const auto encodedAlphabet = builder.build(source, options);

  EXPECT_EQ(encodedAlphabet.dataType, DataType::Uint32);
  EXPECT_EQ(encodedAlphabet.alphabetEncodingType, EncodingType::FixedBitWidth);
  EXPECT_EQ(encodedAlphabet.valueCount, 3);
  EXPECT_EQ(
      EncodingPrefix::encodingType(encodedAlphabet.encodedAlphabet),
      EncodingType::FixedBitWidth);
}

TEST_F(ExternalDictionaryBuilderTest, serializesAndLoadsExternalDictionary) {
  const Vector<int32_t> source{pool_.get(), {30, 10, 20, 10}};
  const ExternalDictionaryBuilder::Options options{
      .sortValues = false,
      .alphabetEncoding = EncodingType::FixedBitWidth,
  };
  const auto builder = dictionaryBuilder();

  const auto encodedAlphabet = builder.build(source, options);
  const auto data = builder.serialize(encodedAlphabet);
  auto loadedAlphabet = builder.deserialize(data);

  EXPECT_EQ(loadedAlphabet.dataType, DataType::Int32);
  EXPECT_EQ(loadedAlphabet.alphabetEncodingType, EncodingType::FixedBitWidth);
  EXPECT_FALSE(loadedAlphabet.sortValues);
  EXPECT_EQ(loadedAlphabet.valueCount, 3);
  EXPECT_EQ(
      materializeAlphabet<int32_t>(std::move(loadedAlphabet)),
      std::vector<int32_t>({30, 10, 20}));
}

TEST_F(ExternalDictionaryBuilderTest, deserializesExternalDictionaryFromFile) {
  const Vector<int32_t> source{pool_.get(), {30, 10, 20, 10}};
  const auto builder = dictionaryBuilder();

  struct TestCase {
    std::string name;
    bool sortValues;
    EncodingType alphabetEncoding;
    std::vector<int32_t> expectedValues;
  };

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "sorted_fixed_bit_width",
               .sortValues = true,
               .alphabetEncoding = EncodingType::FixedBitWidth,
               .expectedValues = {10, 20, 30},
           },
           {
               .name = "first_seen_fixed_bit_width",
               .sortValues = false,
               .alphabetEncoding = EncodingType::FixedBitWidth,
               .expectedValues = {30, 10, 20},
           },
           {
               .name = "sorted_trivial",
               .sortValues = true,
               .alphabetEncoding = EncodingType::Trivial,
               .expectedValues = {10, 20, 30},
           },
           {
               .name = "first_seen_trivial",
               .sortValues = false,
               .alphabetEncoding = EncodingType::Trivial,
               .expectedValues = {30, 10, 20},
           },
       }) {
    SCOPED_TRACE(testCase.name);
    const ExternalDictionaryBuilder::Options options{
        .sortValues = testCase.sortValues,
        .alphabetEncoding = testCase.alphabetEncoding,
    };
    const auto encodedAlphabet = builder.build(source, options);
    const auto data = builder.serialize(encodedAlphabet);
    const auto tempFile = velox::common::testutil::TempFilePath::create();
    tempFile->append(data);

    auto loadedAlphabet = builder.deserializeFromFile(tempFile->getPath());

    EXPECT_EQ(loadedAlphabet.dataType, DataType::Int32);
    EXPECT_EQ(loadedAlphabet.alphabetEncodingType, testCase.alphabetEncoding);
    EXPECT_EQ(loadedAlphabet.sortValues, testCase.sortValues);
    EXPECT_EQ(loadedAlphabet.valueCount, 3);
    EXPECT_EQ(
        materializeAlphabet<int32_t>(std::move(loadedAlphabet)),
        testCase.expectedValues);
  }
}

TEST_F(ExternalDictionaryBuilderTest, rejectsInvalidInputs) {
  const ExternalDictionaryBuilder::Options options;
  const Vector<int32_t> source{pool_.get(), {1}};
  const Vector<int32_t> emptySource{pool_.get()};
  const auto builder = dictionaryBuilder();

  NIMBLE_ASSERT_USER_THROW(
      builder.build(emptySource, options),
      "External dictionary alphabet input is empty.");

  const ExternalDictionaryBuilder::Options conflictingOptions{
      .alphabetEncoding = EncodingType::FixedBitWidth,
      .readFactors = {{EncodingType::Trivial, 1.0f}},
  };
  NIMBLE_ASSERT_USER_THROW(
      builder.build(source, conflictingOptions),
      "alphabet_encoding and read_factors cannot both be set.");
}

} // namespace
} // namespace facebook::nimble

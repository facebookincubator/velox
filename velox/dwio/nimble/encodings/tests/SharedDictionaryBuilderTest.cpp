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

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryBuilder.h"

namespace facebook::nimble {
namespace {

template <typename T>
std::vector<T> toVector(std::span<const T> values) {
  return {values.begin(), values.end()};
}

std::vector<uint32_t> toIndexVector(std::span<const uint32_t> indices) {
  return {indices.begin(), indices.end()};
}

template <typename T>
uint32_t indexForValue(std::vector<T>& alphabet, const T& value) {
  const auto it = std::find(alphabet.begin(), alphabet.end(), value);
  if (it != alphabet.end()) {
    return static_cast<uint32_t>(std::distance(alphabet.begin(), it));
  }

  const auto index = static_cast<uint32_t>(alphabet.size());
  alphabet.push_back(value);
  return index;
}

template <typename T>
struct ExpectedDictionary {
  std::vector<T> alphabet;
  std::vector<uint32_t> indices;
};

template <typename T>
ExpectedDictionary<T> expectedFirstSeenDictionary(std::span<const T> values) {
  ExpectedDictionary<T> expected;
  expected.indices.reserve(values.size());
  for (const auto& value : values) {
    expected.indices.push_back(indexForValue(expected.alphabet, value));
  }
  return expected;
}

template <typename T>
std::vector<T> makeFuzzValues() {
  std::vector<T> values;
  values.reserve(128);
  for (auto i = 0; i < 128; ++i) {
    values.push_back(static_cast<T>((i * 17 + i / 3) % 23));
  }
  return values;
}

template <>
std::vector<std::string_view> makeFuzzValues<std::string_view>() {
  static constexpr std::array<std::string_view, 23> kAlphabet{
      "alpha",  "bravo",    "charlie", "delta",  "echo",   "foxtrot",
      "golf",   "hotel",    "india",   "juliet", "kilo",   "lima",
      "mike",   "november", "oscar",   "papa",   "quebec", "romeo",
      "sierra", "tango",    "uniform", "victor", "whiskey"};
  std::vector<std::string_view> values;
  values.reserve(128);
  for (auto i = 0; i < 128; ++i) {
    values.push_back(kAlphabet[(i * 17 + i / 3) % kAlphabet.size()]);
  }
  return values;
}

template <typename T>
std::vector<T> makeTypedValues();

template <>
std::vector<int32_t> makeTypedValues<int32_t>() {
  return {13, 7, 17, 11, 13, 7};
}

template <>
std::vector<std::string_view> makeTypedValues<std::string_view>() {
  return {"gamma", "alpha", "omega", "beta", "gamma", "alpha"};
}

template <typename T>
void verifyBuildDictionaryIndex(std::vector<T> alphabet, const char* type) {
  SCOPED_TRACE(type);
  const auto alphabetSize = alphabet.size();
  const auto dictionaryIndex = buildDictionaryIndex(
      std::span<const T>{alphabet.data(), alphabet.size()});

  EXPECT_EQ(dictionaryIndex.size(), alphabetSize);
  for (uint32_t i = 0; i < alphabetSize; ++i) {
    const auto it = dictionaryIndex.find(alphabet[i]);
    ASSERT_NE(it, dictionaryIndex.end());
    EXPECT_EQ(it->second, i);
  }
}

template <typename T>
void verifyFuzzedDictionary(velox::memory::MemoryPool* pool, const char* type) {
  SCOPED_TRACE(type);

  const auto values = makeFuzzValues<T>();
  StreamingSharedDictionaryBuilder<T> streamingBuilder{
      SharedDictionaryScope::Stripe, /*dictionaryId=*/7, pool};
  std::vector<T> expectedAlphabet;
  std::vector<uint32_t> expectedIndices;
  expectedIndices.reserve(values.size());

  for (size_t offset = 0; offset < values.size();) {
    const auto chunkSize =
        std::min<size_t>((offset % 7) + 1, values.size() - offset);
    const std::span<const T> chunk{values.data() + offset, chunkSize};
    const auto alphabetBeforeLookup = expectedAlphabet;
    std::vector<uint32_t> expectedChunkIndices;
    expectedChunkIndices.reserve(chunk.size());

    for (const auto& value : chunk) {
      expectedChunkIndices.push_back(indexForValue(expectedAlphabet, value));
    }
    expectedIndices.insert(
        expectedIndices.end(),
        expectedChunkIndices.begin(),
        expectedChunkIndices.end());

    auto mapping = streamingBuilder.lookup(chunk);
    EXPECT_EQ(toIndexVector(mapping.indices()), expectedChunkIndices);
    EXPECT_EQ(
        mapping.newEntryCount(),
        expectedAlphabet.size() - alphabetBeforeLookup.size());
    EXPECT_EQ(toVector(streamingBuilder.alphabet()), expectedAlphabet);

    offset += chunkSize;
  }
  EXPECT_EQ(toVector(streamingBuilder.alphabet()), expectedAlphabet);

  const std::span<const T> expectedAlphabetSpan{
      expectedAlphabet.data(), expectedAlphabet.size()};
  ExternalSharedDictionaryBuilder<T> fileScopeBuilder{
      SharedDictionaryScope::File,
      expectedAlphabetSpan,
      /*dictionaryId=*/7,
      pool};
  auto fileScopeMapping = fileScopeBuilder.lookup(values);
  EXPECT_EQ(toIndexVector(fileScopeMapping.indices()), expectedIndices);
  EXPECT_EQ(fileScopeMapping.newEntryCount(), 0);
  EXPECT_EQ(toVector(fileScopeBuilder.alphabet()), expectedAlphabet);

  ExternalSharedDictionaryBuilder<T> externalBuilder{
      SharedDictionaryScope::External,
      expectedAlphabetSpan,
      /*dictionaryId=*/7,
      pool};
  auto externalMapping = externalBuilder.lookup(values);
  EXPECT_EQ(toIndexVector(externalMapping.indices()), expectedIndices);
  EXPECT_EQ(externalMapping.newEntryCount(), 0);
  EXPECT_EQ(toVector(externalBuilder.alphabet()), expectedAlphabet);
}

class SharedDictionaryBuilderTest : public testing::Test {
 protected:
  void SetUp() final {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

template <typename T>
class SharedDictionaryBuilderTypedTest : public SharedDictionaryBuilderTest {};

using SharedDictionaryBuilderTypedTestTypes =
    ::testing::Types<int32_t, std::string_view>;

TYPED_TEST_SUITE(
    SharedDictionaryBuilderTypedTest,
    SharedDictionaryBuilderTypedTestTypes);

TEST_F(SharedDictionaryBuilderTest, builderKindStringFormats) {
  using Builder = SharedDictionaryBuilder<int32_t>;

  EXPECT_EQ(Builder::kindString(Builder::Kind::Streaming), "Streaming");
  EXPECT_EQ(Builder::kindString(Builder::Kind::External), "External");
  EXPECT_EQ(Builder::kindString(static_cast<Builder::Kind>(42)), "Unknown: 42");
}

TEST_F(SharedDictionaryBuilderTest, buildDictionaryIndex) {
  verifyBuildDictionaryIndex<int16_t>({7, 11, 13}, "int16_t");
  verifyBuildDictionaryIndex<uint16_t>({7, 11, 13}, "uint16_t");
  verifyBuildDictionaryIndex<int32_t>({7, 11, 13}, "int32_t");
  verifyBuildDictionaryIndex<uint32_t>({7, 11, 13}, "uint32_t");
  verifyBuildDictionaryIndex<int64_t>({7, -11, 13}, "int64_t");
  verifyBuildDictionaryIndex<uint64_t>({7, uint64_t{1} << 40, 13}, "uint64_t");
  verifyBuildDictionaryIndex<std::string_view>(
      {"alpha", "bravo", "charlie"}, "std::string_view");
}

TEST_F(SharedDictionaryBuilderTest, buildDictionaryIndexError) {
  const std::vector<int32_t> int32Alphabet{7, 11, 7};
  NIMBLE_ASSERT_USER_THROW(
      buildDictionaryIndex(std::span<const int32_t>{int32Alphabet}),
      "Shared dictionary has duplicate values.");

  const std::vector<uint64_t> uint64Alphabet{7, uint64_t{1} << 40, 7};
  NIMBLE_ASSERT_USER_THROW(
      buildDictionaryIndex(std::span<const uint64_t>{uint64Alphabet}),
      "Shared dictionary has duplicate values.");

  const std::vector<std::string_view> stringAlphabet{"alpha", "bravo", "alpha"};
  NIMBLE_ASSERT_USER_THROW(
      buildDictionaryIndex(std::span<const std::string_view>{stringAlphabet}),
      "Shared dictionary has duplicate values.");
}

TEST_F(SharedDictionaryBuilderTest, externalBuilderWithDuplicateAlphabetError) {
  const std::array<int32_t, 3> alphabet{7, 11, 7};

  NIMBLE_ASSERT_USER_THROW(
      (ExternalSharedDictionaryBuilder<int32_t>{
          SharedDictionaryScope::External,
          alphabet,
          /*dictionaryId=*/7,
          pool_.get()}),
      "Shared dictionary has duplicate values.");
}

TEST_F(SharedDictionaryBuilderTest, externalBuilderWithStripeScopeError) {
  const std::array<int32_t, 3> alphabet{7, 11, 13};

  NIMBLE_ASSERT_THROW(
      (ExternalSharedDictionaryBuilder<int32_t>{
          SharedDictionaryScope::Stripe,
          alphabet,
          /*dictionaryId=*/7,
          pool_.get()}),
      "External shared dictionary builder requires file or external scope.");
}

TEST_F(SharedDictionaryBuilderTest, alphabet) {
  StreamingSharedDictionaryBuilder<int32_t> streamingBuilder{
      SharedDictionaryScope::Stripe, /*dictionaryId=*/7, pool_.get()};
  EXPECT_TRUE(streamingBuilder.alphabet().empty());

  const std::array<int32_t, 4> streamingValues{4, 8, 4, 12};
  auto streamingMapping = streamingBuilder.lookup(streamingValues);
  EXPECT_EQ(
      toVector(streamingBuilder.alphabet()), (std::vector<int32_t>{4, 8, 12}));

  const std::array<int32_t, 3> fileAlphabet{7, 11, 13};
  const std::span<const int32_t> fileAlphabetSpan{fileAlphabet};
  ExternalSharedDictionaryBuilder<int32_t> fileBuilder{
      SharedDictionaryScope::File,
      fileAlphabetSpan,
      /*dictionaryId=*/7,
      pool_.get()};
  EXPECT_EQ(
      fileBuilder.kind(), SharedDictionaryBuilder<int32_t>::Kind::External);
  EXPECT_EQ(
      toVector(fileBuilder.alphabet()), (std::vector<int32_t>{7, 11, 13}));

  const std::array<int32_t, 3> externalAlphabet{7, 11, 13};
  ExternalSharedDictionaryBuilder<int32_t> externalBuilder{
      SharedDictionaryScope::External,
      externalAlphabet,
      /*dictionaryId=*/7,
      pool_.get()};
  EXPECT_EQ(
      externalBuilder.kind(), SharedDictionaryBuilder<int32_t>::Kind::External);
  EXPECT_EQ(
      toVector(externalBuilder.alphabet()), (std::vector<int32_t>{7, 11, 13}));
}

TEST_F(SharedDictionaryBuilderTest, lookupWithFileScopeExternalDictionary) {
  const std::array<int32_t, 4> alphabet{7, 11, 13, 17};
  struct Scenario {
    std::vector<int32_t> values;
    std::optional<std::vector<uint32_t>> expectedIndices;
  };

  const std::vector<Scenario> scenarios{
      {{7, 11, 13, 17}, std::vector<uint32_t>{0, 1, 2, 3}},
      {{13, 7, 17, 11, 13, 7}, std::vector<uint32_t>{2, 0, 3, 1, 2, 0}},
      {{}, std::vector<uint32_t>{}},
      {{13, 99}, std::nullopt},
  };

  for (auto i = 0; i < scenarios.size(); ++i) {
    const auto& scenario = scenarios[i];
    SCOPED_TRACE(fmt::format("scenario={}", i));
    const std::span<const int32_t> alphabetSpan{alphabet};
    ExternalSharedDictionaryBuilder<int32_t> builder{
        SharedDictionaryScope::File,
        alphabetSpan,
        /*dictionaryId=*/7,
        pool_.get()};

    if (!scenario.expectedIndices.has_value()) {
      NIMBLE_ASSERT_USER_THROW(
          builder.lookup(scenario.values),
          "File shared dictionary 7 does not contain value 99.");
      EXPECT_EQ(
          toVector(builder.alphabet()), (std::vector<int32_t>{7, 11, 13, 17}));
      continue;
    }

    const auto mapping = builder.lookup(scenario.values);
    EXPECT_EQ(
        toIndexVector(mapping.indices()), scenario.expectedIndices.value());
    EXPECT_EQ(mapping.newEntryCount(), 0);
    EXPECT_EQ(
        toVector(builder.alphabet()), (std::vector<int32_t>{7, 11, 13, 17}));
  }
}

TEST_F(SharedDictionaryBuilderTest, lookupWithExternalDictionary) {
  const std::array<int32_t, 4> alphabet{7, 11, 13, 17};
  struct Scenario {
    std::vector<int32_t> values;
    std::optional<std::vector<uint32_t>> expectedIndices;
  };

  const std::vector<Scenario> scenarios{
      {{7, 11, 13, 17}, std::vector<uint32_t>{0, 1, 2, 3}},
      {{13, 7, 17, 11, 13, 7}, std::vector<uint32_t>{2, 0, 3, 1, 2, 0}},
      {{}, std::vector<uint32_t>{}},
      {{13, 99}, std::nullopt},
  };

  for (auto i = 0; i < scenarios.size(); ++i) {
    const auto& scenario = scenarios[i];
    SCOPED_TRACE(fmt::format("scenario={}", i));
    const std::span<const int32_t> alphabetSpan{alphabet};
    ExternalSharedDictionaryBuilder<int32_t> builder{
        SharedDictionaryScope::External,
        alphabetSpan,
        /*dictionaryId=*/7,
        pool_.get()};

    if (!scenario.expectedIndices.has_value()) {
      NIMBLE_ASSERT_USER_THROW(
          builder.lookup(scenario.values),
          "External shared dictionary 7 does not contain value 99.");
      EXPECT_EQ(
          toVector(builder.alphabet()), (std::vector<int32_t>{7, 11, 13, 17}));
      continue;
    }

    const auto mapping = builder.lookup(scenario.values);
    EXPECT_EQ(
        toIndexVector(mapping.indices()), scenario.expectedIndices.value());
    EXPECT_EQ(mapping.newEntryCount(), 0);
    EXPECT_EQ(
        toVector(builder.alphabet()), (std::vector<int32_t>{7, 11, 13, 17}));
  }
}

TYPED_TEST(SharedDictionaryBuilderTypedTest, lookupWithStreamingDictionary) {
  using T = TypeParam;
  const auto values = makeTypedValues<T>();
  const auto expected = expectedFirstSeenDictionary<T>(values);
  StreamingSharedDictionaryBuilder<T> builder{
      SharedDictionaryScope::Stripe, /*dictionaryId=*/7, this->pool_.get()};

  auto mapping = builder.lookup(values);
  EXPECT_EQ(toIndexVector(mapping.indices()), expected.indices);
  EXPECT_EQ(mapping.newEntryCount(), expected.alphabet.size());
  EXPECT_EQ(toVector(builder.alphabet()), expected.alphabet);
}

TYPED_TEST(
    SharedDictionaryBuilderTypedTest,
    lookupWithFileScopeExternalDictionary) {
  using T = TypeParam;
  const auto values = makeTypedValues<T>();
  const auto expected = expectedFirstSeenDictionary<T>(values);
  const std::span<const T> alphabet{
      expected.alphabet.data(), expected.alphabet.size()};
  ExternalSharedDictionaryBuilder<T> builder{
      SharedDictionaryScope::File,
      alphabet,
      /*dictionaryId=*/7,
      this->pool_.get()};

  auto mapping = builder.lookup(values);
  EXPECT_EQ(toIndexVector(mapping.indices()), expected.indices);
  EXPECT_EQ(mapping.newEntryCount(), 0);
  EXPECT_EQ(toVector(builder.alphabet()), expected.alphabet);
}

TYPED_TEST(SharedDictionaryBuilderTypedTest, lookupWithExternalDictionary) {
  using T = TypeParam;
  const auto values = makeTypedValues<T>();
  const auto expected = expectedFirstSeenDictionary<T>(values);
  const std::span<const T> alphabet{
      expected.alphabet.data(), expected.alphabet.size()};
  ExternalSharedDictionaryBuilder<T> builder{
      SharedDictionaryScope::External,
      alphabet,
      /*dictionaryId=*/7,
      this->pool_.get()};

  auto mapping = builder.lookup(values);
  EXPECT_EQ(toIndexVector(mapping.indices()), expected.indices);
  EXPECT_EQ(mapping.newEntryCount(), 0);
  EXPECT_EQ(toVector(builder.alphabet()), expected.alphabet);
}

TEST_F(SharedDictionaryBuilderTest, fuzzDifferentTypes) {
  verifyFuzzedDictionary<int16_t>(pool_.get(), "int16_t");
  verifyFuzzedDictionary<uint16_t>(pool_.get(), "uint16_t");
  verifyFuzzedDictionary<int32_t>(pool_.get(), "int32_t");
  verifyFuzzedDictionary<int64_t>(pool_.get(), "int64_t");
  verifyFuzzedDictionary<uint64_t>(pool_.get(), "uint64_t");
  verifyFuzzedDictionary<std::string_view>(pool_.get(), "std::string_view");
}

TEST_F(SharedDictionaryBuilderTest, lookupAndReset) {
  enum class Action {
    Keep,
    Reset,
  };

  struct Step {
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
    uint32_t expectedNewEntryCount;
    std::vector<int32_t> expectedAlphabetAfterLookup;
    Action action;
    std::vector<int32_t> expectedAlphabetAfterAction;
  };

  const std::vector<std::vector<Step>> scenarios{
      {
          {{10, 20, 10, 30},
           {0, 1, 0, 2},
           3,
           {10, 20, 30},
           Action::Keep,
           {10, 20, 30}},
          {{20, 40, 10, 40},
           {1, 3, 0, 3},
           1,
           {10, 20, 30, 40},
           Action::Keep,
           {10, 20, 30, 40}},
          {{50, 30, 50, 60},
           {4, 2, 4, 5},
           2,
           {10, 20, 30, 40, 50, 60},
           Action::Keep,
           {10, 20, 30, 40, 50, 60}},
          {{60, 10, 40},
           {5, 0, 3},
           0,
           {10, 20, 30, 40, 50, 60},
           Action::Keep,
           {10, 20, 30, 40, 50, 60}},
          {{70, 80},
           {6, 7},
           2,
           {10, 20, 30, 40, 50, 60, 70, 80},
           Action::Reset,
           {}},
          {{60, 90}, {0, 1}, 2, {60, 90}, Action::Keep, {60, 90}},
      },
  };

  for (auto i = 0; i < scenarios.size(); ++i) {
    const auto& scenario = scenarios[i];
    StreamingSharedDictionaryBuilder<int32_t> builder{
        SharedDictionaryScope::Stripe, /*dictionaryId=*/7, pool_.get()};
    EXPECT_EQ(
        builder.kind(), SharedDictionaryBuilder<int32_t>::Kind::Streaming);
    EXPECT_EQ(
        SharedDictionaryBuilder<int32_t>::kindString(builder.kind()),
        "Streaming");

    for (auto j = 0; j < scenario.size(); ++j) {
      const auto& step = scenario[j];
      SCOPED_TRACE(fmt::format("scenario={}, step={}", i, j));
      auto mapping = builder.lookup(step.values);
      EXPECT_EQ(
          std::vector<uint32_t>(
              mapping.indices().begin(), mapping.indices().end()),
          step.expectedIndices);
      EXPECT_EQ(mapping.newEntryCount(), step.expectedNewEntryCount);
      EXPECT_EQ(
          std::vector<int32_t>(
              builder.alphabet().begin(), builder.alphabet().end()),
          step.expectedAlphabetAfterLookup);

      switch (step.action) {
        case Action::Keep:
          break;
        case Action::Reset:
          builder.reset();
          break;
      }
      EXPECT_EQ(
          std::vector<int32_t>(
              builder.alphabet().begin(), builder.alphabet().end()),
          step.expectedAlphabetAfterAction);
    }
  }
}

TEST_F(SharedDictionaryBuilderTest, lookupMutatesStreamingAlphabet) {
  StreamingSharedDictionaryBuilder<int32_t> builder{
      SharedDictionaryScope::Stripe, /*dictionaryId=*/7, pool_.get()};

  const std::array<int32_t, 1> values{10};
  auto mapping = builder.lookup(values);
  EXPECT_EQ(
      std::vector<int32_t>(
          builder.alphabet().begin(), builder.alphabet().end()),
      (std::vector<int32_t>{10}));

  const std::array<int32_t, 1> otherValues{20};
  auto otherMapping = builder.lookup(otherValues);
  EXPECT_EQ(
      std::vector<int32_t>(
          builder.alphabet().begin(), builder.alphabet().end()),
      (std::vector<int32_t>{10, 20}));
}

TEST_F(SharedDictionaryBuilderTest, reset) {
  StreamingSharedDictionaryBuilder<int16_t> builder{
      SharedDictionaryScope::Stripe, /*dictionaryId=*/7, pool_.get()};
  const std::array<int16_t, 2> values{4, 8};
  auto mapping = builder.lookup(values);
  EXPECT_FALSE(builder.alphabet().empty());

  builder.reset();
  EXPECT_TRUE(builder.alphabet().empty());

  auto resetMapping = builder.lookup(values);
  EXPECT_FALSE(builder.alphabet().empty());

  builder.reset();
  EXPECT_TRUE(builder.alphabet().empty());

  const std::array<int16_t, 2> afterResetValues{8, 4};
  auto afterResetMapping = builder.lookup(afterResetValues);
  EXPECT_EQ(
      std::vector<uint32_t>(
          afterResetMapping.indices().begin(),
          afterResetMapping.indices().end()),
      (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(
      std::vector<int16_t>(
          builder.alphabet().begin(), builder.alphabet().end()),
      (std::vector<int16_t>{8, 4}));
}

TEST_F(SharedDictionaryBuilderTest, externalBuilderRejectsMissingValue) {
  const std::array<int32_t, 3> alphabet{7, 11, 13};
  const std::span<const int32_t> alphabetSpan{alphabet};
  ExternalSharedDictionaryBuilder<int32_t> fileScopeBuilder{
      SharedDictionaryScope::File,
      alphabetSpan,
      /*dictionaryId=*/7,
      pool_.get()};
  ExternalSharedDictionaryBuilder<int32_t> externalBuilder{
      SharedDictionaryScope::External,
      alphabetSpan,
      /*dictionaryId=*/7,
      pool_.get()};

  struct BuilderCase {
    const char* name;
    SharedDictionaryBuilder<int32_t>* builder;
    SharedDictionaryBuilder<int32_t>::Kind kind;
  };

  const std::array<BuilderCase, 2> builders{{
      {"file",
       &fileScopeBuilder,
       SharedDictionaryBuilder<int32_t>::Kind::External},
      {"external",
       &externalBuilder,
       SharedDictionaryBuilder<int32_t>::Kind::External},
  }};
  const std::array<int32_t, 3> covered{13, 7, 11};
  const std::array<int32_t, 1> missing{17};

  for (const auto& builderCase : builders) {
    SCOPED_TRACE(builderCase.name);
    EXPECT_EQ(builderCase.builder->kind(), builderCase.kind);
    auto mapping = builderCase.builder->lookup(covered);
    EXPECT_EQ(
        std::vector<uint32_t>(
            mapping.indices().begin(), mapping.indices().end()),
        (std::vector<uint32_t>{2, 0, 1}));
    NIMBLE_ASSERT_USER_THROW(
        builderCase.builder->lookup(missing), "does not contain value 17.");
  }

  EXPECT_EQ(
      std::vector<int32_t>(
          fileScopeBuilder.alphabet().begin(),
          fileScopeBuilder.alphabet().end()),
      (std::vector<int32_t>{7, 11, 13}));
  NIMBLE_ASSERT_THROW(fileScopeBuilder.reset(), "does not support reset()");
  EXPECT_EQ(
      std::vector<int32_t>(
          externalBuilder.alphabet().begin(), externalBuilder.alphabet().end()),
      (std::vector<int32_t>{7, 11, 13}));
}

TEST_F(SharedDictionaryBuilderTest, externalBuilderLookupDoesNotGrowAlphabet) {
  const std::array<int32_t, 3> alphabet{7, 11, 13};
  const std::span<const int32_t> alphabetSpan{alphabet};
  ExternalSharedDictionaryBuilder<int32_t> fileScopeBuilder{
      SharedDictionaryScope::File,
      alphabetSpan,
      /*dictionaryId=*/7,
      pool_.get()};
  ExternalSharedDictionaryBuilder<int32_t> externalBuilder{
      SharedDictionaryScope::External,
      alphabetSpan,
      /*dictionaryId=*/7,
      pool_.get()};

  const std::array<int32_t, 3> covered{13, 7, 11};
  auto fileScopeMapping = fileScopeBuilder.lookup(covered);
  EXPECT_EQ(fileScopeMapping.newEntryCount(), 0);
  EXPECT_EQ(
      std::vector<int32_t>(
          fileScopeBuilder.alphabet().begin(),
          fileScopeBuilder.alphabet().end()),
      (std::vector<int32_t>{7, 11, 13}));

  auto externalMapping = externalBuilder.lookup(covered);
  EXPECT_EQ(externalMapping.newEntryCount(), 0);
  EXPECT_EQ(
      std::vector<int32_t>(
          externalBuilder.alphabet().begin(), externalBuilder.alphabet().end()),
      (std::vector<int32_t>{7, 11, 13}));
}

} // namespace
} // namespace facebook::nimble

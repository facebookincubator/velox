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

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SharedDictionaryWriter.h"
#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble {
namespace {

using TestSharedDictionaryWriter = TypedSharedDictionaryWriter<int32_t>;
using TestStringSharedDictionaryWriter =
    TypedSharedDictionaryWriter<std::string_view>;
using EncodingReadFactors = std::vector<std::pair<EncodingType, float>>;

EncodingReadFactors dictionaryValueEncoding() {
  return {{EncodingType::Dictionary, 1.0}};
}

EncodingReadFactors trivialPreferredValueEncoding() {
  return {{EncodingType::Trivial, 0.01}, {EncodingType::Dictionary, 1000.0}};
}

class StripeAbandonValueSelectionPolicy final
    : public EncodingSelectionPolicy<int32_t> {
 public:
  using physicalType = TypeTraits<int32_t>::physicalType;

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& /*statistics*/,
      const Encoding::Options& /*options*/) final {
    return {
        .encodingType = values.size() <= 6 ? EncodingType::Trivial
                                           : EncodingType::Dictionary};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /*values*/,
      std::span<const bool> /*notNulls*/,
      const Statistics<physicalType>& /*statistics*/,
      const Encoding::Options& /*options*/) final {
    return {.encodingType = EncodingType::Nullable};
  }

 private:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /*parentEncodingType*/,
      NestedEncodingIdentifier /*nestedEncodingIdentifier*/,
      DataType nestedDataType) final {
    ManualEncodingSelectionPolicyFactory factory{
        {{EncodingType::Trivial, 1.0}}, std::nullopt};
    return factory.createPolicy(nestedDataType);
  }
};

EncodingSelectionPolicyCreator stripeAbandonEncodingSelectionPolicyCreator() {
  return [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    if (dataType == DataType::Int32) {
      return std::make_unique<StripeAbandonValueSelectionPolicy>();
    }
    ManualEncodingSelectionPolicyFactory factory{
        {{EncodingType::Trivial, 1.0}}, std::nullopt};
    return factory.createPolicy(dataType);
  };
}

EncodingSelectionPolicyCreator testEncodingSelectionPolicyCreator(
    EncodingReadFactors valueEncodingReadFactors = dictionaryValueEncoding()) {
  return
      [valueEncodingReadFactors = std::move(valueEncodingReadFactors)](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        // TODO: Only Int32, Uint32 and String are given tailored read
        // factors; the remaining fixed-width types fall through to the shared
        // default. Extend the pool so they are exercised distinctly too.
        EncodingReadFactors encodingReadFactors;
        switch (dataType) {
          case DataType::Int32:
          case DataType::String:
            encodingReadFactors = valueEncodingReadFactors;
            break;
          case DataType::Uint32:
            encodingReadFactors = {{EncodingType::FixedBitWidth, 1.0}};
            break;
          case DataType::Undefined:
          case DataType::Int8:
          case DataType::Uint8:
          case DataType::Int16:
          case DataType::Uint16:
          case DataType::Int64:
          case DataType::Uint64:
          case DataType::Float:
          case DataType::Double:
          case DataType::Bool:
            encodingReadFactors = {{EncodingType::Trivial, 1.0}};
            break;
        }
        ManualEncodingSelectionPolicyFactory factory{
            std::move(encodingReadFactors), std::nullopt};
        return factory.createPolicy(dataType);
      };
}

TestSharedDictionaryWriter::Options writerOptions(
    SharedDictionaryScope scope,
    uint32_t dictionaryId,
    std::shared_ptr<const ExternalDictionaryResolver> resolver = nullptr,
    std::vector<EncodingType> alphabetEncodings = {},
    bool useExternalAlphabet = false,
    bool useVarintRowCount = false,
    EncodingReadFactors valueEncodingReadFactors = dictionaryValueEncoding()) {
  return TestSharedDictionaryWriter::Options{
      .scope = scope,
      .dictionaryId = dictionaryId,
      .useExternalAlphabet = useExternalAlphabet,
      .alphabetEncodings = std::move(alphabetEncodings),
      .encodingSelectionPolicyCreator = testEncodingSelectionPolicyCreator(
          std::move(valueEncodingReadFactors)),
      .encodingOptions =
          Encoding::Options{.useVarintRowCount = useVarintRowCount},
      .resolver = std::move(resolver)};
}

TestSharedDictionaryWriter createTestWriter(
    velox::memory::MemoryPool* pool,
    const TestSharedDictionaryWriter::Options& options) {
  return TestSharedDictionaryWriter{pool, options};
}

class TestDictionaryResolver final : public ExternalDictionaryResolver {
 public:
  TestDictionaryResolver(
      uint32_t dictionaryId,
      std::span<const int32_t> values,
      velox::memory::MemoryPool* pool,
      std::span<const EncodingType> candidateEncodings = {})
      : dictionaryId_{dictionaryId},
        alphabet_{test::createSharedDictionaryAlphabet<int32_t>(
            values,
            candidateEncodings,
            pool)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final {
    if (dictionaryId != dictionaryId_ || dataType != DataType::Int32) {
      return nullptr;
    }
    return alphabet_;
  }

 private:
  const uint32_t dictionaryId_;
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
};

class TestStringDictionaryResolver final : public ExternalDictionaryResolver {
 public:
  TestStringDictionaryResolver(
      uint32_t dictionaryId,
      std::span<const std::string_view> values,
      velox::memory::MemoryPool* pool,
      std::span<const EncodingType> candidateEncodings = {})
      : dictionaryId_{dictionaryId},
        alphabet_{test::createSharedDictionaryAlphabet<std::string_view>(
            values,
            candidateEncodings,
            pool)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final {
    if (dictionaryId != dictionaryId_ || dataType != DataType::String) {
      return nullptr;
    }
    return alphabet_;
  }

 private:
  const uint32_t dictionaryId_;
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
};

std::string_view bytesOf(std::span<const int32_t> values) {
  return {
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(int32_t)};
}

std::string_view bytesOf(std::span<const std::string_view> values) {
  return {
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(std::string_view)};
}

StreamDataView streamView(
    const StreamDescriptorBuilder& descriptor,
    std::span<const int32_t> values) {
  return StreamDataView{
      descriptor, bytesOf(values), static_cast<uint32_t>(values.size())};
}

StreamDataView streamView(
    const StreamDescriptorBuilder& descriptor,
    std::span<const std::string_view> values) {
  return StreamDataView{
      descriptor, bytesOf(values), static_cast<uint32_t>(values.size())};
}

StreamDataView nullableStreamView(
    const StreamDescriptorBuilder& descriptor,
    std::span<const int32_t> nonNullValues,
    std::span<const bool> notNulls) {
  return StreamDataView{
      descriptor,
      bytesOf(nonNullValues),
      static_cast<uint32_t>(notNulls.size()),
      notNulls,
      static_cast<uint32_t>(
          std::count(notNulls.begin(), notNulls.end(), false))};
}

std::vector<int32_t> repeatedValues(
    std::span<const int32_t> pattern,
    size_t repeatCount) {
  std::vector<int32_t> values;
  values.reserve(pattern.size() * repeatCount);
  for (size_t i = 0; i < repeatCount; ++i) {
    values.insert(values.end(), pattern.begin(), pattern.end());
  }
  return values;
}

template <typename T>
std::vector<typename TypeTraits<T>::physicalType> physicalValues(
    std::span<const T> values) {
  std::vector<typename TypeTraits<T>::physicalType> physical;
  physical.reserve(values.size());
  for (const auto value : values) {
    physical.push_back(EncodingPhysicalType<T>::asEncodingPhysicalType(value));
  }
  return physical;
}

std::vector<uint32_t> repeatedIndices(
    std::span<const uint32_t> pattern,
    size_t repeatCount) {
  std::vector<uint32_t> indices;
  indices.reserve(pattern.size() * repeatCount);
  for (size_t i = 0; i < repeatCount; ++i) {
    indices.insert(indices.end(), pattern.begin(), pattern.end());
  }
  return indices;
}

struct NullableValues {
  std::vector<int32_t> nonNullValues;
  std::vector<int32_t> materializedValues;
  std::unique_ptr<bool[]> notNulls;
  size_t rowCount{};
  uint32_t nullCount{};

  std::span<const bool> notNullsSpan() const {
    return {notNulls.get(), rowCount};
  }
};

NullableValues repeatedNullableValues(
    std::span<const std::optional<int32_t>> pattern,
    size_t repeatCount) {
  NullableValues values;
  values.rowCount = pattern.size() * repeatCount;
  values.notNulls = std::make_unique<bool[]>(values.rowCount);
  values.materializedValues.reserve(values.rowCount);

  size_t row{0};
  for (size_t i = 0; i < repeatCount; ++i) {
    for (const auto& value : pattern) {
      values.notNulls[row++] = value.has_value();
      if (value.has_value()) {
        values.nonNullValues.push_back(*value);
        values.materializedValues.push_back(*value);
        continue;
      }
      ++values.nullCount;
      values.materializedValues.push_back(0);
    }
  }
  return values;
}

struct ExpectedStripeDictionary {
  std::vector<int32_t> values;
  std::vector<uint32_t> indices{};
  std::vector<int32_t> alphabet{};
};

ExpectedStripeDictionary expectedStripeDictionary(std::vector<int32_t> values) {
  ExpectedStripeDictionary expected{
      .values = std::move(values),
      .indices = {},
      .alphabet = {},
  };
  expected.indices.reserve(expected.values.size());
  for (const auto value : expected.values) {
    const auto it =
        std::find(expected.alphabet.begin(), expected.alphabet.end(), value);
    if (it == expected.alphabet.end()) {
      expected.indices.push_back(
          static_cast<uint32_t>(expected.alphabet.size()));
      expected.alphabet.push_back(value);
      continue;
    }
    expected.indices.push_back(
        static_cast<uint32_t>(it - expected.alphabet.begin()));
  }
  return expected;
}

ExpectedStripeDictionary generatedStripeDictionary(size_t stripeIndex) {
  const auto alphabetSize = 3 + (stripeIndex % 5);
  std::vector<int32_t> candidates;
  candidates.reserve(alphabetSize);
  for (size_t i = 0; i < alphabetSize; ++i) {
    candidates.push_back(
        static_cast<int32_t>(1000 * stripeIndex + 17 * i - 31));
  }

  std::vector<int32_t> values;
  values.reserve(1024);
  for (size_t row = 0; row < 1024; ++row) {
    const auto candidateIndex =
        (row * 37 + stripeIndex * 11 + row / 17) % candidates.size();
    values.push_back(candidates[candidateIndex]);
  }
  return expectedStripeDictionary(std::move(values));
}

// Verifies the shared-dictionary envelope and decodes the nested index stream.
std::vector<uint32_t> sharedDictionaryIndices(
    std::string_view encoded,
    uint32_t rowCount,
    velox::memory::MemoryPool* pool,
    bool useVarintRowCount = false,
    DataType dataType = DataType::Int32) {
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  EXPECT_EQ(encodingType, EncodingType::SharedDictionary);
  if (encodingType != EncodingType::SharedDictionary) {
    return {};
  }
  EXPECT_EQ(EncodingPrefix::dataType(encoded), dataType);
  EXPECT_EQ(EncodingPrefix::readRowCount(encoded, useVarintRowCount), rowCount);

  const char* pos =
      encoded.data() + EncodingPrefix::prefixSize(encoded, useVarintRowCount);
  EXPECT_LE(pos, encoded.end());
  const std::string_view encodedIndices{
      pos, static_cast<size_t>(encoded.end() - pos)};

  auto indicesEncoding = EncodingFactory{}.create(
      *pool,
      encodedIndices,
      [](uint32_t /*size*/) -> void* { return nullptr; },
      Encoding::Options{.useVarintRowCount = useVarintRowCount});
  EXPECT_EQ(indicesEncoding->dataType(), DataType::Uint32);
  EXPECT_EQ(indicesEncoding->rowCount(), rowCount);

  std::vector<uint32_t> indices(rowCount);
  indicesEncoding->materialize(rowCount, indices.data());
  return indices;
}

std::unique_ptr<Encoding> createEncoding(
    std::string_view encoded,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options = {}) {
  return EncodingFactory{options}.create(
      *pool, encoded, [](uint32_t /*size*/) -> void* { return nullptr; });
}

std::string_view encodeValues(
    TestSharedDictionaryWriter& writer,
    size_t stripeIndex,
    const StreamData& streamData,
    Buffer& buffer,
    const Encoding::Options& options = {}) {
  NIMBLE_CHECK_EQ(
      streamData.data().size() % sizeof(int32_t),
      0,
      "Test stream has incomplete int32 values.");
  const std::span<const int32_t> values{
      reinterpret_cast<const int32_t*>(streamData.data().data()),
      streamData.data().size() / sizeof(int32_t)};
  auto policy = writer.createEncodingPolicy(stripeIndex);
  if (streamData.hasNulls()) {
    return EncodingFactory::encodeNullable<int32_t>(
        std::move(policy), values, streamData.nonNulls(), buffer, options);
  }
  return EncodingFactory::encode<int32_t>(
      std::move(policy), values, buffer, options);
}

std::string_view encodeValues(
    TestStringSharedDictionaryWriter& writer,
    size_t stripeIndex,
    const StreamData& streamData,
    Buffer& buffer,
    const Encoding::Options& options = {}) {
  NIMBLE_CHECK_EQ(
      streamData.data().size() % sizeof(std::string_view),
      0,
      "Test stream has incomplete string values.");
  const std::span<const std::string_view> values{
      reinterpret_cast<const std::string_view*>(streamData.data().data()),
      streamData.data().size() / sizeof(std::string_view)};
  auto policy = writer.createEncodingPolicy(stripeIndex);
  return EncodingFactory::encode<std::string_view>(
      std::move(policy), values, buffer, options);
}

const Encoding* nullableValuesChild(const Encoding& encoding) {
  EXPECT_EQ(encoding.encodingType(), EncodingType::Nullable);
  const auto* nullableEncoding =
      dynamic_cast<const NullableEncoding<int32_t>*>(&encoding);
  EXPECT_NE(nullableEncoding, nullptr);
  if (nullableEncoding == nullptr) {
    return nullptr;
  }
  return nullableEncoding->nonNulls();
}

std::vector<uint32_t> nullableSharedDictionaryIndices(
    std::string_view encoded,
    uint32_t rowCount,
    std::span<const bool> expectedNonNulls,
    uint32_t nonNullRowCount,
    velox::memory::MemoryPool* pool,
    bool useVarintRowCount) {
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  EXPECT_EQ(encodingType, EncodingType::Nullable);
  if (encodingType != EncodingType::Nullable) {
    return {};
  }
  EXPECT_EQ(EncodingPrefix::dataType(encoded), DataType::Int32);
  EXPECT_EQ(EncodingPrefix::readRowCount(encoded, useVarintRowCount), rowCount);

  const char* pos =
      encoded.data() + EncodingPrefix::prefixSize(encoded, useVarintRowCount);
  const auto nonNullValuesBytes = encoding::readUint32(pos);
  EXPECT_LE(pos + nonNullValuesBytes, encoded.end());
  const std::string_view encodedNonNullValues{pos, nonNullValuesBytes};
  pos += nonNullValuesBytes;
  const std::string_view encodedNonNulls{
      pos, static_cast<size_t>(encoded.end() - pos)};

  auto nonNullsEncoding = EncodingFactory{}.create(
      *pool,
      encodedNonNulls,
      [](uint32_t /*size*/) -> void* { return nullptr; },
      Encoding::Options{.useVarintRowCount = useVarintRowCount});
  EXPECT_EQ(nonNullsEncoding->dataType(), DataType::Bool);
  EXPECT_EQ(nonNullsEncoding->rowCount(), rowCount);

  Vector<bool> actualNonNulls{pool};
  actualNonNulls.resize(rowCount);
  nonNullsEncoding->materialize(rowCount, actualNonNulls.data());
  for (size_t i = 0; i < expectedNonNulls.size(); ++i) {
    EXPECT_EQ(actualNonNulls[i], expectedNonNulls[i]) << "row " << i;
  }

  return sharedDictionaryIndices(
      encodedNonNullValues, nonNullRowCount, pool, useVarintRowCount);
}

template <typename Values>
void expectAlphabetEntries(
    const Chunk& alphabetChunk,
    const Values& expectedValues,
    velox::memory::MemoryPool* pool) {
  using T = typename Values::value_type;
  const std::span<const T> expectedValuesSpan{
      expectedValues.data(), expectedValues.size()};
  EXPECT_EQ(
      alphabetChunk.rowCount, static_cast<uint32_t>(expectedValuesSpan.size()));
  ASSERT_EQ(alphabetChunk.content.size(), 1);

  auto encodedAlphabetOwner =
      std::make_shared<const std::string>(alphabetChunk.content.front());
  const std::string_view encodedAlphabet{*encodedAlphabetOwner};
  const auto alphabet = SharedDictionaryAlphabet::create(
      encodedAlphabet, std::move(encodedAlphabetOwner), pool);
  std::vector<uint32_t> alphabetIndices(expectedValuesSpan.size());
  std::iota(alphabetIndices.begin(), alphabetIndices.end(), 0);
  std::vector<typename TypeTraits<T>::physicalType> entries(
      alphabetIndices.size());
  alphabet->materialize<T>(alphabetIndices, entries.data());
  EXPECT_EQ(entries, physicalValues<T>(expectedValuesSpan));
}

class SharedDictionaryWriterTest : public testing::Test {
 protected:
  void SetUp() final {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  StreamDescriptorBuilder descriptor_{11, ScalarKind::Int32};
};

struct StringScopeTestParam {
  SharedDictionaryScope scope;
  uint32_t dictionaryId;
  bool usesResolver;
};

class StringSharedDictionaryWriterTest
    : public SharedDictionaryWriterTest,
      public testing::WithParamInterface<StringScopeTestParam> {};

TEST_F(SharedDictionaryWriterTest, stripeScope) {
  struct StripeInput {
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
    std::vector<int32_t> expectedAlphabet;
  };
  struct TestCase {
    std::string testName;
    std::vector<StripeInput> stripes;
  };

  const std::vector<TestCase> testSettings{
      {
          "starts fresh dictionary per stripe",
          {
              {
                  repeatedValues(std::array<int32_t, 2>{10, 20}, 512),
                  repeatedIndices(std::array<uint32_t, 2>{0, 1}, 512),
                  {10, 20},
              },
              {
                  repeatedValues(std::array<int32_t, 1>{30}, 512),
                  repeatedIndices(std::array<uint32_t, 1>{0}, 512),
                  {30},
              },
          },
      },
      {
          "assigns indices in first-seen order within each stripe",
          {
              {
                  repeatedValues(
                      std::array<int32_t, 5>{20, 10, 20, 30, 10}, 512),
                  repeatedIndices(std::array<uint32_t, 5>{0, 1, 0, 2, 1}, 512),
                  {20, 10, 30},
              },
              {
                  repeatedValues(std::array<int32_t, 3>{30, 20, 30}, 512),
                  repeatedIndices(std::array<uint32_t, 3>{0, 1, 0}, 512),
                  {30, 20},
              },
          },
      },
      {
          "deduplicates repeated values within a stripe",
          {
              {
                  repeatedValues(std::array<int32_t, 4>{10, 20, 10, 30}, 512),
                  repeatedIndices(std::array<uint32_t, 4>{0, 1, 0, 2}, 512),
                  {10, 20, 30},
              },
          },
      },
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.testName);
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7));
    Buffer buffer{*pool_};

    for (size_t i = 0; i < testData.stripes.size(); ++i) {
      SCOPED_TRACE(fmt::format("stripe={}", i));
      const auto& stripe = testData.stripes[i];
      const auto encoded = encodeValues(
          writer,
          /*stripeIndex=*/i,
          streamView(descriptor_, stripe.values),
          buffer);

      EXPECT_FALSE(encoded.empty());
      EXPECT_EQ(
          sharedDictionaryIndices(encoded, stripe.values.size(), pool_.get()),
          stripe.expectedIndices);

      const auto alphabetChunk = writer.encodeAlphabet(buffer);
      ASSERT_TRUE(alphabetChunk.has_value());
      expectAlphabetEntries(
          *alphabetChunk, stripe.expectedAlphabet, pool_.get());
      NIMBLE_ASSERT_THROW(
          writer.encodeAlphabet(buffer),
          fmt::format(
              "Stripe shared dictionary 7 already finalized its alphabet for "
              "stripe {}.",
              i));
    }
  }
}

TEST_P(StringSharedDictionaryWriterTest, roundTrip) {
  const auto testParam = GetParam();
  StreamDescriptorBuilder descriptor{12, ScalarKind::String};
  const std::array<std::string_view, 3> expectedAlphabet{
      "bravo", "alpha", "charlie"};
  std::shared_ptr<const ExternalDictionaryResolver> resolver;
  if (testParam.usesResolver) {
    resolver = std::make_shared<TestStringDictionaryResolver>(
        testParam.dictionaryId, expectedAlphabet, pool_.get());
  }
  auto writer = TestStringSharedDictionaryWriter{
      pool_.get(),
      writerOptions(testParam.scope, testParam.dictionaryId, resolver)};
  Buffer buffer{*pool_};

  const std::vector<std::string_view> values{
      "bravo", "alpha", "bravo", "charlie", "alpha"};
  const auto encoded = encodeValues(
      writer, /*stripeIndex=*/0, streamView(descriptor, values), buffer);

  EXPECT_FALSE(encoded.empty());
  EXPECT_EQ(
      sharedDictionaryIndices(
          encoded,
          static_cast<uint32_t>(values.size()),
          pool_.get(),
          /*useVarintRowCount=*/false,
          DataType::String),
      (std::vector<uint32_t>{0, 1, 0, 2, 1}));

  Encoding::Options options;
  if (testParam.scope == SharedDictionaryScope::External) {
    NIMBLE_ASSERT_THROW(
        writer.encodeAlphabet(buffer),
        "External shared dictionary 29 cannot encode an alphabet; its "
        "resolver owns the alphabet.");
    options.sharedDictionaryAlphabet =
        resolver->resolve(testParam.dictionaryId, DataType::String);
  } else {
    const auto alphabetChunk = writer.encodeAlphabet(buffer);
    ASSERT_TRUE(alphabetChunk.has_value());
    expectAlphabetEntries(*alphabetChunk, expectedAlphabet, pool_.get());
    auto encodedAlphabetOwner =
        std::make_shared<const std::string>(alphabetChunk->content.front());
    const std::string_view encodedAlphabet{*encodedAlphabetOwner};
    options.sharedDictionaryAlphabet = SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());
  }
  ASSERT_NE(options.sharedDictionaryAlphabet, nullptr);

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding = EncodingFactory{options}.create(
      *pool_, encoded, [&](uint32_t totalLength) -> void* {
        auto& stringBuffer = stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
        return stringBuffer->asMutable<void>();
      });

  std::vector<std::string_view> materialized(encoding->rowCount());
  encoding->materialize(encoding->rowCount(), materialized.data());
  EXPECT_EQ(materialized, values);
}

INSTANTIATE_TEST_SUITE_P(
    Scopes,
    StringSharedDictionaryWriterTest,
    testing::Values(
        StringScopeTestParam{
            .scope = SharedDictionaryScope::Stripe,
            .dictionaryId = 7,
            .usesResolver = false},
        StringScopeTestParam{
            .scope = SharedDictionaryScope::File,
            .dictionaryId = 17,
            .usesResolver = false},
        StringScopeTestParam{
            .scope = SharedDictionaryScope::External,
            .dictionaryId = 29,
            .usesResolver = true}),
    [](const testing::TestParamInfo<StringScopeTestParam>& testInfo) {
      return std::string{
          SharedDictionaryScopeName::toName(testInfo.param.scope)};
    });

TEST_F(SharedDictionaryWriterTest, nullableStripeScope) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7));
  Buffer buffer{*pool_};

  const auto values = repeatedNullableValues(
      std::array<std::optional<int32_t>, 6>{
          10, std::nullopt, 20, 10, std::nullopt, 20},
      256);
  const auto encoded = encodeValues(
      writer,
      /*stripeIndex=*/0,
      nullableStreamView(
          descriptor_, values.nonNullValues, values.notNullsSpan()),
      buffer);

  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  expectAlphabetEntries(*alphabet, std::array<int32_t, 2>{10, 20}, pool_.get());

  Encoding::Options options;
  auto encodedAlphabetOwner =
      std::make_shared<const std::string>(alphabet->content.front());
  const std::string_view encodedAlphabet{*encodedAlphabetOwner};
  options.sharedDictionaryAlphabet = SharedDictionaryAlphabet::create(
      encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());

  auto encoding = createEncoding(encoded, pool_.get(), options);
  const auto* nonNullValuesEncoding = nullableValuesChild(*encoding);
  ASSERT_NE(nonNullValuesEncoding, nullptr);
  EXPECT_EQ(
      nonNullValuesEncoding->encodingType(), EncodingType::SharedDictionary);

  std::vector<int32_t> materialized(encoding->rowCount());
  encoding->materialize(encoding->rowCount(), materialized.data());
  EXPECT_EQ(materialized, values.materializedValues);
}

TEST_F(SharedDictionaryWriterTest, nullableStripeScopeAbandon) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nullptr,
          /*alphabetEncodings=*/{},
          /*useExternalAlphabet=*/false,
          /*useVarintRowCount=*/false,
          trivialPreferredValueEncoding()));
  Buffer buffer{*pool_};

  const auto values = repeatedNullableValues(
      std::array<std::optional<int32_t>, 8>{
          1, std::nullopt, 2, 3, std::nullopt, 4, 5, 6},
      1);
  const auto encoded = encodeValues(
      writer,
      /*stripeIndex=*/0,
      nullableStreamView(
          descriptor_, values.nonNullValues, values.notNullsSpan()),
      buffer);

  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);
  auto encoding = createEncoding(encoded, pool_.get());
  const auto* nonNullValuesEncoding = nullableValuesChild(*encoding);
  ASSERT_NE(nonNullValuesEncoding, nullptr);
  EXPECT_NE(
      nonNullValuesEncoding->encodingType(), EncodingType::SharedDictionary);
  std::vector<int32_t> materialized(encoding->rowCount());
  encoding->materialize(encoding->rowCount(), materialized.data());
  EXPECT_EQ(materialized, values.materializedValues);
  EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
}

TEST_F(SharedDictionaryWriterTest, stripeScopeRequiresDictionaryCandidate) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nullptr,
          /*alphabetEncodings=*/{},
          /*useExternalAlphabet=*/false,
          /*useVarintRowCount=*/false,
          EncodingReadFactors{{EncodingType::Trivial, 1.0}}));
  Buffer buffer{*pool_};

  const std::vector<int32_t> values{1, 2, 3, 4, 5, 6};
  NIMBLE_ASSERT_THROW(
      encodeValues(
          writer,
          /*stripeIndex=*/0,
          streamView(descriptor_, values),
          buffer),
      "Stripe shared dictionary selection requires regular Dictionary in the "
      "non-shared value encoding candidates.");
}

TEST_F(
    SharedDictionaryWriterTest,
    stripeScopeRejectsEncodeValuesAfterAlphabetFinalized) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7));
  Buffer buffer{*pool_};

  const auto values = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  encodeValues(
      writer,
      /*stripeIndex=*/0,
      streamView(descriptor_, values),
      buffer);

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  expectAlphabetEntries(*alphabet, std::array<int32_t, 2>{10, 20}, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "Stripe shared dictionary 7 already finalized its alphabet for stripe "
      "0.");

  NIMBLE_ASSERT_THROW(
      encodeValues(
          writer,
          /*stripeIndex=*/0,
          streamView(descriptor_, values),
          buffer),
      "Stripe shared dictionary 7 cannot encode values after its alphabet was "
      "finalized for stripe 0.");

  const auto nextStripeValues = repeatedValues(std::array<int32_t, 1>{30}, 512);
  const auto nextStripeEncoded = encodeValues(
      writer,
      /*stripeIndex=*/1,
      streamView(descriptor_, nextStripeValues),
      buffer);

  EXPECT_EQ(
      sharedDictionaryIndices(
          nextStripeEncoded, nextStripeValues.size(), pool_.get()),
      repeatedIndices(std::array<uint32_t, 1>{0}, 512));

  const auto nextStripeAlphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(nextStripeAlphabet.has_value());
  expectAlphabetEntries(
      *nextStripeAlphabet, std::array<int32_t, 1>{30}, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "Stripe shared dictionary 7 already finalized its alphabet for stripe "
      "1.");
}

TEST_F(
    SharedDictionaryWriterTest,
    stripeScopeRequiresEncodeAlphabetBeforeNextStripe) {
  {
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7));
    Buffer buffer{*pool_};

    const auto values = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
    encodeValues(
        writer,
        /*stripeIndex=*/0,
        streamView(descriptor_, values),
        buffer);

    NIMBLE_ASSERT_THROW(
        encodeValues(
            writer,
            /*stripeIndex=*/1,
            streamView(descriptor_, values),
            buffer),
        "Stripe shared dictionary 7 reached a stripe boundary before encoding "
        "its alphabet.");
  }

  {
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::Stripe,
            /*dictionaryId=*/7,
            nullptr,
            /*alphabetEncodings=*/{},
            /*useExternalAlphabet=*/false,
            /*useVarintRowCount=*/false,
            trivialPreferredValueEncoding()));
    Buffer buffer{*pool_};

    const std::vector<int32_t> values{1, 2, 3, 4, 5, 6};
    encodeValues(
        writer,
        /*stripeIndex=*/0,
        streamView(descriptor_, values),
        buffer);

    NIMBLE_ASSERT_THROW(
        encodeValues(
            writer,
            /*stripeIndex=*/1,
            streamView(descriptor_, values),
            buffer),
        "Stripe shared dictionary 7 reached a stripe boundary with an active "
        "encoding decision.");
  }
}

TEST_F(SharedDictionaryWriterTest, stripeScopeRejectsBackwardStripeIndex) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7));
  Buffer buffer{*pool_};

  const auto firstStripeValues =
      repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  encodeValues(
      writer,
      /*stripeIndex=*/0,
      streamView(descriptor_, firstStripeValues),
      buffer);
  ASSERT_TRUE(writer.encodeAlphabet(buffer).has_value());

  const auto secondStripeValues =
      repeatedValues(std::array<int32_t, 1>{30}, 512);
  encodeValues(
      writer,
      /*stripeIndex=*/2,
      streamView(descriptor_, secondStripeValues),
      buffer);
  ASSERT_TRUE(writer.encodeAlphabet(buffer).has_value());

  NIMBLE_ASSERT_THROW(
      encodeValues(
          writer,
          /*stripeIndex=*/1,
          streamView(descriptor_, firstStripeValues),
          buffer),
      "Stripe shared dictionary 7 cannot move from stripe 2 back to stripe 1.");
}

TEST_F(SharedDictionaryWriterTest, stripeScopeUsesForcedAlphabetEncoding) {
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nullptr,
          {EncodingType::FixedBitWidth}));
  Buffer buffer{*pool_};

  const std::array<int32_t, 2> pattern{10, 20};
  const auto values = repeatedValues(pattern, 512);
  const auto encoded = encodeValues(
      writer,
      /*stripeIndex=*/0,
      streamView(descriptor_, values),
      buffer);

  EXPECT_FALSE(encoded.empty());
  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  ASSERT_EQ(alphabet->content.size(), 1);
  EXPECT_EQ(
      EncodingPrefix::encodingType(alphabet->content.front()),
      EncodingType::FixedBitWidth);
}

TEST_F(SharedDictionaryWriterTest, stripeAlphabetRoundTripsThroughReader) {
  struct TestParam {
    std::string testName;
    std::vector<EncodingType> alphabetEncodings;

    std::string debugString() const {
      return fmt::format(
          "{}, {} alphabet encoding candidates",
          testName,
          alphabetEncodings.size());
    }
  };

  const std::vector<TestParam> testSettings = {
      {"defaultSelection", {}},
      {"fixedBitWidthCandidate", {EncodingType::FixedBitWidth}}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::Stripe,
            /*dictionaryId=*/7,
            nullptr,
            testData.alphabetEncodings));
    Buffer buffer{*pool_};

    // Distinct values appear in first-use order, which is the order the
    // dictionary assigns indices in.
    const std::array<int32_t, 3> pattern{30, 10, 20};
    const auto values = repeatedValues(pattern, 512);
    encodeValues(
        writer,
        /*stripeIndex=*/0,
        streamView(descriptor_, values),
        buffer);

    const auto encodedAlphabet = writer.encodeAlphabet(buffer);
    ASSERT_TRUE(encodedAlphabet.has_value());
    ASSERT_EQ(encodedAlphabet->content.size(), 1);

    auto encodedAlphabetOwner =
        std::make_shared<const std::string>(encodedAlphabet->content.front());
    const std::string_view encodedAlphabetView{*encodedAlphabetOwner};
    const auto alphabet = SharedDictionaryAlphabet::create(
        encodedAlphabetView, std::move(encodedAlphabetOwner), pool_.get());
    EXPECT_EQ(alphabet->dataType(), DataType::Int32);
    EXPECT_EQ(alphabet->entryCount(), pattern.size());

    std::vector<uint32_t> indices(pattern.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<TypeTraits<int32_t>::physicalType> entries(indices.size());
    alphabet->materialize<int32_t>(indices, entries.data());

    const std::vector<TypeTraits<int32_t>::physicalType> expected{
        pattern.begin(), pattern.end()};
    EXPECT_EQ(entries, expected);
  }
}

TEST_F(
    SharedDictionaryWriterTest,
    stripeAlphabetRoundTripsThroughReaderAcrossStripes) {
  struct TestParam {
    std::string testName;
    std::vector<EncodingType> alphabetEncodings;

    std::string debugString() const {
      return fmt::format(
          "{}, {} alphabet encoding candidates",
          testName,
          alphabetEncodings.size());
    }
  };

  const std::vector<TestParam> testSettings = {
      {"defaultSelection", {}},
      {"fixedBitWidthCandidate", {EncodingType::FixedBitWidth}}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::Stripe,
            /*dictionaryId=*/7,
            nullptr,
            testData.alphabetEncodings));
    Buffer buffer{*pool_};

    for (size_t stripeIndex = 0; stripeIndex < 8; ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripe={}", stripeIndex));
      const auto expected = generatedStripeDictionary(stripeIndex);

      const auto encoded = encodeValues(
          writer,
          stripeIndex,
          streamView(descriptor_, expected.values),
          buffer);

      EXPECT_EQ(
          sharedDictionaryIndices(encoded, expected.values.size(), pool_.get()),
          expected.indices);

      const auto encodedAlphabet = writer.encodeAlphabet(buffer);
      ASSERT_TRUE(encodedAlphabet.has_value());
      expectAlphabetEntries(*encodedAlphabet, expected.alphabet, pool_.get());
      NIMBLE_ASSERT_THROW(
          writer.encodeAlphabet(buffer),
          fmt::format(
              "Stripe shared dictionary 7 already finalized its alphabet for "
              "stripe {}.",
              stripeIndex));
    }
  }
}

TEST_F(SharedDictionaryWriterTest, stripeScopeAbandon) {
  struct Chunk {
    std::string testName;
    std::vector<int32_t> values;
  };

  auto options =
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7);
  options.encodingSelectionPolicyCreator =
      stripeAbandonEncodingSelectionPolicyCreator();
  auto writer = createTestWriter(pool_.get(), options);
  Buffer buffer{*pool_};
  EXPECT_FALSE(writer.hasUsedDictionary());

  const std::vector<int32_t> directFriendlyValues{1, 2, 3, 4, 5, 6};
  const std::vector<Chunk> chunks{
      {
          "beginning chunk chooses direct and abandons stripe",
          directFriendlyValues,
      },
      {
          "middle chunk stays direct after abandon",
          repeatedValues(std::array<int32_t, 2>{10, 20}, 512),
      },
      {
          "ending chunk stays direct after abandon",
          repeatedValues(std::array<int32_t, 1>{30}, 512),
      },
  };

  for (const auto& chunk : chunks) {
    SCOPED_TRACE(chunk.testName);
    const auto encoded = encodeValues(
        writer,
        /*stripeIndex=*/0,
        streamView(descriptor_, chunk.values),
        buffer);
    EXPECT_NE(
        EncodingPrefix::encodingType(encoded), EncodingType::SharedDictionary);
    EXPECT_FALSE(writer.hasUsedDictionary());
  }
  EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
  EXPECT_FALSE(writer.hasUsedDictionary());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "Stripe shared dictionary 7 already finalized its alphabet for stripe "
      "0.");

  const auto repeated = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  const auto nextStripeEncoded = encodeValues(
      writer,
      /*stripeIndex=*/1,
      streamView(descriptor_, repeated),
      buffer);

  const std::array<uint32_t, 2> expectedIndices{0, 1};
  EXPECT_EQ(
      sharedDictionaryIndices(nextStripeEncoded, repeated.size(), pool_.get()),
      repeatedIndices(expectedIndices, 512));
  EXPECT_TRUE(writer.hasUsedDictionary());
  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  EXPECT_EQ(alphabet->rowCount, 2);
  EXPECT_TRUE(writer.hasUsedDictionary());

  const std::vector<uint32_t> directFriendlyIndices{0, 1, 2, 3, 4, 5};
  {
    SCOPED_TRACE("file scope enforces shared dictionary");
    auto fileWriter = createTestWriter(
        pool_.get(),
        writerOptions(SharedDictionaryScope::File, /*dictionaryId=*/17));
    Buffer fileBuffer{*pool_};

    const auto encoded = encodeValues(
        fileWriter,
        /*stripeIndex=*/0,
        streamView(descriptor_, directFriendlyValues),
        fileBuffer);

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded, directFriendlyValues.size(), pool_.get()),
        directFriendlyIndices);
    EXPECT_TRUE(fileWriter.hasUsedDictionary());
    const auto fileAlphabet = fileWriter.encodeAlphabet(fileBuffer);
    ASSERT_TRUE(fileAlphabet.has_value());
    expectAlphabetEntries(*fileAlphabet, directFriendlyValues, pool_.get());
    EXPECT_TRUE(fileWriter.hasUsedDictionary());
  }

  {
    SCOPED_TRACE("file scope external alphabet enforces shared dictionary");
    auto resolver = std::make_shared<TestDictionaryResolver>(
        /*dictionaryId=*/23, directFriendlyValues, pool_.get());
    auto externalAlphabetFileWriter = createTestWriter(
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::File,
            /*dictionaryId=*/23,
            resolver,
            /*alphabetEncodings=*/{},
            /*useExternalAlphabet=*/true));
    Buffer externalAlphabetFileBuffer{*pool_};

    const auto encoded = encodeValues(
        externalAlphabetFileWriter,
        /*stripeIndex=*/0,
        streamView(descriptor_, directFriendlyValues),
        externalAlphabetFileBuffer);

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded, directFriendlyValues.size(), pool_.get()),
        directFriendlyIndices);
    EXPECT_TRUE(externalAlphabetFileWriter.hasUsedDictionary());
    const auto fileAlphabet =
        externalAlphabetFileWriter.encodeAlphabet(externalAlphabetFileBuffer);
    ASSERT_TRUE(fileAlphabet.has_value());
    expectAlphabetEntries(*fileAlphabet, directFriendlyValues, pool_.get());
    EXPECT_TRUE(externalAlphabetFileWriter.hasUsedDictionary());
  }

  {
    SCOPED_TRACE("external scope enforces shared dictionary");
    auto resolver = std::make_shared<TestDictionaryResolver>(
        /*dictionaryId=*/29, directFriendlyValues, pool_.get());
    auto externalWriter = createTestWriter(
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::External, /*dictionaryId=*/29, resolver));
    Buffer externalBuffer{*pool_};

    const auto encoded = encodeValues(
        externalWriter,
        /*stripeIndex=*/0,
        streamView(descriptor_, directFriendlyValues),
        externalBuffer);

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded, directFriendlyValues.size(), pool_.get()),
        directFriendlyIndices);
    EXPECT_TRUE(externalWriter.hasUsedDictionary());
    NIMBLE_ASSERT_THROW(
        externalWriter.encodeAlphabet(externalBuffer),
        "External shared dictionary 29 cannot encode an alphabet; its "
        "resolver owns the alphabet.");
    EXPECT_TRUE(externalWriter.hasUsedDictionary());
  }
}

TEST_F(SharedDictionaryWriterTest, encodeValuesRejectsEmptyValues) {
  struct TestParam {
    SharedDictionaryScope scope;
    uint32_t dictionaryId;
    bool useExternalAlphabet;
    bool usesResolver;

    std::string debugString() const {
      return fmt::format(
          "scope {}, dictionary {}, useExternalAlphabet {}, usesResolver {}",
          scope,
          dictionaryId,
          useExternalAlphabet,
          usesResolver);
    }
  };

  const std::vector<TestParam> testSettings{
      {SharedDictionaryScope::Stripe, /*dictionaryId=*/7, false, false},
      {SharedDictionaryScope::File, /*dictionaryId=*/17, false, false},
      {SharedDictionaryScope::File, /*dictionaryId=*/23, true, true},
      {SharedDictionaryScope::External, /*dictionaryId=*/29, false, true},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const std::vector<int32_t> externalAlphabet{10, 20};
    auto resolver = testData.usesResolver
        ? std::make_shared<TestDictionaryResolver>(
              testData.dictionaryId, externalAlphabet, pool_.get())
        : nullptr;
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            testData.scope,
            testData.dictionaryId,
            resolver,
            /*alphabetEncodings=*/{},
            testData.useExternalAlphabet));
    Buffer buffer{*pool_};

    const std::vector<int32_t> values;
    NIMBLE_ASSERT_THROW(
        encodeValues(
            writer,
            /*stripeIndex=*/0,
            streamView(descriptor_, values),
            buffer),
        fmt::format(
            "{} shared dictionary {} cannot encode an empty value stream.",
            testData.scope,
            testData.dictionaryId));

    const auto validValues =
        repeatedValues(std::array<int32_t, 3>{10, 20, 10}, 512);
    const auto encoded = encodeValues(
        writer,
        /*stripeIndex=*/0,
        streamView(descriptor_, validValues),
        buffer);
    EXPECT_EQ(
        sharedDictionaryIndices(encoded, validValues.size(), pool_.get()),
        repeatedIndices(std::array<uint32_t, 3>{0, 1, 0}, 512));
    if (testData.scope == SharedDictionaryScope::External) {
      NIMBLE_ASSERT_THROW(
          writer.encodeAlphabet(buffer),
          fmt::format(
              "External shared dictionary {} cannot encode an alphabet; its "
              "resolver owns the alphabet.",
              testData.dictionaryId));
    } else {
      ASSERT_TRUE(writer.encodeAlphabet(buffer).has_value());
    }
  }
}

TEST_F(SharedDictionaryWriterTest, fileScope) {
  struct ChunkInput {
    size_t stripeIndex;
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
  };

  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(SharedDictionaryScope::File, /*dictionaryId=*/17));
  Buffer buffer{*pool_};

  // File scope builds one dictionary for all chunks, so new entries append to
  // the same alphabet and existing entries keep their original indices.
  const std::vector<ChunkInput> chunks{
      {
          /*stripeIndex=*/0,
          {10, 10, 20, 10, 20},
          {0, 0, 1, 0, 1},
      },
      {
          /*stripeIndex=*/0,
          {20, 30, 40, 30},
          {1, 2, 3, 2},
      },
      {
          /*stripeIndex=*/1,
          {40, 50, 10, 50},
          {3, 4, 0, 4},
      },
  };

  for (const auto& chunk : chunks) {
    SCOPED_TRACE(fmt::format("stripe={}", chunk.stripeIndex));
    const auto encoded = encodeValues(
        writer,
        chunk.stripeIndex,
        streamView(descriptor_, chunk.values),
        buffer);

    EXPECT_EQ(
        sharedDictionaryIndices(encoded, chunk.values.size(), pool_.get()),
        chunk.expectedIndices);
  }

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  expectAlphabetEntries(
      *alphabet, std::array<int32_t, 5>{10, 20, 30, 40, 50}, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "File shared dictionary 17 already finalized its alphabet.");
  NIMBLE_ASSERT_THROW(
      encodeValues(
          writer,
          /*stripeIndex=*/1,
          streamView(descriptor_, chunks.front().values),
          buffer),
      "File shared dictionary 17 cannot encode values after its alphabet was "
      "finalized.");
}

TEST_F(SharedDictionaryWriterTest, nullableEncodingAcrossScopes) {
  struct TestParam {
    std::string testName;
    SharedDictionaryScope scope;
    uint32_t dictionaryId;
    bool usesResolver;
  };

  const std::vector<TestParam> testSettings{
      {"stripe", SharedDictionaryScope::Stripe, /*dictionaryId=*/7, false},
      {"file", SharedDictionaryScope::File, /*dictionaryId=*/17, false},
      {"external", SharedDictionaryScope::External, /*dictionaryId=*/29, true},
  };

  constexpr size_t repeatCount{512};
  const auto nonNullValues =
      repeatedValues(std::array<int32_t, 4>{10, 20, 10, 30}, repeatCount);
  const auto expectedIndices =
      repeatedIndices(std::array<uint32_t, 4>{0, 1, 0, 2}, repeatCount);
  const std::array<bool, 6> nonNullPattern{
      true, false, true, true, false, true};
  const auto rowCount = nonNullPattern.size() * repeatCount;
  const auto nullCount = 2 * repeatCount;

  for (const auto& testData : testSettings) {
    for (const bool useVarintRowCount : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "testName={}, useVarintRowCount={}",
              testData.testName,
              useVarintRowCount));
      const std::array<int32_t, 3> externalAlphabet{10, 20, 30};
      auto resolver = testData.usesResolver
          ? std::make_shared<TestDictionaryResolver>(
                testData.dictionaryId, externalAlphabet, pool_.get())
          : nullptr;
      auto writer = createTestWriter(
          pool_.get(),
          writerOptions(
              testData.scope,
              testData.dictionaryId,
              resolver,
              /*alphabetEncodings=*/{},
              /*useExternalAlphabet=*/false,
              useVarintRowCount));
      Buffer buffer{*pool_};
      Vector<bool> nonNulls{pool_.get()};
      nonNulls.resize(rowCount);
      for (size_t i = 0; i < rowCount; ++i) {
        nonNulls[i] = nonNullPattern[i % nonNullPattern.size()];
      }
      const StreamDataView stream{
          descriptor_,
          bytesOf(nonNullValues),
          static_cast<uint32_t>(rowCount),
          std::span<const bool>{nonNulls.data(), nonNulls.size()},
          static_cast<uint32_t>(nullCount)};

      const auto encoded = encodeValues(
          writer,
          /*stripeIndex=*/0,
          stream,
          buffer,
          Encoding::Options{.useVarintRowCount = useVarintRowCount});

      EXPECT_EQ(
          nullableSharedDictionaryIndices(
              encoded,
              rowCount,
              std::span<const bool>{nonNulls.data(), nonNulls.size()},
              nonNullValues.size(),
              pool_.get(),
              useVarintRowCount),
          expectedIndices);

      if (testData.scope == SharedDictionaryScope::External) {
        NIMBLE_ASSERT_THROW(
            writer.encodeAlphabet(buffer),
            "External shared dictionary 29 cannot encode an alphabet; its "
            "resolver owns the alphabet.");
      } else {
        const auto alphabet = writer.encodeAlphabet(buffer);
        ASSERT_TRUE(alphabet.has_value());
        expectAlphabetEntries(
            *alphabet, std::array<int32_t, 3>{10, 20, 30}, pool_.get());
      }
    }
  }
}

TEST_F(SharedDictionaryWriterTest, fileScopeWithExternalAlphabet) {
  struct ChunkInput {
    size_t stripeIndex;
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
  };

  const std::vector<int32_t> externalAlphabet{7, 11, 13, 17};
  auto resolver = std::make_shared<TestDictionaryResolver>(
      /*dictionaryId=*/23,
      externalAlphabet,
      pool_.get(),
      std::array{EncodingType::Trivial});
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::File,
          /*dictionaryId=*/23,
          resolver,
          {EncodingType::FixedBitWidth},
          /*useExternalAlphabet=*/true));
  Buffer buffer{*pool_};

  // Nothing has been encoded yet, so there is no alphabet to store.
  EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());

  const std::vector<ChunkInput> chunks{
      {
          /*stripeIndex=*/0,
          {13, 7, 11, 13},
          {2, 0, 1, 2},
      },
      {
          /*stripeIndex=*/1,
          {17, 11, 7},
          {3, 1, 0},
      },
      {
          /*stripeIndex=*/2,
          {7, 13, 17, 11},
          {0, 2, 3, 1},
      },
  };

  for (const auto& chunk : chunks) {
    SCOPED_TRACE(fmt::format("stripe={}", chunk.stripeIndex));
    const auto encoded = encodeValues(
        writer,
        chunk.stripeIndex,
        streamView(descriptor_, chunk.values),
        buffer);

    EXPECT_EQ(
        sharedDictionaryIndices(encoded, chunk.values.size(), pool_.get()),
        chunk.expectedIndices);
  }

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  ASSERT_EQ(alphabet->content.size(), 1);
  EXPECT_EQ(
      EncodingPrefix::encodingType(alphabet->content.front()),
      EncodingType::Trivial);
  expectAlphabetEntries(*alphabet, externalAlphabet, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "File shared dictionary 23 already finalized its alphabet.");
  NIMBLE_ASSERT_THROW(
      encodeValues(
          writer,
          /*stripeIndex=*/3,
          streamView(descriptor_, chunks.front().values),
          buffer),
      "File shared dictionary 23 cannot encode values after its alphabet was "
      "finalized.");
}

TEST_F(SharedDictionaryWriterTest, externalAlphabetRejectsUnknownValue) {
  struct TestParam {
    SharedDictionaryScope scope;
    bool useExternalAlphabet;

    std::string debugString() const {
      return fmt::format(
          "scope {}, useExternalAlphabet {}", scope, useExternalAlphabet);
    }
  };

  const std::vector<TestParam> testSettings = {
      {SharedDictionaryScope::File, true},
      {SharedDictionaryScope::External, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const std::vector<int32_t> externalAlphabet{10, 20, 30};
    auto resolver = std::make_shared<TestDictionaryResolver>(
        /*dictionaryId=*/29, externalAlphabet, pool_.get());
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            testData.scope,
            /*dictionaryId=*/29,
            resolver,
            /*alphabetEncodings=*/{},
            testData.useExternalAlphabet));
    Buffer buffer{*pool_};

    // An external alphabet cannot grow, so an unseen value is a writer input
    // error rather than a new dictionary entry.
    const std::vector<int32_t> values{10, 99};
    NIMBLE_ASSERT_USER_THROW(
        encodeValues(
            writer,
            /*stripeIndex=*/0,
            streamView(descriptor_, values),
            buffer),
        fmt::format(
            "{} shared dictionary 29 does not contain value 99.",
            testData.scope));
  }
}

TEST_F(SharedDictionaryWriterTest, externalScope) {
  struct ChunkInput {
    size_t stripeIndex;
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
  };

  const std::vector<int32_t> externalAlphabet{10, 20, 30, 40};
  auto resolver = std::make_shared<TestDictionaryResolver>(
      /*dictionaryId=*/29, externalAlphabet, pool_.get());
  auto writer = createTestWriter(
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::External,
          /*dictionaryId=*/29,
          resolver));
  Buffer buffer{*pool_};

  const std::vector<ChunkInput> chunks{
      {
          /*stripeIndex=*/0,
          {10, 20, 10},
          {0, 1, 0},
      },
      {
          /*stripeIndex=*/1,
          {30, 20},
          {2, 1},
      },
      {
          /*stripeIndex=*/2,
          {40, 10, 30, 40},
          {3, 0, 2, 3},
      },
  };

  for (const auto& chunk : chunks) {
    SCOPED_TRACE(fmt::format("stripe={}", chunk.stripeIndex));
    const auto encoded = encodeValues(
        writer,
        chunk.stripeIndex,
        streamView(descriptor_, chunk.values),
        buffer);

    EXPECT_EQ(
        sharedDictionaryIndices(encoded, chunk.values.size(), pool_.get()),
        chunk.expectedIndices);
  }
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "External shared dictionary 29 cannot encode an alphabet; its resolver "
      "owns the alphabet.");
}

TEST_F(SharedDictionaryWriterTest, externalAlphabetRequiresResolver) {
  struct TestParam {
    SharedDictionaryScope scope;
    bool useExternalAlphabet;

    std::string debugString() const {
      return fmt::format(
          "scope {}, useExternalAlphabet {}", scope, useExternalAlphabet);
    }
  };

  const std::vector<TestParam> testSettings{
      {SharedDictionaryScope::File, true},
      {SharedDictionaryScope::External, false},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto writer = createTestWriter(
        pool_.get(),
        writerOptions(
            testData.scope,
            /*dictionaryId=*/29,
            nullptr,
            /*alphabetEncodings=*/{},
            testData.useExternalAlphabet));
    Buffer buffer{*pool_};

    if (testData.scope == SharedDictionaryScope::External) {
      NIMBLE_ASSERT_THROW(
          writer.encodeAlphabet(buffer),
          "External shared dictionary 29 cannot encode an alphabet; its "
          "resolver owns the alphabet.");
    } else {
      EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
    }

    const std::vector<int32_t> values{10};
    NIMBLE_ASSERT_USER_THROW(
        encodeValues(
            writer,
            /*stripeIndex=*/0,
            streamView(descriptor_, values),
            buffer),
        fmt::format(
            "{} shared dictionary 29 requires a dictionary resolver.",
            testData.scope));
  }
}

} // namespace
} // namespace facebook::nimble

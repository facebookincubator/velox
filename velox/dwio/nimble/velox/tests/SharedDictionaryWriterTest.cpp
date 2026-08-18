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
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SharedDictionaryWriter.h"
#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble {
namespace {

using TestSharedDictionaryWriter = SharedDictionaryWriter<int32_t>;

EncodingSelectionPolicyCreator testEncodingSelectionPolicyCreator() {
  return [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    const auto encodingType = dataType == DataType::Uint32
        ? EncodingType::FixedBitWidth
        : EncodingType::Trivial;
    ManualEncodingSelectionPolicyFactory factory{
        {{encodingType, 1.0}}, std::nullopt};
    return factory.createPolicy(dataType);
  };
}

TestSharedDictionaryWriter::Options writerOptions(
    SharedDictionaryScope scope,
    uint32_t dictionaryId,
    std::shared_ptr<const SharedDictionaryResolver> resolver = nullptr,
    std::vector<EncodingType> alphabetEncodingCandidates = {},
    bool usesPrebuiltAlphabet = false) {
  return TestSharedDictionaryWriter::Options{
      .scope = scope,
      .dictionaryId = dictionaryId,
      .usesPrebuiltAlphabet = usesPrebuiltAlphabet,
      .alphabetEncodingCandidates = std::move(alphabetEncodingCandidates),
      .encodingSelectionPolicyCreator = testEncodingSelectionPolicyCreator(),
      .encodingOptions = {},
      .resolver = std::move(resolver)};
}

class TestSharedDictionaryResolver final : public SharedDictionaryResolver {
 public:
  TestSharedDictionaryResolver(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      std::span<const int32_t> values,
      velox::memory::MemoryPool* pool)
      : scope_{scope},
        dictionaryId_{dictionaryId},
        alphabet_{test::createSharedDictionaryAlphabet<int32_t>(
            values,
            /*candidateEncodings=*/{},
            pool)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      DataType dataType) const final {
    if (scope != scope_ || dictionaryId != dictionaryId_ ||
        dataType != DataType::Int32) {
      return nullptr;
    }
    return alphabet_;
  }

 private:
  const SharedDictionaryScope scope_;
  const uint32_t dictionaryId_;
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
};

std::string_view bytesOf(std::span<const int32_t> values) {
  return {
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(int32_t)};
}

StreamDataView streamView(
    const StreamDescriptorBuilder& descriptor,
    std::span<const int32_t> values) {
  return StreamDataView{
      descriptor, bytesOf(values), static_cast<uint32_t>(values.size())};
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

std::vector<TypeTraits<int32_t>::physicalType> physicalValues(
    std::span<const int32_t> values) {
  std::vector<TypeTraits<int32_t>::physicalType> physical;
  physical.reserve(values.size());
  for (const auto value : values) {
    physical.push_back(
        EncodingPhysicalType<int32_t>::asEncodingPhysicalType(value));
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

struct ExpectedStripeDictionary {
  std::vector<int32_t> values;
  std::vector<uint32_t> indices{};
  std::vector<int32_t> alphabet{};
};

ExpectedStripeDictionary expectedStripeDictionary(std::vector<int32_t> values) {
  ExpectedStripeDictionary expected{.values = std::move(values)};
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

// Verifies the shared-dictionary envelope and decodes the nested index stream
// after the scope and dictionary id header.
std::vector<uint32_t> sharedDictionaryIndices(
    std::string_view encoded,
    SharedDictionaryScope scope,
    uint32_t dictionaryId,
    uint32_t rowCount,
    velox::memory::MemoryPool* pool) {
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  EXPECT_EQ(encodingType, EncodingType::SharedDictionary);
  if (encodingType != EncodingType::SharedDictionary) {
    return {};
  }
  EXPECT_EQ(EncodingPrefix::dataType(encoded), DataType::Int32);
  EXPECT_EQ(
      EncodingPrefix::readRowCount(encoded, /*useVarint=*/false), rowCount);

  const char* pos =
      encoded.data() + EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
  EXPECT_LT(pos, encoded.end());
  EXPECT_EQ(toSharedDictionaryScope(static_cast<uint8_t>(*pos++)), scope);
  EXPECT_LT(pos, encoded.end());
  EXPECT_EQ(varint::readVarint32(&pos), dictionaryId);
  EXPECT_LE(pos, encoded.end());
  const std::string_view encodedIndices{
      pos, static_cast<size_t>(encoded.end() - pos)};

  auto indicesEncoding = EncodingFactory{}.create(
      *pool,
      encodedIndices,
      [](uint32_t /*size*/) -> void* { return nullptr; },
      Encoding::Options{});
  EXPECT_EQ(indicesEncoding->dataType(), DataType::Uint32);
  EXPECT_EQ(indicesEncoding->rowCount(), rowCount);

  std::vector<uint32_t> indices(rowCount);
  indicesEncoding->materialize(rowCount, indices.data());
  return indices;
}

void expectAlphabetEntries(
    const Chunk& alphabetChunk,
    std::span<const int32_t> expectedValues,
    velox::memory::MemoryPool* pool) {
  EXPECT_EQ(
      alphabetChunk.rowCount, static_cast<uint32_t>(expectedValues.size()));
  ASSERT_EQ(alphabetChunk.content.size(), 1);

  const SharedDictionaryAlphabet alphabet{
      alphabetChunk.content.front(), Encoding::Options{}, pool};
  std::vector<uint32_t> alphabetIndices(expectedValues.size());
  std::iota(alphabetIndices.begin(), alphabetIndices.end(), 0);
  std::vector<TypeTraits<int32_t>::physicalType> entries(
      alphabetIndices.size());
  alphabet.materialize<int32_t>(alphabetIndices, entries.data());
  EXPECT_EQ(entries, physicalValues(expectedValues));
}

class SharedDictionaryWriterTest : public testing::Test {
 protected:
  void SetUp() final {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  StreamDescriptorBuilder descriptor_{11, ScalarKind::Int32};
};

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
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
    Buffer buffer{*pool_};

    for (size_t i = 0; i < testData.stripes.size(); ++i) {
      SCOPED_TRACE(fmt::format("stripe={}", i));
      const auto& stripe = testData.stripes[i];
      const auto encoded = writer.encodeValues(
          /*stripeIndex=*/i, buffer, streamView(descriptor_, stripe.values));

      EXPECT_FALSE(encoded.empty());
      EXPECT_EQ(
          sharedDictionaryIndices(
              encoded,
              SharedDictionaryScope::Stripe,
              /*dictionaryId=*/7,
              stripe.values.size(),
              pool_.get()),
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

TEST_F(
    SharedDictionaryWriterTest,
    stripeScopeRejectsEncodeValuesAfterAlphabetFinalized) {
  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
  Buffer buffer{*pool_};

  const auto values = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  writer.encodeValues(
      /*stripeIndex=*/0, buffer, streamView(descriptor_, values));

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  expectAlphabetEntries(*alphabet, std::array<int32_t, 2>{10, 20}, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "Stripe shared dictionary 7 already finalized its alphabet for stripe "
      "0.");

  NIMBLE_ASSERT_THROW(
      writer.encodeValues(
          /*stripeIndex=*/0, buffer, streamView(descriptor_, values)),
      "Stripe shared dictionary 7 cannot encode values after its alphabet was "
      "finalized for stripe 0.");

  const auto nextStripeValues = repeatedValues(std::array<int32_t, 1>{30}, 512);
  const auto nextStripeEncoded = writer.encodeValues(
      /*stripeIndex=*/1, buffer, streamView(descriptor_, nextStripeValues));

  EXPECT_EQ(
      sharedDictionaryIndices(
          nextStripeEncoded,
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nextStripeValues.size(),
          pool_.get()),
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
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
    Buffer buffer{*pool_};

    const auto values = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
    writer.encodeValues(
        /*stripeIndex=*/0, buffer, streamView(descriptor_, values));

    NIMBLE_ASSERT_THROW(
        writer.encodeValues(
            /*stripeIndex=*/1, buffer, streamView(descriptor_, values)),
        "Stripe shared dictionary 7 reached a stripe boundary before encoding "
        "its alphabet.");
  }

  {
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
    Buffer buffer{*pool_};

    const std::vector<int32_t> values{1, 2, 3, 4, 5, 6};
    writer.encodeValues(
        /*stripeIndex=*/0, buffer, streamView(descriptor_, values));

    NIMBLE_ASSERT_THROW(
        writer.encodeValues(
            /*stripeIndex=*/1, buffer, streamView(descriptor_, values)),
        "Stripe shared dictionary 7 reached a stripe boundary with an active "
        "encoding decision.");
  }
}

TEST_F(SharedDictionaryWriterTest, stripeScopeRejectsBackwardStripeIndex) {
  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
  Buffer buffer{*pool_};

  const auto firstStripeValues =
      repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  writer.encodeValues(
      /*stripeIndex=*/0, buffer, streamView(descriptor_, firstStripeValues));
  ASSERT_TRUE(writer.encodeAlphabet(buffer).has_value());

  const auto secondStripeValues =
      repeatedValues(std::array<int32_t, 1>{30}, 512);
  writer.encodeValues(
      /*stripeIndex=*/2, buffer, streamView(descriptor_, secondStripeValues));
  ASSERT_TRUE(writer.encodeAlphabet(buffer).has_value());

  NIMBLE_ASSERT_THROW(
      writer.encodeValues(
          /*stripeIndex=*/1,
          buffer,
          streamView(descriptor_, firstStripeValues)),
      "Stripe shared dictionary 7 cannot move from stripe 2 back to stripe 1.");
}

TEST_F(SharedDictionaryWriterTest, stripeScopeUsesForcedAlphabetEncoding) {
  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nullptr,
          {EncodingType::FixedBitWidth})};
  Buffer buffer{*pool_};

  const std::array<int32_t, 2> pattern{10, 20};
  const auto values = repeatedValues(pattern, 512);
  const auto encoded = writer.encodeValues(
      /*stripeIndex=*/0, buffer, streamView(descriptor_, values));

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
    std::vector<EncodingType> alphabetEncodingCandidates;

    std::string debugString() const {
      return fmt::format(
          "{}, {} alphabet encoding candidates",
          testName,
          alphabetEncodingCandidates.size());
    }
  };

  const std::vector<TestParam> testSettings = {
      {"defaultSelection", {}},
      {"fixedBitWidthCandidate", {EncodingType::FixedBitWidth}}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::Stripe,
            /*dictionaryId=*/7,
            nullptr,
            testData.alphabetEncodingCandidates)};
    Buffer buffer{*pool_};

    // Distinct values appear in first-use order, which is the order the
    // dictionary assigns indices in.
    const std::array<int32_t, 3> pattern{30, 10, 20};
    const auto values = repeatedValues(pattern, 512);
    writer.encodeValues(
        /*stripeIndex=*/0, buffer, streamView(descriptor_, values));

    const auto encodedAlphabet = writer.encodeAlphabet(buffer);
    ASSERT_TRUE(encodedAlphabet.has_value());
    ASSERT_EQ(encodedAlphabet->content.size(), 1);

    const SharedDictionaryAlphabet alphabet{
        encodedAlphabet->content.front(), Encoding::Options{}, pool_.get()};
    EXPECT_EQ(alphabet.dataType(), DataType::Int32);
    EXPECT_EQ(alphabet.entryCount(), pattern.size());

    std::vector<uint32_t> indices(pattern.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<TypeTraits<int32_t>::physicalType> entries(indices.size());
    alphabet.materialize<int32_t>(indices, entries.data());

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
    std::vector<EncodingType> alphabetEncodingCandidates;

    std::string debugString() const {
      return fmt::format(
          "{}, {} alphabet encoding candidates",
          testName,
          alphabetEncodingCandidates.size());
    }
  };

  const std::vector<TestParam> testSettings = {
      {"defaultSelection", {}},
      {"fixedBitWidthCandidate", {EncodingType::FixedBitWidth}}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::Stripe,
            /*dictionaryId=*/7,
            nullptr,
            testData.alphabetEncodingCandidates)};
    Buffer buffer{*pool_};

    for (size_t stripeIndex = 0; stripeIndex < 8; ++stripeIndex) {
      SCOPED_TRACE(fmt::format("stripe={}", stripeIndex));
      const auto expected = generatedStripeDictionary(stripeIndex);

      const auto encoded = writer.encodeValues(
          stripeIndex, buffer, streamView(descriptor_, expected.values));

      EXPECT_EQ(
          sharedDictionaryIndices(
              encoded,
              SharedDictionaryScope::Stripe,
              /*dictionaryId=*/7,
              expected.values.size(),
              pool_.get()),
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

  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(SharedDictionaryScope::Stripe, /*dictionaryId=*/7)};
  Buffer buffer{*pool_};

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
    const auto encoded = writer.encodeValues(
        /*stripeIndex=*/0, buffer, streamView(descriptor_, chunk.values));
    EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Trivial);
  }
  EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "Stripe shared dictionary 7 already finalized its alphabet for stripe "
      "0.");

  const auto repeated = repeatedValues(std::array<int32_t, 2>{10, 20}, 512);
  const auto nextStripeEncoded = writer.encodeValues(
      /*stripeIndex=*/1, buffer, streamView(descriptor_, repeated));

  const std::array<uint32_t, 2> expectedIndices{0, 1};
  EXPECT_EQ(
      sharedDictionaryIndices(
          nextStripeEncoded,
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          repeated.size(),
          pool_.get()),
      repeatedIndices(expectedIndices, 512));
  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  EXPECT_EQ(alphabet->rowCount, 2);

  const std::vector<uint32_t> directFriendlyIndices{0, 1, 2, 3, 4, 5};
  {
    SCOPED_TRACE("file scope enforces shared dictionary");
    TestSharedDictionaryWriter fileWriter{
        pool_.get(),
        writerOptions(SharedDictionaryScope::File, /*dictionaryId=*/17)};
    Buffer fileBuffer{*pool_};

    const auto encoded = fileWriter.encodeValues(
        /*stripeIndex=*/0,
        fileBuffer,
        streamView(descriptor_, directFriendlyValues));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::File,
            /*dictionaryId=*/17,
            directFriendlyValues.size(),
            pool_.get()),
        directFriendlyIndices);
    const auto fileAlphabet = fileWriter.encodeAlphabet(fileBuffer);
    ASSERT_TRUE(fileAlphabet.has_value());
    expectAlphabetEntries(*fileAlphabet, directFriendlyValues, pool_.get());
  }

  {
    SCOPED_TRACE("file scope prebuilt dictionary enforces shared dictionary");
    auto resolver = std::make_shared<TestSharedDictionaryResolver>(
        SharedDictionaryScope::File,
        /*dictionaryId=*/23,
        directFriendlyValues,
        pool_.get());
    TestSharedDictionaryWriter prebuiltFileWriter{
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::File,
            /*dictionaryId=*/23,
            resolver,
            /*alphabetEncodingCandidates=*/{},
            /*usesPrebuiltAlphabet=*/true)};
    Buffer prebuiltFileBuffer{*pool_};

    const auto encoded = prebuiltFileWriter.encodeValues(
        /*stripeIndex=*/0,
        prebuiltFileBuffer,
        streamView(descriptor_, directFriendlyValues));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::File,
            /*dictionaryId=*/23,
            directFriendlyValues.size(),
            pool_.get()),
        directFriendlyIndices);
    const auto fileAlphabet =
        prebuiltFileWriter.encodeAlphabet(prebuiltFileBuffer);
    ASSERT_TRUE(fileAlphabet.has_value());
    expectAlphabetEntries(*fileAlphabet, directFriendlyValues, pool_.get());
  }

  {
    SCOPED_TRACE("external scope enforces shared dictionary");
    auto resolver = std::make_shared<TestSharedDictionaryResolver>(
        SharedDictionaryScope::External,
        /*dictionaryId=*/29,
        directFriendlyValues,
        pool_.get());
    TestSharedDictionaryWriter externalWriter{
        pool_.get(),
        writerOptions(
            SharedDictionaryScope::External, /*dictionaryId=*/29, resolver)};
    Buffer externalBuffer{*pool_};

    const auto encoded = externalWriter.encodeValues(
        /*stripeIndex=*/0,
        externalBuffer,
        streamView(descriptor_, directFriendlyValues));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::External,
            /*dictionaryId=*/29,
            directFriendlyValues.size(),
            pool_.get()),
        directFriendlyIndices);
    EXPECT_FALSE(externalWriter.encodeAlphabet(externalBuffer).has_value());
  }
}

TEST_F(SharedDictionaryWriterTest, encodeValuesRejectsEmptyValues) {
  struct TestParam {
    SharedDictionaryScope scope;
    uint32_t dictionaryId;
    bool usesPrebuiltAlphabet;
    bool usesResolver;

    std::string debugString() const {
      return fmt::format(
          "scope {}, dictionary {}, usesPrebuiltAlphabet {}, usesResolver {}",
          scope,
          dictionaryId,
          usesPrebuiltAlphabet,
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
    const std::vector<int32_t> prebuiltAlphabet{10, 20};
    auto resolver = testData.usesResolver
        ? std::make_shared<TestSharedDictionaryResolver>(
              testData.scope,
              testData.dictionaryId,
              prebuiltAlphabet,
              pool_.get())
        : nullptr;
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(
            testData.scope,
            testData.dictionaryId,
            resolver,
            /*alphabetEncodingCandidates=*/{},
            testData.usesPrebuiltAlphabet)};
    Buffer buffer{*pool_};

    const std::vector<int32_t> values;
    NIMBLE_ASSERT_THROW(
        writer.encodeValues(
            /*stripeIndex=*/0, buffer, streamView(descriptor_, values)),
        fmt::format(
            "{} shared dictionary {} cannot encode an empty value stream.",
            testData.scope,
            testData.dictionaryId));

    const auto validValues =
        repeatedValues(std::array<int32_t, 3>{10, 20, 10}, 512);
    const auto encoded = writer.encodeValues(
        /*stripeIndex=*/0, buffer, streamView(descriptor_, validValues));
    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            testData.scope,
            testData.dictionaryId,
            validValues.size(),
            pool_.get()),
        repeatedIndices(std::array<uint32_t, 3>{0, 1, 0}, 512));
    if (testData.scope == SharedDictionaryScope::External) {
      EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
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

  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(SharedDictionaryScope::File, /*dictionaryId=*/17)};
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
    const auto encoded = writer.encodeValues(
        chunk.stripeIndex, buffer, streamView(descriptor_, chunk.values));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::File,
            /*dictionaryId=*/17,
            chunk.values.size(),
            pool_.get()),
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
      writer.encodeValues(
          /*stripeIndex=*/1,
          buffer,
          streamView(descriptor_, chunks.front().values)),
      "File shared dictionary 17 cannot encode values after its alphabet was "
      "finalized.");
}

TEST_F(SharedDictionaryWriterTest, fileScopeWithPrebuilt) {
  struct ChunkInput {
    size_t stripeIndex;
    std::vector<int32_t> values;
    std::vector<uint32_t> expectedIndices;
  };

  const std::vector<int32_t> prebuiltAlphabet{7, 11, 13, 17};
  auto resolver = std::make_shared<TestSharedDictionaryResolver>(
      SharedDictionaryScope::File,
      /*dictionaryId=*/23,
      prebuiltAlphabet,
      pool_.get());
  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::File,
          /*dictionaryId=*/23,
          resolver,
          {EncodingType::FixedBitWidth},
          /*usesPrebuiltAlphabet=*/true)};
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
    const auto encoded = writer.encodeValues(
        chunk.stripeIndex, buffer, streamView(descriptor_, chunk.values));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::File,
            /*dictionaryId=*/23,
            chunk.values.size(),
            pool_.get()),
        chunk.expectedIndices);
  }

  const auto alphabet = writer.encodeAlphabet(buffer);
  ASSERT_TRUE(alphabet.has_value());
  ASSERT_EQ(alphabet->content.size(), 1);
  EXPECT_EQ(
      EncodingPrefix::encodingType(alphabet->content.front()),
      EncodingType::FixedBitWidth);
  expectAlphabetEntries(*alphabet, prebuiltAlphabet, pool_.get());
  NIMBLE_ASSERT_THROW(
      writer.encodeAlphabet(buffer),
      "File shared dictionary 23 already finalized its alphabet.");
  NIMBLE_ASSERT_THROW(
      writer.encodeValues(
          /*stripeIndex=*/3,
          buffer,
          streamView(descriptor_, chunks.front().values)),
      "File shared dictionary 23 cannot encode values after its alphabet was "
      "finalized.");
}

TEST_F(SharedDictionaryWriterTest, prebuiltDictionaryRejectsUnknownValue) {
  struct TestParam {
    SharedDictionaryScope scope;
    bool usesPrebuiltAlphabet;

    std::string debugString() const {
      return fmt::format(
          "scope {}, usesPrebuiltAlphabet {}", scope, usesPrebuiltAlphabet);
    }
  };

  const std::vector<TestParam> testSettings = {
      {SharedDictionaryScope::File, true},
      {SharedDictionaryScope::External, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const std::vector<int32_t> prebuiltAlphabet{10, 20, 30};
    auto resolver = std::make_shared<TestSharedDictionaryResolver>(
        testData.scope,
        /*dictionaryId=*/29,
        prebuiltAlphabet,
        pool_.get());
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(
            testData.scope,
            /*dictionaryId=*/29,
            resolver,
            /*alphabetEncodingCandidates=*/{},
            testData.usesPrebuiltAlphabet)};
    Buffer buffer{*pool_};

    // A prebuilt alphabet cannot grow, so an unseen value is a writer input
    // error rather than a new dictionary entry.
    const std::vector<int32_t> values{10, 99};
    NIMBLE_ASSERT_USER_THROW(
        writer.encodeValues(
            /*stripeIndex=*/0, buffer, streamView(descriptor_, values)),
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
  auto resolver = std::make_shared<TestSharedDictionaryResolver>(
      SharedDictionaryScope::External,
      /*dictionaryId=*/29,
      externalAlphabet,
      pool_.get());
  TestSharedDictionaryWriter writer{
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::External,
          /*dictionaryId=*/29,
          resolver)};
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
    const auto encoded = writer.encodeValues(
        chunk.stripeIndex, buffer, streamView(descriptor_, chunk.values));

    EXPECT_EQ(
        sharedDictionaryIndices(
            encoded,
            SharedDictionaryScope::External,
            /*dictionaryId=*/29,
            chunk.values.size(),
            pool_.get()),
        chunk.expectedIndices);
  }
  EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());
}

TEST_F(SharedDictionaryWriterTest, prebuiltScopeRequiresResolver) {
  struct TestParam {
    SharedDictionaryScope scope;
    bool usesPrebuiltAlphabet;

    std::string debugString() const {
      return fmt::format(
          "scope {}, usesPrebuiltAlphabet {}", scope, usesPrebuiltAlphabet);
    }
  };

  const std::vector<TestParam> testSettings{
      {SharedDictionaryScope::File, true},
      {SharedDictionaryScope::External, false},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    TestSharedDictionaryWriter writer{
        pool_.get(),
        writerOptions(
            testData.scope,
            /*dictionaryId=*/29,
            nullptr,
            /*alphabetEncodingCandidates=*/{},
            testData.usesPrebuiltAlphabet)};
    Buffer buffer{*pool_};

    EXPECT_FALSE(writer.encodeAlphabet(buffer).has_value());

    const std::vector<int32_t> values{10};
    NIMBLE_ASSERT_USER_THROW(
        writer.encodeValues(
            /*stripeIndex=*/0, buffer, streamView(descriptor_, values)),
        fmt::format(
            "{} shared dictionary 29 requires a writer resolver.",
            testData.scope));
  }
}

} // namespace
} // namespace facebook::nimble

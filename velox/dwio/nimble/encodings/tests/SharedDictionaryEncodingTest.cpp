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
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/velox/SharedDictionaryWriter.h"

namespace facebook::nimble {
namespace {

class SharedDictionaryEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() final {
    rootPool_ = velox::memory::memoryManager()->addRootPool(
        "SharedDictionaryEncodingTest");
    pool_ = rootPool_->addLeafChild("SharedDictionaryEncodingTestLeaf");
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::unique_ptr<Encoding> createEncoding(std::string_view encoded) {
    return createEncoding(encoded, Encoding::Options{});
  }

  std::unique_ptr<Encoding> createEncoding(
      std::string_view encoded,
      const Encoding::Options& options) {
    return EncodingFactory{options}.create(
        *pool_, encoded, stringBufferFactory());
  }

  std::function<void*(uint32_t)> stringBufferFactory() {
    return [&](uint32_t totalLength) {
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buffer->asMutable<void>();
    };
  }

  std::vector<int32_t> materialize(std::string_view encoded) {
    return materialize(encoded, Encoding::Options{});
  }

  std::vector<int32_t> materialize(
      std::string_view encoded,
      const Encoding::Options& options) {
    auto encoding = createEncoding(encoded, options);
    Vector<int32_t> output{pool_.get(), encoding->rowCount()};
    encoding->materialize(encoding->rowCount(), output.data());
    return {output.begin(), output.end()};
  }

  std::string encodeShared(
      const std::vector<uint32_t>& indices,
      const Encoding::Options& options = {}) {
    const auto encoded = SharedDictionaryEncoding<int32_t>::encode(
        indices, createNestedPolicy, *buffer_, options);
    return std::string{encoded};
  }

  static std::vector<int32_t> sequentialAlphabet(uint32_t size) {
    std::vector<int32_t> values;
    values.reserve(size);
    for (uint32_t i{0}; i < size; ++i) {
      values.push_back(static_cast<int32_t>(i));
    }
    return values;
  }

  std::string_view dictionaryAlphabet(std::string_view encoded) const {
    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
    const uint32_t alphabetBytes = encoding::readUint32(pos);
    return {pos, alphabetBytes};
  }

  std::string_view dictionaryIndices(std::string_view encoded) const {
    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
    const uint32_t alphabetBytes = encoding::readUint32(pos);
    pos += alphabetBytes;
    return {pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};
  }

  std::string_view sharedDictionaryIndices(std::string_view encoded) const {
    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
    return {pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};
  }

  static std::unique_ptr<EncodingSelectionPolicyBase> createNestedPolicy(
      DataType dataType) {
    const auto encodingType = dataType == DataType::Uint32
        ? EncodingType::FixedBitWidth
        : EncodingType::Trivial;
    ManualEncodingSelectionPolicyFactory factory{
        {{encodingType, 1.0}}, std::nullopt};
    return factory.createPolicy(dataType);
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

template <typename T>
EncodingType selectEncoding(
    EncodingSelectionPolicy<T>& policy,
    std::span<const T> values) {
  using PhysicalType = TypeTraits<T>::physicalType;
  static_assert(sizeof(T) == sizeof(PhysicalType));
  const auto physicalValues = std::span<const PhysicalType>{
      reinterpret_cast<const PhysicalType*>(values.data()), values.size()};
  return policy
      .select(
          physicalValues,
          Statistics<PhysicalType>::create(physicalValues),
          Encoding::Options{})
      .encodingType;
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> typedPolicy(
    std::unique_ptr<EncodingSelectionPolicyBase> policy) {
  return std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(policy.release()));
}

template <typename T>
std::span<const T> valueSpan(const std::vector<T>& values) {
  return {values.data(), values.size()};
}

using TestSharedDictionaryWriter = TypedSharedDictionaryWriter<int32_t>;
using EncodingReadFactors = std::vector<std::pair<EncodingType, float>>;

EncodingReadFactors dictionaryValueEncoding() {
  return {{EncodingType::Dictionary, 1.0}};
}

EncodingReadFactors trivialPreferredValueEncoding() {
  return {{EncodingType::Trivial, 0.01}, {EncodingType::Dictionary, 1000.0}};
}

EncodingSelectionPolicyCreator testEncodingSelectionPolicyCreator(
    EncodingReadFactors valueEncodingReadFactors = dictionaryValueEncoding()) {
  return
      [valueEncodingReadFactors = std::move(valueEncodingReadFactors)](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        EncodingReadFactors encodingReadFactors;
        switch (dataType) {
          case DataType::Int32:
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
          case DataType::String:
            encodingReadFactors = {{EncodingType::Trivial, 1.0}};
            break;
        }
        ManualEncodingSelectionPolicyFactory factory{
            std::move(encodingReadFactors), std::nullopt};
        return factory.createPolicy(dataType);
      };
}

class TestDictionaryResolver final : public ExternalDictionaryResolver {
 public:
  TestDictionaryResolver(
      uint32_t dictionaryId,
      std::span<const int32_t> values,
      velox::memory::MemoryPool* pool)
      : dictionaryId_{dictionaryId},
        alphabet_{test::createSharedDictionaryAlphabet<int32_t>(
            values,
            std::span<const EncodingType>{},
            pool)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final {
    if (dictionaryId != dictionaryId_ || dataType != DataType::Int32) {
      return nullptr;
    }
    return alphabet_;
  }

  std::shared_ptr<const SharedDictionaryAlphabet> alphabet() const {
    return alphabet_;
  }

 private:
  const uint32_t dictionaryId_;
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
};

struct GeneratedEncodingBatch {
  bool nullable{};
  std::vector<int32_t> values;
  std::unique_ptr<bool[]> notNulls;
  size_t rowCount{};
  std::vector<int32_t> expected;
  std::string encoded;

  std::span<const bool> notNullsSpan() const {
    return {notNulls.get(), rowCount};
  }
};

std::vector<int32_t> sharedDictionaryValueUniverse() {
  constexpr int32_t kValueCount{256};
  std::vector<int32_t> values;
  values.reserve(kValueCount);
  for (int32_t i{0}; i < kValueCount; ++i) {
    values.push_back(i * 13 - 57);
  }
  return values;
}

int32_t randomValue(std::mt19937& rng, std::span<const int32_t> valueUniverse) {
  std::uniform_int_distribution<size_t> valueIndex{0, valueUniverse.size() - 1};
  return valueUniverse[valueIndex(rng)];
}

GeneratedEncodingBatch generateBatch(
    std::mt19937& rng,
    std::span<const int32_t> valueUniverse,
    bool requireNonNullValue) {
  GeneratedEncodingBatch batch;
  std::uniform_int_distribution<size_t> rowCount{1, 64};
  batch.rowCount = rowCount(rng);
  batch.nullable = std::uniform_int_distribution<int>{0, 2}(rng) != 0;
  batch.expected.reserve(batch.rowCount);

  if (!batch.nullable) {
    batch.values.reserve(batch.rowCount);
    for (size_t i{0}; i < batch.rowCount; ++i) {
      const auto value = randomValue(rng, valueUniverse);
      batch.values.push_back(value);
      batch.expected.push_back(value);
    }
    return batch;
  }

  batch.notNulls = std::make_unique<bool[]>(batch.rowCount);
  const bool allNull = !requireNonNullValue &&
      std::uniform_int_distribution<int>{0, 5}(rng) == 0;
  for (size_t i{0}; i < batch.rowCount; ++i) {
    const bool forcedNonNull = requireNonNullValue && i == 0;
    const bool randomlyNonNull =
        std::uniform_int_distribution<int>{0, 3}(rng) != 0;
    const bool notNull = !allNull && (forcedNonNull || randomlyNonNull);
    batch.notNulls[i] = notNull;
    if (notNull) {
      const auto value = randomValue(rng, valueUniverse);
      batch.values.push_back(value);
      batch.expected.push_back(value);
      continue;
    }
    batch.expected.push_back(0);
  }
  return batch;
}

TestSharedDictionaryWriter::Options writerOptions(
    SharedDictionaryScope scope,
    uint32_t dictionaryId,
    std::shared_ptr<const ExternalDictionaryResolver> resolver = nullptr,
    EncodingReadFactors valueEncodingReadFactors = dictionaryValueEncoding()) {
  return TestSharedDictionaryWriter::Options{
      .scope = scope,
      .dictionaryId = dictionaryId,
      .useExternalAlphabet = false,
      .alphabetEncodings = {},
      .encodingSelectionPolicyCreator = testEncodingSelectionPolicyCreator(
          std::move(valueEncodingReadFactors)),
      .encodingOptions = Encoding::Options{},
      .resolver = std::move(resolver)};
}

class SharedDictionaryWriterPolicyEncodingTest
    : public SharedDictionaryEncodingTest,
      public ::testing::WithParamInterface<SharedDictionaryScope> {
 protected:
  static constexpr uint32_t kDictionaryId{7};

  TestSharedDictionaryWriter createWriter(
      std::span<const int32_t> externalAlphabetValues,
      EncodingReadFactors valueEncodingReadFactors =
          dictionaryValueEncoding()) {
    auto resolver = GetParam() == SharedDictionaryScope::External
        ? std::make_shared<TestDictionaryResolver>(
              kDictionaryId, externalAlphabetValues, pool_.get())
        : nullptr;
    externalResolver_ = resolver;
    return TestSharedDictionaryWriter{
        pool_.get(),
        writerOptions(
            GetParam(),
            kDictionaryId,
            std::move(resolver),
            std::move(valueEncodingReadFactors))};
  }

  void encodeBatch(
      TestSharedDictionaryWriter& writer,
      size_t stripeIndex,
      GeneratedEncodingBatch& batch) {
    if (batch.nullable) {
      batch.encoded = std::string{EncodingFactory::encodeNullable<int32_t>(
          writer.createEncodingPolicy(stripeIndex),
          batch.values,
          batch.notNullsSpan(),
          *buffer_,
          Encoding::Options{})};
      return;
    }
    batch.encoded = std::string{EncodingFactory::encode<int32_t>(
        writer.createEncodingPolicy(stripeIndex),
        batch.values,
        *buffer_,
        Encoding::Options{})};
  }

  void expectRoundTrip(
      const GeneratedEncodingBatch& batch,
      const Encoding::Options& options) {
    EXPECT_EQ(materialize(batch.encoded, options), batch.expected);
  }

  Encoding::Options decodingOptions(TestSharedDictionaryWriter& writer) {
    Encoding::Options options;
    if (GetParam() == SharedDictionaryScope::External) {
      NIMBLE_CHECK_NOT_NULL(externalResolver_);
      options.sharedDictionaryAlphabet = externalResolver_->alphabet();
      return options;
    }

    const auto alphabet = writer.encodeAlphabet(*buffer_);
    NIMBLE_CHECK(alphabet.has_value(), "Expected shared dictionary alphabet.");
    auto encodedAlphabetOwner =
        std::make_shared<const std::string>(alphabet->content.front());
    const std::string_view encodedAlphabet{*encodedAlphabetOwner};
    options.sharedDictionaryAlphabet = SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());
    return options;
  }

 private:
  std::shared_ptr<TestDictionaryResolver> externalResolver_;
};

TEST(SharedDictionaryScopeTest, scopeName) {
  struct Case {
    SharedDictionaryScope scope;
    uint8_t wireValue;
    std::string_view name;
  };

  const std::vector<Case> cases{
      {SharedDictionaryScope::Stripe, 0, "Stripe"},
      {SharedDictionaryScope::File, 2, "File"},
      {SharedDictionaryScope::External, 3, "External"},
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(
        fmt::format(
            "scope={} value={}",
            static_cast<int>(testCase.scope),
            static_cast<int>(testCase.wireValue)));
    EXPECT_EQ(SharedDictionaryScopeName::toName(testCase.scope), testCase.name);
    EXPECT_EQ(
        SharedDictionaryScopeName::toSharedDictionaryScope(testCase.name),
        testCase.scope);
    EXPECT_EQ(fmt::format("{}", testCase.scope), testCase.name);
    EXPECT_EQ(toSharedDictionaryScope(testCase.wireValue), testCase.scope);
  }
}

TEST(SharedDictionaryScopeTest, scopeNameError) {
  EXPECT_FALSE(
      SharedDictionaryScopeName::tryToSharedDictionaryScope("Unknown"));

  for (const auto value : {uint8_t{1}, uint8_t{4}, uint8_t{255}}) {
    SCOPED_TRACE(fmt::format("value={}", static_cast<int>(value)));
    NIMBLE_ASSERT_THROW(
        toSharedDictionaryScope(value),
        fmt::format(
            "Unsupported shared dictionary scope {}.",
            static_cast<int>(value)));
  }
}

TEST_F(SharedDictionaryEncodingTest, alphabetBasic) {
  const std::vector<int32_t> values{10, 20, 30};
  const std::vector<EncodingType> candidateEncodings{
      EncodingType::FixedBitWidth};
  const auto encoded = SharedDictionaryAlphabet::encode<int32_t>(
      values, candidateEncodings, *buffer_);
  auto encodedAlphabetOwner = std::make_shared<const std::string>(encoded);
  const std::string_view encodedAlphabet{*encodedAlphabetOwner};
  const auto alphabet = SharedDictionaryAlphabet::create(
      encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());

  EXPECT_EQ(alphabet->dataType(), DataType::Int32);
  EXPECT_EQ(alphabet->entryCount(), values.size());
  EXPECT_EQ(alphabet->encodingType(), EncodingType::FixedBitWidth);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(1), 20);

  const std::vector<uint32_t> indices{2, 0, 1};
  std::vector<TypeTraits<int32_t>::physicalType> materialized(indices.size());
  alphabet->materialize<int32_t>(indices, materialized.data());

  const std::vector<TypeTraits<int32_t>::physicalType> expected{30, 10, 20};
  EXPECT_EQ(materialized, expected);
}

TEST_F(
    SharedDictionaryEncodingTest,
    encodeRejectsSharedDictionarySelectionWithoutIndices) {
  const std::vector<int32_t> values{10, 20, 30};

  NIMBLE_ASSERT_THROW(
      EncodingFactory::encode<int32_t>(
          std::make_unique<detail::FixedEncodingSelectionPolicy<int32_t>>(
              EncodingType::SharedDictionary),
          values,
          *buffer_,
          Encoding::Options{}),
      "SharedDictionary encoding requires writer-provided dictionary indices.");
}

TEST_P(
    SharedDictionaryWriterPolicyEncodingTest,
    factoryEncodesSelectionWithWriterPolicy) {
  const std::vector<int32_t> values{30, 10, 20, 30};
  const std::vector<int32_t> alphabetValues{30, 10, 20};
  const std::vector<uint32_t> indices{0, 1, 2, 0};
  auto writer = createWriter(alphabetValues);

  const auto encoded = EncodingFactory::encode<int32_t>(
      writer.createEncodingPolicy(/*stripeIndex=*/0),
      values,
      *buffer_,
      Encoding::Options{});

  EXPECT_EQ(
      EncodingPrefix::encodingType(encoded), EncodingType::SharedDictionary);
  EXPECT_EQ(EncodingPrefix::dataType(encoded), DataType::Int32);
  EXPECT_EQ(
      EncodingPrefix::readRowCount(encoded, /*useVarint=*/false),
      values.size());

  const auto encodedIndices = sharedDictionaryIndices(encoded);
  auto indicesEncoding = createEncoding(encodedIndices);
  EXPECT_EQ(indicesEncoding->encodingType(), EncodingType::FixedBitWidth);
  EXPECT_EQ(indicesEncoding->dataType(), DataType::Uint32);

  Vector<uint32_t> materializedIndices{pool_.get(), indices.size()};
  indicesEncoding->materialize(
      static_cast<uint32_t>(indices.size()), materializedIndices.data());
  EXPECT_EQ(
      std::vector<uint32_t>(
          materializedIndices.begin(), materializedIndices.end()),
      indices);

  EXPECT_TRUE(writer.usesDictionary());
  EXPECT_TRUE(writer.hasUsedDictionary());
  EXPECT_EQ(materialize(encoded, decodingOptions(writer)), values);
}

TEST_P(
    SharedDictionaryWriterPolicyEncodingTest,
    factoryEncodesNullableSelectionWithWriterPolicy) {
  const std::vector<int32_t> values{30, 10, 20};
  const std::vector<int32_t> alphabetValues{30, 10, 20};
  const std::array<bool, 4> notNulls{true, false, true, true};
  auto writer = createWriter(alphabetValues);

  const auto encoded = EncodingFactory::encodeNullable<int32_t>(
      writer.createEncodingPolicy(/*stripeIndex=*/0),
      values,
      notNulls,
      *buffer_,
      Encoding::Options{});

  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);
  EXPECT_EQ(EncodingPrefix::dataType(encoded), DataType::Int32);
  EXPECT_EQ(
      EncodingPrefix::readRowCount(encoded, /*useVarint=*/false),
      notNulls.size());

  auto options = decodingOptions(writer);
  auto encoding = createEncoding(encoded, options);
  const auto* nullableEncoding =
      dynamic_cast<const NullableEncoding<int32_t>*>(encoding.get());
  ASSERT_NE(nullableEncoding, nullptr);
  ASSERT_NE(nullableEncoding->nonNulls(), nullptr);
  EXPECT_EQ(
      nullableEncoding->nonNulls()->encodingType(),
      EncodingType::SharedDictionary);

  Vector<int32_t> output{pool_.get(), notNulls.size()};
  encoding->materialize(notNulls.size(), output.data());

  const std::vector<int32_t> expected{30, 0, 10, 20};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_P(
    SharedDictionaryWriterPolicyEncodingTest,
    factoryEncodesAllNullNullableWithWriterPolicy) {
  const std::array<int32_t, 1> alphabetValues{0};
  const std::array<int32_t, 0> values{};
  const std::array<bool, 3> notNulls{false, false, false};
  auto writer = createWriter(alphabetValues);

  const auto encoded = EncodingFactory::encodeNullable<int32_t>(
      writer.createEncodingPolicy(/*stripeIndex=*/0),
      values,
      notNulls,
      *buffer_,
      Encoding::Options{});

  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);
  EXPECT_FALSE(writer.usesDictionary());
  EXPECT_FALSE(writer.hasUsedDictionary());
  if (GetParam() != SharedDictionaryScope::External) {
    EXPECT_FALSE(writer.encodeAlphabet(*buffer_).has_value());
  } else {
    NIMBLE_ASSERT_THROW(
        writer.encodeAlphabet(*buffer_),
        "External shared dictionary 7 cannot encode an alphabet; its resolver "
        "owns the alphabet.");
  }

  auto encoding = createEncoding(encoded);
  const auto* nullableEncoding =
      dynamic_cast<const NullableEncoding<int32_t>*>(encoding.get());
  ASSERT_NE(nullableEncoding, nullptr);
  ASSERT_NE(nullableEncoding->nonNulls(), nullptr);
  EXPECT_NE(
      nullableEncoding->nonNulls()->encodingType(),
      EncodingType::SharedDictionary);
  EXPECT_EQ(nullableEncoding->nonNulls()->rowCount(), 0);
}

TEST_P(
    SharedDictionaryWriterPolicyEncodingTest,
    generatedBatchesAcrossStripesRoundTrip) {
  constexpr uint32_t kSeed{0x9E3779B9};
  std::mt19937 rng{
      kSeed + static_cast<uint32_t>(static_cast<uint8_t>(GetParam()))};
  const auto valueUniverse = sharedDictionaryValueUniverse();
  auto writer = createWriter(valueUniverse);
  std::vector<GeneratedEncodingBatch> pendingBatches;

  SCOPED_TRACE(
      fmt::format(
          "scope={}, seed={}",
          SharedDictionaryScopeName::toName(GetParam()),
          kSeed));

  for (size_t stripeIndex{0}; stripeIndex < 12; ++stripeIndex) {
    SCOPED_TRACE(fmt::format("stripe={}", stripeIndex));
    std::uniform_int_distribution<size_t> batchCount{1, 3};
    const auto batchesInStripe = batchCount(rng);

    for (size_t batchIndex{0}; batchIndex < batchesInStripe; ++batchIndex) {
      SCOPED_TRACE(fmt::format("batch={}", batchIndex));
      auto batch = generateBatch(
          rng,
          valueUniverse,
          /*requireNonNullValue=*/batchIndex == 0);
      encodeBatch(writer, stripeIndex, batch);
      pendingBatches.push_back(std::move(batch));
    }

    if (GetParam() == SharedDictionaryScope::Stripe) {
      const auto options = decodingOptions(writer);
      for (const auto& batch : pendingBatches) {
        expectRoundTrip(batch, options);
      }
      pendingBatches.clear();
    }
  }

  if (GetParam() != SharedDictionaryScope::Stripe) {
    const auto options = decodingOptions(writer);
    for (const auto& batch : pendingBatches) {
      expectRoundTrip(batch, options);
    }
  }
}

TEST_F(
    SharedDictionaryEncodingTest,
    factoryEncodesRegularNullableWhenStripePolicyDoesNotSelectDictionary) {
  auto writer = TestSharedDictionaryWriter{
      pool_.get(),
      writerOptions(
          SharedDictionaryScope::Stripe,
          /*dictionaryId=*/7,
          nullptr,
          trivialPreferredValueEncoding())};
  const std::vector<int32_t> values{1, 2, 3, 4, 5};
  const std::array<bool, 6> notNulls{true, true, false, true, true, true};

  const auto encoded = EncodingFactory::encodeNullable<int32_t>(
      writer.createEncodingPolicy(/*stripeIndex=*/0),
      values,
      notNulls,
      *buffer_,
      Encoding::Options{});

  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);
  EXPECT_FALSE(writer.usesDictionary());
  EXPECT_FALSE(writer.hasUsedDictionary());
  EXPECT_FALSE(writer.encodeAlphabet(*buffer_).has_value());

  auto encoding = createEncoding(encoded);
  const auto* nullableEncoding =
      dynamic_cast<const NullableEncoding<int32_t>*>(encoding.get());
  ASSERT_NE(nullableEncoding, nullptr);
  ASSERT_NE(nullableEncoding->nonNulls(), nullptr);
  EXPECT_NE(
      nullableEncoding->nonNulls()->encodingType(),
      EncodingType::SharedDictionary);

  Vector<int32_t> output{pool_.get(), notNulls.size()};
  encoding->materialize(notNulls.size(), output.data());
  const std::vector<int32_t> expected{1, 2, 0, 3, 4, 5};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

INSTANTIATE_TEST_SUITE_P(
    Scopes,
    SharedDictionaryWriterPolicyEncodingTest,
    ::testing::Values(
        SharedDictionaryScope::Stripe,
        SharedDictionaryScope::File,
        SharedDictionaryScope::External),
    [](const ::testing::TestParamInfo<SharedDictionaryScope>& testInfo) {
      return std::string{SharedDictionaryScopeName::toName(testInfo.param)};
    });

TEST_F(SharedDictionaryEncodingTest, alphabetOwnerKeepsViewBytesAlive) {
  std::shared_ptr<const SharedDictionaryAlphabet> alphabet;
  {
    Buffer buffer{*pool_};
    const std::vector<int32_t> values{10, 20, 30};
    const auto encoded = SharedDictionaryAlphabet::encode<int32_t>(
        values, std::array{EncodingType::FixedBitWidth}, buffer);
    auto encodedAlphabetOwner = std::make_shared<const std::string>(encoded);
    const std::string_view encodedAlphabet{*encodedAlphabetOwner};
    alphabet = SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());
  }

  ASSERT_NE(alphabet, nullptr);
  const auto poolBytesAfterCreate = pool_->usedBytes();
  EXPECT_EQ(alphabet->entryCount(), 3);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(2), 30);
  EXPECT_EQ(pool_->usedBytes(), poolBytesAfterCreate);
}

TEST_F(SharedDictionaryEncodingTest, retainsEncodedAlphabetOwner) {
  const std::vector<int32_t> values{10, 20, 30};
  auto encodedAlphabetOwner = std::make_shared<const std::string>(
      SharedDictionaryAlphabet::encode<int32_t>(
          values, std::array{EncodingType::FixedBitWidth}, *buffer_));
  const auto expectedEncodedAlphabet = *encodedAlphabetOwner;
  std::weak_ptr<const std::string> weakOwner = encodedAlphabetOwner;

  auto alphabet = SharedDictionaryAlphabet::create(
      *encodedAlphabetOwner, encodedAlphabetOwner, pool_.get());
  encodedAlphabetOwner.reset();

  EXPECT_FALSE(weakOwner.expired());
  EXPECT_EQ(std::string{alphabet->encodedAlphabet()}, expectedEncodedAlphabet);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(1), 20);
  alphabet.reset();
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(SharedDictionaryEncodingTest, alphabetSelectsEncodingFromCandidates) {
  const std::vector<int32_t> values{10, 20, 30};

  // An empty candidate list falls back to default encoding selection, which
  // still round-trips every entry.
  const auto defaultAlphabet = test::createSharedDictionaryAlphabet<int32_t>(
      values, /*candidateEncodings=*/{}, pool_.get());
  EXPECT_EQ(defaultAlphabet->entryCount(), values.size());
  EXPECT_EQ(defaultAlphabet->physicalValueAt<int32_t>(2), 30);

  // Several candidates are weighed against each other, so the winner has to be
  // one of them.
  const std::vector<EncodingType> candidateEncodings{
      EncodingType::Trivial, EncodingType::FixedBitWidth};
  const auto alphabet = test::createSharedDictionaryAlphabet<int32_t>(
      values, candidateEncodings, pool_.get());
  EXPECT_NE(
      std::find(
          candidateEncodings.begin(),
          candidateEncodings.end(),
          alphabet->encodingType()),
      candidateEncodings.end());
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(0), 10);
}

TEST_F(
    SharedDictionaryEncodingTest,
    preservedDictionaryEncodingSelectionPolicyPinsNestedEncodings) {
  detail::PreservedDictionaryEncodingSelectionPolicy<int32_t> policy{
      EncodingType::Varint, EncodingType::FixedBitWidth};

  const std::vector<int32_t> alphabetValues{10, 20, 30};
  EXPECT_EQ(
      selectEncoding(policy, valueSpan(alphabetValues)),
      EncodingType::Dictionary);

  auto alphabetPolicy = typedPolicy<int32_t>(policy.create<int32_t>(
      EncodingType::Dictionary, EncodingIdentifiers::Dictionary::Alphabet));
  EXPECT_EQ(
      selectEncoding(*alphabetPolicy, valueSpan(alphabetValues)),
      EncodingType::Varint);

  const std::vector<uint32_t> indices{0, 1, 0, 2};
  auto indicesPolicy = typedPolicy<uint32_t>(policy.create<uint32_t>(
      EncodingType::Dictionary, EncodingIdentifiers::Dictionary::Indices));
  EXPECT_EQ(
      selectEncoding(*indicesPolicy, valueSpan(indices)),
      EncodingType::FixedBitWidth);

  detail::PreservedDictionaryEncodingSelectionPolicy<int32_t> unpinnedPolicy;
  for (const auto nestedEncodingIdentifier :
       {EncodingIdentifiers::Dictionary::Alphabet,
        EncodingIdentifiers::Dictionary::Indices}) {
    SCOPED_TRACE(fmt::format("nested={}", nestedEncodingIdentifier));
    auto fallbackPolicy = unpinnedPolicy.create<uint32_t>(
        EncodingType::Dictionary, nestedEncodingIdentifier);
    const auto* manualPolicy =
        dynamic_cast<ManualEncodingSelectionPolicy<uint32_t>*>(
            fallbackPolicy.get());
    ASSERT_NE(manualPolicy, nullptr);

    const auto& readFactors = manualPolicy->candidateEncodingReadFactors();
    const auto hasEncoding = [&](EncodingType encodingType) {
      return std::find_if(
                 readFactors.begin(),
                 readFactors.end(),
                 [encodingType](const auto& entry) {
                   return entry.first == encodingType;
                 }) != readFactors.end();
    };
    EXPECT_FALSE(hasEncoding(EncodingType::Dictionary));
    EXPECT_TRUE(hasEncoding(EncodingType::Trivial));
  }
}

TEST_F(SharedDictionaryEncodingTest, alphabetEstimateUsesSelectionEstimate) {
  const std::vector<int32_t> values{10, 20, 30};
  using PhysicalType = TypeTraits<int32_t>::physicalType;
  const auto physicalValues = std::span<const PhysicalType>{
      reinterpret_cast<const PhysicalType*>(values.data()), values.size()};
  const auto statistics = Statistics<PhysicalType>::create(physicalValues);
  const Encoding::Options options;

  const auto estimate = SharedDictionaryAlphabet::estimateSize<int32_t>(
      values, std::array{EncodingType::FixedBitWidth}, options);
  const auto expectedEstimate =
      detail::EncodingSizeEstimation<int32_t>::estimateSize(
          EncodingType::FixedBitWidth, physicalValues, statistics, options);

  ASSERT_TRUE(expectedEstimate.has_value());
  EXPECT_EQ(estimate, expectedEstimate.value());
}

TEST_F(
    SharedDictionaryEncodingTest,
    alphabetEstimateUsesSizeModelWhenSelectionHasNoEstimate) {
  const std::vector<int32_t> values{10, 20, 30};
  using PhysicalType = TypeTraits<int32_t>::physicalType;
  const auto physicalValues = std::span<const PhysicalType>{
      reinterpret_cast<const PhysicalType*>(values.data()), values.size()};
  const auto statistics = Statistics<PhysicalType>::create(physicalValues);
  const Encoding::Options options;

  const auto estimate = SharedDictionaryAlphabet::estimateSize<int32_t>(
      values, std::array{EncodingType::ALP}, options);
  const auto expectedEstimate =
      detail::EncodingSizeEstimation<int32_t>::estimateSize(
          EncodingType::Trivial, physicalValues, statistics, options);

  ASSERT_TRUE(expectedEstimate.has_value());
  EXPECT_EQ(estimate, expectedEstimate.value());
}

TEST_F(SharedDictionaryEncodingTest, alphabetRoundTripsWithAndWithoutView) {
  struct TestCase {
    std::string testName;
    EncodingType encodingType;
  };

  const std::vector<TestCase> testCases{
      {"view", EncodingType::FixedBitWidth},
      {"decoded entries", EncodingType::Varint},
  };

  const std::vector<int32_t> values{10, 20, 30};
  const std::vector<uint32_t> indices{2, 0, 1};
  const std::vector<TypeTraits<int32_t>::physicalType> expected{30, 10, 20};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.testName);
    const auto alphabet = test::createSharedDictionaryAlphabet<int32_t>(
        values, std::array{testCase.encodingType}, pool_.get());
    EXPECT_EQ(alphabet->encodingType(), testCase.encodingType);
    EXPECT_EQ(alphabet->entryCount(), values.size());
    EXPECT_EQ(alphabet->physicalValueAt<int32_t>(1), 20);

    std::vector<TypeTraits<int32_t>::physicalType> materialized(indices.size());
    alphabet->materialize<int32_t>(indices, materialized.data());
    EXPECT_EQ(materialized, expected);

    Vector<int32_t> materializedAll{pool_.get()};
    alphabet->materializeAll<int32_t>(materializedAll);
    const std::vector<int32_t> expectedAll{10, 20, 30};
    EXPECT_EQ(
        std::vector<int32_t>(materializedAll.begin(), materializedAll.end()),
        expectedAll);
  }
}

DEBUG_ONLY_TEST_F(
    SharedDictionaryEncodingTest,
    alphabetRejectsOutOfRangeBatchIndicesWithAndWithoutView) {
  struct TestCase {
    std::string testName;
    EncodingType encodingType;
  };

  const std::vector<TestCase> testCases{
      {"view", EncodingType::FixedBitWidth},
      {"decoded entries", EncodingType::Varint},
  };
  const std::vector<int32_t> values{10, 20, 30};
  const std::vector<uint32_t> indices{0, static_cast<uint32_t>(values.size())};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.testName);
    const auto alphabet = test::createSharedDictionaryAlphabet<int32_t>(
        values, std::array{testCase.encodingType}, pool_.get());
    std::vector<TypeTraits<int32_t>::physicalType> output(indices.size());

    NIMBLE_ASSERT_THROW(
        alphabet->materialize<int32_t>(indices, output.data()),
        "Shared dictionary index exceeds alphabet size.");
  }
}

TEST_F(
    SharedDictionaryEncodingTest,
    alphabetMaterializesBlockBitPackingIndexedRuns) {
  const auto values = sequentialAlphabet(/*size=*/2'051);
  const auto alphabet = test::createSharedDictionaryAlphabet<int32_t>(
      values, std::array{EncodingType::BlockBitPacking}, pool_.get());
  EXPECT_EQ(alphabet->encodingType(), EncodingType::BlockBitPacking);

  const std::vector<uint32_t> indices{
      3, 3, 4, 5, 1'023, 1'024, 1'025, 2'050, 1};
  std::vector<TypeTraits<int32_t>::physicalType> materialized(indices.size());
  alphabet->materialize<int32_t>(indices, materialized.data());

  const std::vector<TypeTraits<int32_t>::physicalType> expected{
      3, 3, 4, 5, 1'023, 1'024, 1'025, 2'050, 1};
  EXPECT_EQ(materialized, expected);
}

TEST_F(SharedDictionaryEncodingTest, materializeAllIntegerTypes) {
  auto verify = [this]<typename T>() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    std::vector<T> values;
    if constexpr (std::is_signed_v<T>) {
      values = {
          std::numeric_limits<T>::lowest(),
          static_cast<T>(-1),
          std::numeric_limits<T>::max()};
    } else {
      values = {0, 1, std::numeric_limits<T>::max()};
    }
    const std::string encoded{SharedDictionaryAlphabet::encode<T>(
        values, std::array{EncodingType::Trivial}, *buffer_)};
    auto encodedAlphabetOwner = std::make_shared<const std::string>(encoded);
    const std::string_view encodedAlphabet{*encodedAlphabetOwner};
    const auto alphabet = SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(encodedAlphabetOwner), pool_.get());

    Vector<T> materialized{pool_.get()};
    alphabet->materializeAll<T>(materialized);
    EXPECT_EQ(std::vector<T>(materialized.begin(), materialized.end()), values);
  };

  verify.operator()<int8_t>();
  verify.operator()<uint8_t>();
  verify.operator()<int16_t>();
  verify.operator()<uint16_t>();
  verify.operator()<int32_t>();
  verify.operator()<uint32_t>();
  verify.operator()<int64_t>();
  verify.operator()<uint64_t>();
}

TEST_F(SharedDictionaryEncodingTest, publicApiMaterializesAndSkipsRows) {
  const std::vector<uint32_t> indices{2, 0, 3, 1, 2};
  const auto encoded = test::encodeSharedDictionary(*buffer_, indices);

  const std::vector<int32_t> alphabetValues{10, 20, 30, 40};
  Encoding::Options options;
  options.sharedDictionaryAlphabet =
      test::createSharedDictionaryAlphabet<int32_t>(
          alphabetValues, std::span<const EncodingType>{}, pool_.get());

  auto encoding = createEncoding(encoded, options);
  const auto rowCount = static_cast<uint32_t>(indices.size());
  EXPECT_EQ(encoding->encodingType(), EncodingType::SharedDictionary);
  EXPECT_EQ(encoding->dataType(), DataType::Int32);
  EXPECT_EQ(encoding->rowCount(), rowCount);
  EXPECT_EQ(encoding->dictionarySize(), 4);

  Vector<int32_t> values{pool_.get(), rowCount};
  encoding->materialize(rowCount, values.data());
  const std::vector<int32_t> expected{30, 10, 40, 20, 30};
  EXPECT_EQ(std::vector<int32_t>(values.begin(), values.end()), expected);

  encoding->reset();
  encoding->skip(2);
  Vector<int32_t> suffix{pool_.get(), 2};
  encoding->materialize(2, suffix.data());
  const std::vector<int32_t> expectedSuffix{40, 20};
  EXPECT_EQ(std::vector<int32_t>(suffix.begin(), suffix.end()), expectedSuffix);
}

TEST_F(SharedDictionaryEncodingTest, encodingRejectsInvalidAlphabet) {
  const auto encoded = test::encodeSharedDictionary(*buffer_, {0});

  {
    SCOPED_TRACE("missing alphabet");
    NIMBLE_ASSERT_THROW(
        createEncoding(encoded),
        "Shared dictionary encoding requires an alphabet.");
  }

  {
    SCOPED_TRACE("alphabet type mismatch");
    const std::vector<int64_t> alphabetValues{10, 20};
    Encoding::Options options;
    options.sharedDictionaryAlphabet =
        test::createSharedDictionaryAlphabet<int64_t>(
            alphabetValues, std::span<const EncodingType>{}, pool_.get());

    NIMBLE_ASSERT_THROW(
        createEncoding(encoded, options),
        "Shared dictionary alphabet has unexpected type.");
  }
}

TEST_F(SharedDictionaryEncodingTest, sliceConvertsToLocalDictionary) {
  struct Scenario {
    std::string_view name;
    std::vector<uint32_t> sourceIndices;
    uint32_t offset;
    uint32_t length;
    std::vector<int32_t> alphabetValues;
    // Encodings the shared alphabet may use. The sliced local dictionary must
    // reuse the encoding selected for the source alphabet.
    std::vector<EncodingType> alphabetEncodingCandidates;
    // Shared alphabet entries the slice is expected to pull into its local
    // dictionary, in local index order.
    std::vector<uint32_t> expectedLocalAlphabetIndices;
    std::vector<int32_t> expected;
  };

  const std::vector<Scenario> scenarios{
      {
          "denseSharedIndexRange",
          {2, 1, 2, 3, 1, 4},
          /*offset=*/0,
          /*length=*/6,
          {100, 200, 300, 400, 500, 600},
          {EncodingType::Trivial},
          {1, 2, 3, 4},
          {300, 200, 300, 400, 200, 500},
      },
      {
          "fixedBitWidthAlphabet",
          {0, 1, 0, 2, 3, 1},
          /*offset=*/0,
          /*length=*/4,
          {100, 200, 300, 400, 500, 600},
          {EncodingType::FixedBitWidth},
          {0, 1, 2},
          {100, 200, 100, 300},
      },
      {
          "defaultSelectedAlphabetEncoding",
          {0, 1, 0, 2, 3, 1},
          /*offset=*/0,
          /*length=*/4,
          {100, 200, 300, 400, 500, 600},
          {},
          {0, 1, 2},
          {100, 200, 100, 300},
      },
      {
          "sparseSharedIndexRange",
          {1000, 1, 1000, 500},
          /*offset=*/0,
          /*length=*/4,
          sequentialAlphabet(/*size=*/1001),
          {EncodingType::Trivial},
          {1, 500, 1000},
          {1000, 1, 1000, 500},
      },
  };

  for (const auto& testCase : scenarios) {
    SCOPED_TRACE(testCase.name);
    const auto encoded =
        test::encodeSharedDictionary(*buffer_, testCase.sourceIndices);

    Encoding::Options options;
    options.sharedDictionaryAlphabet =
        test::createSharedDictionaryAlphabet<int32_t>(
            testCase.alphabetValues,
            testCase.alphabetEncodingCandidates,
            pool_.get());

    const auto sliced = EncodingFactory::slice(
        encoded, testCase.offset, testCase.length, *buffer_, options);
    EXPECT_NE(sliced.data(), encoded.data());
    EXPECT_EQ(EncodingPrefix::encodingType(sliced), EncodingType::Dictionary);
    EXPECT_EQ(EncodingPrefix::dataType(sliced), DataType::Int32);
    EXPECT_EQ(
        EncodingPrefix::readRowCount(sliced, /*useVarint=*/false),
        testCase.length);
    EXPECT_EQ(materialize(sliced), testCase.expected);

    const auto alphabet = dictionaryAlphabet(sliced);
    const auto localIndices = dictionaryIndices(sliced);
    std::vector<int32_t> expectedLocalAlphabet;
    expectedLocalAlphabet.reserve(testCase.expectedLocalAlphabetIndices.size());
    for (const auto index : testCase.expectedLocalAlphabetIndices) {
      expectedLocalAlphabet.push_back(testCase.alphabetValues[index]);
    }
    EXPECT_EQ(materialize(alphabet), expectedLocalAlphabet);
    EXPECT_EQ(
        EncodingPrefix::readRowCount(alphabet, /*useVarint=*/false),
        static_cast<uint32_t>(expectedLocalAlphabet.size()));
    // The slice reuses whatever encoding the shared alphabet was stored with.
    EXPECT_EQ(
        EncodingPrefix::encodingType(alphabet),
        options.sharedDictionaryAlphabet->encodingType());
    EXPECT_EQ(
        EncodingPrefix::encodingType(localIndices),
        EncodingType::FixedBitWidth);
  }
}

TEST_F(SharedDictionaryEncodingTest, sliceConvertsToConstantEncoding) {
  const std::vector<uint32_t> indices{0, 0, 0, 1};
  const auto encoded = test::encodeSharedDictionary(*buffer_, indices);

  const std::vector<int32_t> alphabetValues{100, 200, 300};
  Encoding::Options options;
  options.sharedDictionaryAlphabet =
      test::createSharedDictionaryAlphabet<int32_t>(
          alphabetValues, std::span<const EncodingType>{}, pool_.get());

  const auto sliced = EncodingFactory::slice(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, options);
  EXPECT_EQ(EncodingPrefix::encodingType(sliced), EncodingType::Constant);
  EXPECT_EQ(EncodingPrefix::readRowCount(sliced, /*useVarint=*/false), 3);

  const std::vector<int32_t> expected{100, 100, 100};
  EXPECT_EQ(materialize(sliced), expected);
}

TEST_F(SharedDictionaryEncodingTest, capturesIndicesLayout) {
  const std::vector<uint32_t> indices{0, 1};
  const auto encoded = encodeShared(indices);

  const auto layout =
      EncodingLayoutCapture::capture(encoded, Encoding::Options{});

  EXPECT_EQ(layout.encodingType(), EncodingType::SharedDictionary);
  EXPECT_EQ(layout.childrenCount(), 1);
}

} // namespace
} // namespace facebook::nimble

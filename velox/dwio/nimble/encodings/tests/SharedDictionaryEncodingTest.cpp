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
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"

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
    auto encoding = createEncoding(encoded);
    Vector<int32_t> output{pool_.get(), encoding->rowCount()};
    encoding->materialize(encoding->rowCount(), output.data());
    return {output.begin(), output.end()};
  }

  std::string encodeShared(const std::vector<uint32_t>& indices) {
    const auto encoded = SharedDictionaryEncoding<int32_t>::encode(
        indices, createNestedPolicy, *buffer_, Encoding::Options{});
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
  std::weak_ptr<const std::string> weakOwner = encodedAlphabetOwner;

  auto alphabet = SharedDictionaryAlphabet::create(
      *encodedAlphabetOwner, encodedAlphabetOwner, pool_.get());
  encodedAlphabetOwner.reset();

  EXPECT_FALSE(weakOwner.expired());
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

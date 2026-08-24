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
#define NIMBLE_ENCODING_SELECTION_DEBUG

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

using namespace facebook;

namespace {
constexpr uint32_t commonPrefixSize = 6;

nimble::Encoding::Options bitWidthOptions(bool fixedBitWidthUseExactBits) {
  nimble::Encoding::Options options;
  options.fixedBitWidthUseExactBits = fixedBitWidthUseExactBits;
  return options;
}

nimble::Encoding::Options exactBitWidthOptions() {
  return bitWidthOptions(true);
}

template <typename T>
std::unique_ptr<nimble::ManualEncodingSelectionPolicy<T>>
getRootManualSelectionPolicy() {
  return std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Constant, 1.0},
          {nimble::EncodingType::Trivial, 0.7},
          {nimble::EncodingType::FixedBitWidth, 1.05},
          {nimble::EncodingType::MainlyConstant, 1.05},
          {nimble::EncodingType::SparseBool, 1.05},
          {nimble::EncodingType::Dictionary, 1.05},
          {nimble::EncodingType::RLE, 1.05},
          {nimble::EncodingType::Varint, 1.1},
      },
      nimble::CompressionOptions{
          .compressionAcceptRatio = 0.9,
          .internalCompressionLevel = 9,
          .internalDecompressionLevel = 2,
          .useVariableBitWidthCompressor = false,
      },
      std::nullopt);
}
} // namespace

struct EncodingDetails {
  nimble::EncodingType encodingType;
  nimble::DataType dataType;
  uint32_t level;
  std::string nestedEncodingName;
};

void verifyEncodingTree(
    std::string_view stream,
    std::vector<EncodingDetails> expected) {
  ASSERT_GT(expected.size(), 0);
  std::vector<EncodingDetails> actual;
  nimble::tools::traverseEncodings(
      stream,
      [&](auto encodingType,
          auto dataType,
          auto level,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.push_back(
            {.encodingType = encodingType,
             .dataType = dataType,
             .level = level,
             .nestedEncodingName = nestedEncodingName});

        return true;
      });

  ASSERT_EQ(expected.size(), actual.size());
  for (auto i = 0; i < expected.size(); ++i) {
    LOG(INFO) << "Expected: " << expected[i].encodingType << "<"
              << expected[i].dataType << ">[" << expected[i].nestedEncodingName
              << ":" << expected[i].level << "]";
    LOG(INFO) << "Actual: " << actual[i].encodingType << "<"
              << actual[i].dataType << ">[" << actual[i].nestedEncodingName
              << ":" << actual[i].level << "]";
    EXPECT_EQ(expected[i].encodingType, actual[i].encodingType);
    EXPECT_EQ(expected[i].dataType, actual[i].dataType);
    EXPECT_EQ(expected[i].level, actual[i].level);
    EXPECT_EQ(expected[i].nestedEncodingName, actual[i].nestedEncodingName);
  }
}

template <typename T>
typename nimble::TypeTraits<T>::physicalType asPhysicalType(T value) {
  return *reinterpret_cast<typename nimble::TypeTraits<T>::physicalType*>(
      &value);
}

template <typename T>
void verifySizeEstimate(
    std::span<const T> values,
    nimble::Buffer& buffer,
    size_t expectedEstimatedSize,
    size_t expectedActualSize,
    nimble::EncodingType encodingTypeForEstimation,
    const std::vector<std::pair<nimble::EncodingType, float>>& readFactors) {
  // Create a policy that uses no compression
  auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
      readFactors,
      nimble::CompressionOptions{
          .compressionAcceptRatio = 0.0,
      },
      std::nullopt);

  const nimble::Encoding::Options options;

  // Check the serialized uncompressed size
  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, options);
  auto actualSize = serialized.size();
  EXPECT_EQ(actualSize, expectedActualSize);

  // Check the estimated size
  auto estimatedSize = nimble::detail::EncodingSizeEstimation<T>::estimateSize(
      encodingTypeForEstimation,
      values,
      nimble::Statistics<T>::create(values),
      options);
  EXPECT_EQ(estimatedSize, expectedEstimatedSize);
}

template <typename T>
void verifySelectedEstimate(
    std::optional<facebook::nimble::CompressionOptions> compressionOptions) {
  const std::vector<T> values{1, 2, 3, 4, 5, 6};
  const auto valueSpan = std::span<const T>{values.data(), values.size()};
  const auto options = exactBitWidthOptions();
  const auto statistics = nimble::Statistics<T>::create(valueSpan);
  nimble::ManualEncodingSelectionPolicy<T> policy{
      {
          {nimble::EncodingType::FixedBitWidth, 1.0},
          {nimble::EncodingType::Trivial, 100.0},
      },
      std::move(compressionOptions),
      std::nullopt};

  const auto selection = policy.select(valueSpan, statistics, options);

  EXPECT_EQ(selection.encodingType, nimble::EncodingType::FixedBitWidth);
  const auto expectedEstimate =
      nimble::detail::EncodingSizeEstimation<T>::estimateSize(
          selection.encodingType, valueSpan, statistics, options);
  ASSERT_TRUE(expectedEstimate.has_value());
  EXPECT_EQ(selection.estimatedSize, expectedEstimate);
}

TEST(EncodingSelectionTest, manualSelectionReturnsSelectedEstimate) {
  verifySelectedEstimate<uint32_t>(std::nullopt);
  verifySelectedEstimate<uint32_t>(nimble::CompressionOptions{
      .compressionAcceptRatio = 0.0,
  });
}

TEST(EncodingSelectionTest, manualSelectionCanReturnFallbackWithoutEstimate) {
  const std::vector<uint32_t> values{1, 2, 3};
  const auto valueSpan =
      std::span<const uint32_t>{values.data(), values.size()};
  const auto statistics = nimble::Statistics<uint32_t>::create(valueSpan);
  nimble::ManualEncodingSelectionPolicy<uint32_t> policy{
      {{nimble::EncodingType::ALP, 1.0}}, std::nullopt, std::nullopt};

  const auto selection = policy.select(
      valueSpan, statistics, /*options=*/nimble::Encoding::Options{});

  EXPECT_EQ(selection.encodingType, nimble::EncodingType::Trivial);
  EXPECT_FALSE(selection.estimatedSize.has_value());
}

template <typename T>
void test(std::span<const T> values, std::vector<EncodingDetails> expected) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto policy = getRootManualSelectionPolicy<T>();
  nimble::Buffer buffer{*pool};
  const auto options = exactBitWidthOptions();

  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, options);

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  auto expectedSize = values.size_bytes();
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();

  ASSERT_GT(expected.size(), 0);
  std::vector<EncodingDetails> actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto level,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.push_back(
            {.encodingType = encodingType,
             .dataType = dataType,
             .level = level,
             .nestedEncodingName = nestedEncodingName});

        return true;
      });

  ASSERT_EQ(expected.size(), actual.size());
  for (auto i = 0; i < expected.size(); ++i) {
    LOG(INFO) << "Expected: " << expected[i].encodingType << "<"
              << expected[i].dataType << ">[" << expected[i].nestedEncodingName
              << ":" << expected[i].level << "]";
    LOG(INFO) << "Actual: " << actual[i].encodingType << "<"
              << actual[i].dataType << ">[" << actual[i].nestedEncodingName
              << ":" << actual[i].level << "]";
    EXPECT_EQ(expected[i].encodingType, actual[i].encodingType);
    EXPECT_EQ(expected[i].dataType, actual[i].dataType);
    EXPECT_EQ(expected[i].level, actual[i].level);
    EXPECT_EQ(expected[i].nestedEncodingName, actual[i].nestedEncodingName);
  }

  nimble::Vector<T> materialized{pool.get()};
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& stringBuffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, pool.get()));
    return stringBuffer->asMutable<void>();
  };
  auto encoding = nimble::EncodingFactory().create(
      *pool, serialized, stringBufferFactory, options);
  nimble::Vector<T> result{pool.get()};
  result.resize(values.size());
  encoding->materialize(values.size(), result.data());

  for (auto i = 0; i < values.size(); ++i) {
    ASSERT_EQ(asPhysicalType(values[i]), asPhysicalType(result[i])) << i;
  }
}

struct FixedBitWidthSelectionParam {
  bool fixedBitWidthUseExactBits;
  nimble::EncodingType expectedEncodingType;
};

class FixedBitWidthSelectionTest
    : public ::testing::TestWithParam<FixedBitWidthSelectionParam> {};

TEST_P(FixedBitWidthSelectionTest, selectsExpectedEncodingForNarrowValues) {
  using T = uint8_t;

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto policy = getRootManualSelectionPolicy<T>();
  nimble::Buffer buffer{*pool};
  std::vector<T> values;
  values.reserve(10000);
  for (uint32_t i = 0; i < 10000; ++i) {
    values.push_back(i % 32);
  }

  const auto options = bitWidthOptions(GetParam().fixedBitWidthUseExactBits);
  const auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, options);

  verifyEncodingTree(
      serialized,
      {
          {.encodingType = GetParam().expectedEncodingType,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
      });
}

INSTANTIATE_TEST_SUITE_P(
    fixedBitWidthUseExactBits,
    FixedBitWidthSelectionTest,
    ::testing::Values(
        FixedBitWidthSelectionParam{
            .fixedBitWidthUseExactBits = false,
            .expectedEncodingType = nimble::EncodingType::Trivial,
        },
        FixedBitWidthSelectionParam{
            .fixedBitWidthUseExactBits = true,
            .expectedEncodingType = nimble::EncodingType::FixedBitWidth,
        }),
    [](const ::testing::TestParamInfo<FixedBitWidthSelectionParam>& info) {
      return info.param.fixedBitWidthUseExactBits ? "Exact" : "ByteRounded";
    });

using NumericTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double>;

TYPED_TEST_CASE(EncodingSelectionNumericTests, NumericTypes);

template <typename C>
class EncodingSelectionNumericTests : public ::testing::Test {};

TEST(ManualEncodingSelectionPolicyFactoryTest, nestedReadFactorsInheritRoot) {
  nimble::ManualEncodingSelectionPolicyFactory factory{
      {{nimble::EncodingType::Trivial, 1.0},
       {nimble::EncodingType::FixedBitWidth, 1.0}},
      /*compressionOptions=*/std::nullopt};

  auto rootBase = factory.createPolicy(nimble::DataType::Uint32);
  auto* root = dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint32_t>*>(
      rootBase.get());
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(
      root->candidateEncodingReadFactors(),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Trivial, 1.0},
          {nimble::EncodingType::FixedBitWidth, 1.0}}));

  auto nestedBase = rootBase->create<uint32_t>(
      nimble::EncodingType::Trivial,
      nimble::EncodingIdentifiers::Dictionary::Indices);
  auto* nested = dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint32_t>*>(
      nestedBase.get());
  ASSERT_NE(nested, nullptr);
  EXPECT_EQ(
      nested->candidateEncodingReadFactors(),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::FixedBitWidth, 1.0}}));
}

TEST(ManualEncodingSelectionPolicyFactoryTest, nestedReadFactorsOverrideRoot) {
  nimble::ManualEncodingSelectionPolicyFactory factory{
      {{nimble::EncodingType::Trivial, 1.0}},
      /*compressionOptions=*/std::nullopt,
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::FixedBitWidth, 1.0},
          {nimble::EncodingType::Varint, 1.0}}};

  auto rootBase = factory.createPolicy(nimble::DataType::Uint32);
  auto* root = dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint32_t>*>(
      rootBase.get());
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(
      root->candidateEncodingReadFactors(),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Trivial, 1.0}}));

  auto nestedBase = rootBase->create<uint32_t>(
      nimble::EncodingType::Trivial,
      nimble::EncodingIdentifiers::Dictionary::Indices);
  auto* nested = dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint32_t>*>(
      nestedBase.get());
  ASSERT_NE(nested, nullptr);
  EXPECT_EQ(
      nested->candidateEncodingReadFactors(),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::FixedBitWidth, 1.0},
          {nimble::EncodingType::Varint, 1.0}}));

  auto deeperNestedBase = nestedBase->create<uint32_t>(
      nimble::EncodingType::FixedBitWidth,
      nimble::EncodingIdentifiers::Dictionary::Indices);
  auto* deeperNested =
      dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint32_t>*>(
          deeperNestedBase.get());
  ASSERT_NE(deeperNested, nullptr);
  EXPECT_EQ(
      deeperNested->candidateEncodingReadFactors(),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Varint, 1.0}}));
}

TYPED_TEST(EncodingSelectionNumericTests, selectConst) {
  using T = TypeParam;

  for (const T value :
       {std::numeric_limits<T>::min(),
        static_cast<T>(0),
        std::numeric_limits<T>::max()}) {
    std::vector<T> values;
    values.resize(1000);
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }

    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::Constant,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
        });
  }
}

TYPED_TEST(EncodingSelectionNumericTests, selectMainlyConst) {
  using T = TypeParam;

  for (const T value : {
           std::numeric_limits<T>::min(),
           static_cast<T>(0),
           std::numeric_limits<T>::max(),
       }) {
    LOG(INFO) << "Verifying type " << folly::demangle(typeid(T))
              << " with data: " << value;

    std::vector<T> values;
    values.resize(1000);
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }

    for (auto i = 0; i < values.size(); i += 20) {
      if constexpr (nimble::isFloatingPointType<T>()) {
        values[i] = i;
      } else {
        values[i] = i % std::numeric_limits<T>::max();
      }
    }

    if constexpr (nimble::isFloatingPointType<T>() || sizeof(T) < 4) {
      // Floating point types and small types use Trivial encoding to encode
      // the exception values.
      test<T>(
          values,
          {
              {.encodingType = nimble::EncodingType::MainlyConstant,
               .dataType = nimble::TypeTraits<T>::dataType,
               .level = 0,
               .nestedEncodingName = ""},
              {.encodingType = nimble::EncodingType::SparseBool,
               .dataType = nimble::DataType::Bool,
               .level = 1,
               .nestedEncodingName = "IsCommon"},
              {.encodingType = nimble::EncodingType::FixedBitWidth,
               .dataType = nimble::DataType::Uint32,
               .level = 2,
               .nestedEncodingName = "Indices"},
              {.encodingType = nimble::EncodingType::Trivial,
               .dataType = nimble::isFloatingPointType<T>()
                   ? nimble::TypeTraits<T>::dataType
                   : nimble::TypeTraits<typename nimble::EncodingPhysicalType<
                         T>::type>::dataType,
               .level = 1,
               .nestedEncodingName = "OtherValues"},
          });
    } else {
      // All other numeric types use FixedBitWidth encoding to encode the
      // exception values.
      test<T>(
          values,
          {
              {.encodingType = nimble::EncodingType::MainlyConstant,
               .dataType = nimble::TypeTraits<T>::dataType,
               .level = 0,
               .nestedEncodingName = ""},
              {.encodingType = nimble::EncodingType::SparseBool,
               .dataType = nimble::DataType::Bool,
               .level = 1,
               .nestedEncodingName = "IsCommon"},
              {.encodingType = nimble::EncodingType::FixedBitWidth,
               .dataType = nimble::DataType::Uint32,
               .level = 2,
               .nestedEncodingName = "Indices"},
              {.encodingType = nimble::EncodingType::FixedBitWidth,
               .dataType = nimble::TypeTraits<
                   typename nimble::EncodingPhysicalType<T>::type>::dataType,
               .level = 1,
               .nestedEncodingName = "OtherValues"},
          });
    }
  }
}

TYPED_TEST(EncodingSelectionNumericTests, selectTrivial) {
  using T = TypeParam;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<T> values;
  values.resize(10000);
  for (auto i = 0; i < values.size(); ++i) {
    auto random = folly::Random::rand64(rng);
    values[i] = *reinterpret_cast<const T*>(&random);
  }

  test<T>(
      values,
      {
          {.encodingType = nimble::EncodingType::Trivial,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
      });
}

TYPED_TEST(EncodingSelectionNumericTests, selectFixedBitWidth) {
  using T = TypeParam;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (const auto bitWidth : {3, 4, 5}) {
    LOG(INFO) << "Testing with bit width: " << bitWidth;
    uint32_t maxValue = 1 << bitWidth;

    std::vector<T> values;
    values.resize(10000);
    for (auto i = 0; i < values.size(); ++i) {
      auto random = folly::Random::rand64(rng) % maxValue;
      values[i] = *reinterpret_cast<const T*>(&random);
    }

    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
        });
  }
}

TYPED_TEST(EncodingSelectionNumericTests, selectDictionary) {
  using T = TypeParam;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::array<T, 3> uniqueValues{
      std::numeric_limits<T>::min(), 0, std::numeric_limits<T>::max()};

  std::vector<T> values;
  values.resize(10000);
  for (auto i = 0; i < values.size(); ++i) {
    values[i] = uniqueValues[folly::Random::rand32(rng) % uniqueValues.size()];
  }

  test<T>(
      values,
      {
          {.encodingType = nimble::EncodingType::Dictionary,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
          {.encodingType = nimble::EncodingType::Trivial,
           .dataType = nimble::isFloatingPointType<T>()
               ? nimble::TypeTraits<T>::dataType
               : nimble::TypeTraits<
                     typename nimble::EncodingPhysicalType<T>::type>::dataType,
           .level = 1,
           .nestedEncodingName = "Alphabet"},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 1,
           .nestedEncodingName = "Indices"},
      });
}

TYPED_TEST(EncodingSelectionNumericTests, selectRunLength) {
  using T = TypeParam;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::array<T, 3> uniqueValues{
      std::numeric_limits<T>::min(), 0, std::numeric_limits<T>::max()};
  std::vector<uint32_t> runLengths;
  runLengths.resize(20);
  uint32_t valueCount = 0;
  for (auto i = 0; i < runLengths.size(); ++i) {
    runLengths[i] = (folly::Random::rand32(rng) % 700) + 20;
    valueCount += runLengths[i];
  }

  std::vector<T> values;
  values.reserve(valueCount);
  auto index = 0;
  for (const auto length : runLengths) {
    for (auto i = 0; i < length; ++i) {
      values.push_back(uniqueValues[index % uniqueValues.size()]);
    }
    ++index;
  }

  if constexpr (
      nimble::isFloatingPointType<T>() || std::is_same_v<int32_t, T> ||
      std::is_same_v<uint32_t, T> || sizeof(T) > 4) {
    // Floating point, 32-bit range-spanning, and wider types prefer storing the
    // run values as dictionary.
    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::RLE,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 1,
             .nestedEncodingName = "Lengths"},
            {.encodingType = nimble::EncodingType::Dictionary,
             .dataType = nimble::isFloatingPointType<T>()
                 ? nimble::TypeTraits<T>::dataType
                 : nimble::TypeTraits<typename nimble::EncodingPhysicalType<
                       T>::type>::dataType,
             .level = 1,
             .nestedEncodingName = "Values"},
            {.encodingType = nimble::EncodingType::Trivial,
             .dataType = nimble::isFloatingPointType<T>()
                 ? nimble::TypeTraits<T>::dataType
                 : nimble::TypeTraits<typename nimble::EncodingPhysicalType<
                       T>::type>::dataType,
             .level = 2,
             .nestedEncodingName = "Alphabet"},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 2,
             .nestedEncodingName = "Indices"},
        });
  } else {
    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::RLE,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 1,
             .nestedEncodingName = "Lengths"},
            {.encodingType = nimble::EncodingType::Trivial,
             .dataType = nimble::TypeTraits<
                 typename nimble::EncodingPhysicalType<T>::type>::dataType,
             .level = 1,
             .nestedEncodingName = "Values"},
        });
  }
}

TYPED_TEST(EncodingSelectionNumericTests, selectVarint) {
  using T = TypeParam;

  if constexpr (sizeof(T) < 4) {
    return;
  }

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<T> values;
  values.resize(700);
  for (auto i = 0; i < values.size(); ++i) {
    auto value = (std::numeric_limits<T>::max() / 2);
    auto random =
        *reinterpret_cast<const typename nimble::TypeTraits<T>::physicalType*>(
            &value) +
        (folly::Random::rand32(rng) % 128);
    values[i] = *reinterpret_cast<const T*>(&random);
  }

  for (auto i = 0; i < values.size(); i += 30) {
    values[i] = std::numeric_limits<T>::max() - folly::Random::rand32(128, rng);
  }

  test<T>(
      values,
      {
          {.encodingType = nimble::EncodingType::Varint,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
      });
}

TEST(EncodingSelectionBoolTests, selectConst) {
  using T = bool;

  for (const T value : {true, false}) {
    std::array<T, 1000> values;
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }

    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::Constant,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
        });
  }
}

TEST(EncodingSelectionBoolTests, selectSparseBool) {
  using T = bool;

  for (const T value : {false, true}) {
    std::array<T, 10000> values;
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }

    for (auto i = 0; i < values.size(); i += 200) {
      values[i] = !value;
    }

    test<T>(
        values,
        {
            {.encodingType = nimble::EncodingType::SparseBool,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 1,
             .nestedEncodingName = "Indices"},
        });
  }
}

TEST(EncodingSelectionBoolTests, selectTrivial) {
  using T = bool;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<std::unique_ptr<nimble::EncodingSelectionPolicy<T>>> policies;
  policies.push_back(getRootManualSelectionPolicy<T>());
  policies.push_back(
      std::make_unique<nimble::LearnedEncodingSelectionPolicy<T>>());

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  // Size estimate:
  // 1. Encoding header
  // 2. 1 byte for the compression flag
  // 3. Number of bytes to hold one bit per value rounded up to a full byte
  // 4. 7 extra bytes for the safe vectorized extraction
  size_t encodingOverhead = commonPrefixSize + 1 + 7;

  for (auto& policy : policies) {
    std::array<T, 10003> values{};
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = folly::Random::oneIn(2, rng);
    }

    const uint32_t expectedEncodedDataSize = std::ceil(values.size() / 8.0);
    verifySizeEstimate<T>(
        values,
        buffer,
        expectedEncodedDataSize + encodingOverhead,
        expectedEncodedDataSize + encodingOverhead,
        nimble::EncodingType::Trivial,
        {{nimble::EncodingType::Trivial, 1.0}});

    auto serialized =
        nimble::EncodingFactory::encode<T>(std::move(policy), values, buffer);

    // test getRawDataSize
    auto size =
        facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
    auto expectedSize = values.size() * sizeof(T);
    ASSERT_EQ(size, expectedSize);

    LOG(INFO) << "Final size: " << serialized.size();

    verifyEncodingTree(
        serialized,
        {
            {.encodingType = nimble::EncodingType::Trivial,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
        });
  }
}

TEST(EncodingSelectionBoolTests, selectRunLength) {
  using T = bool;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  std::vector<uint32_t> runLengths;
  runLengths.resize(20);
  uint32_t valueCount = 0;
  for (auto i = 0; i < runLengths.size(); ++i) {
    runLengths[i] = (folly::Random::rand32(rng) % 700) + 20;
    valueCount += runLengths[i];
  }

  nimble::Vector<bool> values{pool.get()};
  values.reserve(valueCount);
  auto index = 0;
  for (const auto length : runLengths) {
    for (auto i = 0; i < length; ++i) {
      values.push_back(index % 2 == 0);
    }
    ++index;
  }

  auto policy = getRootManualSelectionPolicy<T>();
  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, exactBitWidthOptions());

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  auto expectedSize = values.size() * sizeof(T);
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();

  verifyEncodingTree(
      serialized,
      {
          {.encodingType = nimble::EncodingType::RLE,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 1,
           .nestedEncodingName = "Lengths"},
      });
}

TEST(EncodingSelectionStringTests, selectConst) {
  using T = std::string_view;

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  for (const T value :
       {std::string(""), std::string("aaaaa"), std::string(5000, '\0')}) {
    LOG(INFO) << "Testing string with value: " << value;
    auto policy = getRootManualSelectionPolicy<T>();

    std::vector<T> values;
    values.resize(1000);
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }

    auto serialized = nimble::EncodingFactory::encode<T>(
        std::move(policy), values, buffer, exactBitWidthOptions());

    // test getRawDataSize
    auto size =
        facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
    auto expectedSize = value.size() * values.size();
    ASSERT_EQ(size, expectedSize);

    LOG(INFO) << "Final size: " << serialized.size();

    verifyEncodingTree(
        serialized,
        {
            {.encodingType = nimble::EncodingType::Constant,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
        });
  }
}

TEST(EncodingSelectionStringTests, selectMainlyConst) {
  using T = std::string_view;

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  for (const T value : {
           std::string("aaaaa"),
           std::string(5000, '\0'),
       }) {
    std::vector<T> values;
    auto expectedSize = 0;

    auto resize = 1000;
    values.resize(resize);
    for (auto i = 0; i < values.size(); ++i) {
      values[i] = value;
    }
    expectedSize += resize * value.size();

    std::vector<std::string> uncommonValues;
    for (auto i = 0; i < values.size() / 20; ++i) {
      uncommonValues.emplace_back(i, 'b');
    }

    for (auto i = 0; i < uncommonValues.size(); ++i) {
      std::string_view val = uncommonValues[i];
      values[i * 20] = val;
      expectedSize += val.size() - value.size();
    }

    auto policy = getRootManualSelectionPolicy<T>();
    auto serialized = nimble::EncodingFactory::encode<T>(
        std::move(policy), values, buffer, exactBitWidthOptions());

    // test getRawDataSize
    auto size =
        facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
    ASSERT_EQ(size, expectedSize);

    LOG(INFO) << "Final size: " << serialized.size();

    verifyEncodingTree(
        serialized,
        {
            {.encodingType = nimble::EncodingType::MainlyConstant,
             .dataType = nimble::TypeTraits<T>::dataType,
             .level = 0,
             .nestedEncodingName = ""},
            {.encodingType = nimble::EncodingType::SparseBool,
             .dataType = nimble::DataType::Bool,
             .level = 1,
             .nestedEncodingName = "IsCommon"},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 2,
             .nestedEncodingName = "Indices"},
            {.encodingType = nimble::EncodingType::Trivial,
             .dataType = nimble::TypeTraits<
                 typename nimble::EncodingPhysicalType<T>::type>::dataType,
             .level = 1,
             .nestedEncodingName = "OtherValues"},
            {.encodingType = nimble::EncodingType::FixedBitWidth,
             .dataType = nimble::DataType::Uint32,
             .level = 2,
             .nestedEncodingName = "Lengths"},
        });
  }
}

TEST(EncodingSelectionStringTests, selectTrivial) {
  using T = std::string_view;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  auto policy = getRootManualSelectionPolicy<T>();

  std::vector<std::string> cache;
  cache.resize(10000);
  for (auto i = 0; i < cache.size(); ++i) {
    std::string value(folly::Random::rand32(128, rng), ' ');
    for (auto j = 0; j < value.size(); ++j) {
      value[j] = static_cast<char>(folly::Random::rand32(256, rng));
    }
    cache[i] = std::move(value);
  }

  std::vector<T> values;
  auto expectedSize = 0;
  values.resize(cache.size());
  for (auto i = 0; i < cache.size(); ++i) {
    values[i] = cache[i];
    expectedSize += cache[i].size();
  }

  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, exactBitWidthOptions());

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();

  verifyEncodingTree(
      serialized,
      {
          {.encodingType = nimble::EncodingType::Trivial,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 1,
           .nestedEncodingName = "Lengths"},
      });
}

TEST(EncodingSelectionStringTests, selectDictionary) {
  using T = std::string_view;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<std::string> uniqueValues{"", "abcdef", std::string(5000, '\0')};

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  auto policy = getRootManualSelectionPolicy<T>();

  std::vector<T> values;
  auto expectedSize = 0;
  values.resize(10000);
  for (auto i = 0; i < values.size(); ++i) {
    T val = uniqueValues[folly::Random::rand32(rng) % uniqueValues.size()];
    values[i] = val;
    expectedSize += val.size();
  }

  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, exactBitWidthOptions());

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();

  verifyEncodingTree(
      serialized,
      {
          {.encodingType = nimble::EncodingType::Dictionary,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
          {.encodingType = nimble::EncodingType::Trivial,
           .dataType = nimble::TypeTraits<
               typename nimble::EncodingPhysicalType<T>::type>::dataType,
           .level = 1,
           .nestedEncodingName = "Alphabet"},
          {.encodingType = nimble::EncodingType::Varint,
           .dataType = nimble::DataType::Uint32,
           .level = 2,
           .nestedEncodingName = "Lengths"},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 1,
           .nestedEncodingName = "Indices"},
      });
}

TEST(EncodingSelectionStringTests, selectRunLength) {
  using T = std::string_view;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  std::vector<uint32_t> runLengths;
  runLengths.resize(20);
  uint32_t valueCount = 0;
  for (auto i = 0; i < runLengths.size(); ++i) {
    runLengths[i] = (folly::Random::rand32(rng) % 700) + 20;
    valueCount += runLengths[i];
  }

  std::vector<T> values;
  auto expectedSize = 0;
  values.reserve(valueCount);
  auto index = 0;
  for (const auto length : runLengths) {
    for (auto i = 0; i < length; ++i) {
      std::string_view val =
          ((index % 2 == 0) ? "abcdefghijklmnopqrstuvwxyz" : "1234567890");
      values.emplace_back(val);
      expectedSize += val.size();
    }
    ++index;
  }

  auto policy = getRootManualSelectionPolicy<T>();
  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, exactBitWidthOptions());

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();

  verifyEncodingTree(
      serialized,
      {
          {.encodingType = nimble::EncodingType::RLE,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 0,
           .nestedEncodingName = ""},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 1,
           .nestedEncodingName = "Lengths"},
          {.encodingType = nimble::EncodingType::Dictionary,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 1,
           .nestedEncodingName = "Values"},
          {.encodingType = nimble::EncodingType::Trivial,
           .dataType = nimble::TypeTraits<T>::dataType,
           .level = 2,
           .nestedEncodingName = "Alphabet"},
          {.encodingType = nimble::EncodingType::Varint,
           .dataType = nimble::DataType::Uint32,
           .level = 3,
           .nestedEncodingName = "Lengths"},
          {.encodingType = nimble::EncodingType::FixedBitWidth,
           .dataType = nimble::DataType::Uint32,
           .level = 2,
           .nestedEncodingName = "Indices"},
      });
}

TEST(EncodingSelectionTests, testNullable) {
  using T = std::string_view;
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  auto policy = getRootManualSelectionPolicy<T>();
  std::vector<T> data{"abcd", "efg", "hijk", "lmno"};
  std::array<bool, 10> nulls{
      true, false, true, true, false, false, true, true, true, false};

  auto serialized = nimble::EncodingFactory::encodeNullable<T>(
      std::move(policy), data, nulls, buffer);

  // test getRawDataSize
  auto size =
      facebook::nimble::test::TestUtils::getRawDataSize(*pool, serialized);
  auto expectedSize = 15 + 10; // 15 bytes for string data, 10 bytes for nulls
  ASSERT_EQ(size, expectedSize);

  LOG(INFO) << "Final size: " << serialized.size();
}

TEST(ManualEncodingSelectionPolicyTest, noCompressFlag) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  // Create data that will trigger compression when enabled.
  std::vector<uint32_t> data(1000, 42);

  // Encode with compression enabled (explicit CompressionOptions).
  auto compressPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          nimble::CompressionOptions{},
          std::nullopt);
  auto result = compressPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto compressionPolicy = result.compressionPolicyFactory();
  // Compression should be attempted (non-Uncompressed type).
  EXPECT_NE(
      compressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);

  // Default factory parameters — compression should also be enabled.
  {
    auto defaultFactory = nimble::ManualEncodingSelectionPolicyFactory{};
    auto defaultPolicy = defaultFactory.createPolicy(nimble::DataType::Uint32);
    auto* typedDefault =
        dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(
            defaultPolicy.get());
    ASSERT_NE(typedDefault, nullptr);
    auto defaultResult = typedDefault->select(
        data,
        nimble::Statistics<uint32_t>::create(data),
        nimble::Encoding::Options{});
    auto defaultCompressionPolicy = defaultResult.compressionPolicyFactory();
    EXPECT_NE(
        defaultCompressionPolicy->config().compressionType,
        nimble::CompressionType::Uncompressed);
  }

  // Encode with compression disabled (nullopt).
  auto noCompressPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          /*compressionOptions=*/std::nullopt,
          std::nullopt);
  auto noCompressResult = noCompressPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto noCompressionPolicy = noCompressResult.compressionPolicyFactory();
  // Compression should be Uncompressed (default NoCompressionPolicy).
  EXPECT_EQ(
      noCompressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);

  // Both should select the same encoding type.
  EXPECT_EQ(result.encodingType, noCompressResult.encodingType);
}

TEST(ManualEncodingSelectionPolicyTest, noCompressFlagPropagesToNested) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  // Create a no-compress policy and verify nested policies also have
  // compression disabled.
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          /*compressionOptions=*/std::nullopt,
          std::nullopt);

  // Create a nested policy (simulating what happens during encoding selection).
  auto nested = policy->template create<uint32_t>(
      nimble::EncodingType::Dictionary,
      nimble::EncodingIdentifiers::Dictionary::Alphabet);
  auto* typedNested =
      dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(nested.get());
  ASSERT_NE(typedNested, nullptr);

  // The nested policy should also disable compression.
  std::vector<uint32_t> data(100, 7);
  auto nestedResult = typedNested->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto nestedCompressionPolicy = nestedResult.compressionPolicyFactory();
  EXPECT_EQ(
      nestedCompressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);
}

TEST(ManualEncodingSelectionPolicyTest, perEncodingCompressionAcceptRatio) {
  // Default CompressionOptions has BlockBitPacking=0.7 in overrides.
  // The accept ratio is enforced by shouldAccept(), not by changing the
  // compression type — compression is always attempted, but rejected if
  // the compressed size exceeds the ratio threshold.

  // Force BlockBitPacking selection via read factors.
  auto cbpPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::BlockBitPacking, 1.0}},
          nimble::CompressionOptions{},
          std::nullopt);
  std::vector<uint32_t> data(2048);
  for (uint32_t i = 0; i < data.size(); ++i) {
    data[i] = i % 100;
  }
  auto cbpResult = cbpPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  EXPECT_EQ(cbpResult.encodingType, nimble::EncodingType::BlockBitPacking);
  auto cbpCompressionPolicy = cbpResult.compressionPolicyFactory();
  // BlockBitPacking ratio is 0.7: reject compression that only saves 20%.
  EXPECT_FALSE(cbpCompressionPolicy->shouldAccept(
      nimble::CompressionType::MetaInternal, 100, 80));
  // Accept compression that saves 40%.
  EXPECT_TRUE(cbpCompressionPolicy->shouldAccept(
      nimble::CompressionType::MetaInternal, 100, 60));

  // Force Trivial selection — default ratio is 0.98, more permissive.
  auto trivialPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::Trivial, 1.0}},
          nimble::CompressionOptions{},
          std::nullopt);
  auto trivialResult = trivialPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  EXPECT_EQ(trivialResult.encodingType, nimble::EncodingType::Trivial);
  auto trivialCompressionPolicy = trivialResult.compressionPolicyFactory();
  // Trivial uses default ratio 0.98: accept compression that saves 20%.
  EXPECT_TRUE(trivialCompressionPolicy->shouldAccept(
      nimble::CompressionType::MetaInternal, 100, 80));
}

TEST(ManualEncodingSelectionPolicyTest, customCompressionAcceptRatioOverride) {
  // Override: FixedBitWidth=0.0 (reject all compression).
  nimble::CompressionOptions options;
  options.compressionAcceptRatioOverrides = {
      {nimble::EncodingType::FixedBitWidth, 0.0f}};

  auto fbwPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::FixedBitWidth, 1.0}},
          options,
          std::nullopt);
  std::vector<uint32_t> data(1000);
  for (uint32_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }
  auto result = fbwPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  EXPECT_EQ(result.encodingType, nimble::EncodingType::FixedBitWidth);
  auto compressionPolicy = result.compressionPolicyFactory();
  // FixedBitWidth ratio is 0.0: reject all compression.
  EXPECT_FALSE(compressionPolicy->shouldAccept(
      nimble::CompressionType::MetaInternal, 100, 1));

  // BlockBitPacking NOT in this override list — uses default ratio (0.98).
  auto cbpPolicy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::BlockBitPacking, 1.0}},
          options,
          std::nullopt);
  auto cbpResult = cbpPolicy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto cbpCompressionPolicy = cbpResult.compressionPolicyFactory();
  // BlockBitPacking has no override here, uses default 0.98: accept moderate
  // savings.
  EXPECT_TRUE(cbpCompressionPolicy->shouldAccept(
      nimble::CompressionType::MetaInternal, 100, 80));
}

TEST(ManualEncodingSelectionPolicyFactoryTest, noCompressFactory) {
  // Factory with nullopt compressionOptions should create policies that skip
  // compression.
  nimble::ManualEncodingSelectionPolicyFactory factory{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      /*compressionOptions=*/std::nullopt};

  auto policy = factory.createPolicy(nimble::DataType::Uint32);
  auto* typed =
      dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(policy.get());
  ASSERT_NE(typed, nullptr);

  std::vector<uint32_t> data(100, 42);
  auto result = typed->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto compressionPolicy = result.compressionPolicyFactory();
  EXPECT_EQ(
      compressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);
}

TEST(
    ManualEncodingSelectionPolicyFactoryTest,
    noCompressFactoryPropagesToNested) {
  // Factory with nullopt compressionOptions should create policies whose
  // nested children also have compression disabled.
  nimble::ManualEncodingSelectionPolicyFactory factory{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      /*compressionOptions=*/std::nullopt};

  auto policy = factory.createPolicy(nimble::DataType::Uint32);
  auto* typed =
      dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(policy.get());
  ASSERT_NE(typed, nullptr);

  // Create a nested policy (simulating what happens during encoding selection).
  auto nested = typed->template create<uint32_t>(
      nimble::EncodingType::Dictionary,
      nimble::EncodingIdentifiers::Dictionary::Alphabet);
  auto* typedNested =
      dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(nested.get());
  ASSERT_NE(typedNested, nullptr);

  // The nested policy should also disable compression.
  std::vector<uint32_t> data(100, 7);
  auto nestedResult = typedNested->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto nestedCompressionPolicy = nestedResult.compressionPolicyFactory();
  EXPECT_EQ(
      nestedCompressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);
}

TEST(ReplayedEncodingSelectionPolicyTest, encodingRoundTrip) {
  // Verify that ReplayedEncodingSelectionPolicy with a known encoding type
  // produces valid encoded output and round-trips correctly.
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer(*pool);

  nimble::ManualEncodingSelectionPolicyFactory factory{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      /*compressionOptions=*/std::nullopt};
  nimble::EncodingSelectionPolicyCreator policyCreator =
      [&factory](nimble::DataType dataType) {
        return factory.createPolicy(dataType);
      };

  struct TestParam {
    nimble::EncodingType encodingType;
    std::optional<nimble::CompressionOptions> compressionOptions;
    std::vector<uint32_t> data;
    // Children layouts for compound encodings (e.g., RLE needs RunLengths
    // and RunValues slots). nullopt children fall back to policyCreator.
    std::vector<std::optional<const nimble::EncodingLayout>> children;
    std::string debugString() const {
      return fmt::format(
          "encodingType {}, compress {}",
          nimble::toString(encodingType),
          compressionOptions.has_value());
    }
  };
  std::vector<TestParam> testSettings = {
      {nimble::EncodingType::Trivial, std::nullopt, {10, 20, 30, 40, 50}, {}},
      {nimble::EncodingType::Trivial,
       nimble::CompressionOptions{},
       {10, 20, 30, 40, 50},
       {}},
      {nimble::EncodingType::Constant, std::nullopt, {42, 42, 42, 42, 42}, {}},
      {nimble::EncodingType::RLE,
       std::nullopt,
       {1, 1, 1, 2, 2, 2, 3, 3, 3},
       {std::nullopt, std::nullopt}},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buffer.reset();

    nimble::EncodingLayout layout{
        testData.encodingType,
        {},
        nimble::CompressionType::Uncompressed,
        testData.children};
    auto policy =
        std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint32_t>>(
            std::move(layout), testData.compressionOptions, policyCreator);

    auto encoded = nimble::EncodingFactory::encode<uint32_t>(
        std::move(policy), std::span<const uint32_t>(testData.data), buffer);

    // Decode and verify round-trip correctness.
    auto encoding = nimble::EncodingFactory().create(
        *pool,
        encoded,
        [&](uint32_t totalLength) -> void* {
          return pool->allocate(totalLength);
        },
        nimble::Encoding::Options{});
    EXPECT_EQ(encoding->encodingType(), testData.encodingType);
    EXPECT_EQ(encoding->dataType(), nimble::DataType::Uint32);
    EXPECT_EQ(encoding->rowCount(), testData.data.size());

    // For leaf encodings (Trivial), verify compression type stored in the
    // encoded data. The compression type byte is at dataOffset().
    if (testData.encodingType == nimble::EncodingType::Trivial) {
      auto compressionType =
          static_cast<nimble::CompressionType>(encoded[encoding->dataOffset()]);
      if (!testData.compressionOptions.has_value()) {
        EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);
      }
    }

    std::vector<uint32_t> decoded(testData.data.size());
    encoding->materialize(testData.data.size(), decoded.data());
    EXPECT_EQ(decoded, testData.data);
  }
}

TEST(ManualEncodingSelectionPolicyTest, nestedEncodingCompressionType) {
  // Verify that ManualEncodingSelectionPolicy propagates compression options
  // to nested encodings and the compression type is correctly stored in the
  // encoded output.
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer(*pool);

  // Data with repeated values — ManualEncodingSelectionPolicy may select
  // Dictionary or RLE, both of which have nested leaf encodings.
  std::vector<uint32_t> data = {1, 2, 3, 1, 2, 3, 1, 2, 3};

  struct TestParam {
    std::optional<nimble::CompressionOptions> compressionOptions;
    std::string debugString() const {
      return fmt::format("compress {}", compressionOptions.has_value());
    }
  };
  std::vector<TestParam> testSettings = {
      {std::nullopt},
      {nimble::CompressionOptions{}},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buffer.reset();

    nimble::ManualEncodingSelectionPolicyFactory factory{
        nimble::ManualEncodingSelectionPolicyFactory::
            defaultEncodingReadFactors(),
        testData.compressionOptions};
    auto basePolicy = factory.createPolicy(nimble::DataType::Uint32);
    auto* rawTyped = dynamic_cast<nimble::EncodingSelectionPolicy<uint32_t>*>(
        basePolicy.get());
    ASSERT_NE(rawTyped, nullptr);
    basePolicy.release();
    auto policy =
        std::unique_ptr<nimble::EncodingSelectionPolicy<uint32_t>>(rawTyped);

    auto encoded = nimble::EncodingFactory::encode<uint32_t>(
        std::move(policy), std::span<const uint32_t>(data), buffer);

    // Capture the encoding layout tree from the encoded output and verify
    // compression types at each level.
    auto capturedLayout = nimble::EncodingLayoutCapture::capture(
        encoded, nimble::Encoding::Options{});

    // Verify leaf compression types in the captured tree.
    // Walk through all children and check leaf nodes.
    std::function<void(const nimble::EncodingLayout&)> verifyCompression =
        [&](const nimble::EncodingLayout& layout) {
          // Leaf encodings (Trivial, FixedBitWidth) store compression type.
          if (layout.encodingType() == nimble::EncodingType::Trivial ||
              layout.encodingType() == nimble::EncodingType::FixedBitWidth) {
            if (!testData.compressionOptions.has_value()) {
              EXPECT_EQ(
                  layout.compressionType(),
                  nimble::CompressionType::Uncompressed);
            }
          }
          for (uint32_t i = 0; i < layout.childrenCount(); ++i) {
            const auto& child = layout.child(i);
            if (child.has_value()) {
              verifyCompression(child.value());
            }
          }
        };
    verifyCompression(capturedLayout);

    // Verify round-trip correctness.
    auto encoding = nimble::EncodingFactory().create(
        *pool,
        encoded,
        [&](uint32_t totalLength) -> void* {
          return pool->allocate(totalLength);
        },
        nimble::Encoding::Options{});
    EXPECT_EQ(encoding->rowCount(), data.size());

    std::vector<uint32_t> decoded(data.size());
    encoding->materialize(data.size(), decoded.data());
    EXPECT_EQ(decoded, data);
  }
}

TEST(ReplayedEncodingSelectionPolicyTest, nestedEncodingCompressionType) {
  // Verify that ReplayedEncodingSelectionPolicy propagates compression options
  // to nested encodings and the compression type is correctly stored in the
  // encoded output.
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer(*pool);

  nimble::ManualEncodingSelectionPolicyFactory fallbackFactory{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      /*compressionOptions=*/std::nullopt};
  auto policyFactory = [&fallbackFactory](nimble::DataType dataType) {
    return fallbackFactory.createPolicy(dataType);
  };

  // Dictionary encoding with Trivial children for alphabet and indices.
  auto makeLayout = []() {
    return nimble::EncodingLayout{
        nimble::EncodingType::Dictionary,
        {},
        nimble::CompressionType::Uncompressed,
        {
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::Uncompressed},
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::Uncompressed},
        }};
  };

  // Data with repeated values suitable for dictionary encoding.
  std::vector<uint32_t> data = {1, 2, 3, 1, 2, 3, 1, 2, 3};

  struct TestParam {
    std::optional<nimble::CompressionOptions> compressionOptions;
    std::string debugString() const {
      return fmt::format("compress {}", compressionOptions.has_value());
    }
  };
  std::vector<TestParam> testSettings = {
      {std::nullopt},
      {nimble::CompressionOptions{}},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buffer.reset();

    auto policy =
        std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint32_t>>(
            makeLayout(), testData.compressionOptions, policyFactory);

    auto encoded = nimble::EncodingFactory::encode<uint32_t>(
        std::move(policy), std::span<const uint32_t>(data), buffer);

    // Capture the encoding layout tree and verify compression types.
    auto capturedLayout = nimble::EncodingLayoutCapture::capture(
        encoded, nimble::Encoding::Options{});
    EXPECT_EQ(capturedLayout.encodingType(), nimble::EncodingType::Dictionary);

    // Verify leaf compression types in the captured tree.
    std::function<void(const nimble::EncodingLayout&)> verifyCompression =
        [&](const nimble::EncodingLayout& layout) {
          if (layout.encodingType() == nimble::EncodingType::Trivial ||
              layout.encodingType() == nimble::EncodingType::FixedBitWidth) {
            if (!testData.compressionOptions.has_value()) {
              EXPECT_EQ(
                  layout.compressionType(),
                  nimble::CompressionType::Uncompressed);
            }
          }
          for (uint32_t i = 0; i < layout.childrenCount(); ++i) {
            const auto& child = layout.child(i);
            if (child.has_value()) {
              verifyCompression(child.value());
            }
          }
        };
    verifyCompression(capturedLayout);

    // Verify round-trip correctness.
    auto encoding = nimble::EncodingFactory().create(
        *pool,
        encoded,
        [&](uint32_t totalLength) -> void* {
          return pool->allocate(totalLength);
        },
        nimble::Encoding::Options{});
    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Dictionary);
    EXPECT_EQ(encoding->rowCount(), data.size());

    std::vector<uint32_t> decoded(data.size());
    encoding->materialize(data.size(), decoded.data());
    EXPECT_EQ(decoded, data);
  }
}

TEST(ManualEncodingSelectionPolicyTest, defaultCompressionType) {
  // Verify that default CompressionOptions uses the compile-time default
  // compression type.
  std::vector<uint32_t> data(1000, 42);

  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          nimble::CompressionOptions{},
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto compressionPolicy = result.compressionPolicyFactory();
  auto info = compressionPolicy->config();

#ifndef DISABLE_META_INTERNAL_COMPRESSOR
  EXPECT_EQ(info.compressionType, nimble::CompressionType::MetaInternal);
#else
  EXPECT_EQ(info.compressionType, nimble::CompressionType::Zstd);
#endif
}

TEST(ManualEncodingSelectionPolicyTest, explicitZstdCompressionType) {
  // Verify that explicitly setting compressionType to Zstd overrides the
  // compile-time default.
  std::vector<uint32_t> data(1000, 42);

  nimble::CompressionOptions options{
      .compressionType = nimble::CompressionType::Zstd,
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto compressionPolicy = result.compressionPolicyFactory();
  auto info = compressionPolicy->config();

  EXPECT_EQ(info.compressionType, nimble::CompressionType::Zstd);
  EXPECT_EQ(
      info.parameters.zstd.compressionLevel, options.zstdCompressionLevel);
}

TEST(ManualEncodingSelectionPolicyTest, explicitMetaInternalCompressionType) {
  // Verify that explicitly setting compressionType to MetaInternal overrides
  // the compile-time default.
  std::vector<uint32_t> data(1000, 42);

  nimble::CompressionOptions options{
      .compressionType = nimble::CompressionType::MetaInternal,
      .internalCompressionLevel = 7,
      .internalDecompressionLevel = 3,
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  auto compressionPolicy = result.compressionPolicyFactory();
  auto info = compressionPolicy->config();

  EXPECT_EQ(info.compressionType, nimble::CompressionType::MetaInternal);
  EXPECT_EQ(
      info.parameters.metaInternal.compressionLevel,
      options.internalCompressionLevel);
  EXPECT_EQ(
      info.parameters.metaInternal.decompressionLevel,
      options.internalDecompressionLevel);
}

TEST(ManualEncodingSelectionPolicyTest, compressionTypeRoundTrip) {
  // Verify that data encoded with an explicit compression type can be
  // decoded correctly.
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer(*pool);

  std::vector<uint32_t> data(1000, 42);

  nimble::CompressionOptions options{
      .compressionType = nimble::CompressionType::Zstd,
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);

  auto encoded = nimble::EncodingFactory::encode<uint32_t>(
      std::move(policy), std::span<const uint32_t>(data), buffer);

  auto encoding = nimble::EncodingFactory().create(
      *pool,
      encoded,
      [&](uint32_t totalLength) -> void* {
        return pool->allocate(totalLength);
      },
      nimble::Encoding::Options{});
  EXPECT_EQ(encoding->rowCount(), data.size());

  std::vector<uint32_t> decoded(data.size());
  encoding->materialize(static_cast<uint32_t>(data.size()), decoded.data());
  EXPECT_EQ(decoded, data);
}

TEST(ManualEncodingSelectionPolicyTest, compressionAcceptRatioOverride) {
  std::vector<uint32_t> data(1000, 42);

  nimble::CompressionOptions options{
      .compressionAcceptRatioOverrides =
          {
              {nimble::EncodingType::Constant, 0.0f},
          },
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});

  // Data is all-constant, so Constant encoding is selected.
  EXPECT_EQ(result.encodingType, nimble::EncodingType::Constant);

  auto compressionPolicy = result.compressionPolicyFactory();
  // Constant encoding has override ratio=0.0, so compression is attempted but
  // shouldAccept() will always reject (0 < compressedSize).
  EXPECT_NE(
      compressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);
  EXPECT_FALSE(compressionPolicy->shouldAccept(
      nimble::CompressionType::Zstd,
      /*uncompressedSize=*/100,
      /*compressedSize=*/1));
}

TEST(ManualEncodingSelectionPolicyFactoryTest, fsstRequiresExplicitOptIn) {
  const auto defaultEncodingReadFactors = nimble::
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  EXPECT_EQ(
      std::find_if(
          defaultEncodingReadFactors.begin(),
          defaultEncodingReadFactors.end(),
          [](const auto& entry) {
            return entry.first == nimble::EncodingType::Fsst;
          }),
      defaultEncodingReadFactors.end());

  EXPECT_EQ(
      nimble::ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
          "Fsst=1.0"),
      (std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Fsst, 1.0}}));
}

TEST(ManualEncodingSelectionPolicyFactoryTest, rejectsReadOnlyEncodings) {
  NIMBLE_ASSERT_THROW(
      nimble::ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
          "PFOR=1.0"),
      "Encoding is read-only and cannot be used for new writes: PFOR");
}

TEST(
    ManualEncodingSelectionPolicyTest,
    compressionAcceptRatioOverrideFallback) {
  std::vector<uint32_t> data(1000, 42);

  nimble::CompressionOptions options{
      .compressionAcceptRatioOverrides =
          {
              {nimble::EncodingType::Trivial, 0.0f},
          },
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});

  // Data is all-constant, so Constant encoding is selected — not Trivial.
  EXPECT_EQ(result.encodingType, nimble::EncodingType::Constant);

  auto compressionPolicy = result.compressionPolicyFactory();
  // No override for Constant, so the global ratio applies — compression
  // should be attempted.
  EXPECT_NE(
      compressionPolicy->config().compressionType,
      nimble::CompressionType::Uncompressed);
}

TEST(
    ManualEncodingSelectionPolicyTest,
    compressionAcceptRatioOverrideShouldAccept) {
  std::vector<uint32_t> data(1000, 42);

  // Override with a very strict ratio that will reject most compression.
  nimble::CompressionOptions options{
      .compressionAcceptRatioOverrides =
          {
              {nimble::EncodingType::Constant, 0.01f},
          },
  };
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          options,
          std::nullopt);
  auto result = policy->select(
      data,
      nimble::Statistics<uint32_t>::create(data),
      nimble::Encoding::Options{});
  EXPECT_EQ(result.encodingType, nimble::EncodingType::Constant);

  auto compressionPolicy = result.compressionPolicyFactory();
  // With ratio=0.01, compressed size must be < 1% of original — even 50%
  // compression (100→50) is rejected.
  EXPECT_FALSE(compressionPolicy->shouldAccept(
      nimble::CompressionType::Zstd,
      /*uncompressedSize=*/100,
      /*compressedSize=*/50));
}

// Regression: a float stream of all-zeros with mixed sign bits (-0.0f/+0.0f) is
// logically constant (float ==) but physically distinct. Without ALP,
// ConstantEncoding must not treat it as constant, since doing so silently drops
// per-row ±0.0 sign bits. Verifies bit-exact (not float-==) round-trip.
TEST(EncodingSelectionFloatSignTest, mixedSignedZeroPreservesBitsWithoutAlp) {
  using T = float;
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  std::vector<T> values(1000);
  for (auto i = 0; i < values.size(); ++i) {
    values[i] = (i % 2 == 0) ? -0.0f : 0.0f;
  }

  // Default options: allowNestedAlpSelection is false (prod/default).
  const nimble::Encoding::Options options;
  auto policy = getRootManualSelectionPolicy<T>();
  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, options);

  auto encoding = nimble::EncodingFactory().create(
      *pool, serialized, [](uint32_t) -> void* { return nullptr; }, options);
  // Distinct sign bits must not collapse into a single Constant value.
  EXPECT_NE(encoding->encodingType(), nimble::EncodingType::Constant);

  nimble::Vector<T> result{pool.get()};
  result.resize(values.size());
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  for (auto i = 0; i < values.size(); ++i) {
    EXPECT_EQ(asPhysicalType(values[i]), asPhysicalType(result[i]))
        << "row " << i;
  }
}

// Companion: with allowNestedAlpSelection=true, floats are encoded by logical
// value, so logically-equal ±0.0 are legitimately collapsed into a single
// canonical ConstantEncoding value. Covers the ALP gate in the on state.
TEST(EncodingSelectionFloatSignTest, mixedSignedZeroCollapsesWithAlp) {
  using T = float;
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  std::vector<T> values(1000);
  for (auto i = 0; i < values.size(); ++i) {
    values[i] = (i % 2 == 0) ? -0.0f : 0.0f;
  }

  nimble::Encoding::Options options;
  options.allowNestedAlpSelection = true;
  auto policy = getRootManualSelectionPolicy<T>();
  auto serialized = nimble::EncodingFactory::encode<T>(
      std::move(policy), values, buffer, options);

  auto encoding = nimble::EncodingFactory().create(
      *pool, serialized, [](uint32_t) -> void* { return nullptr; }, options);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Constant);

  nimble::Vector<T> result{pool.get()};
  result.resize(values.size());
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  const auto canonicalBits = asPhysicalType(result[0]);
  for (auto i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], 0.0f) << "row " << i;
    // All values collapse to a single canonical bit pattern under ALP.
    EXPECT_EQ(asPhysicalType(result[i]), canonicalBits) << "row " << i;
  }
}

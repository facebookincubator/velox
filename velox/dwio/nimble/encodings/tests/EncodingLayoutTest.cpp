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
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/Buffer.h"

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#endif
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace facebook;

namespace {

void verifyEncodingLayout(
    const std::optional<const nimble::EncodingLayout>& expected,
    const std::optional<const nimble::EncodingLayout>& actual) {
  ASSERT_EQ(expected.has_value(), actual.has_value());

  if (!expected.has_value()) {
    return;
  }

  ASSERT_EQ(expected->encodingType(), actual->encodingType());

  // When MetaInternal is not available, it gets redirected to Zstd.
  // For tests, we need to account for this mapping.
  auto expectedCompression = expected->compressionType();
  auto actualCompression = actual->compressionType();

#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  // If expected is MetaInternal but we don't have it, accept Zstd or
  // Uncompressed (Uncompressed can happen if the data is too small to benefit
  // from compression)
  if (expectedCompression == nimble::CompressionType::MetaInternal) {
    ASSERT_TRUE(
        actualCompression == nimble::CompressionType::Zstd ||
        actualCompression == nimble::CompressionType::Uncompressed)
        << "Expected MetaInternal (mapped to Zstd or Uncompressed), but got "
        << nimble::toString(actualCompression);
  } else {
    ASSERT_EQ(expectedCompression, actualCompression);
  }
#else
  ASSERT_EQ(expectedCompression, actualCompression);
#endif

  ASSERT_EQ(expected->childrenCount(), actual->childrenCount());

  for (auto i = 0; i < expected->childrenCount(); ++i) {
    verifyEncodingLayout(expected->child(i), actual->child(i));
  }
}

void testSerialization(nimble::EncodingLayout expected) {
  std::string output;
  output.resize(1024);
  auto size = expected.serialize(output);
  auto actual = nimble::EncodingLayout::create(
      {output.data(), static_cast<size_t>(size)});
  verifyEncodingLayout(expected, actual.first);
}

template <typename T, typename TCollection = std::vector<T>>
nimble::EncodingLayout encodeAndCapture(
    nimble::EncodingLayout encodingLayout,
    TCollection data) {
  nimble::EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      [encodingFactory = nimble::ManualEncodingSelectionPolicyFactory{}](
          nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };

  auto defaultPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*defaultPool};
  auto encoding = nimble::EncodingFactory::encode<T>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<T>>(
          std::move(encodingLayout),
          nimble::CompressionOptions{
              .compressionAcceptRatio = 100, .internalMinCompressionSize = 0},
          encodingSelectionPolicyCreator),
      data,
      buffer);

  return nimble::EncodingLayoutCapture::capture(
      encoding, nimble::Encoding::Options{});
}

template <typename T, typename TCollection = std::vector<T>>
void testCapture(nimble::EncodingLayout expected, TCollection data) {
  verifyEncodingLayout(
      expected, encodeAndCapture<T>(expected, std::move(data)));
}

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
template <typename T>
class ForceSubIntSplitPolicy final : public nimble::EncodingSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::SubIntSplit};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* encodingType */,
      nimble::NestedEncodingIdentifier /* identifier */,
      nimble::DataType type) override {
    nimble::ManualEncodingSelectionPolicyFactory factory{
        nimble::ManualEncodingSelectionPolicyFactory::
            defaultEncodingReadFactors(),
        std::nullopt};
    return factory.createPolicy(type);
  }
};
#endif

} // namespace

TEST(EncodingLayoutTests, trivial) {
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::Trivial,
        {},
        nimble::CompressionType::Uncompressed};

    testSerialization(expected);
    testCapture<uint32_t>(expected, {1, 2, 3});
  }

  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::Trivial,
        {},
        nimble::CompressionType::MetaInternal};

    testSerialization(expected);
    testCapture<uint32_t>(expected, {1, 2, 3});
  }
}

TEST(EncodingLayoutTests, trivialString) {
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::Trivial,
        {},
        nimble::CompressionType::Uncompressed,
        {
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::Uncompressed},
        }};

    testSerialization(expected);
    testCapture<std::string_view>(expected, {"a", "b", "c"});
  }
}

TEST(EncodingLayoutTests, fixedBitWidth) {
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::FixedBitWidth,
        {},
        nimble::CompressionType::Uncompressed,
    };

    testSerialization(expected);
    testCapture<uint32_t>(expected, {1, 2, 3});
  }

  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::FixedBitWidth,
        {},
        nimble::CompressionType::MetaInternal,
    };

    testSerialization(expected);
    // NOTE: We need this artitifical long input data, because if MetaInternal
    // compressed buffer is bigger than the uncompressed buffer, it is not
    // picked up, which then leads to the captured encloding layout to be
    // uncompressed.
    testCapture<uint32_t>(
        expected, {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF});
  }
}

TEST(EncodingLayoutTests, varint) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::Varint,
      {},
      nimble::CompressionType::Uncompressed,
  };

  testSerialization(expected);
  testCapture<uint32_t>(expected, {1, 2, 3});
}

TEST(EncodingLayoutTests, constant) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::Constant,
      {},
      nimble::CompressionType::Uncompressed,
  };

  testSerialization(expected);
  testCapture<uint32_t>(expected, {1, 1, 1});
}

TEST(EncodingLayoutTests, pfor) {
  // PFOR nests two exception sub-streams (positions, residual values). Capture
  // records each present sub-stream recursively, like the compound encodings,
  // so a captured layout reproduces the full PFOR tree.
  nimble::EncodingLayout pforLayout{
      nimble::EncodingType::PFOR,
      {},
      nimble::CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};

  // Serialization preserves the PFOR node and its two child slots.
  testSerialization(pforLayout);

  // Sparse outliers so the encoding emits exceptions and both nested
  // sub-streams are present.
  std::vector<uint32_t> data;
  data.reserve(500);
  for (uint32_t i = 0; i < 500; ++i) {
    data.push_back(i % 10 == 7 ? 100000 + i : 50 + (i % 50));
  }

  // The nullopt children drive selection to PFOR while letting each sub-stream
  // re-select; capture must then record both sub-streams' encodings (not
  // nullopt), proving recursive capture of the nested layout.
  auto captured = encodeAndCapture<uint32_t>(pforLayout, data);
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::PFOR);
  ASSERT_EQ(captured.childrenCount(), 2);
  EXPECT_TRUE(captured.child(0).has_value());
  EXPECT_TRUE(captured.child(1).has_value());
}

TEST(EncodingLayoutTests, forEncoding) {
  // FOR nests three frame metadata sub-streams: bit widths, references, and
  // bit offsets. Capture records each present sub-stream recursively so a
  // captured layout reproduces the full FOR tree.
  nimble::EncodingLayout forLayout{
      nimble::EncodingType::FOR,
      {},
      nimble::CompressionType::Uncompressed,
      {std::nullopt, std::nullopt, std::nullopt}};

  testSerialization(forLayout);

  std::vector<uint32_t> data;
  data.reserve(500);
  for (uint32_t i = 0; i < 500; ++i) {
    data.push_back((i / 128) * 1000 + (i % 37));
  }

  auto captured = encodeAndCapture<uint32_t>(forLayout, data);
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::FOR);
  ASSERT_EQ(captured.childrenCount(), 3);
  EXPECT_TRUE(captured.child(0).has_value());
  EXPECT_TRUE(captured.child(1).has_value());
  EXPECT_TRUE(captured.child(2).has_value());
}

TEST(EncodingLayoutTests, fsst) {
  nimble::EncodingLayout fsstLayout{
      nimble::EncodingType::Fsst,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(fsstLayout);

  std::vector<std::string> storage;
  storage.reserve(500);
  for (uint32_t i = 0; i < 500; ++i) {
    storage.emplace_back(
        "common/prefix/for/fsst/layout/" + std::to_string(i % 31));
  }

  std::vector<std::string_view> data;
  data.reserve(storage.size());
  for (const auto& value : storage) {
    data.push_back(value);
  }

  auto captured = encodeAndCapture<std::string_view>(fsstLayout, data);
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::Fsst);
  ASSERT_EQ(captured.childrenCount(), 1);
  ASSERT_TRUE(
      captured.child(nimble::EncodingIdentifiers::Fsst::Lengths).has_value());
  EXPECT_EQ(
      captured.child(nimble::EncodingIdentifiers::Fsst::Lengths)
          ->encodingType(),
      nimble::EncodingType::Trivial);
}

TEST(EncodingLayoutTests, blockBitPacking) {
  // BlockBitPacking nests three per-block metadata sub-streams (baselines, bit
  // widths, data offsets). Capture records each present sub-stream recursively,
  // like the compound encodings, so a captured layout reproduces the full tree.
  nimble::EncodingLayout bbpLayout{
      nimble::EncodingType::BlockBitPacking,
      {},
      nimble::CompressionType::Uncompressed,
      {std::nullopt, std::nullopt, std::nullopt}};

  // Serialization preserves the BlockBitPacking node and its three child slots.
  testSerialization(bbpLayout);

  // Locally narrow per-block ranges so blocks get distinct baselines / widths.
  std::vector<uint32_t> data;
  data.reserve(1000);
  for (uint32_t i = 0; i < 1000; ++i) {
    data.push_back((i / 200) * 1000 + (i % 37));
  }

  // The nullopt children drive selection to BlockBitPacking while letting each
  // sub-stream re-select; capture must then record all three sub-streams'
  // encodings (not nullopt), proving recursive capture of the nested layout.
  auto captured = encodeAndCapture<uint32_t>(bbpLayout, data);
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::BlockBitPacking);
  ASSERT_EQ(captured.childrenCount(), 3);
  EXPECT_TRUE(captured.child(0).has_value());
  EXPECT_TRUE(captured.child(1).has_value());
  EXPECT_TRUE(captured.child(2).has_value());
}

TEST(EncodingLayoutTests, sparseBool) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::MetaInternal},
      }};
  testSerialization(expected);

  // Test actual capture with uncompressed
  nimble::EncodingLayout captureTest{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
      }};
  testCapture<bool>(
      captureTest, std::array<bool, 5>{false, false, false, true, false});
}

TEST(EncodingLayoutTests, mainlyConst) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::MainlyConstant,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::MetaInternal},
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
      }};
  testSerialization(expected);

  // Test actual capture with uncompressed
  nimble::EncodingLayout captureTest{
      nimble::EncodingType::MainlyConstant,
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
  testCapture<uint32_t>(captureTest, {1, 1, 1, 1, 5, 1});
}

TEST(EncodingLayoutTests, dictionary) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::Dictionary,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(expected);
  testCapture<uint32_t>(expected, {1, 1, 1, 1, 5, 1});
}

TEST(EncodingLayoutTests, sharedDictionary) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::SharedDictionary,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(expected);

  auto defaultPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*defaultPool};
  const std::vector<uint32_t> indices{0, 1, 0, 2, 1, 2};
  const auto encoded = nimble::test::encodeSharedDictionary(buffer, indices);

  verifyEncodingLayout(
      expected,
      nimble::EncodingLayoutCapture::capture(
          encoded, nimble::Encoding::Options{}));

  struct TraversedEncoding {
    nimble::EncodingType encodingType;
    nimble::DataType dataType;
    uint32_t level;
    uint32_t index;
    std::string nestedEncodingName;
  };
  std::vector<TraversedEncoding> traversed;
  nimble::tools::traverseEncodings(
      encoded,
      [&](nimble::EncodingType encodingType,
          nimble::DataType dataType,
          uint32_t level,
          uint32_t index,
          std::string nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        traversed.push_back(
            TraversedEncoding{
                .encodingType = encodingType,
                .dataType = dataType,
                .level = level,
                .index = index,
                .nestedEncodingName = std::move(nestedEncodingName)});
        return true;
      });
  ASSERT_EQ(traversed.size(), 2);
  EXPECT_EQ(traversed[0].encodingType, nimble::EncodingType::SharedDictionary);
  EXPECT_EQ(traversed[0].dataType, nimble::DataType::Int32);
  EXPECT_EQ(traversed[0].level, 0);
  EXPECT_TRUE(traversed[0].nestedEncodingName.empty());
  EXPECT_EQ(traversed[1].encodingType, nimble::EncodingType::FixedBitWidth);
  EXPECT_EQ(traversed[1].dataType, nimble::DataType::Uint32);
  EXPECT_EQ(traversed[1].level, 1);
  EXPECT_EQ(traversed[1].index, 0);
  EXPECT_EQ(traversed[1].nestedEncodingName, "Indices");
}

TEST(EncodingLayoutTests, replayDictionaryRejectsEmpty) {
  // A replayed Dictionary layout applied to an EMPTY value stream cannot build
  // a dictionary. It must reject the stream with an incompatible-encoding error
  // -- like the other data-requiring encodings in EncodingFactory -- so the
  // writer retries the stream without the captured layout. Before the fix this
  // was a NIMBLE_DCHECK that aborted the process (and was silently ignored in
  // opt).
  nimble::EncodingLayout dictionary{
      nimble::EncodingType::Dictionary,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  NIMBLE_ASSERT_THROW(
      encodeAndCapture<uint32_t>(
          std::move(dictionary), std::vector<uint32_t>{}),
      "Dictionary encoding cannot be used with 0 rows.");
}

TEST(
    EncodingLayoutTests,
    replayMainlyConstantDictionaryRejectsEmptyOtherValues) {
  // Replay a MainlyConstant whose OtherValues stream is a nested Dictionary.
  // When every value equals the common value, OtherValues is empty, so the
  // nested Dictionary replay has nothing to encode -- the data shape that made
  // fuzzMainlyConstantDictionaryVector flake. The empty inner Dictionary must
  // reject with an incompatible-encoding error (propagated out of the
  // MainlyConstant encode) instead of aborting, so the writer can retry.
  nimble::EncodingLayout mainlyConstant{
      nimble::EncodingType::MainlyConstant,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::Dictionary,
              {},
              nimble::CompressionType::Uncompressed,
              {
                  nimble::EncodingLayout{
                      nimble::EncodingType::Trivial,
                      {},
                      nimble::CompressionType::Uncompressed},
                  nimble::EncodingLayout{
                      nimble::EncodingType::FixedBitWidth,
                      {},
                      nimble::CompressionType::Uncompressed},
              }},
      }};

  // All values identical -> MainlyConstant OtherValues stream is empty.
  std::vector<uint32_t> data(64, 7);
  NIMBLE_ASSERT_THROW(
      encodeAndCapture<uint32_t>(std::move(mainlyConstant), data),
      "Dictionary encoding cannot be used with 0 rows.");
}

TEST(EncodingLayoutTests, rle) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::RLE,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(expected);
  testCapture<uint32_t>(expected, {1, 1, 1, 1, 5, 1});
}

TEST(EncodingLayoutTests, rleBool) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::RLE,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(expected);
  testCapture<bool>(expected, std::array<bool, 4>{false, false, true, true});
}

TEST(EncodingLayoutTests, nullable) {
  nimble::EncodingLayout expected{
      nimble::EncodingType::Nullable,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::EncodingLayout{
           nimble::EncodingType::FixedBitWidth,
           {},
           nimble::CompressionType::Uncompressed},
       nimble::EncodingLayout{
           nimble::EncodingType::SparseBool,
           {},
           nimble::CompressionType::Uncompressed,
           {
               nimble::EncodingLayout{
                   nimble::EncodingType::Trivial,
                   {},
                   nimble::CompressionType::MetaInternal},
           }}}};

  testSerialization(expected);

  nimble::EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      [encodingFactory = nimble::ManualEncodingSelectionPolicyFactory{}](
          nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };

  auto defaultPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*defaultPool};
  auto encoding = nimble::EncodingFactory::encodeNullable<uint32_t>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint32_t>>(
          expected.child(nimble::EncodingIdentifiers::Nullable::Data).value(),
          nimble::CompressionOptions{},
          encodingSelectionPolicyCreator),
      std::vector<uint32_t>{1, 1, 1, 1, 5, 1},
      std::array<bool, 6>{false, false, true, false, false, false},
      buffer);

  std::string output;
  output.resize(1024);
  auto captured = nimble::EncodingLayoutCapture::capture(
      encoding, nimble::Encoding::Options{});
  auto size = captured.serialize(output);

  auto actual = nimble::EncodingLayout::create(
      {output.data(), static_cast<size_t>(size)});

  // For nullable, captured encoding layout strips out the nullable node and
  // just captures the data node.
  verifyEncodingLayout(
      expected.child(nimble::EncodingIdentifiers::Nullable::Data),
      actual.first);
}

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
// SubIntSplitEncoding layout tests -------------------------------------------

// Verify that an EncodingLayout tree with a SubIntSplit root and two named
// child sections can be serialized and deserialized correctly.
TEST(EncodingLayoutTests, subIntSplitSerialization) {
  nimble::EncodingLayout layout{
      nimble::EncodingType::SubIntSplit,
      {},
      nimble::CompressionType::Uncompressed,
      {
          // section 0: lower bits — typically FixedBitWidth or Trivial
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
          // section 1: upper bits — often Constant for structured IDs
          nimble::EncodingLayout{
              nimble::EncodingType::Constant,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(layout);

  // Three sections
  nimble::EncodingLayout layout3{
      nimble::EncodingType::SubIntSplit,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::Dictionary,
              {},
              nimble::CompressionType::Uncompressed,
              {
                  nimble::EncodingLayout{
                      nimble::EncodingType::Trivial,
                      {},
                      nimble::CompressionType::Uncompressed},
                  nimble::EncodingLayout{
                      nimble::EncodingType::FixedBitWidth,
                      {},
                      nimble::CompressionType::Uncompressed},
              }},
          nimble::EncodingLayout{
              nimble::EncodingType::Constant,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  testSerialization(layout3);
}

// Verify that EncodingLayoutCapture correctly reads the SubIntSplit binary
// format and that the captured tree round-trips through serialize/deserialize.
TEST(EncodingLayoutTests, subIntSplitCapture) {
  nimble::EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      [encodingFactory = nimble::ManualEncodingSelectionPolicyFactory{}](
          nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };

  auto defaultPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*defaultPool};

  // Structured int64 values: upper 40 bits are a fixed datacenter/timestamp
  // prefix; lower 24 bits are a monotone counter. This should trigger
  // SubIntSplit selection (range << typeWidth) and produce a multi-section
  // encoding where the upper section is Constant or FixedBitWidth with small
  // range and the lower section is FixedBitWidth or Trivial.
  std::vector<int64_t> data;
  data.reserve(300);
  for (int64_t i = 0; i < 300; ++i) {
    data.push_back(static_cast<int64_t>(0x1234567890000000LL) | i);
  }

  auto encoding = nimble::EncodingFactory::encode<int64_t>(
      std::make_unique<ForceSubIntSplitPolicy<int64_t>>(), data, buffer);

  // Capture must succeed and not throw.
  auto captured = nimble::EncodingLayoutCapture::capture(
      encoding, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  ASSERT_GT(captured.childrenCount(), 0u);
  EXPECT_EQ(
      captured.config().get(
          std::string(nimble::detail::subintsplit::kSplitModeConfigKey)),
      nimble::detail::subintsplit::kSplitModePreserve);
  ASSERT_TRUE(
      captured.config()
          .get(
              std::string(
                  nimble::detail::subintsplit::kSplitBoundariesConfigKey))
          .has_value());

  // The encoded stream must round-trip through the encoding factory.
  auto decoded = nimble::EncodingFactory().create(
      *defaultPool,
      encoding,
      [](uint32_t) { return nullptr; },
      nimble::Encoding::Options{});
  nimble::Vector<int64_t> decodedValues{defaultPool.get()};
  decodedValues.resize(data.size());
  decoded->materialize(data.size(), decodedValues.data());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data[i], decodedValues[i]) << i;
  }

  std::vector<nimble::EncodingType> traversedTypes;
  nimble::tools::traverseEncodings(
      encoding,
      [&](nimble::EncodingType encodingType,
          nimble::DataType dataType,
          uint32_t level,
          uint32_t /* index */,
          std::string nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        if (level == 0) {
          EXPECT_EQ(encodingType, nimble::EncodingType::SubIntSplit);
          EXPECT_EQ(dataType, nimble::DataType::Int64);
          EXPECT_TRUE(nestedEncodingName.empty());
        }
        traversedTypes.push_back(encodingType);
        return true;
      });
  ASSERT_GE(traversedTypes.size(), 2u);

  // The captured layout must round-trip through serialize → deserialize.
  std::string output(4096, '\0');
  const auto serializedSize = captured.serialize(output);
  ASSERT_GT(serializedSize, 0);
  auto [deserialized, bytesRead] = nimble::EncodingLayout::create(
      {output.data(), static_cast<size_t>(serializedSize)});
  verifyEncodingLayout(captured, deserialized);
  auto preserveMode = deserialized.config().get(
      std::string(nimble::detail::subintsplit::kSplitModeConfigKey));
  ASSERT_TRUE(preserveMode.has_value());
  EXPECT_EQ(*preserveMode, nimble::detail::subintsplit::kSplitModePreserve);

  auto deserializedBoundaries = deserialized.config().get(
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
  auto capturedBoundaries = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(deserializedBoundaries.has_value());
  ASSERT_TRUE(capturedBoundaries.has_value());
  EXPECT_EQ(*deserializedBoundaries, *capturedBoundaries);

  // Each child must be non-nullopt (all sections were encoded).
  for (nimble::NestedEncodingIdentifier id = 0; id < captured.childrenCount();
       ++id) {
    EXPECT_TRUE(captured.child(id).has_value());
  }
}

TEST(EncodingLayoutTests, subIntSplitPreserveBoundariesReplay) {
  nimble::EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      [encodingFactory = nimble::ManualEncodingSelectionPolicyFactory{}](
          nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };

  auto defaultPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*defaultPool};

  std::vector<int64_t> data;
  data.reserve(300);
  for (int64_t i = 0; i < 300; ++i) {
    data.push_back(static_cast<int64_t>(0x1234567890000000LL) | i);
  }

  const std::string preserveBoundaries = "0-15;16-47;48-63";
  nimble::EncodingLayout preserveLayout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{
          {{std::string(nimble::detail::subintsplit::kSplitModeConfigKey),
            std::string(nimble::detail::subintsplit::kSplitModePreserve)},
           {std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey),
            preserveBoundaries}}},
      nimble::CompressionType::Uncompressed,
      {std::nullopt, std::nullopt, std::nullopt}};

  auto encoding = nimble::EncodingFactory::encode<int64_t>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<int64_t>>(
          preserveLayout, std::nullopt, encodingSelectionPolicyCreator),
      data,
      buffer);

  auto captured = nimble::EncodingLayoutCapture::capture(
      encoding, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  auto replayedMode = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitModeConfigKey));
  auto replayedBoundaries = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(replayedMode.has_value());
  ASSERT_TRUE(replayedBoundaries.has_value());
  EXPECT_EQ(*replayedMode, nimble::detail::subintsplit::kSplitModePreserve);
  EXPECT_EQ(*replayedBoundaries, preserveBoundaries);
}
#endif

TEST(EncodingLayoutTests, sizeTooSmall) {
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::Trivial,
        {},
        nimble::CompressionType::Uncompressed,
    };

    std::string output;
    // Encoding needs minimum of 5 bytes. 4 is not enough.
    output.resize(4);
    EXPECT_THROW(
        expected.serialize(output), facebook::nimble::NimbleInternalError);
  }
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::Trivial,
        {},
        nimble::CompressionType::Uncompressed,
    };

    std::string output;
    // Encoding needs minimum of 5 bytes. Should not throw.
    output.resize(5);
    EXPECT_EQ(5, expected.serialize(output));
  }
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::MainlyConstant,
        {},
        nimble::CompressionType::Uncompressed,
        {
            std::nullopt,
            std::nullopt,
        }};

    std::string output;
    // 5 bytes for the top level encoding, plus 2 "exists" bytes.
    // Total of 7 bytes. 6 bytes is not enough.
    output.resize(6);
    EXPECT_THROW(
        expected.serialize(output), facebook::nimble::NimbleInternalError);
  }
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::MainlyConstant,
        {},
        nimble::CompressionType::Uncompressed,
        {
            std::nullopt,
            std::nullopt,
        }};

    std::string output;
    // 5 bytes for the top level encoding, plus 2 "exists" bytes.
    // Total of 7 bytes. 7 bytes is enough.
    output.resize(7);
    EXPECT_EQ(7, expected.serialize(output));
  }
  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::MainlyConstant,
        {},
        nimble::CompressionType::Uncompressed,
        {
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::MetaInternal},
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::Uncompressed},
        }};

    std::string output;
    // Each sub-encoding is 5 bytes (total of 10), plus 5 for the top level one.
    // Plus 2 "exists" bytes. Total of 17 bytes. 16 bytes is not enough.
    output.resize(16);
    EXPECT_THROW(
        expected.serialize(output), facebook::nimble::NimbleInternalError);
  }

  {
    nimble::EncodingLayout expected{
        nimble::EncodingType::MainlyConstant,
        {},
        nimble::CompressionType::Uncompressed,
        {
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::MetaInternal},
            nimble::EncodingLayout{
                nimble::EncodingType::Trivial,
                {},
                nimble::CompressionType::Uncompressed},
        }};

    std::string output;
    // Each sub-encoding is 5 bytes (total of 10), plus 5 for the top level one.
    // Plus 2 "exists" bytes. Total of 17 bytes. 17 bytes is enough.
    output.resize(17);
    EXPECT_EQ(17, expected.serialize(output));
  }
}

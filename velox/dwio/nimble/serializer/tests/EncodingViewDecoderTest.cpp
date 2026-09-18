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

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"

namespace facebook::nimble {
namespace {

class EncodingViewDecoderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "encoding_view_decoder_test");
  }

  template <typename T>
  std::string encode(std::span<const T> values) {
    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), values, buffer);
    return std::string{encoded};
  }

  template <typename T>
  std::string encodeNullable(const std::vector<std::optional<T>>& values) {
    std::vector<T> nonNullValues;
    Vector<bool> isNonNull{pool_.get(), values.size()};
    for (size_t i{0}; i < values.size(); ++i) {
      isNonNull[i] = values[i].has_value();
      if (values[i].has_value()) {
        nonNullValues.push_back(*values[i]);
      }
    }

    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded = EncodingFactory::encodeNullable<T>(
        std::move(policy), nonNullValues, isNonNull, buffer);
    return std::string{encoded};
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(std::string_view stream) {
    return std::make_unique<EncodingViewDecoder>(
        stream, pool_.get(), [this](std::string_view encoded) {
          return createEncodingView(encoded, pool_.get());
        });
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(EncodingViewDecoderTest, readsSelectedNullableRows) {
  const auto stream =
      encodeNullable<int64_t>({10, std::nullopt, 12, 13, std::nullopt, 15});
  auto decoder = makeDecoder(stream);
  const std::array<uint32_t, 4> rows{1, 2, 4, 5};
  std::array<int64_t, 4> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::vector<velox::BufferPtr> stringBuffers;

  const auto numNonNulls = decoder->read(
      rows,
      DataType::Int64,
      output.data(),
      [&outputNulls]() { return outputNulls.data(); },
      stringBuffers);

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{0, 12, 0, 15}));
  std::array<bool, 4> actualNotNulls{};
  for (size_t i{0}; i < actualNotNulls.size(); ++i) {
    actualNotNulls[i] = velox::bits::isBitSet(outputNulls.data(), i);
  }
  EXPECT_EQ(actualNotNulls, (std::array<bool, 4>{false, true, false, true}));
  EXPECT_TRUE(stringBuffers.empty());
}

TEST_F(EncodingViewDecoderTest, retainsSelectedStrings) {
  const auto stream = encode<std::string_view>(
      std::vector<std::string_view>{"first string", "unused", "third string"});
  const std::array<uint32_t, 2> rows{0, 2};
  std::array<std::string_view, 2> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  {
    auto decoder = makeDecoder(stream);
    EXPECT_EQ(
        decoder->read(
            rows,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        rows.size());
  }

  EXPECT_EQ(
      output,
      (std::array<std::string_view, 2>{"first string", "third string"}));
  ASSERT_EQ(stringBuffers.size(), 1);
  const std::string_view retainedContent{
      stringBuffers[0]->as<char>(), stringBuffers[0]->size()};
  EXPECT_EQ(retainedContent, "first stringthird string");
}

} // namespace
} // namespace facebook::nimble

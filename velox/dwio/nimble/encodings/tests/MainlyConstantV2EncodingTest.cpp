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

#include "velox/dwio/nimble/encodings/MainlyConstantV2Encoding.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;
using testing::ElementsAreArray;

namespace {

class MainlyConstantV2EncodingTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  nimble::Vector<int64_t> makeValues(std::initializer_list<int64_t> values) {
    nimble::Vector<int64_t> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(MainlyConstantV2EncodingTest, roundTripAndSequentialState) {
  const auto values = makeValues({9, 7, 7, 11, 7, 7, 7, 13});

  for (const bool useVarintRowCount : {false, true}) {
    SCOPED_TRACE(
        testing::Message() << "useVarintRowCount=" << useVarintRowCount);
    nimble::Buffer buffer{*pool_};
    const nimble::Encoding::Options options{
        .useVarintRowCount = useVarintRowCount};
    auto encoding = nimble::test::
        Encoder<nimble::MainlyConstantV2Encoding<int64_t>>::createEncoding(
            buffer,
            values,
            [](uint32_t /*totalLength*/) -> void* { return nullptr; },
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Vector<int64_t> decoded{pool_.get(), values.size()};
    encoding->materialize(values.size(), decoded.data());
    EXPECT_THAT(decoded, ElementsAreArray(values));

    encoding->reset();
    encoding->skip(2);
    nimble::Vector<int64_t> middle{pool_.get(), 4};
    encoding->materialize(middle.size(), middle.data());
    EXPECT_THAT(middle, testing::ElementsAre(7, 11, 7, 7));

    encoding->reset();
    nimble::Vector<int64_t> first{pool_.get(), 3};
    encoding->materialize(first.size(), first.data());
    EXPECT_THAT(first, testing::ElementsAre(9, 7, 7));
    nimble::Vector<int64_t> second{pool_.get(), 5};
    encoding->materialize(second.size(), second.data());
    EXPECT_THAT(second, testing::ElementsAre(11, 7, 7, 7, 13));
  }
}

TEST_F(MainlyConstantV2EncodingTest, allCommonHasNoChildren) {
  const auto values = makeValues({7, 7, 7, 7});
  nimble::Buffer buffer{*pool_};
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantV2Encoding<int64_t>>::encode(
          buffer, values);

  const auto layout = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  EXPECT_EQ(layout.encodingType(), nimble::EncodingType::MainlyConstantV2);
  EXPECT_EQ(layout.childrenCount(), 2);
  EXPECT_FALSE(layout.child(0).has_value());
  EXPECT_FALSE(layout.child(1).has_value());

  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, [](uint32_t /*totalLength*/) -> void* {
        return nullptr;
      });
  nimble::Vector<int64_t> decoded{pool_.get(), values.size()};
  encoding->materialize(values.size(), decoded.data());
  EXPECT_THAT(decoded, ElementsAreArray(values));
}

TEST_F(MainlyConstantV2EncodingTest, rejectsEmptyInput) {
  nimble::Vector<int64_t> values{pool_.get()};
  nimble::Buffer buffer{*pool_};

  NIMBLE_ASSERT_THROW(
      (nimble::test::Encoder<nimble::MainlyConstantV2Encoding<int64_t>>::encode(
          buffer, values)),
      "MainlyConstantV2 cannot be empty");
}

TEST_F(MainlyConstantV2EncodingTest, rejectsMismatchedPositionsType) {
  const auto values = makeValues({7, 7, 9, 7});
  nimble::Buffer buffer{*pool_};
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantV2Encoding<int64_t>>::encode(
          buffer, values);
  std::string malformed{encoded};
  const auto positionsOffset = nimble::EncodingPrefix::prefixSize(
                                   malformed, /*useVarintRowCount=*/false) +
      sizeof(int64_t) + sizeof(uint32_t);
  malformed[positionsOffset + 1] = static_cast<char>(nimble::DataType::Int32);

  NIMBLE_ASSERT_THROW(
      nimble::EncodingFactory{}.create(
          *pool_,
          malformed,
          [](uint32_t /*totalLength*/) -> void* { return nullptr; }),
      "positions child must use Uint32");
}

TEST_F(MainlyConstantV2EncodingTest, rejectsReadPastEnd) {
  const auto values = makeValues({7, 7, 9, 7});
  nimble::Buffer buffer{*pool_};
  auto encoding =
      nimble::test::Encoder<nimble::MainlyConstantV2Encoding<int64_t>>::
          createEncoding(buffer, values, [](uint32_t /*totalLength*/) -> void* {
            return nullptr;
          });
  nimble::Vector<int64_t> decoded{pool_.get(), values.size() + 1};

  NIMBLE_ASSERT_THROW(
      encoding->materialize(decoded.size(), decoded.data()),
      "Cannot materialize beyond MainlyConstantV2 row count");
}

} // namespace

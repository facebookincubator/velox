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

#include "velox/dwio/nimble/encodings/SliceEncoding.h"

#include <gtest/gtest.h>

#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class SliceEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("SliceEncodingTest");
    pool_ = rootPool_->addLeafChild("SliceEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> makeVector(std::initializer_list<T> values) {
    nimble::Vector<T> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(std::string_view encoded) {
    return nimble::EncodingFactory{}.create(
        *pool_, encoded, [](uint32_t /*totalLength*/) -> void* {
          return nullptr;
        });
  }

  template <typename T>
  std::vector<T> materialize(nimble::Encoding& encoding, uint32_t rowCount) {
    nimble::Vector<T> output{pool_.get(), rowCount};
    encoding.materialize(rowCount, output.data());
    return std::vector<T>(output.begin(), output.end());
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(SliceEncodingTest, wrapsWithoutSlicing) {
  const auto values = makeVector<int32_t>({10, 10, 12, 10, 14, 10});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {});

  // The payload still carries every source row, so it is larger than the
  // source rather than smaller: the slice was recorded, not performed.
  EXPECT_GT(wrapped.size(), encoded.size());

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  // The row count is the slice length, not the source's.
  EXPECT_EQ(encoding->rowCount(), 4);

  const std::vector<int32_t> expected{10, 12, 10, 14};
  EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, wrapsZeroOffset) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);

  const std::vector<int32_t> expected{10, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, wrapsFullRange) {
  const auto values = makeVector<int32_t>({10, 11, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);

  const std::vector<int32_t> expected{10, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, wrapsRle) {
  const auto values = makeVector<int32_t>({10, 10, 11, 11, 11, 12, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/5, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->rowCount(), 5);

  const std::vector<int32_t> expected{10, 11, 11, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 5), expected);
}

TEST_F(SliceEncodingTest, wrapsBoolRle) {
  // Bool is the null-stream case: StreamSlicer reads a sliced bool stream back
  // through skip() + materializeBoolsAsBits() before any consumer sees it, so
  // the wrapper has to honour both against the slice rather than the source.
  const auto values =
      makeVector<bool>({false, false, true, true, true, false, true});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<bool>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
  EXPECT_EQ(encoding->rowCount(), 4);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/4, &bits, /*begin=*/0);
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 1));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 2));
  EXPECT_FALSE(velox::bits::isBitSet(&bits, 3));
}

TEST_F(SliceEncodingTest, boolSkipIsRelativeToSlice) {
  const auto values =
      makeVector<bool>({false, false, true, true, true, false, true});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<bool>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  encoding->skip(2);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/2, &bits, /*begin=*/0);
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0));
  EXPECT_FALSE(velox::bits::isBitSet(&bits, 1));
}

TEST_F(SliceEncodingTest, resetReturnsToSliceStart) {
  const auto values = makeVector<int32_t>({10, 10, 12, 10, 14, 10});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);

  // reset() must return to the slice start, not to the source's row zero.
  const std::vector<int32_t> expected{10, 12, 10, 14};
  for (int pass = 0; pass < 2; ++pass) {
    EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected) << "pass " << pass;
    encoding->reset();
  }
}

TEST_F(SliceEncodingTest, skipsWithinSlice) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14, 15});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  encoding->skip(1);

  const std::vector<int32_t> expected{13, 14, 15};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

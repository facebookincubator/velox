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
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/SortedPositionSlots.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook::nimble;

TEST(EncodingUtilsTest, dataTypeSizeOneByte) {
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Int8));
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Uint8));
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Bool));
}

TEST(EncodingUtilsTest, dataTypeSizeTwoBytes) {
  EXPECT_EQ(2, detail::dataTypeSize(DataType::Int16));
  EXPECT_EQ(2, detail::dataTypeSize(DataType::Uint16));
}

TEST(EncodingUtilsTest, dataTypeSizeFourBytes) {
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Int32));
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Uint32));
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Float));
}

TEST(EncodingUtilsTest, dataTypeSizeEightBytes) {
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Int64));
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Uint64));
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Double));
}

TEST(EncodingUtilsTest, dataTypeSizeUnsupported) {
  EXPECT_THROW(detail::dataTypeSize(DataType::String), NimbleUserError);
  EXPECT_THROW(detail::dataTypeSize(DataType::Undefined), NimbleUserError);
}

TEST(EncodingUtilsTest, writeVarintString) {
  const std::vector<std::string> values{
      "",
      "a",
      std::string(127, 'x'),
      std::string(128, 'y'),
      std::string(130, 'z')};
  size_t totalSize{0};
  for (const auto& value : values) {
    totalSize += varint::varintSize(value.size()) + value.size();
  }
  std::vector<char> buffer(totalSize);
  char* writePos = buffer.data();

  for (const auto& value : values) {
    encoding::writeVarintString(value, writePos);
  }

  const char* readPos = buffer.data();
  for (const auto& value : values) {
    SCOPED_TRACE(testing::Message() << "size=" << value.size());
    EXPECT_EQ(varint::readVarint32(&readPos), value.size());
    EXPECT_EQ(std::string_view(readPos, value.size()), value);
    readPos += value.size();
  }
  EXPECT_EQ(readPos, writePos);
}

TEST(EncodingUtilsTest, copyPackedBits) {
  struct TestCase {
    const char* name;
    uint8_t bitWidth;
    uint32_t sourceValueCount;
    uint32_t sourceValueBase;
    uint64_t sourceBitOffset;
    uint64_t bitCount;
    std::vector<uint64_t> expected;
  };

  for (const auto& testCase : {
           TestCase{
               .name = "byteAligned",
               .bitWidth = 4,
               .sourceValueCount = 16,
               .sourceValueBase = 0,
               .sourceBitOffset = 16,
               .bitCount = 32,
               .expected = {4, 5, 6, 7, 8, 9, 10, 11}},
           TestCase{
               .name = "misaligned",
               .bitWidth = 5,
               .sourceValueCount = 12,
               .sourceValueBase = 3,
               .sourceBitOffset = 5,
               .bitCount = 35,
               .expected = {4, 5, 6, 7, 8, 9, 10}},
       }) {
    SCOPED_TRACE(testCase.name);

    std::array<char, 8> source{};
    std::array<char, 8> output{};
    FixedBitArray sourceBits{source.data(), testCase.bitWidth};
    for (uint32_t i = 0; i < testCase.sourceValueCount; ++i) {
      sourceBits.set(i, i + testCase.sourceValueBase);
    }

    encoding::copyPackedBits(
        {source.data(), source.size()},
        testCase.sourceBitOffset,
        testCase.bitCount,
        output.data());

    FixedBitArray outputBits{output.data(), testCase.bitWidth};
    std::vector<uint64_t> actual;
    actual.reserve(testCase.expected.size());
    for (uint32_t i = 0; i < testCase.expected.size(); ++i) {
      actual.push_back(outputBits.get(i));
    }
    EXPECT_EQ(actual, testCase.expected);
  }
}

TEST(EncodingUtilsTest, copyPackedBitsRejectsEmptyRange) {
  std::array<char, 8> source{};
  std::array<char, 8> output{};
  VELOX_ASSERT_THROW(
      encoding::copyPackedBits(
          {source.data(), source.size()},
          /*sourceBitOffset=*/0,
          /*bitCount=*/0,
          output.data()),
      "Cannot copy zero bits.");
}

class FindSortedPositionSlotsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  Vector<uint32_t> makePositions(std::initializer_list<uint32_t> values) {
    Vector<uint32_t> v{pool_.get()};
    v.insert(v.end(), values.begin(), values.end());
    return v;
  }

  // Trivial encoding is viewable, so it drives the view + readAt binary-search
  // branch of findSortedPositionSlots.
  std::string_view encodeAsView(const Vector<uint32_t>& values) {
    return test::Encoder<TrivialEncoding<uint32_t>>::encode(*buffer_, values);
  }

  // Varint encoding is NOT viewable, so it drives the materialize + binary-
  // search-on-decoded-values fallback branch.
  std::string_view encodeAsMaterialize(const Vector<uint32_t>& values) {
    return test::Encoder<VarintEncoding<uint32_t>>::encode(*buffer_, values);
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

// Exercises both branches with the same behavioral test suite so the view and
// materialize paths cannot silently diverge.
TEST_F(FindSortedPositionSlotsTest, matchesLowerBoundAcrossBranches) {
  const auto positions = makePositions({3, 7, 12, 18, 25, 30});
  const auto positionCount = static_cast<uint32_t>(positions.size());

  struct BranchCase {
    const char* name;
    std::string_view encoded;
  };
  const BranchCase branches[] = {
      {"viewBranch", encodeAsView(positions)},
      {"materializeBranch", encodeAsMaterialize(positions)},
  };

  struct RangeCase {
    const char* name;
    uint32_t lo;
    uint32_t hi;
    uint32_t expectedStart;
    uint32_t expectedEnd;
  };
  const RangeCase ranges[] = {
      // Strictly interior range: 12 and 18 fall inside [10, 20).
      {"interior", 10, 20, 2, 4},
      // lo hits the smallest position value; retains positions >= 3 and < 30.
      {"loEqualsFirst", 3, 30, 0, 5},
      // hi hits the largest position value; retains everything before it.
      {"hiEqualsLast", 0, 30, 0, 5},
      // hi past the max keeps every slot.
      {"fullRange", 0, 100, 0, positionCount},
      // Range that starts before the first value.
      {"loBelowMin", 0, 5, 0, 1},
      // Range that starts after the last value returns (positionCount,
      // positionCount).
      {"loAboveMax", 40, 50, positionCount, positionCount},
      // Empty range (lo == hi) collapses both slots to the same lower_bound.
      {"emptyAtValue", 12, 12, 2, 2},
      {"emptyBetweenValues", 20, 20, 4, 4},
      // Range that ends before any position keeps no slots.
      {"hiBelowMin", 0, 3, 0, 0},
      // Single-slot range around the last value.
      {"singleAtEnd", 25, 30, 4, 5},
  };

  for (const auto& branch : branches) {
    SCOPED_TRACE(branch.name);
    for (const auto& range : ranges) {
      SCOPED_TRACE(
          testing::Message()
          << range.name << " lo=" << range.lo << " hi=" << range.hi);
      const auto [slotStart, slotEnd] = detail::findSortedPositionSlots(
          branch.encoded,
          positionCount,
          range.lo,
          range.hi,
          *pool_,
          /*options=*/{});
      EXPECT_EQ(slotStart, range.expectedStart);
      EXPECT_EQ(slotEnd, range.expectedEnd);
    }
  }
}

TEST_F(FindSortedPositionSlotsTest, singlePositionArray) {
  const auto positions = makePositions({42});
  const auto viewEncoded = encodeAsView(positions);
  const auto materializeEncoded = encodeAsMaterialize(positions);

  for (const auto encoded : {viewEncoded, materializeEncoded}) {
    // Range strictly before the single value.
    auto slots = detail::findSortedPositionSlots(encoded, 1, 0, 42, *pool_, {});
    EXPECT_EQ(slots.slotStart, 0);
    EXPECT_EQ(slots.slotEnd, 0);

    // Range that spans the single value.
    slots = detail::findSortedPositionSlots(encoded, 1, 0, 100, *pool_, {});
    EXPECT_EQ(slots.slotStart, 0);
    EXPECT_EQ(slots.slotEnd, 1);

    // Range strictly after the single value.
    slots = detail::findSortedPositionSlots(encoded, 1, 43, 100, *pool_, {});
    EXPECT_EQ(slots.slotStart, 1);
    EXPECT_EQ(slots.slotEnd, 1);
  }
}

TEST_F(FindSortedPositionSlotsTest, positionCountSubsetSlicesLeadingPrefix) {
  // Callers may request a search over a leading prefix of the encoded
  // positions by passing positionCount < actual encoded row count. This mirrors
  // the sentinel-aware usage where SparseBool keeps the trailing sentinel out
  // of the search domain in some workflows.
  const auto positions = makePositions({1, 2, 3, 4, 5, 6, 7, 8});
  const auto viewEncoded = encodeAsView(positions);
  const auto materializeEncoded = encodeAsMaterialize(positions);

  for (const auto encoded : {viewEncoded, materializeEncoded}) {
    const auto [slotStart, slotEnd] = detail::findSortedPositionSlots(
        encoded,
        /*positionCount=*/4,
        /*valueStartOffset=*/2,
        /*valueEndOffset=*/10,
        *pool_,
        {});
    // Only positions {1, 2, 3, 4} are considered; slots [1..4) are >= 2 and
    // < 10.
    EXPECT_EQ(slotStart, 1);
    EXPECT_EQ(slotEnd, 4);
  }
}

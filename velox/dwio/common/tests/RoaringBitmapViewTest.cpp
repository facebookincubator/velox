/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/dwio/common/RoaringBitmapView.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::dwio::common {
namespace {

using Format = RoaringBitmapView::Format;

template <typename T>
void appendLittleEndian(std::string& data, T value) {
  for (size_t i = 0; i < sizeof(T); ++i) {
    data.push_back(
        static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF));
  }
}

// A container of a 32-bit RoaringBitmap, which holds the low 16 bits of the
// positions whose high 16 bits are 'key'.
struct TestContainer {
  enum class Kind { kArray, kBitset, kRun };

  uint16_t key{0};
  Kind kind{Kind::kArray};
  // Values of an array or bitset container.
  std::vector<uint16_t> values{};
  // Start and length minus one of each run of a run container.
  std::vector<std::pair<uint16_t, uint16_t>> runs{};

  uint32_t cardinality() const {
    if (kind != Kind::kRun) {
      return static_cast<uint32_t>(values.size());
    }
    uint32_t cardinality{0};
    for (const auto& [_, lengthMinusOne] : runs) {
      cardinality += lengthMinusOne + 1;
    }
    return cardinality;
  }
};

// Serializes 'containers' as a portable 32-bit RoaringBitmap, with the run
// cookie if any container is a run container. The view skips the offset
// section, so it is written as zeros.
std::string serialize32Bit(const std::vector<TestContainer>& containers) {
  const bool hasRuns = std::any_of(
      containers.begin(), containers.end(), [](const auto& container) {
        return container.kind == TestContainer::Kind::kRun;
      });
  const auto numContainers = static_cast<uint32_t>(containers.size());
  std::string data;
  if (hasRuns) {
    appendLittleEndian(data, uint32_t{12'347} | ((numContainers - 1) << 16));
    std::string runBitmap((numContainers + 7) / 8, '\0');
    for (uint32_t i = 0; i < numContainers; ++i) {
      if (containers[i].kind == TestContainer::Kind::kRun) {
        runBitmap[i / 8] = static_cast<char>(runBitmap[i / 8] | (1 << (i % 8)));
      }
    }
    data += runBitmap;
  } else {
    appendLittleEndian(data, uint32_t{12'346});
    appendLittleEndian(data, numContainers);
  }
  for (const auto& container : containers) {
    appendLittleEndian(data, container.key);
    appendLittleEndian(
        data, static_cast<uint16_t>(container.cardinality() - 1));
  }
  if (!hasRuns || numContainers >= 4) {
    for (uint32_t i = 0; i < numContainers; ++i) {
      appendLittleEndian(data, uint32_t{0});
    }
  }
  for (const auto& container : containers) {
    switch (container.kind) {
      case TestContainer::Kind::kArray:
        for (auto value : container.values) {
          appendLittleEndian(data, value);
        }
        break;
      case TestContainer::Kind::kBitset: {
        std::vector<uint64_t> words(1'024);
        for (auto value : container.values) {
          words[value / 64] |= uint64_t{1} << (value % 64);
        }
        for (auto word : words) {
          appendLittleEndian(data, word);
        }
        break;
      }
      case TestContainer::Kind::kRun:
        appendLittleEndian(data, static_cast<uint16_t>(container.runs.size()));
        for (const auto& [start, lengthMinusOne] : container.runs) {
          appendLittleEndian(data, start);
          appendLittleEndian(data, lengthMinusOne);
        }
        break;
    }
  }
  return data;
}

// Returns the positions the view sets for consecutive ranges of 'batchSize'
// rows over [0, end), checking exhausted() after each range.
std::vector<int64_t> applyInBatches(
    RoaringBitmapView& view,
    int64_t end,
    int64_t batchSize,
    int64_t lastPosition) {
  std::vector<int64_t> positions;
  std::vector<uint8_t> bitmap(bits::nbytes(static_cast<uint64_t>(batchSize)));
  for (int64_t begin = 0; begin < end; begin += batchSize) {
    std::fill(bitmap.begin(), bitmap.end(), 0);
    view.apply(begin, begin + batchSize, bitmap.data());
    for (int64_t i = 0; i < batchSize; ++i) {
      if (bits::isBitSet(bitmap.data(), i)) {
        positions.push_back(begin + i);
      }
    }
    EXPECT_EQ(view.exhausted(), begin + batchSize > lastPosition)
        << "range starting at " << begin;
  }
  return positions;
}

TEST(RoaringBitmapViewTest, containerKinds) {
  std::vector<uint16_t> evenValues;
  for (uint16_t value = 0; value < 10'000; value += 2) {
    evenValues.push_back(value);
  }
  const auto data = serialize32Bit({
      {.key = 0, .kind = TestContainer::Kind::kArray, .values = {1, 5, 70}},
      {.key = 1, .kind = TestContainer::Kind::kBitset, .values = evenValues},
      {.key = 2,
       .kind = TestContainer::Kind::kRun,
       .runs = {{10, 4}, {100, 0}}},
  });

  std::vector<int64_t> expected{1, 5, 70};
  for (auto value : evenValues) {
    expected.push_back(65'536 + value);
  }
  for (int64_t value = 10; value <= 14; ++value) {
    expected.push_back(2 * 65'536 + value);
  }
  expected.push_back(2 * 65'536 + 100);

  RoaringBitmapView view(data, Format::k32Bit);
  EXPECT_EQ(view.positions(), expected);
  EXPECT_EQ(view.cardinality(), expected.size());
  EXPECT_FALSE(view.exhausted());

  // Ranges of 1'000 rows straddle every container boundary.
  EXPECT_EQ(applyInBatches(view, 3 * 65'536, 1'000, expected.back()), expected);
}

TEST(RoaringBitmapViewTest, roaring64Groups) {
  constexpr int64_t kGroupBase = int64_t{1} << 32;
  std::string data;
  appendLittleEndian(data, uint64_t{2});
  appendLittleEndian(data, uint32_t{0});
  data += serialize32Bit(
      {{.key = 0, .kind = TestContainer::Kind::kArray, .values = {3}}});
  appendLittleEndian(data, uint32_t{1});
  data += serialize32Bit(
      {{.key = 0, .kind = TestContainer::Kind::kRun, .runs = {{0, 2}}}});

  RoaringBitmapView view(data, Format::k64Bit);
  EXPECT_THAT(
      view.positions(),
      testing::ElementsAre(3, kGroupBase, kGroupBase + 1, kGroupBase + 2));
  EXPECT_EQ(view.cardinality(), uint64_t{4});

  std::vector<uint8_t> bitmap(1);
  view.apply(0, 8, bitmap.data());
  EXPECT_EQ(bitmap[0], 0b1000);
  EXPECT_FALSE(view.exhausted());
  bitmap[0] = 0;
  view.apply(kGroupBase - 2, kGroupBase + 6, bitmap.data());
  EXPECT_EQ(bitmap[0], 0b11100);
  EXPECT_TRUE(view.exhausted());
}

TEST(RoaringBitmapViewTest, empty) {
  RoaringBitmapView defaultView;
  EXPECT_TRUE(defaultView.exhausted());
  EXPECT_EQ(defaultView.cardinality(), uint64_t{0});
  EXPECT_THAT(defaultView.positions(), testing::IsEmpty());

  const auto data = serialize32Bit({});
  RoaringBitmapView view(data, Format::k32Bit);
  EXPECT_TRUE(view.exhausted());
  EXPECT_EQ(view.cardinality(), uint64_t{0});
  std::vector<uint8_t> bitmap(1);
  view.apply(0, 8, bitmap.data());
  EXPECT_EQ(bitmap[0], 0);
}

TEST(RoaringBitmapViewTest, corruptBitmaps) {
  struct TestParam {
    std::string name;
    std::string data;
    Format format;
    std::string error;

    std::string debugString() const {
      return fmt::format(
          "{}, {}-bit", name, format == Format::k32Bit ? 32 : 64);
    }
  };

  std::string tooManyContainers;
  appendLittleEndian(tooManyContainers, uint32_t{12'346});
  appendLittleEndian(tooManyContainers, uint32_t{70'000});

  std::string truncatedMetadata;
  appendLittleEndian(truncatedMetadata, uint32_t{12'346});
  appendLittleEndian(truncatedMetadata, uint32_t{2});
  appendLittleEndian(truncatedMetadata, uint32_t{0});

  std::string unknownCookie;
  appendLittleEndian(unknownCookie, uint64_t{1});

  std::string groupKeyTooLarge;
  appendLittleEndian(groupKeyTooLarge, uint64_t{1});
  appendLittleEndian(groupKeyTooLarge, RoaringBitmapView::kMaxGroupKey + 1);
  groupKeyTooLarge += serialize32Bit({});

  // Two groups with the same high bits, each with a container for key 0.
  std::string repeatedGroup;
  appendLittleEndian(repeatedGroup, uint64_t{2});
  for (int i = 0; i < 2; ++i) {
    appendLittleEndian(repeatedGroup, uint32_t{0});
    repeatedGroup += serialize32Bit(
        {{.key = 0, .kind = TestContainer::Kind::kArray, .values = {3}}});
  }

  const std::vector<TestParam> testSettings{
      {"too many containers",
       tooManyContainers,
       Format::k32Bit,
       "Roaring bitmap container count exceeds the 16-bit key space: 70000"},
      {"run past its container",
       serialize32Bit(
           {{.key = 0,
             .kind = TestContainer::Kind::kRun,
             .runs = {{65'535, 1}}}}),
       Format::k32Bit,
       "Roaring run extends past its container."},
      {"repeated container key",
       serialize32Bit({
           {.key = 0, .kind = TestContainer::Kind::kArray, .values = {1, 5}},
           {.key = 0, .kind = TestContainer::Kind::kArray, .values = {3}},
       }),
       Format::k32Bit,
       "Roaring bitmap has two containers for the same key."},
      {"truncated metadata",
       truncatedMetadata,
       Format::k32Bit,
       "Truncated container metadata."},
      {"unknown cookie",
       unknownCookie,
       Format::k32Bit,
       "Unknown roaring bitmap cookie: 1. Expected 12346 or 12347."},
      {"truncated group count",
       std::string(4, '\0'),
       Format::k64Bit,
       "Truncated Roaring64Bitmap group count."},
      {"group key too large",
       groupKeyTooLarge,
       Format::k64Bit,
       "Roaring64Bitmap group key exceeds the maximum the Iceberg "
       "deletion-vector format can represent: 2147483647"},
      {"repeated group",
       repeatedGroup,
       Format::k64Bit,
       "Roaring bitmap has two containers for the same key."},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    VELOX_ASSERT_THROW(
        RoaringBitmapView(testData.data, testData.format), testData.error);
  }
}

} // namespace
} // namespace facebook::velox::dwio::common

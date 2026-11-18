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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/StringPageArena.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble::test {
namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;

// Returns a writer for StringPageArena::write() that writes 'value'.
auto writerOf(std::string_view value) {
  return [value](char* output) {
    std::memcpy(output, value.data(), value.size());
    return value.size();
  };
}

// Owns the pages that a StringPageArena takes from its factory, as the string
// buffers of an output vector do, and records the size of each.
class PageOwner {
 public:
  // Returns a factory that allocates pages that this object owns.
  StringPageArena::PageFactory factory() {
    return [this](uint32_t bytes) -> void* {
      pageBytes_.emplace_back(bytes);
      return pages_.emplace_back(std::make_unique<char[]>(bytes)).get();
    };
  }

  // Returns the sizes of the pages that the factory allocated, in order.
  const std::vector<uint32_t>& pageBytes() const {
    return pageBytes_;
  }

 private:
  std::vector<uint32_t> pageBytes_;
  std::vector<std::unique_ptr<char[]>> pages_;
};

class StringPageArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(pool_->trackUsage());
  }

  // Owns the pools, so that the tests can count what the arena allocates,
  // which a pool from deprecatedAddDefaultLeafMemoryPool() does not count.
  velox::memory::MemoryManager manager_;
  const std::shared_ptr<velox::memory::MemoryPool> rootPool_{
      manager_.addRootPool()};
  const std::shared_ptr<velox::memory::MemoryPool> pool_{
      rootPool_->addLeafChild("string-page-arena")};
};

TEST_F(StringPageArenaTest, writeStringFillsPageThenStartsNewOne) {
  StringPageArena arena{pool_.get(), /*minPageBytes=*/64, /*inlinePage=*/{}};
  EXPECT_EQ(pool_->usedBytes(), 0);

  // Six 10-byte strings take 60 of the first page's 64 bytes, one after
  // another.
  std::vector<std::string> values;
  std::vector<std::string_view> copies;
  for (char letter = 'a'; letter < 'g'; ++letter) {
    values.emplace_back(10, letter);
    copies.emplace_back(arena.writeString(values.back()));
  }
  for (size_t i = 1; i < copies.size(); ++i) {
    EXPECT_EQ(copies[i].data(), copies[i - 1].data() + 10);
  }
  const auto onePageBytes = pool_->usedBytes();
  EXPECT_GT(onePageBytes, 0);

  // A seventh does not fit in the last 4 bytes, so it starts a second page.
  values.emplace_back(10, 'g');
  copies.emplace_back(arena.writeString(values.back()));
  EXPECT_EQ(pool_->usedBytes(), 2 * onePageBytes);
  EXPECT_THAT(copies, ElementsAreArray(values));
}

TEST_F(StringPageArenaTest, writeStringLargerThanPageGetsItsOwnPage) {
  PageOwner pageOwner;
  StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};
  const std::string small(10, 's');
  const std::string large(100, 'l');
  EXPECT_EQ(arena.writeString(small), small);
  EXPECT_EQ(arena.writeString(large), large);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 100));
}

TEST_F(StringPageArenaTest, writeFitsInPlace) {
  StringPageArena arena{pool_.get(), /*minPageBytes=*/64, /*inlinePage=*/{}};

  // Each write has room for 20 bytes in the page and writes 10 there, so each
  // string starts where the one before ends.
  const std::vector<std::string> values{
      std::string(10, 'a'), std::string(10, 'b'), std::string(10, 'c')};
  std::vector<std::string_view> copies;
  for (const auto& value : values) {
    char* writerOutput{nullptr};
    copies.emplace_back(arena.write(/*maxBytes=*/20, [&](char* output) {
      writerOutput = output;
      return writerOf(value)(output);
    }));
    EXPECT_EQ(copies.back(), value);
    EXPECT_EQ(copies.back().data(), writerOutput);
  }
  EXPECT_EQ(copies[1].data(), copies[0].data() + 10);
  EXPECT_EQ(copies[2].data(), copies[1].data() + 10);
}

TEST_F(StringPageArenaTest, writeTakesNextPageWhenBoundDoesNotFit) {
  PageOwner pageOwner;
  StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};
  arena.writeString(std::string(54, 'a'));

  // A bound of 20 bytes does not fit in the page's last 10, so the write goes
  // to a new page, though the 8 bytes written would fit.
  char* writerOutput{nullptr};
  const auto copy = arena.write(/*maxBytes=*/20, [&](char* output) {
    writerOutput = output;
    return writerOf("abcdefgh")(output);
  });
  EXPECT_EQ(copy, "abcdefgh");
  EXPECT_EQ(copy.data(), writerOutput);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 64));

  // The next string goes right after the 8 bytes in the new page.
  EXPECT_EQ(arena.writeString("xyz").data(), copy.data() + 8);
}

TEST_F(StringPageArenaTest, writeWithBoundAbovePageTakesPageOfBound) {
  PageOwner pageOwner;
  StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};

  // A bound of 1,000 bytes takes a page of 1,000 bytes. The 990 bytes that
  // the write leaves hold the next string, but not the next bound of 1,000
  // bytes, which takes another page.
  const std::string value(10, 'v');
  const auto first = arena.write(/*maxBytes=*/1'000, writerOf(value));
  EXPECT_EQ(arena.writeString(value).data(), first.data() + 10);
  EXPECT_EQ(arena.write(/*maxBytes=*/1'000, writerOf(value)), value);
  EXPECT_EQ(first, value);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(1'000, 1'000));
}

TEST_F(StringPageArenaTest, inlinePageAllocatesNothing) {
  std::array<char, 32> inlinePage{};
  StringPageArena arena{pool_.get(), /*minPageBytes=*/64, inlinePage};

  // Three 10-byte strings fit in the 32-byte inline page.
  const auto first = arena.writeString("0123456789");
  const auto second = arena.write(/*maxBytes=*/10, writerOf("abcdefghij"));
  const auto third = arena.writeString("ABCDEFGHIJ");
  EXPECT_EQ(first.data(), inlinePage.data());
  EXPECT_EQ(second.data(), inlinePage.data() + 10);
  EXPECT_EQ(third.data(), inlinePage.data() + 20);
  EXPECT_EQ(pool_->usedBytes(), 0);

  // A fourth does not fit in the last 2 bytes, so it takes a page from the
  // pool.
  const auto fourth = arena.writeString("klmnopqrst");
  EXPECT_GT(pool_->usedBytes(), 0);
  EXPECT_EQ(first, "0123456789");
  EXPECT_EQ(second, "abcdefghij");
  EXPECT_EQ(third, "ABCDEFGHIJ");
  EXPECT_EQ(fourth, "klmnopqrst");
}

TEST_F(StringPageArenaTest, resetReusesPages) {
  PageOwner pageOwner;
  StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};

  // Two 40-byte strings do not fit in one 64-byte page, so each takes a page.
  const std::string value(40, 'v');
  const auto first = arena.writeString(value);
  arena.writeString(value);
  arena.writeString(value);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 64, 64));

  // The same writes after reset() take the same pages in the same order.
  arena.reset();
  EXPECT_EQ(arena.writeString(value).data(), first.data());
  arena.writeString(value);
  arena.writeString(value);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 64, 64));

  // A string larger than the second page replaces it with a page of its size,
  // which the next reset() keeps.
  arena.reset();
  arena.writeString(value);
  const auto large = arena.writeString(std::string(100, 'l'));
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 64, 64, 100));
  arena.reset();
  arena.writeString(value);
  EXPECT_EQ(arena.writeString(std::string(100, 'm')).data(), large.data());
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 64, 64, 100));
}

TEST_F(StringPageArenaTest, resetKeepsPoolPages) {
  const std::string value(40, 'v');
  {
    StringPageArena arena{pool_.get(), /*minPageBytes=*/64, /*inlinePage=*/{}};
    arena.writeString(value);
    arena.writeString(value);
    const auto usedBytes = pool_->usedBytes();
    EXPECT_GT(usedBytes, 0);

    // The same writes after reset() allocate nothing.
    arena.reset();
    arena.writeString(value);
    arena.writeString(value);
    EXPECT_EQ(pool_->usedBytes(), usedBytes);
  }

  // The arena frees its pages on destruction.
  EXPECT_EQ(pool_->usedBytes(), 0);
}

TEST_F(StringPageArenaTest, factoryPagesOutliveArena) {
  PageOwner pageOwner;
  const std::string large(100, 'l');
  std::string_view smallCopy;
  std::string_view largeCopy;
  {
    StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};
    smallCopy = arena.writeString("small");
    largeCopy = arena.writeString(large);
  }
  EXPECT_EQ(smallCopy, "small");
  EXPECT_EQ(largeCopy, large);
  EXPECT_THAT(pageOwner.pageBytes(), ElementsAre(64, 100));
}

TEST_F(StringPageArenaTest, emptyFactoryFailsFirstPage) {
  // An arena without a factory works until a write needs a page.
  StringPageArena arena{StringPageArena::PageFactory{}, /*minPageBytes=*/64};
  EXPECT_TRUE(arena.writeString("").empty());
  NIMBLE_ASSERT_THROW(
      arena.writeString("value"), "String page factory is empty.");
}

TEST_F(StringPageArenaTest, emptyStringsTakeNoPage) {
  StringPageArena arena{pool_.get(), /*minPageBytes=*/64, /*inlinePage=*/{}};
  EXPECT_TRUE(arena.writeString("").empty());

  bool writerCalled{false};
  const auto copy = arena.write(/*maxBytes=*/0, [&](char* /*output*/) {
    writerCalled = true;
    return size_t{0};
  });
  EXPECT_TRUE(copy.empty());
  EXPECT_FALSE(writerCalled);
  EXPECT_EQ(pool_->usedBytes(), 0);
}

TEST_F(StringPageArenaTest, writeRejectsSizeAboveBound) {
  StringPageArena arena{pool_.get(), /*minPageBytes=*/64, /*inlinePage=*/{}};

  // The writer reports one byte more than the bound and writes nothing.
  NIMBLE_ASSERT_THROW(
      arena.write(
          /*maxBytes=*/10, [](char* /*output*/) { return size_t{11}; }),
      "Write exceeds its bound.");
}

TEST_F(StringPageArenaTest, writeRejectsBoundAboveFactoryPageLimit) {
  PageOwner pageOwner;
  StringPageArena arena{pageOwner.factory(), /*minPageBytes=*/64};

  // The factory takes a uint32 size, so a larger bound fails before the
  // writer runs.
  bool writerCalled{false};
  NIMBLE_ASSERT_FILE_THROW(
      arena.write(
          /*maxBytes=*/size_t{std::numeric_limits<uint32_t>::max()} + 1,
          [&](char* /*output*/) {
            writerCalled = true;
            return size_t{0};
          }),
      "String page exceeds the supported size.");
  EXPECT_FALSE(writerCalled);
  EXPECT_TRUE(pageOwner.pageBytes().empty());
}

TEST_F(StringPageArenaTest, rejectsInvalidArguments) {
  {
    NIMBLE_ASSERT_THROW(
        StringPageArena(nullptr, /*minPageBytes=*/64, /*inlinePage=*/{}),
        "Memory pool must not be null.");
  }
  {
    NIMBLE_ASSERT_THROW(
        StringPageArena(
            pool_.get(), /*minPageBytes=*/size_t{0}, /*inlinePage=*/{}),
        "Minimum page size must be positive.");
  }
  {
    NIMBLE_ASSERT_THROW(
        StringPageArena(
            StringPageArena::PageFactory{}, /*minPageBytes=*/size_t{0}),
        "Minimum page size must be positive.");
  }
  {
    StringPageArena arena{
        [](uint32_t /*bytes*/) -> void* { return nullptr; },
        /*minPageBytes=*/64};
    NIMBLE_ASSERT_THROW(
        arena.writeString("value"), "String page factory returned null.");
  }
}

} // namespace
} // namespace facebook::nimble::test

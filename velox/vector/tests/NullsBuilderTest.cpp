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
#include "velox/vector/NullsBuilder.h"

#include <gtest/gtest.h>

#include "velox/common/base/Nulls.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox {
namespace {

constexpr vector_size_t kSize = 8;

class NullsBuilderTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  // A bitmap of kSize rows with the given rows null.
  BufferPtr makeNulls(const std::vector<vector_size_t>& nullRows) {
    auto buffer =
        AlignedBuffer::allocate<bool>(kSize, pool_.get(), bits::kNotNull);
    auto* raw = buffer->asMutable<uint64_t>();
    for (auto row : nullRows) {
      bits::setNull(raw, row, true);
    }
    return buffer;
  }

  std::vector<vector_size_t> nullRowsOf(const BufferPtr& nulls) {
    std::vector<vector_size_t> rows;
    const auto* raw = nulls->as<uint64_t>();
    for (vector_size_t row = 0; row < kSize; ++row) {
      if (bits::isBitNull(raw, row)) {
        rows.push_back(row);
      }
    }
    return rows;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(NullsBuilderTest, nothingAdded) {
  NullsBuilder builder(kSize, pool_.get());
  EXPECT_EQ(builder.build(), nullptr);
}

TEST_F(NullsBuilderTest, setNullMarksRows) {
  NullsBuilder builder(kSize, pool_.get());
  builder.setNull(1);
  builder.setNull(5);

  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_EQ(nullRowsOf(nulls), (std::vector<vector_size_t>{1, 5}));
}

TEST_F(NullsBuilderTest, setNullIsIdempotent) {
  NullsBuilder builder(kSize, pool_.get());
  builder.setNull(3);
  builder.setNull(3);

  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_EQ(nullRowsOf(nulls), (std::vector<vector_size_t>{3}));
}

TEST_F(NullsBuilderTest, addNullsIgnoresNullptr) {
  NullsBuilder builder(kSize, pool_.get());
  builder.addNulls(nullptr);
  builder.addNulls(nullptr);
  EXPECT_EQ(builder.build(), nullptr);
}

TEST_F(NullsBuilderTest, addNullsUnionsBitmaps) {
  auto first = makeNulls({1, 4});
  auto second = makeNulls({4, 6});

  NullsBuilder builder(kSize, pool_.get());
  builder.addNulls(first->as<uint64_t>());
  builder.addNulls(second->as<uint64_t>());

  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_EQ(nullRowsOf(nulls), (std::vector<vector_size_t>{1, 4, 6}));
}

TEST_F(NullsBuilderTest, addNullsSkipsNullptrAmongBitmaps) {
  auto first = makeNulls({2});

  NullsBuilder builder(kSize, pool_.get());
  builder.addNulls(nullptr);
  builder.addNulls(first->as<uint64_t>());
  builder.addNulls(nullptr);

  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_EQ(nullRowsOf(nulls), (std::vector<vector_size_t>{2}));
}

TEST_F(NullsBuilderTest, addNullsWithoutAnyNullRow) {
  auto none = makeNulls({});

  NullsBuilder builder(kSize, pool_.get());
  builder.addNulls(none->as<uint64_t>());

  // A bitmap with no nulls still allocates: the caller passed one.
  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_TRUE(nullRowsOf(nulls).empty());
}

TEST_F(NullsBuilderTest, setNullAndAddNullsShareTheBuffer) {
  auto first = makeNulls({3});

  NullsBuilder builder(kSize, pool_.get());
  builder.setNull(0);
  builder.addNulls(first->as<uint64_t>());

  auto nulls = builder.build();
  ASSERT_NE(nulls, nullptr);
  EXPECT_EQ(nullRowsOf(nulls), (std::vector<vector_size_t>{0, 3}));
}

} // namespace
} // namespace facebook::velox

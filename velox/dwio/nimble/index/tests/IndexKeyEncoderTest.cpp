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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index::test {
namespace {

class IndexKeyEncoderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("IndexKeyEncoderTest");
    leafPool_ = rootPool_->addLeafChild("leaf");
  }

  velox::RowTypePtr rowType() const {
    return velox::ROW({{"id", velox::INTEGER()}});
  }

  velox::RowVectorPtr makeBatch() const {
    constexpr int32_t kNumRows{20};
    auto ids =
        velox::BaseVector::create(velox::INTEGER(), kNumRows, leafPool_.get());
    auto* flatIds = ids->asFlatVector<int32_t>();
    for (int32_t i = 0; i < kNumRows; ++i) {
      flatIds->set(i, i);
    }
    return std::make_shared<velox::RowVector>(
        leafPool_.get(),
        rowType(),
        nullptr,
        kNumRows,
        std::vector<velox::VectorPtr>{ids});
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(IndexKeyEncoderTest, encodeBatchInSortOrder) {
  auto encoder = createNimbleIndexKeyEncoder(
      {"id"}, rowType(), {SortOrder{.ascending = true}}, leafPool_.get());
  Buffer buffer{*leafPool_};
  std::vector<std::string_view> keys;
  encoder->encode(makeBatch(), keys, [&buffer](size_t size) {
    return buffer.reserve(size);
  });

  ASSERT_EQ(keys.size(), 20);
  for (size_t i = 1; i < keys.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_LT(keys[i - 1], keys[i]);
  }
}

TEST_F(IndexKeyEncoderTest, rejectMismatchedSortOrders) {
  NIMBLE_ASSERT_THROW(
      createNimbleIndexKeyEncoder({"id"}, rowType(), {}, leafPool_.get()),
      "sortOrders and columns must have the same size");
}

} // namespace
} // namespace facebook::nimble::index::test

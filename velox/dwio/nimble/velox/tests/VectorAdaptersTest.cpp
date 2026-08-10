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

#include "velox/dwio/nimble/velox/VectorAdapters.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook;

class VectorAdaptersTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::initialize(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
    vectorMaker_ =
        std::make_unique<velox::test::VectorMaker>(this->pool_.get());
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

// DecodedAdapter tests

TEST_F(VectorAdaptersTest, decodedAdapterFlatInt32NoNulls) {
  auto vector = vectorMaker_->flatVector<int32_t>({1, 2, 3, 4, 5});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  for (velox::vector_size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(adapter.valueAt(i), i + 1);
    EXPECT_EQ(adapter.index(i), i);
  }
}

TEST_F(VectorAdaptersTest, decodedAdapterFlatInt32WithNulls) {
  auto vector = vectorMaker_->flatVectorNullable<int32_t>(
      {1, std::nullopt, 3, std::nullopt, 5});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_TRUE(adapter.hasNulls());
  EXPECT_FALSE(adapter.isNullAt(0));
  EXPECT_TRUE(adapter.isNullAt(1));
  EXPECT_FALSE(adapter.isNullAt(2));
  EXPECT_TRUE(adapter.isNullAt(3));
  EXPECT_FALSE(adapter.isNullAt(4));

  EXPECT_EQ(adapter.valueAt(0), 1);
  EXPECT_EQ(adapter.valueAt(2), 3);
  EXPECT_EQ(adapter.valueAt(4), 5);
}

TEST_F(VectorAdaptersTest, decodedAdapterInt64) {
  auto vector = vectorMaker_->flatVector<int64_t>({100L, 200L, 300L});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int64_t> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_EQ(adapter.valueAt(0), 100L);
  EXPECT_EQ(adapter.valueAt(1), 200L);
  EXPECT_EQ(adapter.valueAt(2), 300L);
}

TEST_F(VectorAdaptersTest, decodedAdapterDouble) {
  auto vector = vectorMaker_->flatVector<double>({1.5, 2.5, 3.5});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<double> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_DOUBLE_EQ(adapter.valueAt(0), 1.5);
  EXPECT_DOUBLE_EQ(adapter.valueAt(1), 2.5);
  EXPECT_DOUBLE_EQ(adapter.valueAt(2), 3.5);
}

TEST_F(VectorAdaptersTest, decodedAdapterBool) {
  auto vector = vectorMaker_->flatVector<bool>({true, false, true, false});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<bool> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_TRUE(adapter.valueAt(0));
  EXPECT_FALSE(adapter.valueAt(1));
  EXPECT_TRUE(adapter.valueAt(2));
  EXPECT_FALSE(adapter.valueAt(3));
}

TEST_F(VectorAdaptersTest, decodedAdapterBoolWithNulls) {
  auto vector = vectorMaker_->flatVectorNullable<bool>(
      {true, std::nullopt, false, std::nullopt});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<bool> adapter(decoded);

  EXPECT_TRUE(adapter.hasNulls());
  EXPECT_FALSE(adapter.isNullAt(0));
  EXPECT_TRUE(adapter.isNullAt(1));
  EXPECT_FALSE(adapter.isNullAt(2));
  EXPECT_TRUE(adapter.isNullAt(3));

  EXPECT_TRUE(adapter.valueAt(0));
  EXPECT_FALSE(adapter.valueAt(2));
}

TEST_F(VectorAdaptersTest, decodedAdapterDictionaryVector) {
  auto baseVector = vectorMaker_->flatVector<int32_t>({100, 200, 300});

  auto indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(5, pool_.get());
  auto rawIndices = indices->asMutable<velox::vector_size_t>();
  rawIndices[0] = 0;
  rawIndices[1] = 1;
  rawIndices[2] = 2;
  rawIndices[3] = 0;
  rawIndices[4] = 1;

  auto dictVector =
      velox::BaseVector::wrapInDictionary(nullptr, indices, 5, baseVector);

  velox::DecodedVector decoded;
  decoded.decode(*dictVector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_EQ(adapter.valueAt(0), 100);
  EXPECT_EQ(adapter.valueAt(1), 200);
  EXPECT_EQ(adapter.valueAt(2), 300);
  EXPECT_EQ(adapter.valueAt(3), 100);
  EXPECT_EQ(adapter.valueAt(4), 200);

  EXPECT_EQ(adapter.index(0), 0);
  EXPECT_EQ(adapter.index(1), 1);
  EXPECT_EQ(adapter.index(2), 2);
  EXPECT_EQ(adapter.index(3), 0);
  EXPECT_EQ(adapter.index(4), 1);
}

TEST_F(VectorAdaptersTest, decodedAdapterDictionaryWithNulls) {
  auto baseVector = vectorMaker_->flatVector<int32_t>({100, 200, 300});

  auto indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(5, pool_.get());
  auto rawIndices = indices->asMutable<velox::vector_size_t>();
  rawIndices[0] = 0;
  rawIndices[1] = 1;
  rawIndices[2] = 2;
  rawIndices[3] = 0;
  rawIndices[4] = 1;

  auto nulls = velox::allocateNulls(5, pool_.get());
  auto rawNulls = nulls->asMutable<uint64_t>();
  velox::bits::fillBits(rawNulls, 0, 5, true);
  velox::bits::setNull(rawNulls, 1);
  velox::bits::setNull(rawNulls, 3);

  auto dictVector =
      velox::BaseVector::wrapInDictionary(nulls, indices, 5, baseVector);

  velox::DecodedVector decoded;
  decoded.decode(*dictVector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_TRUE(adapter.hasNulls());
  EXPECT_FALSE(adapter.isNullAt(0));
  EXPECT_TRUE(adapter.isNullAt(1));
  EXPECT_FALSE(adapter.isNullAt(2));
  EXPECT_TRUE(adapter.isNullAt(3));
  EXPECT_FALSE(adapter.isNullAt(4));

  EXPECT_EQ(adapter.valueAt(0), 100);
  EXPECT_EQ(adapter.valueAt(2), 300);
  EXPECT_EQ(adapter.valueAt(4), 200);
}

TEST_F(VectorAdaptersTest, decodedAdapterIgnoreNulls) {
  auto vector = vectorMaker_->flatVectorNullable<int32_t>(
      {1, std::nullopt, 3, std::nullopt, 5});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int32_t, true> adapter(decoded);

  // IgnoreNulls=true means hasNulls() and isNullAt() always return false
  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_FALSE(adapter.isNullAt(0));
  EXPECT_FALSE(adapter.isNullAt(1));
  EXPECT_FALSE(adapter.isNullAt(2));
  EXPECT_FALSE(adapter.isNullAt(3));
  EXPECT_FALSE(adapter.isNullAt(4));
}

TEST_F(VectorAdaptersTest, decodedAdapterDefaultTemplateParam) {
  auto vector = vectorMaker_->flatVector<int32_t>({1, 2, 3});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_EQ(adapter.index(0), 0);
  EXPECT_EQ(adapter.index(1), 1);
  EXPECT_EQ(adapter.index(2), 2);
}

TEST_F(VectorAdaptersTest, decodedAdapterConstantVector) {
  auto vector =
      velox::BaseVector::createConstant(velox::INTEGER(), 42, 5, pool_.get());

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  for (velox::vector_size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(adapter.valueAt(i), 42);
  }
}

TEST_F(VectorAdaptersTest, decodedAdapterConstantNullVector) {
  auto vector =
      velox::BaseVector::createNullConstant(velox::INTEGER(), 5, pool_.get());

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<int32_t> adapter(decoded);

  EXPECT_TRUE(adapter.hasNulls());
  for (velox::vector_size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(adapter.isNullAt(i));
  }
}

TEST_F(VectorAdaptersTest, decodedAdapterStringView) {
  auto vector = vectorMaker_->flatVector<velox::StringView>(
      {velox::StringView("hello"),
       velox::StringView("world"),
       velox::StringView("test")});

  velox::DecodedVector decoded;
  decoded.decode(*vector);

  nimble::DecodedAdapter<velox::StringView> adapter(decoded);

  EXPECT_FALSE(adapter.hasNulls());
  EXPECT_EQ(adapter.valueAt(0), velox::StringView("hello"));
  EXPECT_EQ(adapter.valueAt(1), velox::StringView("world"));
  EXPECT_EQ(adapter.valueAt(2), velox::StringView("test"));
}

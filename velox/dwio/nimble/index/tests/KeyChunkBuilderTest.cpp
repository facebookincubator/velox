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
#include "velox/dwio/nimble/index/KeyChunkBuilder.h"

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/KeyReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::index::test {
namespace {

// Orders keys in reverse bytewise order.
class ReverseKeyComparator final : public KeyComparator {
 public:
  int compare(std::string_view lhs, std::string_view rhs) const override {
    return rhs.compare(lhs);
  }
};

class KeyChunkBuilderTest : public testing::Test,
                            public velox::test::VectorTestBase {
 public:
  static void SetUpTestSuite() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::testingSetInstance({});
    }
  }
};

TEST_F(KeyChunkBuilderTest, flat) {
  const std::vector<std::string> keyColumns{"key"};
  const std::vector<SortOrder> sortOrders{{.ascending = true}};
  const auto type = velox::ROW({{"key", velox::BIGINT()}});
  auto builder = createFlatKeyChunkBuilder(
      createNimbleIndexKeyEncoder(keyColumns, type, sortOrders, pool()),
      EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
      /*enforceKeyOrder=*/true,
      /*noDuplicateKey=*/false,
      /*keyComparator=*/nullptr,
      pool());

  builder->append(makeRowVector({makeFlatVector<int64_t>({-2, 0, 3})}));
  ASSERT_EQ(builder->size(), 3);
  std::vector<std::string> expected;
  expected.reserve(builder->size());
  for (size_t row = 0; row < builder->size(); ++row) {
    expected.push_back(builder->keyAt(row));
  }

  Buffer buffer{*pool()};
  const auto encoded =
      builder->encode(0, static_cast<uint32_t>(builder->size()), buffer);
  auto reader = createFlatKeyReader(
      encoded,
      [](uint32_t) -> void* { return nullptr; },
      /*comparator=*/nullptr,
      pool());
  EXPECT_NE(dynamic_cast<FlatKeyReader*>(reader.get()), nullptr);
  EXPECT_EQ(reader->materialize(0, reader->rowCount()), expected);
}

TEST_F(KeyChunkBuilderTest, flatWithKeyComparator) {
  const ReverseKeyComparator comparator;
  const std::vector<std::string> keyColumns{"key"};
  const std::vector<SortOrder> sortOrders{{.ascending = true}};
  const auto type = velox::ROW({{"key", velox::BIGINT()}});
  const auto makeBuilder = [&](bool noDuplicateKey) {
    return createFlatKeyChunkBuilder(
        createNimbleIndexKeyEncoder(keyColumns, type, sortOrders, pool()),
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
        /*enforceKeyOrder=*/true,
        noDuplicateKey,
        &comparator,
        pool());
  };

  // The encoded keys of ascending integers ascend bytewise, so descending
  // integers ascend under the reverse comparator, across batches and after
  // clear() too.
  {
    auto builder = makeBuilder(/*noDuplicateKey=*/true);
    builder->append(makeRowVector({makeFlatVector<int64_t>({9, 3, 0})}));
    builder->clear();
    builder->append(makeRowVector({makeFlatVector<int64_t>({-1, -7})}));
    EXPECT_EQ(builder->size(), 2);
    NIMBLE_ASSERT_USER_THROW(
        builder->append(makeRowVector({makeFlatVector<int64_t>({-7})})),
        "Encoded keys must be in strictly ascending order");
  }
  {
    auto builder = makeBuilder(/*noDuplicateKey=*/false);
    builder->append(makeRowVector({makeFlatVector<int64_t>({5, 5, 2})}));
    EXPECT_EQ(builder->size(), 3);
    NIMBLE_ASSERT_USER_THROW(
        builder->append(makeRowVector({makeFlatVector<int64_t>({1, 4})})),
        "Encoded keys must be in ascending order");
  }
}

TEST_F(KeyChunkBuilderTest, encodeFlatKeys) {
  const std::vector<std::string_view> keys{"aa", "bb", "bb", "cc"};
  const std::vector<std::string> expected{"aa", "bb", "bb", "cc"};
  for (const auto encodingType :
       {EncodingType::Prefix, EncodingType::Trivial}) {
    SCOPED_TRACE(toString(encodingType));
    auto layout =
        EncodingLayout{encodingType, {}, CompressionType::Uncompressed};
    if (encodingType == EncodingType::Trivial) {
      layout = EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Uncompressed,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
    }

    Buffer buffer{*pool()};
    const auto encoded = encodeFlatKeys(layout, keys, buffer);
    std::vector<velox::BufferPtr> stringBuffers;
    const auto reader = createFlatKeyReader(
        encoded,
        [&](uint32_t bytes) -> void* {
          auto& stringBuffer = stringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(bytes, pool()));
          return stringBuffer->asMutable<void>();
        },
        /*comparator=*/nullptr,
        pool());
    EXPECT_EQ(reader->materialize(0, reader->rowCount()), expected);
  }
}

} // namespace
} // namespace facebook::nimble::index::test

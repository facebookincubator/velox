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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/KeyReader.h"

namespace facebook::nimble::index {
namespace {

using testing::ElementsAre;

class KeyReaderTest : public testing::TestWithParam<EncodingType> {
 protected:
  static void SetUpTestSuite() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::testingSetInstance({});
    }
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("KeyReaderTest");
    pool_ = rootPool_->addLeafChild("leaf");
  }

  std::string encode(
      EncodingType encodingType,
      std::span<const std::string_view> keys) {
    EncodingLayout layout{encodingType, {}, CompressionType::Uncompressed};
    if (encodingType == EncodingType::Trivial) {
      layout = EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Uncompressed,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
    }
    Buffer buffer{*pool_};
    auto policy =
        std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
            layout,
            CompressionOptions{},
            [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
              return nullptr;
            });
    const auto encoded = EncodingFactory::encode<std::string_view>(
        std::move(policy), keys, buffer);
    return std::string{encoded};
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_P(KeyReaderTest, readsFlatKeys) {
  const std::vector<std::string_view> keys{"aa", "bb", "bb", "cc"};
  const auto encodedKeys = encode(GetParam(), keys);
  std::vector<velox::BufferPtr> stringBuffers;
  const auto reader = createFlatKeyReader(
      encodedKeys,
      [&](uint32_t bytes) -> void* {
        auto& buffer = stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(bytes, pool_.get()));
        return buffer->asMutable<void>();
      },
      pool_.get());

  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->rowCount(), keys.size());
  EXPECT_EQ(reader->seek("bb", /*inclusive=*/true), 1);
  EXPECT_EQ(reader->seek("bb", /*inclusive=*/false), 3);
  EXPECT_EQ(reader->get(3), "cc");
  EXPECT_THAT(reader->materialize(1, 2), ElementsAre("bb", "bb"));

  auto cursor = reader->cursor(1);
  EXPECT_EQ(cursor->next(), "bb");
  EXPECT_EQ(cursor->next(), "bb");
  EXPECT_EQ(cursor->next(), "cc");
  EXPECT_FALSE(cursor->hasNext());
}

INSTANTIATE_TEST_SUITE_P(
    EncodingTypes,
    KeyReaderTest,
    testing::Values(EncodingType::Trivial, EncodingType::Prefix),
    [](const testing::TestParamInfo<EncodingType>& info) {
      return info.param == EncodingType::Trivial ? "Trivial" : "Prefix";
    });

} // namespace
} // namespace facebook::nimble::index

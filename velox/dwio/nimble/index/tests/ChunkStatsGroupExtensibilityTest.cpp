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

#include <memory>
#include <type_traits>

#include "velox/dwio/nimble/index/ChunkStatsGroup.h"

namespace facebook::nimble::index::test {
namespace {

class TestStreamIndex final : public StreamIndex {
 public:
  TestStreamIndex() : StreamIndex{7} {}

  ChunkLocation lookupChunk(uint32_t rowId) const final {
    return {rowId, 11, 13, 17};
  }

  std::optional<uint32_t> chunkNullCount(uint32_t chunkIndex) const final {
    return chunkIndex + 1;
  }

  uint32_t rowCount() const final {
    return 19;
  }
};

class TestChunkStatsGroup final : public ChunkStatsGroup {
 public:
  TestChunkStatsGroup() : ChunkStatsGroup{3, 5, 7} {}

  std::shared_ptr<StreamIndex> createStreamIndex(uint32_t, uint32_t, uint32_t)
      const final {
    return std::make_shared<TestStreamIndex>();
  }
};

static_assert(std::has_virtual_destructor_v<ChunkStatsGroup>);
static_assert(std::has_virtual_destructor_v<StreamIndex>);
static_assert(std::is_abstract_v<ChunkStatsGroup>);
static_assert(std::is_abstract_v<StreamIndex>);

TEST(ChunkStatsGroupExtensibilityTest, dispatchesThroughBaseTypes) {
  std::shared_ptr<ChunkStatsGroup> group =
      std::make_shared<TestChunkStatsGroup>();
  const auto stream = group->createStreamIndex(3, 4, 5);

  EXPECT_EQ(stream->streamId(), 7);
  const auto location = stream->lookupChunk(23);
  EXPECT_EQ(location.chunkIndex, 23);
  EXPECT_EQ(location.chunkOffset, 11);
  EXPECT_EQ(location.chunkSize, 13);
  EXPECT_EQ(location.rowOffset, 17);
  EXPECT_EQ(stream->chunkNullCount(29), 30);
  EXPECT_EQ(stream->rowCount(), 19);
}

} // namespace
} // namespace facebook::nimble::index::test

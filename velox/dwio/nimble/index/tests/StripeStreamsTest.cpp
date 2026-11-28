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

#include "velox/dwio/nimble/index/StripeStreams.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble {
namespace {

// A buffer of each non-empty loaded stream of strings points at the stream's
// bytes and keeps them alive through the data handle after every other owner
// is gone. A stream of other values gets none, because
// StreamReader::setStreams() would not keep it.
TEST(StripeStreamsTest, streamBufferHoldersKeepTheLoadedBytesAlive) {
  auto bytes = std::make_shared<std::string>("abcdefghij");
  const std::weak_ptr<std::string> loadedBytes{bytes};
  const std::string_view data{*bytes};
  const std::vector<std::string_view> streams{
      data.substr(0, 3), std::string_view{}, data.substr(3, 2), data.substr(5)};
  const std::vector<bool> stringStreams{true, true, false, true};
  DataInput::Handle dataHandle{bytes};

  auto holders = makeStreamBufferHolders(streams, stringStreams, dataHandle);

  ASSERT_EQ(holders.size(), streams.size());
  EXPECT_EQ(holders[1], nullptr);
  EXPECT_EQ(holders[2], nullptr);
  for (const auto i : {0, 3}) {
    SCOPED_TRACE(fmt::format("stream {}", i));
    ASSERT_NE(holders[i], nullptr);
    EXPECT_EQ(holders[i]->as<char>(), streams[i].data());
    EXPECT_EQ(holders[i]->size(), streams[i].size());
  }

  bytes.reset();
  dataHandle.reset();
  ASSERT_FALSE(loadedBytes.expired());
  EXPECT_EQ(
      std::string_view(holders[3]->as<char>(), holders[3]->size()), "fghij");

  holders.clear();
  EXPECT_TRUE(loadedBytes.expired());
}

// Without a loaded stream of strings there is nothing to hold, so no buffers
// come back, which setStreams() takes as none, and no data handle is needed.
TEST(StripeStreamsTest, returnsNoBuffersWithoutLoadedStringStreams) {
  const std::vector<std::string_view> streams{std::string_view{}, "abc"};
  const std::vector<bool> stringStreams{true, false};

  EXPECT_TRUE(
      makeStreamBufferHolders(streams, stringStreams, /*dataHandle=*/nullptr)
          .empty());
}

TEST(StripeStreamsTest, loadedStringStreamsRequireDataHandle) {
  const std::vector<std::string_view> streams{"abc"};
  const std::vector<bool> stringStreams{true};

  NIMBLE_ASSERT_THROW(
      makeStreamBufferHolders(streams, stringStreams, /*dataHandle=*/nullptr),
      "Loaded streams require the handle that keeps them alive");
}

TEST(StripeStreamsTest, rejectsStringStreamsNotAlignedWithStreams) {
  const std::vector<std::string_view> streams{"abc"};
  const std::vector<bool> stringStreams{true, false};
  const DataInput::Handle dataHandle{std::make_shared<std::string>("abc")};

  NIMBLE_ASSERT_THROW(
      makeStreamBufferHolders(streams, stringStreams, dataHandle),
      "String stream flags must align with the streams");
}

} // namespace
} // namespace facebook::nimble

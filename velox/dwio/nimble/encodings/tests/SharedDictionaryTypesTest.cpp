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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"

namespace facebook::nimble {
namespace {

std::string encodeSharedDictionaryIdForTest(uint32_t dictionaryId) {
  std::string encoded(varint::varintSize(dictionaryId), '\0');
  char* pos = encoded.data();
  varint::writeVarint(dictionaryId, &pos);
  encoded.resize(static_cast<size_t>(pos - encoded.data()));
  return encoded;
}

TEST(SharedDictionaryScopeTest, scopeName) {
  struct Case {
    SharedDictionaryScope scope;
    uint8_t wireValue;
    std::string_view name;
  };

  const std::vector<Case> cases{
      {SharedDictionaryScope::Stripe, 0, "Stripe"},
      {SharedDictionaryScope::File, 2, "File"},
      {SharedDictionaryScope::External, 3, "External"},
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(
        fmt::format(
            "scope={} value={}",
            static_cast<int>(testCase.scope),
            static_cast<int>(testCase.wireValue)));
    EXPECT_EQ(SharedDictionaryScopeName::toName(testCase.scope), testCase.name);
    EXPECT_EQ(
        SharedDictionaryScopeName::toSharedDictionaryScope(testCase.name),
        testCase.scope);
    EXPECT_EQ(fmt::format("{}", testCase.scope), testCase.name);
    EXPECT_EQ(toSharedDictionaryScope(testCase.wireValue), testCase.scope);
  }
}

TEST(SharedDictionaryScopeTest, scopeNameError) {
  EXPECT_FALSE(
      SharedDictionaryScopeName::tryToSharedDictionaryScope("Unknown"));

  for (const auto value : {uint8_t{1}, uint8_t{4}, uint8_t{255}}) {
    SCOPED_TRACE(fmt::format("value={}", static_cast<int>(value)));
    NIMBLE_ASSERT_THROW(
        toSharedDictionaryScope(value),
        fmt::format(
            "Unsupported shared dictionary scope {}.",
            static_cast<int>(value)));
  }
}

TEST(SharedDictionaryScopeTest, readsScope) {
  const std::string data{
      static_cast<char>(0), static_cast<char>(2), static_cast<char>(3)};
  const char* pos = data.data();

  EXPECT_EQ(
      readSharedDictionaryScope(data, pos), SharedDictionaryScope::Stripe);
  EXPECT_EQ(pos, data.data() + 1);
  EXPECT_EQ(readSharedDictionaryScope(data, pos), SharedDictionaryScope::File);
  EXPECT_EQ(pos, data.data() + 2);
  EXPECT_EQ(
      readSharedDictionaryScope(data, pos), SharedDictionaryScope::External);
  EXPECT_EQ(pos, data.data() + 3);
}

TEST(SharedDictionaryScopeTest, readScopeError) {
  {
    const std::string data;
    const char* pos = data.data();

    NIMBLE_ASSERT_THROW(
        readSharedDictionaryScope(data, pos),
        "Shared dictionary encoding is missing its scope.");
  }

  {
    const std::string data{static_cast<char>(1)};
    const char* pos = data.data();

    NIMBLE_ASSERT_THROW(
        readSharedDictionaryScope(data, pos),
        "Unsupported shared dictionary scope 1.");
  }
}

TEST(SharedDictionaryScopeTest, readsDictionaryId) {
  std::string data = encodeSharedDictionaryIdForTest(0);
  const auto secondIdOffset = data.size();
  data += encodeSharedDictionaryIdForTest(300);
  const char* pos = data.data();

  EXPECT_EQ(readSharedDictionaryId(data, pos), 0);
  EXPECT_EQ(pos, data.data() + secondIdOffset);
  EXPECT_EQ(readSharedDictionaryId(data, pos), 300);
  EXPECT_EQ(pos, data.data() + data.size());
}

TEST(SharedDictionaryScopeTest, readDictionaryIdError) {
  struct Case {
    std::string data;
    std::string_view message;
  };

  const std::vector<Case> cases{
      {"", "Truncated shared dictionary ID varint."},
      {std::string{static_cast<char>(0x80)},
       "Truncated shared dictionary ID varint."},
      {std::string(
           varint::maxVarintSizeForBitWidth(/*bitWidth=*/32),
           static_cast<char>(0x80)),
       "Shared dictionary ID varint is too long."},
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(fmt::format("bytes={}", testCase.data.size()));
    const char* pos = testCase.data.data();
    NIMBLE_ASSERT_THROW(
        readSharedDictionaryId(testCase.data, pos), testCase.message);
  }
}

} // namespace
} // namespace facebook::nimble

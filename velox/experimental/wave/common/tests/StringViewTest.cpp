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

#include <gtest/gtest.h>

#include "velox/experimental/wave/common/StringView.h"
#include "velox/type/StringView.h"

namespace facebook::velox::wave {
namespace {

TEST(StringViewTest, empty) {
  StringView sv{};
  sv.init(reinterpret_cast<const char*>(0xDEADBEEFBADEFEEDll), 0);
  ASSERT_EQ(sv.size(), 0);
  ASSERT_EQ(sv, StringView{});
}

TEST(StringViewTest, inlined) {
  StringView sv{};
  sv.init("foobarquux", 3);
  ASSERT_EQ(sv.size(), 3);
  ASSERT_EQ(strncmp(sv.data(), "foo", 3), 0);
  ASSERT_LT(
      reinterpret_cast<uintptr_t>(&sv), reinterpret_cast<uintptr_t>(sv.data()));
  ASSERT_LT(
      reinterpret_cast<uintptr_t>(sv.data()),
      reinterpret_cast<uintptr_t>(&sv + 1));
  ASSERT_NE(sv, StringView{});
  StringView sv2{};
  sv2.init("foobar", 3);
  ASSERT_EQ(sv, sv2);
  sv2.init("foobar", 4);
  ASSERT_NE(sv, sv2);
  sv2.init("quux", 3);
  ASSERT_NE(sv, sv2);
}

TEST(StringViewTest, inlined12) {
  StringView sv{};
  sv.init("abcdefghijkl", 12);
  ASSERT_TRUE(sv.isInline());
  ASSERT_EQ(std::string_view(sv), "abcdefghijkl");
  StringView sv2{};
  sv2.init("abcdefghijkx", 12);
  ASSERT_NE(sv, sv2);
  sv2.init("abcdefghijklmn", 12);
  ASSERT_EQ(sv, sv2);
}

TEST(StringViewTest, nonInlined) {
  StringView sv{};
  const char* data = "foobarquux and more";
  sv.init(data, 19);
  ASSERT_FALSE(sv.isInline());
  ASSERT_EQ(sv.size(), 19);
  ASSERT_EQ(sv.data(), data);
  const std::string copy(data);
  StringView sv2{};
  sv2.init(copy.data(), 19);
  ASSERT_EQ(sv, sv2);
  sv2.init(copy.data(), 18);
  ASSERT_NE(sv, sv2);
  // Same prefix, different tail.
  const std::string other("foobarquux and moRe");
  sv2.init(other.data(), 19);
  ASSERT_NE(sv, sv2);
  // Different prefix.
  sv2.init("Foobarquux and more", 19);
  ASSERT_NE(sv, sv2);
}

TEST(StringViewTest, large) {
  const std::string big(100'000, 'x');
  StringView sv{};
  sv.init(big.data(), static_cast<uint32_t>(big.size()));
  ASSERT_EQ(sv.size(), big.size());
  ASSERT_EQ(std::string_view(sv), big);
}

// The layout is the same as velox::StringView, so vectors of strings can be
// copied between device and host as is.
TEST(StringViewTest, veloxLayout) {
  for (const std::string value : {"", "abc", "abcdefghijkl", "abcdefghijklm"}) {
    StringView sv{};
    sv.init(value.data(), static_cast<uint32_t>(value.size()));
    velox::StringView veloxView(
        value.data(), static_cast<int32_t>(value.size()));
    ASSERT_EQ(memcmp(&sv, &veloxView, sizeof(StringView)), 0) << value;
  }
}

} // namespace
} // namespace facebook::velox::wave

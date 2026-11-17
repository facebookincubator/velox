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

#include "velox/common/base/Padded.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace facebook::velox {
namespace {

struct Base {
  int32_t value;
};

struct Derived : Base {
  int32_t other;
};

// A raw pointer does not convert to Padded: callers say where the padding
// comes from.
static_assert(!std::is_convertible_v<int32_t*, Padded<int32_t>>);
static_assert(!std::is_constructible_v<Padded<int32_t>, int32_t*>);

// Padded converts to its const and void forms only.
static_assert(std::is_convertible_v<Padded<int32_t>, Padded<const int32_t>>);
static_assert(std::is_convertible_v<Padded<int32_t>, Padded<void>>);
static_assert(std::is_convertible_v<Padded<const int32_t>, Padded<const void>>);
static_assert(!std::is_convertible_v<Padded<const int32_t>, Padded<int32_t>>);
static_assert(!std::is_convertible_v<Padded<int32_t>, Padded<uint32_t>>);
// A base class would make operator+ step by the base class size.
static_assert(!std::is_convertible_v<Padded<Derived>, Padded<Base>>);

TEST(PaddedTest, assumePadded) {
  int32_t data[4] = {1, 2, 3, 4};
  const auto padded = Padded<int32_t>::assumePadded(data);
  EXPECT_EQ(padded.get(), data);

  const Padded<const int32_t> constPadded = padded;
  EXPECT_EQ(constPadded.get(), data);
  const Padded<const void> voidPadded = constPadded;
  EXPECT_EQ(voidPadded.get(), data);
}

TEST(PaddedTest, offset) {
  Derived data[4]{};
  const auto padded = Padded<Derived>::assumePadded(data);
  EXPECT_EQ((padded + 3).get(), &data[3]);
  EXPECT_EQ((padded + 0).get(), data);
}

TEST(PaddedTest, nullptr) {
  const Padded<const int32_t> padded = nullptr;
  EXPECT_EQ(padded.get(), nullptr);
}

} // namespace
} // namespace facebook::velox

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
#include "velox/dwio/nimble/common/ExceptionHelper.h"
#include <gtest/gtest.h>

using namespace facebook::nimble;

TEST(ExceptionHelperTest, noArgs) {
  auto result = errorMessage();
  // CompileTimeEmptyString is convertible to const char*, string_view, string.
  const char* asCharPtr = result;
  EXPECT_STREQ(asCharPtr, "");

  std::string_view asSv = result;
  EXPECT_EQ(asSv, "");
  EXPECT_EQ(asSv.size(), 0);

  std::string asStr = result;
  EXPECT_EQ(asStr, "");
}

TEST(ExceptionHelperTest, constCharPtr) {
  const char* input = "hello";
  const char* result = errorMessage(input);
  EXPECT_EQ(result, input);
  EXPECT_STREQ(result, "hello");
}

TEST(ExceptionHelperTest, stdString) {
  std::string input = "test message";
  std::string result = errorMessage(input);
  EXPECT_EQ(result, "test message");
}

TEST(ExceptionHelperTest, formatArgs) {
  std::string result = errorMessage("val={}", 42);
  EXPECT_EQ(result, "val=42");
}

TEST(ExceptionHelperTest, formatArgsMultiple) {
  std::string result = errorMessage("{} + {} = {}", 1, 2, 3);
  EXPECT_EQ(result, "1 + 2 = 3");
}

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
#include "velox/type/Tokenizer.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::common;

namespace {

std::vector<std::unique_ptr<Subfield::PathElement>> tokenize(
    const std::string& path) {
  std::vector<std::unique_ptr<Subfield::PathElement>> elements;
  Tokenizer tokenizer(path, Separators::get());
  while (tokenizer.hasNext()) {
    elements.push_back(tokenizer.next());
  }
  return elements;
}

} // namespace

TEST(TokenizerTest, simpleSubscript) {
  auto elements = tokenize("a[1]");
  ASSERT_EQ(elements.size(), 2);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::LongSubscript(1));
}

TEST(TokenizerTest, subscriptAfterLongPath) {
  auto elements = tokenize("some.long.path[42]");
  ASSERT_EQ(elements.size(), 4);
  EXPECT_EQ(*elements[0], Subfield::NestedField("some"));
  EXPECT_EQ(*elements[1], Subfield::NestedField("long"));
  EXPECT_EQ(*elements[2], Subfield::NestedField("path"));
  EXPECT_EQ(*elements[3], Subfield::LongSubscript(42));
}

TEST(TokenizerTest, multipleSubscripts) {
  auto elements = tokenize("a[1][2][3]");
  ASSERT_EQ(elements.size(), 4);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::LongSubscript(1));
  EXPECT_EQ(*elements[2], Subfield::LongSubscript(2));
  EXPECT_EQ(*elements[3], Subfield::LongSubscript(3));
}

TEST(TokenizerTest, negativeSubscript) {
  auto elements = tokenize("a[-1]");
  ASSERT_EQ(elements.size(), 2);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::LongSubscript(-1));
}

TEST(TokenizerTest, invalidSubscriptErrorMessage) {
  VELOX_ASSERT_THROW(tokenize("a[b]"), "Invalid index b");
  VELOX_ASSERT_THROW(tokenize("some.long.path[xyz]"), "Invalid index xyz");
}

TEST(TokenizerTest, pathSegment) {
  auto elements = tokenize("a.b.c");
  ASSERT_EQ(elements.size(), 3);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::NestedField("b"));
  EXPECT_EQ(*elements[2], Subfield::NestedField("c"));
}

TEST(TokenizerTest, quotedSubscript) {
  auto elements = tokenize("a[\"key\"]");
  ASSERT_EQ(elements.size(), 2);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::StringSubscript("key"));
}

TEST(TokenizerTest, wildcardSubscript) {
  auto elements = tokenize("a[*]");
  ASSERT_EQ(elements.size(), 2);
  EXPECT_EQ(*elements[0], Subfield::NestedField("a"));
  EXPECT_EQ(*elements[1], Subfield::AllSubscripts());
}

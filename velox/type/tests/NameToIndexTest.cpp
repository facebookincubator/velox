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
#include "velox/type/NameToIndex.h"

#include <gtest/gtest.h>

using namespace facebook::velox::detail;

class NameToIndexTest : public testing::Test {
 protected:
  NameToIndex nameToIndex_;
};

TEST_F(NameToIndexTest, emptyLookup) {
  EXPECT_FALSE(nameToIndex_.contains("foo"));
  EXPECT_FALSE(nameToIndex_.find("foo").has_value());
}

TEST_F(NameToIndexTest, insertAndContains) {
  nameToIndex_.insert("foo", 0);
  nameToIndex_.insert("bar", 1);
  nameToIndex_.insert("baz", 2);

  EXPECT_TRUE(nameToIndex_.contains("foo"));
  EXPECT_TRUE(nameToIndex_.contains("bar"));
  EXPECT_TRUE(nameToIndex_.contains("baz"));
  EXPECT_FALSE(nameToIndex_.contains("qux"));
}

TEST_F(NameToIndexTest, insertAndFind) {
  nameToIndex_.insert("foo", 0);
  nameToIndex_.insert("bar", 1);
  nameToIndex_.insert("baz", 2);

  EXPECT_EQ(nameToIndex_.find("foo"), 0);
  EXPECT_EQ(nameToIndex_.find("bar"), 1);
  EXPECT_EQ(nameToIndex_.find("baz"), 2);
  EXPECT_FALSE(nameToIndex_.find("qux").has_value());
}

TEST_F(NameToIndexTest, caseSensitivity) {
  nameToIndex_.insert("Foo", 0);

  EXPECT_TRUE(nameToIndex_.contains("Foo"));
  EXPECT_FALSE(nameToIndex_.contains("foo"));
  EXPECT_FALSE(nameToIndex_.contains("FOO"));
}

TEST_F(NameToIndexTest, emptyString) {
  nameToIndex_.insert("", 0);

  EXPECT_TRUE(nameToIndex_.contains(""));
  EXPECT_EQ(nameToIndex_.find(""), 0);
}

TEST_F(NameToIndexTest, reserve) {
  nameToIndex_.reserve(100);

  nameToIndex_.insert("foo", 0);
  EXPECT_TRUE(nameToIndex_.contains("foo"));
  EXPECT_EQ(nameToIndex_.find("foo"), 0);
}

TEST_F(NameToIndexTest, duplicateInsert) {
  nameToIndex_.insert("foo", 0);
  nameToIndex_.insert("foo", 1);

  // The first insert should win since we use emplace.
  EXPECT_EQ(nameToIndex_.find("foo"), 0);
}

TEST_F(NameToIndexTest, size) {
  EXPECT_EQ(nameToIndex_.size(), 0);

  nameToIndex_.insert("foo", 0);
  EXPECT_EQ(nameToIndex_.size(), 1);

  nameToIndex_.insert("bar", 1);
  EXPECT_EQ(nameToIndex_.size(), 2);

  nameToIndex_.insert("baz", 2);
  EXPECT_EQ(nameToIndex_.size(), 3);

  // Duplicate insert should not increase size.
  nameToIndex_.insert("foo", 3);
  EXPECT_EQ(nameToIndex_.size(), 3);
}

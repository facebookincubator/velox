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

#include "velox/dwio/nimble/index/HashIndexUtils.h"

namespace facebook::nimble::index {
namespace {

TEST(HashIndexUtilsTest, deterministic) {
  const uint32_t mask = 15;
  const auto bucket1 = bucketIndex("key_abc", mask);
  const auto bucket2 = bucketIndex("key_abc", mask);
  EXPECT_EQ(bucket1, bucket2);
}

TEST(HashIndexUtilsTest, resultWithinMask) {
  const uint32_t mask = 7;
  for (int i = 0; i < 100; ++i) {
    const auto key = "key_" + std::to_string(i);
    const auto bucket = bucketIndex(key, mask);
    EXPECT_LE(bucket, mask);
  }
}

TEST(HashIndexUtilsTest, singleBucket) {
  const uint32_t mask = 0;
  EXPECT_EQ(bucketIndex("any_key", mask), 0);
  EXPECT_EQ(bucketIndex("another_key", mask), 0);
}

TEST(HashIndexUtilsTest, emptyKey) {
  const uint32_t mask = 15;
  const auto bucket = bucketIndex("", mask);
  EXPECT_LE(bucket, mask);
}

TEST(HashIndexUtilsTest, differentKeysDistribute) {
  const uint32_t mask = 255;
  std::set<uint32_t> buckets;
  for (int i = 0; i < 100; ++i) {
    buckets.insert(bucketIndex("key_" + std::to_string(i), mask));
  }
  EXPECT_GT(buckets.size(), 1);
}

} // namespace
} // namespace facebook::nimble::index

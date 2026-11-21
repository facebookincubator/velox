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

#include "velox/dwio/nimble/tablet/StripeGroupFlushPolicy.h"

namespace facebook::nimble {
namespace {

TEST(StripeGroupFlushPolicyTest, flushesAtThirteenBytesPerEntry) {
  // 4 bytes plus 13 bytes per entry reach 1'000 bytes at 77 entries.
  const StripeGroupFlushPolicy policy{1'000};
  EXPECT_FALSE(policy.shouldFlush(/*numEntries=*/76));
  EXPECT_TRUE(policy.shouldFlush(/*numEntries=*/77));
}

TEST(StripeGroupFlushPolicyTest, calibratesWithWrittenGroup) {
  StripeGroupFlushPolicy policy{1'000};

  // A group of 100 entries serialized to 500 bytes, so later groups take 5
  // bytes per entry and flush at 200 entries.
  policy.onFlush(/*numEntries=*/100, /*encodedBytes=*/500);
  EXPECT_FALSE(policy.shouldFlush(/*numEntries=*/199));
  EXPECT_TRUE(policy.shouldFlush(/*numEntries=*/200));

  // The last written group calibrates: 50 entries at 20 bytes each.
  policy.onFlush(/*numEntries=*/50, /*encodedBytes=*/1'000);
  EXPECT_FALSE(policy.shouldFlush(/*numEntries=*/49));
  EXPECT_TRUE(policy.shouldFlush(/*numEntries=*/50));

  // A group without entries has no bytes per entry to calibrate with.
  policy.onFlush(/*numEntries=*/0, /*encodedBytes=*/100);
  EXPECT_FALSE(policy.shouldFlush(/*numEntries=*/49));
  EXPECT_TRUE(policy.shouldFlush(/*numEntries=*/50));
}

} // namespace

} // namespace facebook::nimble

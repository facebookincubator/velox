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

#include "velox/common/base/SuccinctPrintUtil.h"

#include <gtest/gtest.h>

namespace facebook {
namespace velox {
namespace printer {

TEST(SuccintPrintTest, testSuccintNanos) {
  EXPECT_EQ(succinctNanos(123), "123ns");
  EXPECT_EQ(succinctNanos(1'000), "1.0000us");
  EXPECT_EQ(succinctNanos(1'234), "1.2340us");
  EXPECT_EQ(succinctNanos(123'456), "123.4560us");
  EXPECT_EQ(succinctNanos(1'000'000), "1.0000ms");
  EXPECT_EQ(succinctNanos(12'345'678), "12.3457ms");
  EXPECT_EQ(succinctNanos(12'345'678, 6), "12.345678ms");
  EXPECT_EQ(succinctNanos(1'000'000'000), "1.0000s");
  EXPECT_EQ(succinctNanos(1'234'567'890), "1.2346s");
  EXPECT_EQ(succinctNanos(123'456'789'000'000), "123456.7890s");
}

TEST(SuccintPrintTest, testSuccintMillis) {
  EXPECT_EQ(succinctMillis(123), "123ms");
  EXPECT_EQ(succinctMillis(1'000), "1.0000s");
  EXPECT_EQ(succinctMillis(1234), "1.2340s");
  EXPECT_EQ(succinctMillis(123456), "123.4560s");
  EXPECT_EQ(succinctMillis(12345678), "12345.6780s");
  EXPECT_EQ(succinctMillis(12345678, 2), "12345.68s");
}

TEST(SuccintPrintTest, testSuccintBytes) {
  EXPECT_EQ(succinctBytes(123), "123B");
  EXPECT_EQ(succinctBytes(1'024), "1.0000KB");
  EXPECT_EQ(succinctBytes(123'456), "120.5625KB");
  EXPECT_EQ(succinctBytes(1'048'576), "1.0000MB");
  EXPECT_EQ(succinctBytes(12'345'678, 2), "11.77MB");
  EXPECT_EQ(succinctBytes(1'073'741'824), "1.0000GB");
  EXPECT_EQ(succinctBytes(1'234'567'890), "1.1498GB");
}

} // namespace printer
} // namespace velox
} // namespace facebook

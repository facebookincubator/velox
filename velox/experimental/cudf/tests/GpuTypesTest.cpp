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
#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/types/GpuStringView.cuh"
#include "velox/experimental/cudf/types/GpuTimestamp.cuh"

using namespace facebook::velox::gpu;

TEST(GpuTypesTest, resolverPrimitives) {
  static_assert(std::is_same_v<GpuExec::resolver<double>::in_type, double>);
  static_assert(std::is_same_v<GpuExec::resolver<int64_t>::in_type, int64_t>);
  static_assert(std::is_same_v<GpuExec::resolver<int32_t>::in_type, int32_t>);
  static_assert(std::is_same_v<GpuExec::resolver<float>::in_type, float>);
  static_assert(std::is_same_v<GpuExec::resolver<bool>::in_type, bool>);
  static_assert(
      std::is_same_v<GpuExec::resolver<double>::null_free_in_type, double>);
}

TEST(GpuTypesTest, resolverVarchar) {
  using R = GpuExec::resolver<facebook::velox::Varchar>;
  static_assert(std::is_same_v<R::in_type, GpuStringView>);
  static_assert(std::is_same_v<R::out_type, GpuStringView>);
  static_assert(std::is_same_v<R::null_free_in_type, GpuStringView>);
}

TEST(GpuTypesTest, resolverVarbinary) {
  using R = GpuExec::resolver<facebook::velox::Varbinary>;
  static_assert(std::is_same_v<R::in_type, GpuStringView>);
}

TEST(GpuTypesTest, resolverDate) {
  using R = GpuExec::resolver<facebook::velox::Date>;
  static_assert(std::is_same_v<R::in_type, int32_t>);
}

TEST(GpuTypesTest, resolverIntervalDayTime) {
  using R = GpuExec::resolver<facebook::velox::IntervalDayTime>;
  static_assert(std::is_same_v<R::in_type, int64_t>);
}

TEST(GpuTypesTest, resolverIntervalYearMonth) {
  using R = GpuExec::resolver<facebook::velox::IntervalYearMonth>;
  static_assert(std::is_same_v<R::in_type, int32_t>);
}

TEST(GpuTypesTest, resolverTime) {
  using R = GpuExec::resolver<facebook::velox::Time>;
  static_assert(std::is_same_v<R::in_type, int64_t>);
}

TEST(GpuTypesTest, resolverTimestamp) {
  using R = GpuExec::resolver<facebook::velox::Timestamp>;
  static_assert(std::is_same_v<R::in_type, GpuTimestamp>);
  static_assert(std::is_same_v<R::out_type, GpuTimestamp>);
  static_assert(std::is_same_v<R::null_free_in_type, GpuTimestamp>);
}

TEST(GpuTypesTest, gpuStringViewBasic) {
  const char* s = "hello";
  GpuStringView sv(s, 5);
  EXPECT_EQ(sv.size(), 5);
  EXPECT_FALSE(sv.empty());
  EXPECT_EQ(sv.data(), s);
  EXPECT_EQ(sv.begin(), s);
  EXPECT_EQ(sv.end(), s + 5);
}

TEST(GpuTypesTest, gpuStringViewEquality) {
  const char* s1 = "hello";
  const char* s2 = "hello";
  const char* s3 = "world";
  GpuStringView sv1(s1, 5);
  GpuStringView sv2(s2, 5);
  GpuStringView sv3(s3, 5);
  GpuStringView empty;

  EXPECT_EQ(sv1, sv2);
  EXPECT_NE(sv1, sv3);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);
}

TEST(GpuTypesTest, gpuTimestampComparison) {
  GpuTimestamp a(100, 500);
  GpuTimestamp b(100, 600);
  GpuTimestamp c(101, 0);
  GpuTimestamp d(100, 500);

  EXPECT_EQ(a, d);
  EXPECT_NE(a, b);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b < c);
  EXPECT_TRUE(a <= d);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(c > b);
  EXPECT_TRUE(c >= b);
  EXPECT_TRUE(a >= d);
}

TEST(GpuTypesTest, gpuTimestampDefault) {
  GpuTimestamp t;
  EXPECT_EQ(t.seconds, 0);
  EXPECT_EQ(t.nanos, 0u);
}

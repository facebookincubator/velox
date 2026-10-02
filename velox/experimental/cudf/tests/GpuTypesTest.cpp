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

#include "velox/experimental/cudf/functions/GpuExec.h"

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/SimpleFunctionTags.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <gtest/gtest.h>

namespace facebook::velox::gpu {
namespace {

// True when GpuExec maps T to Expected in every position a call() sees it.
template <typename T, typename Expected>
constexpr bool resolvesTo =
    std::is_same_v<typename GpuExec::resolver<T>::in_type, Expected> &&
    std::is_same_v<typename GpuExec::resolver<T>::out_type, Expected> &&
    std::is_same_v<typename GpuExec::resolver<T>::null_free_in_type, Expected>;

// Passes primitives through, maps each Velox type tag to the physical type a
// kernel reads, and keeps Velox's own Timestamp, as on the CPU.
TEST(GpuTypesTest, resolver) {
  static_assert(resolvesTo<bool, bool>);
  static_assert(resolvesTo<int32_t, int32_t>);
  static_assert(resolvesTo<int64_t, int64_t>);
  static_assert(resolvesTo<float, float>);
  static_assert(resolvesTo<double, double>);
  static_assert(resolvesTo<Date, int32_t>);
  static_assert(resolvesTo<IntervalYearMonth, int32_t>);
  static_assert(resolvesTo<IntervalDayTime, int64_t>);
  static_assert(resolvesTo<Time, int64_t>);
  static_assert(resolvesTo<ShortDecimal<P1, S1>, int64_t>);
  static_assert(resolvesTo<LongDecimal<P1, S1>, __int128>);
  static_assert(resolvesTo<Timestamp, Timestamp>);
}

// True when GpuExec hands a T argument to a body as Velox's own StringView and
// lets a kernel produce none.
template <typename T>
constexpr bool resolvesToStringArgument =
    std::is_same_v<typename GpuExec::resolver<T>::in_type, StringView> &&
    std::is_same_v<
        typename GpuExec::resolver<T>::null_free_in_type,
        StringView> &&
    std::is_same_v<typename GpuExec::resolver<T>::out_type, void>;

// A VARCHAR or VARBINARY argument arrives as Velox's own StringView, as on the
// CPU, and a kernel produces neither.
TEST(GpuTypesTest, resolverStrings) {
  static_assert(resolvesToStringArgument<Varchar>);
  static_assert(resolvesToStringArgument<Varbinary>);
}

// A custom type with a custom comparison arrives in a view, as on the CPU,
// and leaves as its physical type.
TEST(GpuTypesTest, resolverCustomType) {
  using R = GpuExec::resolver<TimestampWithTimezone>;
  static_assert(
      std::is_same_v<R::in_type, GpuCustomTypeView<TimestampWithTimezoneT>>);
  static_assert(std::is_same_v<R::out_type, int64_t>);
  static_assert(std::is_same_v<R::null_free_in_type, R::in_type>);
}

// The view compares by instant, as TimestampWithTimeZoneType does, and does
// not convert to the packed bits, which a generic comparison would otherwise
// order by zone key.
TEST(GpuTypesTest, customTypeViewComparesByInstant) {
  using View = GpuCustomTypeView<TimestampWithTimezoneT>;
  static_assert(!std::is_convertible_v<View, int64_t>);
  static_assert(std::is_trivially_copyable_v<View>);

  const auto utc = tz::getTimeZoneID("UTC");
  const auto kolkata = tz::getTimeZoneID("Asia/Kolkata");
  const View instantInUtc{pack(1'000, utc)};
  const View instantInKolkata{pack(1'000, kolkata)};
  const View laterInUtc{pack(1'001, utc)};

  EXPECT_EQ(*instantInUtc, pack(1'000, utc));
  EXPECT_TRUE(instantInUtc == instantInKolkata);
  EXPECT_FALSE(instantInUtc != instantInKolkata);
  EXPECT_TRUE(instantInUtc <= instantInKolkata);
  EXPECT_TRUE(instantInUtc >= instantInKolkata);
  EXPECT_FALSE(instantInUtc < instantInKolkata);
  EXPECT_TRUE(instantInKolkata < laterInUtc);
  EXPECT_TRUE(laterInUtc > instantInKolkata);
  EXPECT_FALSE(laterInUtc <= instantInKolkata);
}

} // namespace
} // namespace facebook::velox::gpu

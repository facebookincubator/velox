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
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/Comparisons.h"
#include "velox/functions/prestosql/types/IPAddressRegistration.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixRegistration.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRegistration.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::functions {
namespace {

template <template <class> class T, typename TReturn>
void registerNonSimdizableScalar(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner) {
  registerFunction<T, TReturn, Varchar, Varchar>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Varbinary, Varbinary>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, bool, bool>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Timestamp, Timestamp>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, TimestampWithTimezone, TimestampWithTimezone>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Time, Time>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, TimeWithTimezone, TimeWithTimezone>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, IPAddress, IPAddress>(
      aliases, {}, true, defaultOwner);
}
} // namespace

void registerComparisonFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  // Comparison functions also need TimestampWithTimezoneType,
  // independent of DateTimeFunctions
  registerTimestampWithTimeZoneType();
  registerTimeWithTimezoneType();
  registerIPAddressType();
  registerIPPrefixType();

  registerNonSimdizableScalar<EqFunction, bool>({prefix + "eq"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_eq, prefix + "eq", defaultOwner);
  registerFunction<EqFunction, bool, Generic<T1>, Generic<T1>>(
      {prefix + "eq"}, {}, true, defaultOwner);

  registerNonSimdizableScalar<NeqFunction, bool>(
      {prefix + "neq"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_neq, prefix + "neq", defaultOwner);
  registerFunction<NeqFunction, bool, Generic<T1>, Generic<T1>>(
      {prefix + "neq"}, {}, true, defaultOwner);

  registerNonSimdizableScalar<LtFunction, bool>({prefix + "lt"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_lt, prefix + "lt", defaultOwner);
  registerFunction<LtFunction, bool, Orderable<T1>, Orderable<T1>>(
      {prefix + "lt"}, {}, true, defaultOwner);

  registerNonSimdizableScalar<GtFunction, bool>({prefix + "gt"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_gt, prefix + "gt", defaultOwner);
  registerFunction<GtFunction, bool, Orderable<T1>, Orderable<T1>>(
      {prefix + "gt"}, {}, true, defaultOwner);

  registerNonSimdizableScalar<LteFunction, bool>(
      {prefix + "lte"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_lte, prefix + "lte", defaultOwner);
  registerFunction<LteFunction, bool, Orderable<T1>, Orderable<T1>>(
      {prefix + "lte"}, {}, true, defaultOwner);

  registerNonSimdizableScalar<GteFunction, bool>(
      {prefix + "gte"}, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_simd_comparison_gte, prefix + "gte", defaultOwner);
  registerFunction<GteFunction, bool, Orderable<T1>, Orderable<T1>>(
      {prefix + "gte"}, {}, true, defaultOwner);

  registerFunction<DistinctFromFunction, bool, Generic<T1>, Generic<T1>>(
      {prefix + "distinct_from"}, {}, true, defaultOwner);

  registerFunction<BetweenFunction, bool, int8_t, int8_t, int8_t>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, int16_t, int16_t, int16_t>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, int32_t, int32_t, int32_t>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, int64_t, int64_t, int64_t>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, double, double, double>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, float, float, float>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, Varchar, Varchar, Varchar>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, Date, Date, Date>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, Timestamp, Timestamp, Timestamp>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, Time, Time, Time>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      TimeWithTimezone,
      TimeWithTimezone,
      TimeWithTimezone>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      LongDecimal<P1, S1>,
      LongDecimal<P1, S1>,
      LongDecimal<P1, S1>>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      IntervalDayTime,
      IntervalDayTime,
      IntervalDayTime>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      IntervalYearMonth,
      IntervalYearMonth,
      IntervalYearMonth>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, IPAddress, IPAddress, IPAddress>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<BetweenFunction, bool, IPPrefix, IPPrefix, IPPrefix>(
      {prefix + "between"}, {}, true, defaultOwner);
  registerFunction<
      BetweenFunction,
      bool,
      UnknownValue,
      UnknownValue,
      UnknownValue>({prefix + "between"}, {}, true, defaultOwner);
}

} // namespace facebook::velox::functions

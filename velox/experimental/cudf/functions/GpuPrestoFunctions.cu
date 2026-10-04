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

// PrestoSQL simple functions compiled for GPU, registered the way
// velox/functions/prestosql/registration/*.cpp registers them on the CPU.
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuDecimalRegistration.cuh"
#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"
#include "velox/experimental/cudf/functions/GpuRegistrationHelpers.cuh"

// Bitwise.h calls bits::countBits without including BitUtil.h.
#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"
// The calendar functions alone: DateTimeFunctions.h reaches the formatters,
// parsers and vector machinery, which do not parse under nvcc.
#include "velox/functions/prestosql/detail/DateTimeCalendarFunctions.h"
#include "velox/functions/prestosql/detail/DecimalMathFunctions.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

namespace {

/// Registers a calendar field extractor over the three types the CPU registers
/// it for, as DateTimeFunctionsRegistration.cpp does.
template <template <class> typename Fn>
void registerGpuCalendarField(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, int64_t, Timestamp>(aliases);
  registerGpuFunction<Fn, int64_t, Date>(aliases);
  registerGpuFunction<Fn, int64_t, TimestampWithTimezone>(aliases);
}

/// Registers the interval operators over one timestamp type, in both operand
/// orders, and the difference of two values, as registerTimestampPlusInterval
/// and registerTimestampMinusInterval register them.
template <typename TTimestamp>
void registerGpuTimestampIntervalOperators(const std::string& prefix) {
  registerGpuFunction<
      TimestampPlusInterval,
      TTimestamp,
      TTimestamp,
      IntervalDayTime>({prefix + "plus"});
  registerGpuFunction<
      TimestampPlusInterval,
      TTimestamp,
      TTimestamp,
      IntervalYearMonth>({prefix + "plus"});
  registerGpuFunction<
      IntervalPlusTimestamp,
      TTimestamp,
      IntervalDayTime,
      TTimestamp>({prefix + "plus"});
  registerGpuFunction<
      IntervalPlusTimestamp,
      TTimestamp,
      IntervalYearMonth,
      TTimestamp>({prefix + "plus"});
  registerGpuFunction<
      TimestampMinusInterval,
      TTimestamp,
      TTimestamp,
      IntervalDayTime>({prefix + "minus"});
  registerGpuFunction<
      TimestampMinusInterval,
      TTimestamp,
      TTimestamp,
      IntervalYearMonth>({prefix + "minus"});
  registerGpuFunction<
      TimestampMinusFunction,
      IntervalDayTime,
      TTimestamp,
      TTimestamp>({prefix + "minus"});
}

} // namespace

void registerPrestoGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  // Type sets follow MathematicalOperatorsRegistration.cpp and
  // MathematicalFunctionsRegistration.cpp helper for helper: the plain structs
  // for floating point, the Checked* ones for the integral overloads.
  registerGpuBinaryFloatingPoint<PlusFunction>({prefix + "plus"});
  registerGpuBinaryFloatingPoint<MinusFunction>({prefix + "minus"});
  registerGpuBinaryFloatingPoint<MultiplyFunction>({prefix + "multiply"});
  registerGpuBinaryFloatingPoint<DivideFunction>({prefix + "divide"});
  registerGpuBinaryFloatingPoint<ModulusFunction>({prefix + "mod"});

  // negate is floating point only upstream; its integral overloads come from
  // CheckedNegateFunction below.
  registerGpuUnaryFloatingPoint<NegateFunction>({prefix + "negate"});

  registerGpuUnaryNumeric<AbsFunction>({prefix + "abs"});
  registerGpuUnaryNumeric<CeilFunction>({prefix + "ceil", prefix + "ceiling"});
  registerGpuUnaryNumeric<FloorFunction>({prefix + "floor"});
  registerGpuUnaryNumeric<SignFunction>({prefix + "sign"});
  registerGpuTernaryNumeric<ClampFunction>({prefix + "clamp"});

  registerGpuFunction<PowerFunction, double, double, double>(
      {prefix + "power", prefix + "pow"});
  registerGpuFunction<PowerFunction, double, int64_t, int64_t>(
      {prefix + "power", prefix + "pow"});

  // One call() with a defaulted trailing parameter serves both arities
  // upstream, so each type appears once per arity.
  registerGpuUnaryNumeric<RoundFunction>({prefix + "round"});
  registerGpuNumericWithDecimals<RoundFunction>({prefix + "round"});
  // truncate is floating point only, both arities.
  registerGpuUnaryFloatingPoint<TruncateFunction>({prefix + "truncate"});
  registerGpuFunction<TruncateFunction, double, double, int32_t>(
      {prefix + "truncate"});
  registerGpuFunction<TruncateFunction, float, float, int32_t>(
      {prefix + "truncate"});

  // --- Checked integral arithmetic ----------------------------------------
  registerGpuBinaryIntegral<CheckedPlusFunction>({prefix + "plus"});
  registerGpuBinaryIntegral<CheckedMinusFunction>({prefix + "minus"});
  registerGpuBinaryIntegral<CheckedMultiplyFunction>({prefix + "multiply"});
  registerGpuBinaryIntegral<CheckedDivideFunction>({prefix + "divide"});
  registerGpuBinaryIntegral<CheckedModulusFunction>({prefix + "mod"});
  registerGpuUnaryIntegral<CheckedNegateFunction>({prefix + "negate"});

  // --- Math and trigonometry ---------------------------------------------
  registerGpuFunction<ExpFunction, double, double>({prefix + "exp"});
  registerGpuFunction<LnFunction, double, double>({prefix + "ln"});
  registerGpuFunction<Log2Function, double, double>({prefix + "log2"});
  registerGpuFunction<Log10Function, double, double>({prefix + "log10"});
  registerGpuFunction<SqrtFunction, double, double>({prefix + "sqrt"});
  registerGpuFunction<CbrtFunction, double, double>({prefix + "cbrt"});
  registerGpuFunction<SinFunction, double, double>({prefix + "sin"});
  registerGpuFunction<CosFunction, double, double>({prefix + "cos"});
  registerGpuFunction<TanFunction, double, double>({prefix + "tan"});
  registerGpuFunction<AsinFunction, double, double>({prefix + "asin"});
  registerGpuFunction<AcosFunction, double, double>({prefix + "acos"});
  registerGpuFunction<AtanFunction, double, double>({prefix + "atan"});
  registerGpuFunction<CoshFunction, double, double>({prefix + "cosh"});
  registerGpuFunction<TanhFunction, double, double>({prefix + "tanh"});
  registerGpuFunction<DegreesFunction, double, double>({prefix + "degrees"});
  registerGpuFunction<RadiansFunction, double, double>({prefix + "radians"});
  registerGpuFunction<Atan2Function, double, double, double>(
      {prefix + "atan2"});

  // --- Floating-point predicates -----------------------------------------
  registerGpuFunction<IsNanFunction, bool, double>({prefix + "is_nan"});
  registerGpuFunction<IsFiniteFunction, bool, double>({prefix + "is_finite"});
  registerGpuFunction<IsInfiniteFunction, bool, double>(
      {prefix + "is_infinite"});

  // --- Comparisons --------------------------------------------------------
  registerGpuBinaryNumericWithTReturn<LtFunction, bool>({prefix + "lt"});
  registerGpuBinaryNumericWithTReturn<LteFunction, bool>({prefix + "lte"});
  registerGpuBinaryNumericWithTReturn<GtFunction, bool>({prefix + "gt"});
  registerGpuBinaryNumericWithTReturn<GteFunction, bool>({prefix + "gte"});
  registerGpuTernaryNumericWithTReturn<BetweenFunction, bool>(
      {prefix + "between"});

  // --- Comparisons, TIMESTAMP WITH TIME ZONE -------------------------------
  // Velox's own comparisons over the custom-type view, whose operators order
  // the type by instant as TimestampWithTimeZoneType::compare() does. Names
  // follow ComparisonFunctionsRegistration.cpp.
  registerGpuFunction<
      EqFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "eq"});
  registerGpuFunction<
      NeqFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "neq"});
  registerGpuFunction<
      LtFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "lt"});
  registerGpuFunction<
      GtFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "gt"});
  registerGpuFunction<
      LteFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "lte"});
  registerGpuFunction<
      GteFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "gte"});
  registerGpuFunction<
      BetweenFunction,
      bool,
      TimestampWithTimezone,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "between"});

  // --- Bitwise ------------------------------------------------------------
  registerGpuFunction<BitwiseAndFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_and"});
  registerGpuFunction<BitwiseOrFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_or"});
  registerGpuFunction<BitwiseXorFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_xor"});
  registerGpuFunction<BitwiseNotFunction, int64_t, int64_t>(
      {prefix + "bitwise_not"});
  registerGpuFunction<BitwiseLeftShiftFunction, int64_t, int64_t, int32_t>(
      {prefix + "bitwise_left_shift"});
  registerGpuFunction<BitwiseRightShiftFunction, int64_t, int64_t, int32_t>(
      {prefix + "bitwise_right_shift"});
  registerGpuFunction<
      BitwiseRightShiftArithmeticFunction,
      int64_t,
      int64_t,
      int32_t>({prefix + "bitwise_right_shift_arithmetic"});
  // The checked bitwise functions follow BitwiseFunctionsRegistration.cpp:
  // bit_count widens every integral pair to bigint, the rest are bigint only.
  registerGpuFunction<BitCountFunction, int64_t, int8_t, int8_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int16_t, int16_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int32_t, int32_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int64_t, int64_t>(
      {prefix + "bit_count"});
  registerGpuFunction<
      BitwiseArithmeticShiftRightFunction,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_arithmetic_shift_right"});
  registerGpuFunction<
      BitwiseLogicalShiftRightFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_logical_shift_right"});
  registerGpuFunction<
      BitwiseShiftLeftFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_shift_left"});

  // --- Datetime -----------------------------------------------------------
  // Velox's own calendar functions, over the device forms of Timestamp and
  // the time zone database: the TIMESTAMP overloads apply the session time
  // zone and the TIMESTAMP WITH TIME ZONE ones render each value in its own
  // zone or the session zone, as legacy_timestamp_with_timezone selects, both
  // read by the structs' initialize(). Names follow
  // DateTimeFunctionsRegistration.cpp; the type has to be registered before
  // these signatures are parsed, which registerCudf() ensures.
  registerGpuCalendarField<YearFunction>({prefix + "year"});
  registerGpuCalendarField<QuarterFunction>({prefix + "quarter"});
  registerGpuCalendarField<MonthFunction>({prefix + "month"});
  registerGpuCalendarField<DayFunction>(
      {prefix + "day", prefix + "day_of_month"});
  registerGpuCalendarField<DayOfWeekFunction>(
      {prefix + "day_of_week", prefix + "dow"});
  registerGpuCalendarField<DayOfYearFunction>(
      {prefix + "day_of_year", prefix + "doy"});
  registerGpuCalendarField<WeekFunction>(
      {prefix + "week", prefix + "week_of_year"});
  registerGpuCalendarField<YearOfWeekFunction>(
      {prefix + "year_of_week", prefix + "yow"});
  registerGpuCalendarField<HourFunction>({prefix + "hour"});
  registerGpuCalendarField<MinuteFunction>({prefix + "minute"});
  registerGpuCalendarField<SecondFunction>({prefix + "second"});
  registerGpuCalendarField<MillisecondFunction>({prefix + "millisecond"});
  registerGpuFunction<ToUnixtimeFunction, double, Timestamp>(
      {prefix + "to_unixtime"});
  registerGpuFunction<ToUnixtimeFunction, double, TimestampWithTimezone>(
      {prefix + "to_unixtime"});
  registerGpuFunction<TimeZoneHourFunction, int64_t, TimestampWithTimezone>(
      {prefix + "timezone_hour"});
  registerGpuFunction<TimeZoneMinuteFunction, int64_t, TimestampWithTimezone>(
      {prefix + "timezone_minute"});
  // A zone name or a unit binds only as a literal, which initialize() resolves
  // on the host; a column there leaves the call to the CPU, since a kernel
  // cannot read a strings column.
  registerGpuFunction<FromUnixtimeFunction, Timestamp, double>(
      {prefix + "from_unixtime"});
  registerGpuFunction<
      FromUnixtimeFunction,
      TimestampWithTimezone,
      double,
      Constant<Varchar>>({prefix + "from_unixtime"});
  registerGpuFunction<
      FromUnixtimeFunction,
      TimestampWithTimezone,
      double,
      int64_t,
      int64_t>({prefix + "from_unixtime"});
  registerGpuFunction<
      AtTimezoneFunction,
      TimestampWithTimezone,
      TimestampWithTimezone,
      Constant<Varchar>>({prefix + "at_timezone"});
  registerGpuFunction<
      DateTruncFunction,
      Timestamp,
      Constant<Varchar>,
      Timestamp>({prefix + "date_trunc"});
  registerGpuFunction<DateTruncFunction, Date, Constant<Varchar>, Date>(
      {prefix + "date_trunc"});
  registerGpuFunction<
      DateTruncFunction,
      TimestampWithTimezone,
      Constant<Varchar>,
      TimestampWithTimezone>({prefix + "date_trunc"});

  // --- Datetime arithmetic -------------------------------------------------
  // Velox's own date_add, date_diff and interval operators, converting through
  // the device zone's to_local(), to_sys() and correct_nonexistent_time(). The
  // unit binds only as a literal, as above. DATE plus or minus an interval is
  // declared in DateTimeFunctions.h, which this unit cannot include, and stays
  // with the function tier.
  registerGpuFunction<
      DateAddFunction,
      Timestamp,
      Constant<Varchar>,
      int64_t,
      Timestamp>({prefix + "date_add"});
  registerGpuFunction<DateAddFunction, Date, Constant<Varchar>, int64_t, Date>(
      {prefix + "date_add"});
  registerGpuFunction<
      DateAddFunction,
      TimestampWithTimezone,
      Constant<Varchar>,
      int64_t,
      TimestampWithTimezone>({prefix + "date_add"});
  registerGpuFunction<
      DateDiffFunction,
      int64_t,
      Constant<Varchar>,
      Timestamp,
      Timestamp>({prefix + "date_diff"});
  registerGpuFunction<DateDiffFunction, int64_t, Constant<Varchar>, Date, Date>(
      {prefix + "date_diff"});
  registerGpuFunction<
      DateDiffFunction,
      int64_t,
      Constant<Varchar>,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "date_diff"});
  registerGpuTimestampIntervalOperators<Timestamp>(prefix);
  registerGpuTimestampIntervalOperators<TimestampWithTimezone>(prefix);

  // --- Logical -------------------------------------------------------------
  // See GpuLogicalFunctions.cuh. TODO: register is_null for every input type,
  // not only BOOLEAN.
  registerGpuFunction<GpuAndFunction, bool, Variadic<bool>>({prefix + "and"});
  registerGpuFunction<GpuOrFunction, bool, Variadic<bool>>({prefix + "or"});
  registerGpuFunction<GpuNotFunction, bool, bool>({prefix + "not"});
  registerGpuFunction<GpuIsNullFunction, bool, bool>({prefix + "is_null"});

  // --- Decimal -------------------------------------------------------------
  // Five type combinations each, as registerDecimalBinary registers them, with
  // Velox's result precision and scale constraints. initialize() derives the
  // rescale factors from the argument types.
  registerGpuDecimalBinary<functions::detail::DecimalPlusFunction>(
      {prefix + "plus"}, plusMinusConstraints());
  registerGpuDecimalBinary<functions::detail::DecimalMinusFunction>(
      {prefix + "minus"}, plusMinusConstraints());
  registerGpuDecimalBinary<functions::detail::DecimalMultiplyFunction>(
      {prefix + "multiply"}, multiplyConstraints());
  registerGpuDecimalBinary<functions::detail::DecimalDivideFunction>(
      {prefix + "divide"}, divideConstraints());
  registerGpuDecimalBinary<functions::detail::DecimalModulusFunction>(
      {prefix + "mod"}, modulusConstraints());

  registerGpuDecimalToInteger<functions::detail::DecimalFloorFunction>(
      {prefix + "floor"}, roundToIntegerConstraints());
  registerGpuDecimalToInteger<functions::detail::DecimalCeilFunction>(
      {prefix + "ceil"}, roundToIntegerConstraints());
  registerGpuDecimalToInteger<functions::detail::DecimalRoundFunction>(
      {prefix + "round"}, roundToIntegerConstraints());
  registerGpuDecimalToInteger<functions::detail::DecimalTruncateFunction>(
      {prefix + "truncate"}, truncateToIntegerConstraints());

  registerGpuDecimalRoundWithDigits<functions::detail::DecimalRoundFunction>(
      {prefix + "round"});
  registerGpuDecimalTruncateWithDigits<
      functions::detail::DecimalTruncateFunction>({prefix + "truncate"});

  // eq and neq over numbers are absent because their call() bodies are
  // host-only.
}

} // namespace facebook::velox::cudf_velox::gpu_sfi

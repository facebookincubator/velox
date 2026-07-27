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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/CalendarInterval.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

/// Implements Spark's make_interval(years, months, weeks, days, hours, mins,
/// secs) function.
///
/// Returns CalendarIntervalType (int128_t packed as months|days|microseconds).
///
/// Arguments (all optional, default 0):
///   years  - INTEGER
///   months - INTEGER
///   weeks  - INTEGER
///   days   - INTEGER
///   hours  - INTEGER
///   mins   - INTEGER
///   secs   - DECIMAL(precision, 6); unscaled int64 = microseconds
///
/// Overflow behavior: Always throws VELOX_USER_FAIL on overflow. The
/// expression evaluation framework controls whether this propagates as an
/// exception (ANSI mode / throwOnError=true) or gets caught and converted
/// to null (non-ANSI mode / wrapped in TryExpr / throwOnError=false).
/// to null (non-ANSI mode / wrapped in TryExpr / throwOnError=false).
class MakeIntervalFunction : public exec::VectorFunction {
 public:
  MakeIntervalFunction() {}

  /// The output type: CalendarIntervalType (int128_t packed).
  static TypePtr outputType() {
    return CALENDAR_INTERVAL();
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->asFlatVector<int128_t>();

    // Handle zero-argument case: all fields are 0.
    if (args.empty()) {
      rows.applyToSelected([&](vector_size_t row) {
        flatResult->set(row, CalendarInterval(0, 0, 0).pack());
      });
      return;
    }

    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto numArgs = args.size();

    // Use applyToSelectedNoThrow so exceptions are handled by the
    // framework: throwOnError=true re-throws (ANSI), throwOnError=false
    // stores error and nulls the row (TryExpr/non-ANSI).
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      int32_t years = 0;
      int32_t months = 0;
      int32_t weeks = 0;
      int32_t days = 0;
      int32_t hours = 0;
      int32_t mins = 0;
      int64_t secsUnscaled = 0; // microseconds from DECIMAL(p,6)

      if (numArgs >= 1) {
        years = decodedArgs.at(0)->valueAt<int32_t>(row);
      }
      if (numArgs >= 2) {
        months = decodedArgs.at(1)->valueAt<int32_t>(row);
      }
      if (numArgs >= 3) {
        weeks = decodedArgs.at(2)->valueAt<int32_t>(row);
      }
      if (numArgs >= 4) {
        days = decodedArgs.at(3)->valueAt<int32_t>(row);
      }
      if (numArgs >= 5) {
        hours = decodedArgs.at(4)->valueAt<int32_t>(row);
      }
      if (numArgs >= 6) {
        mins = decodedArgs.at(5)->valueAt<int32_t>(row);
      }
      if (numArgs >= 7) {
        secsUnscaled = decodedArgs.at(6)->valueAt<int64_t>(row);
      }

      // Compute totalMonths with overflow check.
      // Spark: totalMonths = Math.addExact(months, years * 12)
      int32_t yearsAsMonths32 = 0;
      if (__builtin_mul_overflow(
              years, static_cast<int32_t>(12), &yearsAsMonths32)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: years({}) * 12 overflows int32",
            years);
      }

      int32_t totalMonths32 = 0;
      if (__builtin_add_overflow(months, yearsAsMonths32, &totalMonths32)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: months({}) + years({}) * 12 overflows int32",
            months,
            years);
      }

      // Compute totalDays with overflow check.
      // Spark: totalDays = Math.addExact(days, Math.multiplyExact(weeks, 7))
      int32_t weeksAsDays32 = 0;
      if (__builtin_mul_overflow(
              weeks, static_cast<int32_t>(7), &weeksAsDays32)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: weeks({}) * 7 overflows int32",
            weeks);
      }

      int32_t totalDays32 = 0;
      if (__builtin_add_overflow(days, weeksAsDays32, &totalDays32)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: days({}) + weeks({}) * 7 overflows int32",
            days,
            weeks);
      }

      // Compute microseconds matching Spark's IntervalUtils.scala:779-781
      // left-fold order: start from secs, add hours*MICROS_PER_HOUR, then
      // add mins*MICROS_PER_MINUTE. This ensures identical overflow points.
      int64_t hoursMicros = 0;
      if (__builtin_mul_overflow(hours, util::kMicrosPerHour, &hoursMicros)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: hours({}) * MICROS_PER_HOUR overflows",
            hours);
      }

      int64_t minsMicros = 0;
      if (__builtin_mul_overflow(mins, util::kMicrosPerMinute, &minsMicros)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: mins({}) * MICROS_PER_MINUTE overflows",
            mins);
      }

      // Start from secsUnscaled, then add hoursMicros, then add minsMicros
      // (matching Spark's left-fold order for identical overflow semantics).
      int64_t totalMicros = secsUnscaled;
      if (__builtin_add_overflow(totalMicros, hoursMicros, &totalMicros)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: secs + hours micros overflows");
      }
      if (__builtin_add_overflow(totalMicros, minsMicros, &totalMicros)) {
        VELOX_USER_FAIL(
            "Integer overflow in make_interval: secs + hours + mins micros overflows");
      }

      flatResult->set(
          row,
          CalendarInterval(totalMonths32, totalDays32, totalMicros).pack());
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    auto outputInterval = "INTERVAL";
    return {
        // make_interval()
        exec::FunctionSignatureBuilder().returnType(outputInterval).build(),
        // make_interval(years)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .build(),
        // make_interval(years, months)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .build(),
        // make_interval(years, months, weeks)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .build(),
        // make_interval(years, months, weeks, days)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .build(),
        // make_interval(years, months, weeks, days, hours)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .build(),
        // make_interval(years, months, weeks, days, hours, mins)
        exec::FunctionSignatureBuilder()
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .build(),
        // make_interval(years, months, weeks, days, hours, mins, secs)
        exec::FunctionSignatureBuilder()
            // Gluten always casts secs to Decimal(8,6), so precision ≤ 18.
            // Restrict to short decimals (int64-backed) since we use
            // valueAt<int64_t>. HUGEINT (precision > 18) would misread.
            .integerVariable("precision", "min(precision, 18)")
            .returnType(outputInterval)
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("decimal(precision, 6)")
            .build(),
    };
  }

 private:
};

} // namespace

/// Factory function for make_interval.
/// Overflow always throws; the expression framework handles null-on-overflow
/// via TryExpr wrapping (when failOnError=false in Spark/non-ANSI mode).
std::shared_ptr<exec::VectorFunction> makeMakeInterval(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& /*config*/) {
  return std::make_shared<MakeIntervalFunction>();
}

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_make_interval,
    MakeIntervalFunction::signatures(),
    makeMakeInterval);

} // namespace facebook::velox::functions::sparksql

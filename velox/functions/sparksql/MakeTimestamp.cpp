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
#include "velox/functions/sparksql/AnsiMode.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/TimestampUtils.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {
namespace {

std::optional<Timestamp> makeTimeStampFromDecodedArgs(
    vector_size_t row,
    DecodedVector* yearVector,
    DecodedVector* monthVector,
    DecodedVector* dayVector,
    DecodedVector* hourVector,
    DecodedVector* minuteVector,
    DecodedVector* microsVector,
    bool ansiEnabled) {
  // Check hour.
  auto hour = hourVector->valueAt<int32_t>(row);
  if (hour < 0 || hour >= 24) {
    return nullOrUserFail(
        ansiEnabled, "Invalid value for hour, must be in [0, 24): {}", hour);
  }
  // Check minute.
  auto minute = minuteVector->valueAt<int32_t>(row);
  if (minute < 0 || minute >= 60) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for minute, must be in [0, 60): {}",
        minute);
  }
  // Check microseconds.
  auto micros = microsVector->valueAt<int64_t>(row);
  if (micros < 0) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for second microseconds, must be non-negative: {}",
        micros);
  }
  auto seconds = micros / util::kMicrosPerSec;
  if (seconds > 60 || (seconds == 60 && micros % util::kMicrosPerSec != 0)) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for second, must be in [0, 60] with 0 microseconds at 60: {}.{:06d}",
        seconds,
        micros % util::kMicrosPerSec);
  }

  // Year, month, day will be checked in utils::daysSinceEpochFromDate.
  Expected<int64_t> daysSinceEpoch = util::daysSinceEpochFromDate(
      yearVector->valueAt<int32_t>(row),
      monthVector->valueAt<int32_t>(row),
      dayVector->valueAt<int32_t>(row));
  if (daysSinceEpoch.hasError()) {
    VELOX_DCHECK(daysSinceEpoch.error().isUserError());
    return nullOrUserFail(ansiEnabled, "{}", daysSinceEpoch.error().message());
  }

  // Micros has at most 8 digits (2 for seconds + 6 for microseconds),
  // thus it's safe to cast micros from int64_t to int32_t.
  auto localMicros =
      util::fromTime(hour, minute, 0, static_cast<int32_t>(micros));
  return util::fromDatetime(daysSinceEpoch.value(), localMicros);
}

// Builds the timestamp from the datetime fields and adjusts it to GMT using
// 'timeZone'. 'timeZone' is nullptr when the timezone argument did not resolve
// to a known zone; this follows the same ANSI rule as invalid datetime fields.
// Fields are validated before the timezone to match Spark's evaluation order.
std::optional<Timestamp> makeTimestampWithTimeZone(
    vector_size_t row,
    DecodedVector* yearVector,
    DecodedVector* monthVector,
    DecodedVector* dayVector,
    DecodedVector* hourVector,
    DecodedVector* minuteVector,
    DecodedVector* microsVector,
    const tz::TimeZone* timeZone,
    std::string_view timeZoneName,
    bool ansiEnabled) {
  auto timestamp = makeTimeStampFromDecodedArgs(
      row,
      yearVector,
      monthVector,
      dayVector,
      hourVector,
      minuteVector,
      microsVector,
      ansiEnabled);
  if (!timestamp.has_value()) {
    return std::nullopt;
  }
  if (timeZone == nullptr) {
    return nullOrUserFail(ansiEnabled, "Unknown time zone: '{}'", timeZoneName);
  }
  toGMTWithGapCorrection(timestamp.value(), *timeZone);
  return timestamp;
}

void setTimestampOrNull(
    int32_t row,
    std::optional<Timestamp> timestamp,
    FlatVector<Timestamp>* result) {
  if (timestamp.has_value()) {
    result->set(row, *timestamp);
  } else {
    result->setNull(row, true);
  }
}

class MakeTimestampFunction : public exec::VectorFunction {
 public:
  MakeTimestampFunction(const tz::TimeZone* sessionTimeZone, bool ansiEnabled)
      : sessionTimeZone_(sessionTimeZone), ansiEnabled_(ansiEnabled) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, TIMESTAMP(), result);
    auto* resultFlatVector = result->as<FlatVector<Timestamp>>();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* year = decodedArgs.at(0);
    auto* month = decodedArgs.at(1);
    auto* day = decodedArgs.at(2);
    auto* hour = decodedArgs.at(3);
    auto* minute = decodedArgs.at(4);
    auto* micros = decodedArgs.at(5);

    if (args.size() == 7) {
      // If the timezone argument is specified, treat the input timestamp as the
      // time in that timezone.
      if (args[6]->isConstantEncoding()) {
        const auto timeZoneName =
            args[6]->asUnchecked<ConstantVector<StringView>>()->valueAt(0);
        const auto* constantTimeZone = tz::locateZone(
            std::string_view(timeZoneName), /*failOnError=*/false);
        context.applyToSelectedNoThrow(rows, [&](auto row) {
          auto timestamp = makeTimestampWithTimeZone(
              row,
              year,
              month,
              day,
              hour,
              minute,
              micros,
              constantTimeZone,
              std::string_view(timeZoneName),
              ansiEnabled_);
          setTimestampOrNull(row, timestamp, resultFlatVector);
        });
      } else {
        auto* timeZone = decodedArgs.at(6);
        context.applyToSelectedNoThrow(rows, [&](auto row) {
          const auto timeZoneName = timeZone->valueAt<StringView>(row);
          const auto* rowTimeZone = tz::locateZone(
              std::string_view(timeZoneName), /*failOnError=*/false);
          auto timestamp = makeTimestampWithTimeZone(
              row,
              year,
              month,
              day,
              hour,
              minute,
              micros,
              rowTimeZone,
              std::string_view(timeZoneName),
              ansiEnabled_);
          setTimestampOrNull(row, timestamp, resultFlatVector);
        });
      }
    } else {
      // Otherwise use session timezone, which is validated at function
      // creation and is never null.
      context.applyToSelectedNoThrow(rows, [&](auto row) {
        auto timestamp = makeTimestampWithTimeZone(
            row,
            year,
            month,
            day,
            hour,
            minute,
            micros,
            sessionTimeZone_,
            /*timeZoneName=*/{},
            ansiEnabled_);
        setTimestampOrNull(row, timestamp, resultFlatVector);
      });
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            // precision <= 18.
            .integerVariable("precision", "min(precision, 18)")
            .returnType("timestamp")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("decimal(precision, 6)")
            .build(),
        exec::FunctionSignatureBuilder()
            // precision <= 18.
            .integerVariable("precision", "min(precision, 18)")
            .returnType("timestamp")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("decimal(precision, 6)")
            .argumentType("varchar")
            .build(),
    };
  }

 private:
  const tz::TimeZone* sessionTimeZone_;
  const bool ansiEnabled_;
};

std::shared_ptr<exec::VectorFunction> createMakeTimestampFunction(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  const auto sessionTzName = config.sessionTimezone();
  VELOX_USER_CHECK(
      !sessionTzName.empty(),
      "make_timestamp requires session time zone to be set.");
  const auto* sessionTimeZone = tz::locateZone(sessionTzName);

  const auto& secondsType = inputArgs[5].type;
  VELOX_USER_CHECK(
      secondsType->isShortDecimal(),
      "Seconds must be short decimal type but got {}",
      secondsType->toString());
  auto secondsScale = secondsType->asShortDecimal().scale();
  VELOX_USER_CHECK_EQ(
      secondsScale,
      6,
      "Seconds fraction must have 6 digits for microseconds but got {}",
      secondsScale);

  const bool ansiEnabled = SparkQueryConfig{config}.ansiEnabled();
  return std::make_shared<MakeTimestampFunction>(sessionTimeZone, ansiEnabled);
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_make_timestamp,
    MakeTimestampFunction::signatures(),
    createMakeTimestampFunction);

} // namespace facebook::velox::functions::sparksql

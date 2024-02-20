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

#include "velox/functions/sparksql/DateTimeFunctions.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {
namespace {

std::optional<Timestamp> makeTimeStampFromDecodedArgs(
    vector_size_t row,
    DecodedVector* yearVector,
    DecodedVector* monthVector,
    DecodedVector* dayVector,
    DecodedVector* hourVector,
    DecodedVector* minuteVector,
    DecodedVector* microsVector) {
  // Check hour.
  auto hour = hourVector->valueAt<int32_t>(row);
  if (hour < 0 || hour > 24) {
    return std::nullopt;
  }
  // Check miniute.
  auto minute = minuteVector->valueAt<int32_t>(row);
  if (minute < 0 || minute > 60) {
    return std::nullopt;
  }
  // Check microseconds.
  auto micros = microsVector->valueAt<int64_t>(row);
  if (micros < 0) {
    return std::nullopt;
  }
  auto seconds = micros / util::kMicrosPerSec;
  if (seconds > 60 || (seconds == 60 && micros % util::kMicrosPerSec != 0)) {
    return std::nullopt;
  }

  // year, month, day will be checked in utils::daysSinceEpochFromDate;
  try {
    auto daysSinceEpoch = util::daysSinceEpochFromDate(
        yearVector->valueAt<int32_t>(row),
        monthVector->valueAt<int32_t>(row),
        dayVector->valueAt<int32_t>(row));
    // micros has at most 8 digits (2 for seconds + 6 for microseconds),
    // thus it's safe to cast micros from int64_t to int32_t.
    auto localMicros = util::fromTime(hour, minute, 0, (int32_t)micros);
    return util::fromDatetime(daysSinceEpoch, localMicros);
  } catch (const VeloxUserError& e) {
    return std::nullopt;
  } catch (const std::exception&) {
    throw;
  }
}

void setTimestampOrNull(
    int32_t row,
    std::optional<Timestamp> timestamp,
    int64_t tzID,
    FlatVector<Timestamp>* result) {
  if (timestamp.has_value()) {
    (*timestamp).toGMT(tzID);
    result->set(row, *timestamp);
  } else {
    result->setNull(row, true);
  }
}

class MakeTimestampFunction : public exec::VectorFunction {
 public:
  MakeTimestampFunction(int64_t sessionTzID) : sessionTzID_(sessionTzID) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, TIMESTAMP(), result);
    auto* resultFlatVector = result->as<FlatVector<Timestamp>>();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto year = decodedArgs.at(0);
    auto month = decodedArgs.at(1);
    auto day = decodedArgs.at(2);
    auto hour = decodedArgs.at(3);
    auto minute = decodedArgs.at(4);
    auto micros = decodedArgs.at(5);

    if (args.size() == 7) {
      // If the timezone argument is specified, treat the input timestamp as the
      // time in that timezone.
      if (args[6]->isConstantEncoding()) {
        auto constantTzID =
            util::getTimeZoneID(args[6]
                                    ->asUnchecked<ConstantVector<StringView>>()
                                    ->valueAt(0)
                                    .str());
        rows.applyToSelected([&](vector_size_t row) {
          auto timestamp = makeTimeStampFromDecodedArgs(
              row, year, month, day, hour, minute, micros);
          setTimestampOrNull(row, timestamp, constantTzID, resultFlatVector);
        });
      } else {
        auto timeZone = decodedArgs.at(6);
        rows.applyToSelected([&](vector_size_t row) {
          auto timestamp = makeTimeStampFromDecodedArgs(
              row, year, month, day, hour, minute, micros);
          auto tzID =
              util::getTimeZoneID(timeZone->valueAt<StringView>(row).str());
          setTimestampOrNull(row, timestamp, tzID, resultFlatVector);
        });
      }
    } else {
      // Otherwise use session timezone.
      rows.applyToSelected([&](vector_size_t row) {
        auto timestamp = makeTimeStampFromDecodedArgs(
            row, year, month, day, hour, minute, micros);
        setTimestampOrNull(row, timestamp, sessionTzID_, resultFlatVector);
      });
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .integerVariable("precision")
            .integerVariable("scale")
            .returnType("timestamp")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("decimal(precision, scale)")
            .build(),
        exec::FunctionSignatureBuilder()
            .integerVariable("precision")
            .integerVariable("scale")
            .returnType("timestamp")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("integer")
            .argumentType("decimal(precision, scale)")
            .argumentType("varchar")
            .build(),
    };
  }

 private:
  int64_t sessionTzID_;
};

std::shared_ptr<exec::VectorFunction> createMakeTimestampFunction(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  const auto sessionTzName = config.sessionTimezone();
  VELOX_USER_CHECK(
      !sessionTzName.empty(),
      "make_timestamp requires session time zone to be set.")
  const auto sessionTzID = util::getTimeZoneID(sessionTzName);

  VELOX_USER_CHECK(
      inputArgs[5].type->isShortDecimal(),
      "Seconds must be short decimal type but got {}",
      inputArgs[5].type->toString());
  auto microsType = inputArgs[5].type->asShortDecimal();
  VELOX_USER_CHECK(
      microsType.scale() == 6,
      "Seconds fraction must have 6 digits for microseconds but got {}",
      microsType.scale());

  return std::make_shared<MakeTimestampFunction>(sessionTzID);
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_make_timestamp,
    MakeTimestampFunction::signatures(),
    createMakeTimestampFunction);

} // namespace facebook::velox::functions::sparksql

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
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"

#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/fuzzer_utils/TimeWithTimezoneInputGenerator.h"
#include "velox/type/Time.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox {

namespace {
void castToTime(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<int64_t>();

  auto convertToLocalTime = [](int64_t timeWithTimezone) {
    int64_t millisUtc = util::unpackMillisUtc(timeWithTimezone);
    auto timezoneMinutes = util::unpackZoneKeyId(timeWithTimezone);
    int16_t offsetMinutes = util::decodeTimezoneOffset(timezoneMinutes);
    return util::utcToLocalTime(millisUtc, offsetMinutes);
  };

  if (input.isConstantEncoding()) {
    const auto timeWithTimezone =
        input.as<ConstantVector<int64_t>>()->valueAt(0);
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      flatResult->set(row, convertToLocalTime(timeWithTimezone));
    });
    return;
  }
  const auto timeWithTimezones = input.as<FlatVector<int64_t>>();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeWithTimezone = timeWithTimezones->valueAt(row);
    flatResult->set(row, convertToLocalTime(timeWithTimezone));
  });
}

void castToString(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<StringView>();
  DecodedVector decoded(input, rows);
  Buffer* buffer = flatResult->getBufferWithSpace(
      rows.countSelected() *
          TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize,
      true /*exactSize*/);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeWithTimezone = decoded.valueAt<int64_t>(row);
    auto output =
        TIME_WITH_TIME_ZONE()->valueToString(timeWithTimezone, rawBuffer);
    flatResult->setNoCopy(row, output);
    rawBuffer += output.size();
  });
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
}

void castFromString(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<int64_t>();
  DecodedVector decoded(input, rows);

  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto stringValue = decoded.valueAt<StringView>(row);

    auto parseResult = util::fromTimeWithTimezoneString(stringValue);
    if (parseResult.hasError()) {
      context.setStatus(row, parseResult.error());
      return;
    }

    flatResult->set(row, parseResult.value());
  });
}

class TimeWithTimeZoneCastOperator final : public exec::CastOperator {
  TimeWithTimeZoneCastOperator() = default;

 public:
  static std::shared_ptr<const exec::CastOperator> get() {
    VELOX_CONSTEXPR_SINGLETON TimeWithTimeZoneCastOperator kInstance;
    return {std::shared_ptr<const exec::CastOperator>{}, &kInstance};
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      case TypeKind::BIGINT:
        return other->isTime();
      default:
        return false;
    }
  }

  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    switch (input.typeKind()) {
      case TypeKind::VARCHAR:
        castFromString(input, context, rows, *result);
        break;
      default:
        VELOX_UNREACHABLE(
            "Cast to TIME WITH TIME ZONE from {} not yet supported",
            input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    switch (resultType->kind()) {
      case TypeKind::VARCHAR:
        castToString(input, context, rows, *result);
        break;
      case TypeKind::BIGINT:
        VELOX_CHECK(resultType->isTime());
        castToTime(input, context, rows, *result);
        break;
      default:
        VELOX_UNREACHABLE(
            "Cast from TIME WITH TIME ZONE to {} not yet supported",
            resultType->toString());
    }
  }
};

class TimeWithTimezoneTypeFactory : public CustomTypeFactory {
 public:
  TimeWithTimezoneTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return TIME_WITH_TIME_ZONE();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return TimeWithTimeZoneCastOperator::get();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::make_shared<fuzzer::TimeWithTimezoneInputGenerator>(
        config.seed_, config.nullRatio_);
  }
};
} // namespace

void registerTimeWithTimezoneType() {
  registerCustomType(
      "time with time zone",
      std::make_unique<const TimeWithTimezoneTypeFactory>());
}

} // namespace facebook::velox

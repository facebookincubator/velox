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

#pragma once

#include "velox/expression/CastKernel.h"

namespace facebook::velox::exec {

class PrestoCastKernel : public CastKernel {
 public:
  explicit PrestoCastKernel(const core::QueryConfig& config);

  bool applyTryCastRecursively() const override {
    return false;
  }

  const TimestampToStringOptions& timestampToStringOptions() const override {
    return timestampToStringOptions_;
  }

  VectorPtr castFromDate(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToDate(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    const auto& fromType = input.type();

    VectorPtr result;
    context.ensureWritable(rows, DATE(), result);
    (*result).clearNulls(rows);
    auto* resultFlatVector = result->as<FlatVector<int32_t>>();

    switch (fromType->kind()) {
      case TypeKind::VARCHAR: {
        auto* inputVector = input.as<SimpleVector<StringView>>();
        applyToSelectedNoThrowLocal(
            rows, context, result, setNullInResultAtError, [&](int row) {
              // Cast from string to date allows only complete ISO 8601
              // formatted strings :
              // [+-](YYYY-MM-DD).
              const auto resultValue = util::fromDateString(
                  inputVector->valueAt(row), util::ParseMode::kPrestoCast);

              if (resultValue.hasValue()) {
                resultFlatVector->set(row, resultValue.value());
              } else {
                if (setNullInResultAtError) {
                  resultFlatVector->setNull(row, true);
                } else if (context.captureErrorDetails()) {
                  context.setStatus(
                      row,
                      Status::UserError(
                          "{} {}",
                          makeErrorMessage(input, row, DATE()),
                          resultValue.error().message()));
                } else {
                  context.setStatus(row, Status::UserError());
                }
              }
            });

        return result;
      }
      case TypeKind::TIMESTAMP: {
        auto* inputVector = input.as<SimpleVector<Timestamp>>();
        const auto* timeZone =
            getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
        applyToSelectedNoThrowLocal(
            rows, context, result, setNullInResultAtError, [&](int row) {
              const auto days =
                  util::toDate(inputVector->valueAt(row), timeZone);
              resultFlatVector->set(row, days);
            });

        return result;
      }
      default:
        VELOX_UNSUPPORTED(
            "Cast from {} to DATE is not supported", fromType->toString());
    }
  }

  VectorPtr castFromIntervalDayTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToIntervalDayTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override;

  VectorPtr castFromTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override;

  VectorPtr castFromDecimal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToDecimal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToBoolean(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::BOOLEAN>(
        input.type(), BOOLEAN(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToTinyInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::TINYINT>(
        input.type(), TINYINT(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToSmallInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::SMALLINT>(
        input.type(), SMALLINT(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToInteger(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::INTEGER>(
        input.type(), INTEGER(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToBigInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::BIGINT>(
        input.type(), BIGINT(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToHugeInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::HUGEINT>(
        input.type(), HUGEINT(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToReal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::REAL>(
        input.type(), REAL(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToDouble(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::DOUBLE>(
        input.type(), DOUBLE(), rows, context, input, setNullInResultAtError);
  }

  VectorPtr castToVarchar(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override;

  VectorPtr castToVarbinary(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToTimestamp(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::TIMESTAMP>(
        input.type(),
        TIMESTAMP(),
        rows,
        context,
        input,
        setNullInResultAtError);
  }

 private:
  template <typename FromNativeType>
  VectorPtr applyDecimalToVarcharCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      bool setNullInResultAtError) const;

  template <typename FromNativeType, TypeKind ToKind>
  VectorPtr applyDecimalToFloatCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename FromNativeType, TypeKind ToKind>
  VectorPtr applyDecimalToIntegralCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename FromNativeType>
  VectorPtr applyDecimalToBooleanCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const;

  template <typename ToNativeType, typename FromNativeType>
  VectorPtr applyIntToDecimalCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename FromNativeType, typename ToNativeType>
  VectorPtr applyDecimalToDecimalCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename ToNativeType, typename FromNativeType>
  VectorPtr applyFloatingPointToDecimalCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename ToNativeType>
  VectorPtr applyVarcharToDecimalCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  VectorPtr applyTimestampToVarcharCast(
      const TypePtr& toType,
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      const BaseVector& input,
      bool setNullInResultAtError) const;

  template <typename TInput>
  VectorPtr applyIntToBinaryCast(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      const BaseVector& input,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToKind>
  VectorPtr applyCastPrimitivesDispatch(
      const TypePtr& fromType,
      const TypePtr& toType,
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      const BaseVector& input,
      bool setNullInResultAtError) const;

  template <TypeKind ToKind, TypeKind FromKind>
  VectorPtr applyCastPrimitivesPolicyDispatch(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      const BaseVector& input,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToKind, TypeKind FromKind, typename TPolicy>
  void applyCastPrimitives(
      vector_size_t row,
      EvalCtx& context,
      const SimpleVector<typename TypeTraits<FromKind>::NativeType>* input,
      bool setNullInResultAtError,
      FlatVector<typename TypeTraits<ToKind>::NativeType>* result) const;

  template <typename T>
  static Expected<T> doCastToFloatingPoint(const StringView& data);

  static inline const tz::TimeZone* FOLLY_NULLABLE
  getTimeZoneFromConfig(const core::QueryConfig& config) {
    if (config.adjustTimestampToTimezone()) {
      const auto sessionTzName = config.sessionTimezone();
      if (!sessionTzName.empty()) {
        return tz::locateZone(sessionTzName);
      }
    }
    return nullptr;
  }

  const bool legacyCast_;
  TimestampToStringOptions timestampToStringOptions_ = {
      .precision = TimestampToStringOptions::Precision::kMilliseconds};
};
} // namespace facebook::velox::exec

#include "velox/expression/PrestoCastKernel-inl.h"

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
      bool setNullInResultAtError) const override;

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
      const TypePtr& toType,
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
      const TypePtr& toType,
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
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::BOOLEAN>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToTinyInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::TINYINT>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToSmallInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::SMALLINT>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToInteger(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::INTEGER>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToBigInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::BIGINT>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToHugeInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::HUGEINT>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToReal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::REAL>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToDouble(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::DOUBLE>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

  VectorPtr castToVarchar(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
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
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return applyCastPrimitivesDispatch<TypeKind::TIMESTAMP>(
        rows, input, context, input.type(), toType, setNullInResultAtError);
  }

 private:
  template <typename FromNativeType>
  VectorPtr applyDecimalToVarcharCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
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
      const TypePtr& toType,
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
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename TInput>
  VectorPtr applyIntToBinaryCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToKind>
  VectorPtr applyCastPrimitivesDispatch(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToKind, TypeKind FromKind>
  VectorPtr applyCastPrimitivesPolicyDispatch(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
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

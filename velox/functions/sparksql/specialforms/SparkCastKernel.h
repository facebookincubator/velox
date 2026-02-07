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

#include "velox/expression/PrestoCastKernel.h"

namespace facebook::velox::functions::sparksql {

// This class provides cast hooks following Spark semantics.
class SparkCastKernel : public exec::PrestoCastKernel {
 public:
  explicit SparkCastKernel(
      const velox::core::QueryConfig& config,
      bool allowOverflow);

  /// Returns the options to cast from timestamp to string.
  const TimestampToStringOptions& timestampToStringOptions() const override {
    return timestampToStringOptions_;
  }

  VectorPtr castToDate(
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
      const TypePtr& toType,
      bool setNullInResultAtError) const override {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        castToBooleanImpl,
        input.type()->kind(),
        rows,
        input,
        context,
        toType,
        setNullInResultAtError);
  }

  VectorPtr castToTinyInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToSmallInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToInteger(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToBigInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToHugeInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToReal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const override;

  VectorPtr castToDouble(
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
      bool setNullInResultAtError) const override;

 private:
  template <typename FromNativeType, TypeKind ToKind>
  VectorPtr applyDecimalToIntegralCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <typename ToNativeType>
  void applyVarcharToDecimalCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind FromTypeKind>
  VectorPtr castToBooleanImpl(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToTypeKind>
  void applyTimestampToIntegerCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind ToTypeKind>
  void applyStringToIntegerCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind FromTypeKind, TypeKind ToTypeKind>
  void applyIntegerToIntegerCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind FromTypeKind, TypeKind ToTypeKind>
  void applyFloatingPointToIntegerCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  /// Handles cast from various source types to an integer type.
  template <TypeKind ToTypeKind, TypeKind FromTypeKind>
  VectorPtr castToIntegerImpl(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  template <TypeKind ToTypeKind>
  void applyStringToFloatingPointCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind FromTypeKind>
  void applyNumberToTimestampCast(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError,
      VectorPtr& result) const;

  template <TypeKind FromTypeKind>
  VectorPtr castToTimestampImpl(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const;

  StringView removeWhiteSpaces(const StringView& view) const;

  // Casts a number to a timestamp. The number is treated as the number of
  // seconds since the epoch (1970-01-01 00:00:00 UTC).
  // Supports integer and floating-point types.
  template <typename T>
  Timestamp castNumberToTimestamp(T seconds) const;

  const core::QueryConfig& config_;

  // If true, the cast will truncate the overflow value to fit the target type.
  const bool allowOverflow_;

  /// 1) Does not follow 'isLegacyCast'. 2) The conversion precision is
  /// microsecond. 3) Does not append trailing zeros. 4) Adds a positive
  /// sign at first if the year exceeds 9999. 5) Respects the configured
  /// session timezone.
  TimestampToStringOptions timestampToStringOptions_ = {
      .precision = TimestampToStringOptions::Precision::kMicroseconds,
      .leadingPositiveSign = true,
      .skipTrailingZeros = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' '};
};
} // namespace facebook::velox::functions::sparksql

#include "velox/functions/sparksql/specialforms/SparkCastKernel-inl.h"

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

#include "velox/expression/EvalCtx.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {

/// This class provides cast hooks to allow different behaviors of CastExpr and
/// SparkCastExpr. The main purpose is to create customized cast implementation
/// by taking full usage of existing cast expression.
class CastKernel {
 public:
  virtual ~CastKernel() = default;

  using ApplyCallback = std::function<void(
      const SelectivityVector& rows,
      const VectorPtr& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      VectorPtr& result)>;

  /// Returns whether to apply try_cast recursively rather than only at the top
  /// level. E.g. if true, an element inside an array would be null rather than
  /// the entire array if the cast of that element fails.
  virtual bool applyTryCastRecursively() const = 0;

  /// Returns the options to cast from timestamp to string.
  virtual const TimestampToStringOptions& timestampToStringOptions() const = 0;

  virtual VectorPtr castFromDate(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToDate(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castFromIntervalDayTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToIntervalDayTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castFromTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToTime(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castFromDecimal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToDecimal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToBoolean(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToTinyInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToSmallInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToInteger(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToBigInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToHugeInt(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToReal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToDouble(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToVarchar(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToVarbinary(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castToTimestamp(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError) const = 0;

  virtual VectorPtr castArray(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError,
      ApplyCallback apply) const = 0;

  virtual VectorPtr castMap(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError,
      ApplyCallback apply) const = 0;

  virtual VectorPtr castRow(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& toType,
      bool setNullInResultAtError,
      ApplyCallback apply) const = 0;

 protected:
  static inline std::string makeErrorMessage(
      const BaseVector& input,
      vector_size_t row,
      const TypePtr& toType,
      const std::string& details = "") {
    return fmt::format(
        "Cannot cast {} '{}' to {}. {}",
        input.type()->toString(),
        input.toString(row),
        toType->toString(),
        details);
  }

  template <typename Func>
  void applyToSelectedNoThrowLocal(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const VectorPtr& result,
      bool setNullInResultAtError,
      Func&& func) const {
    if (setNullInResultAtError) {
      rows.applyToSelected([&](auto row) INLINE_LAMBDA {
        try {
          func(row);
        } catch (const VeloxException& e) {
          if (!e.isUserError()) {
            throw;
          }
          result->setNull(row, true);
        } catch (const std::exception&) {
          result->setNull(row, true);
        }
      });
    } else {
      rows.applyToSelected([&](auto row) INLINE_LAMBDA {
        try {
          func(row);
        } catch (const VeloxException& e) {
          if (!e.isUserError()) {
            throw;
          }

          context.setStatus(row, Status::UserError(e.message()));
        } catch (const std::exception& e) {
          context.setStatus(row, Status::UserError(e.what()));
        }
      });
    }
  }
};
} // namespace facebook::velox::exec

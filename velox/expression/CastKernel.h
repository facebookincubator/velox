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

/// Implementations of this class should provide the logic to handle casting
/// between the core primitive types provided by Velox. This is inteded to be
/// used by CastExpr and custom CastOperators.
///
/// For logical Types (Date, Time, Decimal, etc.) the functions castFromType
/// and castToType are provided. These should handle any supported conversions
/// between core primitive Types in Velox and Type.
///
/// For physical Types (Boolean, Integer, Varchar, etc.) only the function
/// castToType is provided. These only need to handle converting from core
/// physical primitive Types in Velox to Type.
class CastKernel {
 public:
  virtual ~CastKernel() = default;

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

 protected:
  /// Initializes a result vector with the specified type and clears nulls
  /// for the selected rows.
  static FOLLY_ALWAYS_INLINE void initializeResultVector(
      const SelectivityVector& rows,
      const TypePtr& toType,
      exec::EvalCtx& context,
      VectorPtr& result) {
    context.ensureWritable(rows, toType, result);
    result->clearNulls(rows);
  }

  /// Constructs a helpful error message containing the Types involved in the
  /// cast, the value being casted, and any additional details the caller
  /// provides.
  static FOLLY_ALWAYS_INLINE std::string makeErrorMessage(
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

  /// Inokes `func` passing each `row` in `rows`. For each `row` handles any
  /// exceptions by either setting the value in `result` to NULL if
  /// `setNullInResultAtError` is true, or setting the status in `context` if
  /// `setNullInResultAtError` is false.
  ///
  /// If the exception is a VeloxException but not a UserError, the exception
  /// will not be handled.
  template <typename Func>
  static FOLLY_ALWAYS_INLINE void applyToSelectedNoThrowLocal(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      const VectorPtr& result,
      bool setNullInResultAtError,
      Func&& func) {
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

          context.setStatus(row, Status::UserError("{}", e.message()));
        } catch (const std::exception& e) {
          context.setStatus(row, Status::UserError("{}", e.what()));
        }
      });
    }
  }

  static FOLLY_ALWAYS_INLINE void setError(
      const BaseVector& input,
      exec::EvalCtx& context,
      BaseVector& result,
      vector_size_t row,
      const std::string& details,
      bool setNullInResultAtError) {
    if (setNullInResultAtError) {
      result.setNull(row, true);
    } else if (context.captureErrorDetails()) {
      const auto errorDetails =
          makeErrorMessage(input, row, result.type(), details);
      context.setStatus(row, Status::UserError("{}", errorDetails));
    } else {
      context.setStatus(row, Status::UserError());
    }
  }
};
} // namespace facebook::velox::exec

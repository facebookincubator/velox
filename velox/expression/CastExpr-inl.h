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

#include "velox/common/base/CountBits.h"
#include "velox/common/base/Exceptions.h"
#include "velox/core/CoreTypeSystem.h"
#include "velox/expression/CastExpr.h"
#include "velox/expression/StringWriter.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {
namespace {

inline std::string makeErrorMessage(
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

inline std::exception_ptr makeBadCastException(
    const TypePtr& resultType,
    const BaseVector& input,
    vector_size_t row,
    const std::string& errorDetails) {
  return std::make_exception_ptr(VeloxUserError(
      std::current_exception(),
      makeErrorMessage(input, row, resultType, errorDetails),
      false));
}

} // namespace

template <typename Func>
void CastExpr::applyToSelectedNoThrowLocal(
    EvalCtx& context,
    const SelectivityVector& rows,
    VectorPtr& result,
    Func&& func) {
  if (setNullInResultAtError()) {
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
        // Avoid double throwing.
        context.setVeloxExceptionError(row, std::current_exception());
      } catch (const std::exception&) {
        context.setError(row, std::current_exception());
      }
    });
  }
}

/// The per-row level Kernel
/// @tparam ToKind The cast target type
/// @tparam FromKind The expression type
/// @tparam TPolicy The policy used by the cast
/// @param row The index of the current row
/// @param input The input vector (of type FromKind)
/// @param result The output vector (of type ToKind)
template <TypeKind ToKind, TypeKind FromKind, typename TPolicy>
void CastExpr::applyCastKernel(
    vector_size_t row,
    EvalCtx& context,
    const SimpleVector<typename TypeTraits<FromKind>::NativeType>* input,
    FlatVector<typename TypeTraits<ToKind>::NativeType>* result) {
  bool wrapException = true;
  auto setError = [&](const std::string& details) INLINE_LAMBDA {
    if (setNullInResultAtError()) {
      result->setNull(row, true);
    } else {
      wrapException = false;
      if (context.captureErrorDetails()) {
        const auto errorDetails =
            makeErrorMessage(*input, row, result->type(), details);
        context.setStatus(row, Status::UserError("{}", errorDetails));
      } else {
        context.setStatus(row, Status::UserError());
      }
    }
  };

  // If castResult has an error, set the error in context. Otherwise, set the
  // value in castResult directly to result. This lambda should be called only
  // when ToKind is primitive and is not VARCHAR or VARBINARY.
  auto setResultOrError = [&](const auto& castResult, vector_size_t row)
                              INLINE_LAMBDA {
                                if (castResult.hasError()) {
                                  setError(castResult.error().message());
                                } else {
                                  result->set(row, castResult.value());
                                }
                              };

  try {
    auto inputRowValue = input->valueAt(row);

    if constexpr (
        (FromKind == TypeKind::TINYINT || FromKind == TypeKind::SMALLINT ||
         FromKind == TypeKind::INTEGER || FromKind == TypeKind::BIGINT) &&
        ToKind == TypeKind::TIMESTAMP) {
      const auto castResult =
          hooks_->castIntToTimestamp((int64_t)inputRowValue);
      setResultOrError(castResult, row);
      return;
    }

    if constexpr (
        (FromKind == TypeKind::BOOLEAN) && ToKind == TypeKind::TIMESTAMP) {
      const auto castResult = hooks_->castBooleanToTimestamp(inputRowValue);
      setResultOrError(castResult, row);
      return;
    }

    if constexpr (
        (ToKind == TypeKind::TINYINT || ToKind == TypeKind::SMALLINT ||
         ToKind == TypeKind::INTEGER || ToKind == TypeKind::BIGINT) &&
        FromKind == TypeKind::TIMESTAMP) {
      const auto castResult = hooks_->castTimestampToInt(inputRowValue);
      setResultOrError(castResult, row);
      return;
    }

    if constexpr (
        (FromKind == TypeKind::DOUBLE || FromKind == TypeKind::REAL) &&
        ToKind == TypeKind::TIMESTAMP) {
      const auto castResult =
          hooks_->castDoubleToTimestamp(static_cast<double>(inputRowValue));
      if (castResult.hasError()) {
        setError(castResult.error().message());
      } else {
        if (castResult.value().has_value()) {
          result->set(row, castResult.value().value());
        } else {
          result->setNull(row, true);
        }
      }
      return;
    }

    // Optimize empty input strings casting by avoiding throwing exceptions.
    if constexpr (is_string_kind(FromKind)) {
      if constexpr (
          TypeTraits<ToKind>::isPrimitiveType &&
          TypeTraits<ToKind>::isFixedWidth) {
        inputRowValue = hooks_->removeWhiteSpaces(inputRowValue);
        if (inputRowValue.size() == 0) {
          setError("Empty string");
          return;
        }
      }
      if constexpr (ToKind == TypeKind::TIMESTAMP) {
        const auto castResult = hooks_->castStringToTimestamp(inputRowValue);
        setResultOrError(castResult, row);
        return;
      }
      if constexpr (ToKind == TypeKind::REAL) {
        const auto castResult = hooks_->castStringToReal(inputRowValue);
        setResultOrError(castResult, row);
        return;
      }
      if constexpr (ToKind == TypeKind::DOUBLE) {
        const auto castResult = hooks_->castStringToDouble(inputRowValue);
        setResultOrError(castResult, row);
        return;
      }

      if constexpr (
          ToKind == TypeKind::TINYINT || ToKind == TypeKind::SMALLINT ||
          ToKind == TypeKind::INTEGER || ToKind == TypeKind::BIGINT ||
          ToKind == TypeKind::HUGEINT) {
        if constexpr (TPolicy::throwOnUnicode) {
          if (!functions::stringCore::isAscii(
                  inputRowValue.data(), inputRowValue.size())) {
            VELOX_USER_FAIL(
                "Unicode characters are not supported for conversion to integer types");
          }
        }
      }
    }

    const auto castResult =
        util::Converter<ToKind, void, TPolicy>::tryCast(inputRowValue);
    if (castResult.hasError()) {
      setError(castResult.error().message());
      return;
    }

    const auto& output = castResult.value();

    if constexpr (is_string_kind(ToKind)) {
      // Write the result output to the output vector
      auto writer = exec::StringWriter(result, row);
      writer.copy_from(output);
      writer.finalize();
    } else {
      result->set(row, output);
    }

  } catch (const VeloxException& ue) {
    if (!ue.isUserError() || !wrapException) {
      throw;
    }
    setError(ue.message());
  } catch (const std::exception& e) {
    setError(e.what());
  }
}

template <TypeKind ToKind, TypeKind FromKind>
void CastExpr::applyCastPrimitives(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    VectorPtr& result) {
  using To = typename TypeTraits<ToKind>::NativeType;
  using From = typename TypeTraits<FromKind>::NativeType;
  auto* resultFlatVector = result->as<FlatVector<To>>();
  auto* inputSimpleVector = input.as<SimpleVector<From>>();

  switch (hooks_->getPolicy()) {
    case LegacyCastPolicy:
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, util::LegacyCastPolicy>(
            row, context, inputSimpleVector, resultFlatVector);
      });
      break;
    case PrestoCastPolicy:
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, util::PrestoCastPolicy>(
            row, context, inputSimpleVector, resultFlatVector);
      });
      break;
    case SparkCastPolicy:
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, util::SparkCastPolicy>(
            row, context, inputSimpleVector, resultFlatVector);
      });
      break;
    case SparkTryCastPolicy:
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, util::SparkTryCastPolicy>(
            row, context, inputSimpleVector, resultFlatVector);
      });
      break;

    default:
      VELOX_NYI("Policy {} not yet implemented.", hooks_->getPolicy());
  }
}

template <TypeKind ToKind>
void CastExpr::applyCastPrimitivesDispatch(
    const TypePtr& fromType,
    const TypePtr& toType,
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    VectorPtr& result) {
  context.ensureWritable(rows, toType, result);

  // This already excludes complex types, hugeint and unknown from type kinds.
  VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      applyCastPrimitives,
      ToKind,
      fromType->kind() /*dispatched*/,
      rows,
      context,
      input,
      result);
}

} // namespace facebook::velox::exec

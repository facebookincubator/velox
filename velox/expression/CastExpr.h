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
#include "velox/expression/ExprConstants.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/PeeledEncoding.h"
#include "velox/expression/ScopedVarSetter.h"
#include "velox/expression/SpecialForm.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox::exec {

/// Custom operator for casts from and to custom types.
class CastOperator {
 public:
  virtual ~CastOperator() = default;

  /// Determines whether the cast operator supports casting to the custom type
  /// from the other type.
  virtual bool isSupportedFromType(const TypePtr& other) const = 0;

  /// Determines whether the cast operator supports casting from the custom type
  /// to the other type.
  virtual bool isSupportedToType(const TypePtr& other) const = 0;

  /// Casts an input vector to the custom type. This function should not throw
  /// when processing input rows, but report errors via context.setError().
  /// @param input The flat or constant input vector
  /// @param context The context
  /// @param rows Non-null rows of input
  /// @param resultType The result type.
  /// @param result The result vector of the custom type
  virtual void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const = 0;

  virtual void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result,
      const std::shared_ptr<CastKernel>& /* kernel */) const {
    castTo(input, context, rows, resultType, result);
  }

  /// Casts a vector of the custom type to another type. This function should
  /// not throw when processing input rows, but report errors via
  /// context.setError().
  /// @param input The flat or constant input vector
  /// @param context The context
  /// @param rows Non-null rows of input
  /// @param resultType The result type
  /// @param result The result vector of the destination type
  virtual void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const = 0;
};

class CastExpr : public SpecialForm {
 public:
  /// @param type The target type of the cast expression
  /// @param expr The expression to cast
  /// @param trackCpuUsage Whether to track CPU usage
  CastExpr(
      TypePtr type,
      ExprPtr&& expr,
      bool trackCpuUsage,
      bool isTryCast,
      std::shared_ptr<CastKernel> kernel)
      : SpecialForm(
            SpecialFormKind::kCast,
            type,
            std::vector<ExprPtr>({expr}),
            isTryCast ? expression::kTryCast : expression::kCast,
            false /* supportsFlatNoNullsFastPath */,
            trackCpuUsage),
        isTryCast_(isTryCast),
        kernel_(std::move(kernel)) {}

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override {
    VectorPtr input;
    inputs_[0]->eval(rows, context, input);
    auto fromType = inputs_[0]->type();
    auto toType = std::const_pointer_cast<const Type>(type_);

    inTopLevel = true;
    if (isTryCast()) {
      ScopedVarSetter holder{context.mutableThrowOnError(), false};
      ScopedVarSetter captureErrorDetails(
          context.mutableCaptureErrorDetails(), false);

      ScopedThreadSkipErrorDetails skipErrorDetails(true);

      apply(rows, input, context, fromType, toType, result);
    } else {
      apply(rows, input, context, fromType, toType, result);
    }
    // Return 'input' back to the vector pool in 'context' so it can be
    // reused.
    context.releaseVector(input);
  }

  std::string toString(bool recursive = true) const override {
    std::stringstream out;
    out << name() << "(";
    if (recursive) {
      appendInputs(out);
    } else {
      out << inputs_[0]->toString(false);
    }
    out << " as " << type_->toString() << ")";
    return out.str();
  }

  std::string toSql(std::vector<VectorPtr>* complexConstants) const override {
    std::stringstream out;
    out << name() << "(";
    appendInputsSql(out, complexConstants);
    out << " as ";
    toTypeSql(type_, out);
    out << ")";
    return out.str();
  }

 private:
  /// Apply the cast after generating the input vectors
  /// @param rows The list of rows being processed
  /// @param input The input vector to be casted
  /// @param context The context
  /// @param fromType the input type
  /// @param toType the target type
  /// @param result The result vector
  void apply(
      const SelectivityVector& rows,
      const VectorPtr& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      VectorPtr& result) {
    LocalSelectivityVector remainingRows(context, rows);

    context.deselectErrors(*remainingRows);

    LocalDecodedVector decoded(context, *input, *remainingRows);
    auto* rawNulls = decoded->nulls(remainingRows.get());

    if (rawNulls) {
      remainingRows->deselectNulls(
          rawNulls, remainingRows->begin(), remainingRows->end());
    }

    VectorPtr localResult;
    if (!remainingRows->hasSelections()) {
      localResult =
          BaseVector::createNullConstant(toType, rows.end(), context.pool());
    } else if (decoded->isIdentityMapping()) {
      applyPeeled(
          *remainingRows,
          *decoded->base(),
          context,
          fromType,
          toType,
          localResult);
    } else {
      withContextSaver([&](ContextSaver& saver) {
        LocalSelectivityVector newRowsHolder(*context.execCtx());

        LocalDecodedVector localDecoded(context);
        std::vector<VectorPtr> peeledVectors;
        auto peeledEncoding = PeeledEncoding::peel(
            {input}, *remainingRows, localDecoded, true, peeledVectors);
        VELOX_CHECK_EQ(peeledVectors.size(), 1);
        if (peeledVectors[0]->isLazy()) {
          peeledVectors[0] =
              peeledVectors[0]->as<LazyVector>()->loadedVectorShared();
        }
        auto newRows =
            peeledEncoding->translateToInnerRows(*remainingRows, newRowsHolder);
        // Save context and set the peel.
        context.saveAndReset(saver, *remainingRows);
        context.setPeeledEncoding(peeledEncoding);
        applyPeeled(
            *newRows,
            *peeledVectors[0],
            context,
            fromType,
            toType,
            localResult);

        localResult = context.getPeeledEncoding()->wrap(
            toType, context.pool(), localResult, *remainingRows);
      });
    }
    context.moveOrCopyResult(localResult, *remainingRows, result);
    context.releaseVector(localResult);

    // If there are nulls or rows that encountered errors in the input,
    // add nulls to the result at the same rows.
    VELOX_CHECK_NOT_NULL(result);
    if (rawNulls || context.errors()) {
      EvalCtx::addNulls(
          rows, remainingRows->asRange().bits(), context, toType, result);
    }
  }

  // Apply the cast to a vector after vector encodings being peeled off. The
  // input vector is guaranteed to be flat or constant.
  void applyPeeled(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      const TypePtr& fromType,
      const TypePtr& toType,
      VectorPtr& result) {
    auto castFromOperator = getCastOperator(fromType);
    if (castFromOperator && !castFromOperator->isSupportedToType(toType)) {
      VELOX_USER_FAIL(
          "Cannot cast {} to {}.", fromType->toString(), toType->toString());
    }

    auto castToOperator = getCastOperator(toType);
    if (castToOperator && !castToOperator->isSupportedFromType(fromType)) {
      VELOX_USER_FAIL(
          "Cannot cast {} to {}.", fromType->toString(), toType->toString());
    }

    if (castFromOperator || castToOperator) {
      VELOX_USER_CHECK(
          *fromType != *toType,
          "Attempting to cast from {} to itself.",
          fromType->toString());

      auto applyCustomCast = [&]() {
        if (castToOperator) {
          castToOperator->castTo(input, context, rows, toType, result, kernel_);
        } else {
          castFromOperator->castFrom(input, context, rows, toType, result);
        }
      };

      if (setNullInResultAtError()) {
        // This can be optimized by passing setNullInResultAtError() to
        // castTo and castFrom operations.

        EvalErrorsPtr oldErrors;
        context.swapErrors(oldErrors);

        applyCustomCast();

        if (context.errors()) {
          auto errors = context.errors();
          auto rawNulls = result->mutableRawNulls();

          rows.applyToSelected([&](auto row) {
            if (errors->hasErrorAt(row)) {
              bits::setNull(rawNulls, row, true);
            }
          });
        };
        // Restore original state.
        context.swapErrors(oldErrors);

      } else {
        applyCustomCast();
      }
    } else if (fromType->isDate()) {
      result = kernel_->castFromDate(
          rows, input, context, toType, setNullInResultAtError());
    } else if (toType->isDate()) {
      result = kernel_->castToDate(
          rows, input, context, toType, setNullInResultAtError());
    } else if (fromType->isIntervalDayTime()) {
      result = kernel_->castFromIntervalDayTime(
          rows, input, context, toType, setNullInResultAtError());
    } else if (toType->isIntervalDayTime()) {
      result = kernel_->castToIntervalDayTime(
          rows, input, context, toType, setNullInResultAtError());
    } else if (fromType->isTime()) {
      result = kernel_->castFromTime(
          rows, input, context, toType, setNullInResultAtError());
    } else if (toType->isTime()) {
      result = kernel_->castToTime(
          rows, input, context, toType, setNullInResultAtError());
    } else if (fromType->isDecimal()) {
      result = kernel_->castFromDecimal(
          rows, input, context, toType, setNullInResultAtError());
    } else if (toType->isDecimal()) {
      result = kernel_->castToDecimal(
          rows, input, context, toType, setNullInResultAtError());
    } else {
      switch (toType->kind()) {
        case TypeKind::BOOLEAN:
          result = kernel_->castToBoolean(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::TINYINT:
          result = kernel_->castToTinyInt(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::SMALLINT:
          result = kernel_->castToSmallInt(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::INTEGER:
          result = kernel_->castToInteger(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::BIGINT:
          result = kernel_->castToBigInt(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::HUGEINT:
          result = kernel_->castToHugeInt(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::REAL:
          result = kernel_->castToReal(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::DOUBLE:
          result = kernel_->castToDouble(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::VARCHAR:
          result = kernel_->castToVarchar(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::VARBINARY:
          result = kernel_->castToVarbinary(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::TIMESTAMP:
          result = kernel_->castToTimestamp(
              rows, input, context, toType, setNullInResultAtError());
          break;
        case TypeKind::ARRAY:
          result = kernel_->castArray(
              rows,
              input,
              context,
              toType,
              setNullInResultAtError(),
              [this](
                  const SelectivityVector& rows,
                  const VectorPtr& input,
                  exec::EvalCtx& context,
                  const TypePtr& fromType,
                  const TypePtr& toType,
                  VectorPtr& result) {
                ScopedVarSetter holder(&inTopLevel, false);
                apply(rows, input, context, fromType, toType, result);
              });
          break;
        case TypeKind::MAP:
          result = kernel_->castMap(
              rows,
              input,
              context,
              toType,
              setNullInResultAtError(),
              [this](
                  const SelectivityVector& rows,
                  const VectorPtr& input,
                  exec::EvalCtx& context,
                  const TypePtr& fromType,
                  const TypePtr& toType,
                  VectorPtr& result) {
                ScopedVarSetter holder(&inTopLevel, false);
                apply(rows, input, context, fromType, toType, result);
              });
          break;
        case TypeKind::ROW:
          result = kernel_->castRow(
              rows,
              input,
              context,
              toType,
              setNullInResultAtError(),
              [this](
                  const SelectivityVector& rows,
                  const VectorPtr& input,
                  exec::EvalCtx& context,
                  const TypePtr& fromType,
                  const TypePtr& toType,
                  VectorPtr& result) {
                ScopedVarSetter holder(&inTopLevel, false);
                apply(rows, input, context, fromType, toType, result);
              });
          break;
        default:
          VELOX_UNREACHABLE(
              "Unsupported cast from {} to {}.",
              fromType->toString(),
              toType->toString());
      }
    }
  }

  bool isTryCast() const {
    return isTryCast_;
  }

  bool setNullInResultAtError() const {
    return isTryCast() && (inTopLevel || kernel_->applyTryCastRecursively());
  }

  CastOperatorPtr getCastOperator(const TypePtr& type) {
    const auto* key = type->name();

    auto it = castOperators_.find(key);
    if (it != castOperators_.end()) {
      return it->second;
    }

    auto castOperator = getCustomTypeCastOperator(key);
    if (castOperator == nullptr) {
      return nullptr;
    }

    castOperators_.emplace(key, castOperator);
    return castOperator;
  }

  // Custom cast operators for to and from top-level as well as nested
  // types.
  folly::F14FastMap<std::string, CastOperatorPtr> castOperators_;

  bool isTryCast_;
  std::shared_ptr<CastKernel> kernel_;

  bool inTopLevel = false;
};

class CastCallToSpecialForm : public FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;
};

class TryCastCallToSpecialForm : public FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;
};
} // namespace facebook::velox::exec

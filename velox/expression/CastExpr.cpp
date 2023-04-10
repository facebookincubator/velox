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

#include "velox/expression/CastExpr.h"

#include <stdexcept>

#include <fmt/format.h>

#include <velox/common/base/VeloxException.h>
#include "velox/common/base/Exceptions.h"
#include "velox/core/CoreTypeSystem.h"
#include "velox/expression/PeeledEncoding.h"
#include "velox/expression/StringWriter.h"
#include "velox/external/date/tz.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FunctionVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {

namespace {

/// The per-row level Kernel
/// @tparam To The cast target type
/// @tparam From The expression type
/// @param row The index of the current row
/// @param input The input vector (of type From)
/// @param result The output vector (of type To)
/// @return False if the result is null
template <typename To, typename From, bool Truncate>
void applyCastKernel(
    vector_size_t row,
    const SimpleVector<From>* input,
    FlatVector<To>* result,
    bool& nullOutput) {
  // Special handling for string target type
  if constexpr (CppToType<To>::typeKind == TypeKind::VARCHAR) {
    auto output =
        util::Converter<CppToType<To>::typeKind, void, Truncate>::cast(
            input->valueAt(row), nullOutput);
    if (!nullOutput) {
      // Write the result output to the output vector
      auto writer = exec::StringWriter<>(result, row);
      writer.resize(output.size());
      if (output.size()) {
        std::memcpy(writer.data(), output.data(), output.size());
      }
      writer.finalize();
    }
  } else {
    auto output =
        util::Converter<CppToType<To>::typeKind, void, Truncate>::cast(
            input->valueAt(row), nullOutput);
    if (!nullOutput) {
      result->set(row, output);
    }
  }
}

std::string makeErrorMessage(
    const BaseVector& input,
    vector_size_t row,
    const TypePtr& toType) {
  return fmt::format(
      "Failed to cast from {} to {}: {}.",
      input.type()->toString(),
      toType->toString(),
      input.toString(row));
}

template <typename TInput, typename TOutput>
void applyDecimalCastKernel(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    VectorPtr castResult) {
  auto sourceVector = input.as<SimpleVector<TInput>>();
  auto castResultRawBuffer =
      castResult->asUnchecked<FlatVector<TOutput>>()->mutableRawValues();
  const auto& fromPrecisionScale = getDecimalPrecisionScale(*fromType);
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    auto rescaledValue = DecimalUtil::rescaleWithRoundUp<TInput, TOutput>(
        sourceVector->valueAt(row),
        fromPrecisionScale.first,
        fromPrecisionScale.second,
        toPrecisionScale.first,
        toPrecisionScale.second);
    if (rescaledValue.has_value()) {
      castResultRawBuffer[row] = rescaledValue.value();
    } else {
      castResult->setNull(row, true);
    }
  });
}

template <typename TInput, typename TOutput>
void applyIntToDecimalCastKernel(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    VectorPtr castResult) {
  auto sourceVector = input.as<SimpleVector<TInput>>();
  auto castResultRawBuffer =
      castResult->asUnchecked<FlatVector<TOutput>>()->mutableRawValues();
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    auto rescaledValue = DecimalUtil::rescaleInt<TInput, TOutput>(
        sourceVector->valueAt(row),
        toPrecisionScale.first,
        toPrecisionScale.second);
    if (rescaledValue.has_value()) {
      castResultRawBuffer[row] = rescaledValue.value();
    } else {
      castResult->setNull(row, true);
    }
  });
}
} // namespace

template <typename To, typename From>
void CastExpr::applyCastWithTry(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    FlatVector<To>* resultFlatVector) {
  const auto& queryConfig = context.execCtx()->queryCtx()->queryConfig();
  auto isCastIntByTruncate = queryConfig.isCastIntByTruncate();

  auto* inputSimpleVector = input.as<SimpleVector<From>>();

  if (!isCastIntByTruncate) {
    context.applyToSelectedNoThrow(rows, [&](int row) {
      bool nullOutput = false;
      try {
        // Passing a false truncate flag
        applyCastKernel<To, From, false>(
            row, inputSimpleVector, resultFlatVector, nullOutput);
      } catch (const VeloxRuntimeError& re) {
        VELOX_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            re.message());
      } catch (const VeloxUserError& ue) {
        VELOX_USER_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            ue.message());
      } catch (const std::exception& e) {
        VELOX_USER_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            e.what());
      }

      if (nullOutput) {
        VELOX_USER_FAIL(makeErrorMessage(input, row, resultFlatVector->type()));
      }
    });
  } else {
    context.applyToSelectedNoThrow(rows, [&](int row) {
      bool nullOutput = false;
      try {
        // Passing a true truncate flag
        applyCastKernel<To, From, true>(
            row, inputSimpleVector, resultFlatVector, nullOutput);
      } catch (const VeloxRuntimeError& re) {
        VELOX_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            re.message());
      } catch (const VeloxUserError& ue) {
        VELOX_USER_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            ue.message());
      } catch (const std::exception& e) {
        VELOX_USER_FAIL(
            makeErrorMessage(input, row, resultFlatVector->type()) + " " +
            e.what());
      }

      if (nullOutput) {
        VELOX_USER_FAIL(makeErrorMessage(input, row, resultFlatVector->type()));
      }
    });
  }

  // If we're converting to a TIMESTAMP, check if we need to adjust the current
  // GMT timezone to the user provided session timezone.
  if constexpr (CppToType<To>::typeKind == TypeKind::TIMESTAMP) {
    // If user explicitly asked us to adjust the timezone.
    if (queryConfig.adjustTimestampToTimezone()) {
      auto sessionTzName = queryConfig.sessionTimezone();
      if (!sessionTzName.empty()) {
        // locate_zone throws runtime_error if the timezone couldn't be found
        // (so we're safe to dereference the pointer).
        auto* timeZone = date::locate_zone(sessionTzName);
        auto rawTimestamps = resultFlatVector->mutableRawValues();

        rows.applyToSelected(
            [&](int row) { rawTimestamps[row].toGMT(*timeZone); });
      }
    }
  }
}

template <TypeKind Kind>
void CastExpr::applyCast(
    const TypePtr& fromType,
    const TypePtr& toType,
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    VectorPtr& result) {
  using To = typename TypeTraits<Kind>::NativeType;
  context.ensureWritable(rows, toType, result);
  auto* resultFlatVector = result->as<FlatVector<To>>();

  // Unwrapping fromType pointer. VERY IMPORTANT: dynamic type pointer and
  // static type templates in each cast must match exactly
  // @TODO Add support for needed complex types in T74045702
  switch (fromType->kind()) {
    case TypeKind::TINYINT: {
      return applyCastWithTry<To, int8_t>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::SMALLINT: {
      return applyCastWithTry<To, int16_t>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::INTEGER: {
      return applyCastWithTry<To, int32_t>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::BIGINT: {
      return applyCastWithTry<To, int64_t>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::BOOLEAN: {
      return applyCastWithTry<To, bool>(rows, context, input, resultFlatVector);
    }
    case TypeKind::REAL: {
      return applyCastWithTry<To, float>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::DOUBLE: {
      return applyCastWithTry<To, double>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      return applyCastWithTry<To, StringView>(
          rows, context, input, resultFlatVector);
    }
    case TypeKind::DATE: {
      return applyCastWithTry<To, Date>(rows, context, input, resultFlatVector);
    }
    case TypeKind::TIMESTAMP: {
      return applyCastWithTry<To, Timestamp>(
          rows, context, input, resultFlatVector);
    }
    default: {
      VELOX_UNSUPPORTED("Invalid from type in casting: {}", fromType);
    }
  }
}

VectorPtr CastExpr::applyMap(
    const SelectivityVector& rows,
    const MapVector* input,
    exec::EvalCtx& context,
    const MapType& fromType,
    const MapType& toType) {
  // Cast input keys/values vector to output keys/values vector using their
  // element selectivity vector

  // Initialize nested rows
  auto mapKeys = input->mapKeys();
  auto mapValues = input->mapValues();

  SelectivityVector nestedRows;
  BufferPtr elementToTopLevelRows;
  if (fromType.keyType() != toType.keyType() ||
      fromType.valueType() != toType.valueType()) {
    nestedRows = functions::toElementRows(mapKeys->size(), rows, input);
    elementToTopLevelRows = functions::getElementToTopLevelRows(
        mapKeys->size(), rows, input, context.pool());
  }

  ErrorVectorPtr oldErrors;
  context.swapErrors(oldErrors);

  // Cast keys
  VectorPtr newMapKeys;
  if (fromType.keyType() == toType.keyType()) {
    newMapKeys = input->mapKeys();
  } else {
    apply(
        nestedRows,
        mapKeys,
        context,
        fromType.keyType(),
        toType.keyType(),
        newMapKeys);
  }

  // Cast values
  VectorPtr newMapValues;
  if (fromType.valueType() == toType.valueType()) {
    newMapValues = mapValues;
  } else {
    apply(
        nestedRows,
        mapValues,
        context,
        fromType.valueType(),
        toType.valueType(),
        newMapValues);
  }

  context.addElementErrorsToTopLevel(
      nestedRows, elementToTopLevelRows, oldErrors);
  context.swapErrors(oldErrors);

  // Returned map vector should be addressable for every element, even those
  // that are not selected.
  BufferPtr sizes = input->sizes();
  if (newMapKeys->isConstantEncoding() && newMapValues->isConstantEncoding()) {
    // We extends size since that is cheap.
    newMapKeys->resize(input->mapKeys()->size());
    newMapValues->resize(input->mapValues()->size());

  } else if (
      newMapKeys->size() < input->mapKeys()->size() ||
      newMapValues->size() < input->mapValues()->size()) {
    sizes =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), context.pool(), 0);
    auto* inputSizes = input->rawSizes();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    rows.applyToSelected(
        [&](vector_size_t row) { rawSizes[row] = inputSizes[row]; });
  }

  // Assemble the output map
  return std::make_shared<MapVector>(
      context.pool(),
      MAP(toType.keyType(), toType.valueType()),
      input->nulls(),
      rows.end(),
      input->offsets(),
      sizes,
      newMapKeys,
      newMapValues);
}

VectorPtr CastExpr::applyArray(
    const SelectivityVector& rows,
    const ArrayVector* input,
    exec::EvalCtx& context,
    const ArrayType& fromType,
    const ArrayType& toType) {
  // Cast input array elements to output array elements based on their types
  // using their linear selectivity vector
  auto arrayElements = input->elements();

  auto nestedRows =
      functions::toElementRows(arrayElements->size(), rows, input);
  auto elementToTopLevelRows = functions::getElementToTopLevelRows(
      arrayElements->size(), rows, input, context.pool());

  ErrorVectorPtr oldErrors;
  context.swapErrors(oldErrors);

  VectorPtr newElements;
  apply(
      nestedRows,
      arrayElements,
      context,
      fromType.elementType(),
      toType.elementType(),
      newElements);

  if (context.errors()) {
    context.addElementErrorsToTopLevel(
        nestedRows, elementToTopLevelRows, oldErrors);
  }
  context.swapErrors(oldErrors);

  // Returned array vector should be addressable for every element, even those
  // that are not selected.
  BufferPtr sizes = input->sizes();
  if (newElements->isConstantEncoding()) {
    // If the newElements we extends its size since that is cheap.
    newElements->resize(input->elements()->size());
  } else if (newElements->size() < input->elements()->size()) {
    sizes =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), context.pool(), 0);
    auto* inputSizes = input->rawSizes();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    rows.applyToSelected(
        [&](vector_size_t row) { rawSizes[row] = inputSizes[row]; });
  }

  return std::make_shared<ArrayVector>(
      context.pool(),
      ARRAY(toType.elementType()),
      input->nulls(),
      rows.end(),
      input->offsets(),
      sizes,
      newElements);
}

VectorPtr CastExpr::applyRow(
    const SelectivityVector& rows,
    const RowVector* input,
    exec::EvalCtx& context,
    const RowType& fromType,
    const TypePtr& toType) {
  const RowType& toRowType = toType->asRow();
  int numInputChildren = input->children().size();
  int numOutputChildren = toRowType.size();

  // Extract the flag indicating matching of children must be done by name or
  // position
  auto matchByName =
      context.execCtx()->queryCtx()->queryConfig().isMatchStructByName();

  // Cast each row child to its corresponding output child
  std::vector<VectorPtr> newChildren;
  newChildren.reserve(numOutputChildren);

  for (auto toChildrenIndex = 0; toChildrenIndex < numOutputChildren;
       toChildrenIndex++) {
    // For each child, find the corresponding column index in the output
    const auto& toFieldName = toRowType.nameOf(toChildrenIndex);
    bool matchNotFound = false;

    // If match is by field name and the input field name is not found
    // in the output row type, do not consider it in the output
    int fromChildrenIndex = -1;
    if (matchByName) {
      if (!fromType.containsChild(toFieldName)) {
        matchNotFound = true;
      } else {
        fromChildrenIndex = fromType.getChildIdx(toFieldName);
        toChildrenIndex = toRowType.getChildIdx(toFieldName);
      }
    } else {
      fromChildrenIndex = toChildrenIndex;
      if (fromChildrenIndex >= numInputChildren) {
        matchNotFound = true;
      }
    }

    // Updating output types and names
    VectorPtr outputChild;
    const auto& toChildType = toRowType.childAt(toChildrenIndex);

    if (matchNotFound) {
      // Create a vector for null for this child
      context.ensureWritable(rows, toChildType, outputChild);
      outputChild->addNulls(nullptr, rows);
    } else {
      const auto& inputChild = input->children()[fromChildrenIndex];
      if (toChildType == inputChild->type()) {
        outputChild = inputChild;
      } else {
        // Apply cast for the child
        apply(
            rows,
            inputChild,
            context,
            inputChild->type(),
            toChildType,
            outputChild);
      }
    }
    newChildren.emplace_back(std::move(outputChild));
  }

  // Assemble the output row
  return std::make_shared<RowVector>(
      context.pool(),
      toType,
      input->nulls(),
      rows.end(),
      std::move(newChildren));
}

template <typename DecimalType>
VectorPtr CastExpr::applyDecimal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType) {
  VectorPtr castResult;
  context.ensureWritable(rows, toType, castResult);
  (*castResult).clearNulls(rows);
  switch (fromType->kind()) {
    case TypeKind::SHORT_DECIMAL:
      applyDecimalCastKernel<UnscaledShortDecimal, DecimalType>(
          rows, input, context, fromType, toType, castResult);
      break;
    case TypeKind::LONG_DECIMAL:
      applyDecimalCastKernel<UnscaledLongDecimal, DecimalType>(
          rows, input, context, fromType, toType, castResult);
      break;
    case TypeKind::TINYINT:
      applyIntToDecimalCastKernel<int8_t, DecimalType>(
          rows, input, context, toType, castResult);
      break;
    case TypeKind::SMALLINT:
      applyIntToDecimalCastKernel<int16_t, DecimalType>(
          rows, input, context, toType, castResult);
      break;
    case TypeKind::INTEGER:
      applyIntToDecimalCastKernel<int32_t, DecimalType>(
          rows, input, context, toType, castResult);
      break;
    case TypeKind::BIGINT:
      applyIntToDecimalCastKernel<int64_t, DecimalType>(
          rows, input, context, toType, castResult);
      break;
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
  }
  return castResult;
}

void CastExpr::applyPeeled(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    VectorPtr& result) {
  if (castFromOperator_ || castToOperator_) {
    VELOX_CHECK_NE(
        fromType,
        toType,
        "Attempting to cast from {} to itself.",
        fromType->toString());

    if (castToOperator_) {
      castToOperator_->castTo(input, context, rows, toType, result);
    } else {
      castFromOperator_->castFrom(input, context, rows, toType, result);
    }
  } else {
    switch (toType->kind()) {
      case TypeKind::MAP:
        result = applyMap(
            rows,
            input.asUnchecked<MapVector>(),
            context,
            fromType->asMap(),
            toType->asMap());
        break;
      case TypeKind::ARRAY:
        result = applyArray(
            rows,
            input.asUnchecked<ArrayVector>(),
            context,
            fromType->asArray(),
            toType->asArray());
        break;
      case TypeKind::ROW:
        result = applyRow(
            rows,
            input.asUnchecked<RowVector>(),
            context,
            fromType->asRow(),
            toType);
        break;
      case TypeKind::SHORT_DECIMAL:
        result = applyDecimal<UnscaledShortDecimal>(
            rows, input, context, fromType, toType);
        break;
      case TypeKind::LONG_DECIMAL:
        result = applyDecimal<UnscaledLongDecimal>(
            rows, input, context, fromType, toType);
        break;
      default: {
        // Handle primitive type conversions.
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            applyCast,
            toType->kind(),
            fromType,
            toType,
            rows,
            context,
            input,
            result);
      }
    }
  }
}

void CastExpr::apply(
    const SelectivityVector& rows,
    const VectorPtr& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    VectorPtr& result) {
  LocalDecodedVector decoded(context, *input, rows);
  auto* rawNulls = decoded->nulls();

  LocalSelectivityVector nonNullRows(*context.execCtx(), rows.end());
  *nonNullRows = rows;
  if (rawNulls) {
    nonNullRows->deselectNulls(rawNulls, rows.begin(), rows.end());
  }

  VectorPtr localResult;
  if (!nonNullRows->hasSelections()) {
    localResult =
        BaseVector::createNullConstant(toType, rows.end(), context.pool());
  } else if (decoded->isIdentityMapping()) {
    applyPeeled(
        *nonNullRows, *decoded->base(), context, fromType, toType, localResult);
  } else {
    ScopedContextSaver saver;
    LocalSelectivityVector newRowsHolder(*context.execCtx());

    LocalDecodedVector localDecoded(context);
    std::vector<VectorPtr> peeledVectors;
    auto peeledEncoding = PeeledEncoding::Peel(
        {input}, *nonNullRows, localDecoded, true, peeledVectors);
    VELOX_CHECK_EQ(peeledVectors.size(), 1);
    auto newRows =
        peeledEncoding->translateToInnerRows(*nonNullRows, newRowsHolder);
    // Save context and set the peel.
    context.saveAndReset(saver, *nonNullRows);
    context.setPeeledEncoding(peeledEncoding);
    applyPeeled(
        *newRows, *peeledVectors[0], context, fromType, toType, localResult);

    localResult = context.getPeeledEncoding()->wrap(
        toType, context.pool(), localResult, *nonNullRows);
  }
  context.moveOrCopyResult(localResult, *nonNullRows, result);
  context.releaseVector(localResult);

  // If there are nulls in input, add nulls to the result at the same rows.
  VELOX_CHECK_NOT_NULL(result);
  if (rawNulls) {
    Expr::addNulls(
        rows, nonNullRows->asRange().bits(), context, toType, result);
  }
}

void CastExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  VectorPtr input;
  inputs_[0]->eval(rows, context, input);
  auto fromType = inputs_[0]->type();
  auto toType = std::const_pointer_cast<const Type>(type_);

  apply(rows, input, context, fromType, toType, result);
  // Return 'input' back to the vector pool in 'context' so it can be reused.
  context.releaseVector(input);
}

std::string CastExpr::toString(bool recursive) const {
  std::stringstream out;
  out << "cast(";
  if (recursive) {
    appendInputs(out);
  } else {
    out << inputs_[0]->toString(false);
  }
  out << " as " << type_->toString() << ")";
  return out.str();
}

std::string CastExpr::toSql(std::vector<VectorPtr>* complexConstants) const {
  std::stringstream out;
  out << "cast(";
  appendInputsSql(out, complexConstants);
  out << " as ";
  toTypeSql(type_, out);
  out << ")";
  return out.str();
}

TypePtr CastCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /* argTypes */) {
  VELOX_FAIL("CAST expressions do not support type resolution.");
}

ExprPtr CastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool trackCpuUsage) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "CAST statements expect exactly 1 argument, received {}",
      compiledChildren.size());
  return std::make_shared<CastExpr>(
      type, std::move(compiledChildren[0]), trackCpuUsage);
}
} // namespace facebook::velox::exec

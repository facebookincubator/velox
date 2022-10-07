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
#include "velox/expression/LambdaExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::exec {

void ExprCallable::apply(
    const SelectivityVector& rows,
    const SelectivityVector& finalSelection,
    BufferPtr wrapCapture,
    EvalCtx* context,
    const std::vector<VectorPtr>& args,
    VectorPtr* result) {
  doApply(rows, finalSelection, wrapCapture, context, args, result);

  context->addErrors(rows, *lambdaCtx_->errorsPtr(), *context->errorsPtr());
}

void ExprCallable::apply(
    const SelectivityVector& rows,
    const SelectivityVector& finalSelection,
    BufferPtr wrapCapture,
    EvalCtx* context,
    const std::vector<VectorPtr>& args,
    VectorPtr* result,
    const BufferPtr& elementToTopLevelRows) {
  doApply(rows, finalSelection, wrapCapture, context, args, result);

  // Transform error vector to map element rows back to top-level rows.
  lambdaCtx_->addElementErrorsToTopLevel(
      rows, elementToTopLevelRows, *context->errorsPtr());
}

void ExprCallable::doApply(
    const SelectivityVector& rows,
    const SelectivityVector& finalSelection,
    BufferPtr wrapCapture,
    EvalCtx* context,
    const std::vector<VectorPtr>& args,
    VectorPtr* result) {
  std::vector<VectorPtr> allVectors = args;
  for (auto index = args.size(); index < capture_->childrenSize(); ++index) {
    auto values = capture_->childAt(index);
    if (wrapCapture) {
      values = BaseVector::wrapInDictionary(
          BufferPtr(nullptr), wrapCapture, rows.end(), values);
    }
    allVectors.push_back(values);
  }
  auto row = std::make_shared<RowVector>(
      context->pool(),
      capture_->type(),
      BufferPtr(nullptr),
      rows.end(),
      std::move(allVectors));
  lambdaCtx_.emplace(context->execCtx(), context->exprSet(), row.get());
  *lambdaCtx_->mutableThrowOnError() = context->throwOnError();
  if (!context->isFinalSelection()) {
    *lambdaCtx_->mutableIsFinalSelection() = false;
    *lambdaCtx_->mutableFinalSelection() = &finalSelection;
  }
  body_->eval(rows, *lambdaCtx_, *result);
}

std::string LambdaExpr::toString(bool recursive) const {
  if (!recursive) {
    return name_;
  }

  std::string inputs;
  for (int i = 0; i < signature_->size(); ++i) {
    inputs.append(signature_->nameOf(i));
    if (!inputs.empty()) {
      inputs.append(", ");
    }
  }

  for (const auto& field : capture_) {
    inputs.append(field->field());
    if (!inputs.empty()) {
      inputs.append(", ");
    }
  }
  inputs.pop_back();
  inputs.pop_back();

  return fmt::format("({}) -> {}", inputs, body_->toString());
}

std::string LambdaExpr::toSql() const {
  std::ostringstream out;
  out << "(";
  // Inputs.
  for (auto i = 0; i < signature_->size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << signature_->nameOf(i);
  }
  out << ") -> " << body_->toSql();

  return out.str();
}

void LambdaExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  if (!typeWithCapture_) {
    makeTypeWithCapture(context);
  }
  std::vector<VectorPtr> values(typeWithCapture_->size());
  for (auto i = 0; i < captureChannels_.size(); ++i) {
    assert(!values.empty());
    values[signature_->size() + i] = context.getField(captureChannels_[i]);
  }
  auto capture = std::make_shared<RowVector>(
      context.pool(),
      typeWithCapture_,
      BufferPtr(nullptr),
      rows.end(),
      values,
      0);
  auto callable = std::make_shared<ExprCallable>(signature_, capture, body_);
  std::shared_ptr<FunctionVector> functions;
  if (!result) {
    functions = std::make_shared<FunctionVector>(context.pool(), type_);
    result = functions;
  } else {
    VELOX_CHECK(result->encoding() == VectorEncoding::Simple::FUNCTION);
    functions = std::static_pointer_cast<FunctionVector>(result);
  }
  functions->addFunction(callable, rows);
}

void LambdaExpr::makeTypeWithCapture(EvalCtx& context) {
  // On first use, compose the type of parameters + capture and set
  // the indices of captures in the context row.
  if (capture_.empty()) {
    typeWithCapture_ = signature_;
  } else {
    auto& contextType = context.row()->type()->as<TypeKind::ROW>();
    auto parameterNames = signature_->names();
    auto parameterTypes = signature_->children();
    for (auto& reference : capture_) {
      auto& name = reference->field();
      auto channel = contextType.getChildIdx(name);
      captureChannels_.push_back(channel);
      parameterNames.push_back(name);
      parameterTypes.push_back(contextType.childAt(channel));
    }
    typeWithCapture_ =
        ROW(std::move(parameterNames), std::move(parameterTypes));
  }
}
} // namespace facebook::velox::exec

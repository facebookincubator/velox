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

#include "velox/expression/FieldReference.h"

#include "velox/expression/PeeledEncoding.h"

namespace facebook::velox::exec {

void FieldReference::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  if (result) {
    context.ensureWritable(rows, type_, result);
  }
  const RowVector* row;
  DecodedVector decoded;
  VectorPtr input;
  std::shared_ptr<PeeledEncoding> peeledEncoding;
  VectorRecycler inputRecycler(input, context.vectorPool());
  bool useDecode = false;
  LocalSelectivityVector nonNullRowsHolder(*context.execCtx());
  const SelectivityVector* nonNullRows = &rows;
  if (inputs_.empty()) {
    row = context.row();
  } else {
    inputs_[0]->eval(rows, context, input);

    if (auto rowTry = input->as<RowVector>()) {
      // Make sure output is not copied
      if (rowTry->isCodegenOutput()) {
        auto rowType = dynamic_cast<const RowType*>(rowTry->type().get());
        index_ = rowType->getChildIdx(field_);
        result = std::move(rowTry->childAt(index_));
        VELOX_CHECK(result.unique());
        return;
      }
    }

    decoded.decode(*input, rows);
    if (decoded.mayHaveNulls()) {
      nonNullRowsHolder.get(rows);
      nonNullRowsHolder->deselectNulls(
          decoded.nulls(), rows.begin(), rows.end());
      nonNullRows = nonNullRowsHolder.get();
      if (!nonNullRows->hasSelections()) {
        addNulls(rows, decoded.nulls(), context, result);
        return;
      }
    }
    useDecode = !decoded.isIdentityMapping();
    if (useDecode) {
      std::vector<VectorPtr> peeledVectors;
      LocalDecodedVector localDecoded{context};
      peeledEncoding = PeeledEncoding::peel(
          {input}, *nonNullRows, localDecoded, true, peeledVectors);
      VELOX_CHECK_NOT_NULL(peeledEncoding);
      VELOX_CHECK(peeledVectors[0]->encoding() == VectorEncoding::Simple::ROW);
      row = peeledVectors[0]->as<const RowVector>();
    } else {
      VELOX_CHECK(input->encoding() == VectorEncoding::Simple::ROW);
      row = input->as<const RowVector>();
    }
  }
  if (index_ == -1) {
    auto rowType = dynamic_cast<const RowType*>(row->type().get());
    VELOX_CHECK(rowType);
    index_ = rowType->getChildIdx(field_);
  }
  VectorPtr child =
      inputs_.empty() ? context.getField(index_) : row->childAt(index_);
  if (child->encoding() == VectorEncoding::Simple::LAZY) {
    child = BaseVector::loadedVectorShared(child);
  }
  if (result.get()) {
    if (useDecode) {
      child = peeledEncoding->wrap(type_, context.pool(), child, *nonNullRows);
    }
    result->copy(child.get(), *nonNullRows, nullptr);
  } else {
    // The caller relies on vectors having a meaningful size. If we
    // have a constant that is not wrapped in anything we set its size
    // to correspond to rows.end().
    if (!useDecode && child->isConstantEncoding()) {
      child = BaseVector::wrapInConstant(nonNullRows->end(), 0, child);
    }
    result = useDecode ? std::move(peeledEncoding->wrap(
                             type_, context.pool(), child, *nonNullRows))
                       : std::move(child);
  }

  // Check for nulls in the input struct. Propagate these nulls to 'result'.
  if (!inputs_.empty() && decoded.mayHaveNulls()) {
    addNulls(rows, decoded.nulls(), context, result);
  }
}

void FieldReference::evalSpecialFormSimplified(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  ExceptionContextSetter exceptionContext(
      {[](VeloxException::Type /*exceptionType*/, auto* expr) {
         return static_cast<Expr*>(expr)->toString();
       },
       this});
  VectorPtr input;
  const RowVector* row;
  if (inputs_.empty()) {
    row = context.row();
  } else {
    VELOX_CHECK_EQ(inputs_.size(), 1);
    inputs_[0]->evalSimplified(rows, context, input);
    BaseVector::flattenVector(input);
    row = input->as<RowVector>();
    VELOX_CHECK(row);
  }
  auto index = row->type()->asRow().getChildIdx(field_);
  if (index_ == -1) {
    index_ = index;
  } else {
    VELOX_CHECK_EQ(index_, index);
  }
  auto& child = row->childAt(index_);
  context.ensureWritable(rows, type_, result);
  result->copy(child.get(), rows, nullptr);
  if (row->mayHaveNulls()) {
    addNulls(rows, row->rawNulls(), context, result);
  }
}

} // namespace facebook::velox::exec

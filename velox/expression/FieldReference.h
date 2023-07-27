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

#include "velox/expression/SpecialForm.h"

namespace facebook::velox::exec {

class FieldReference : public SpecialForm {
 public:
  FieldReference(
      TypePtr type,
      const std::vector<ExprPtr>& inputs,
      const std::string& field)
      : SpecialForm(
            std::move(type),
            inputs,
            field,
            inputs.empty() ? true : false,
            false /* trackCpuUsage */),
        field_(field) {
    if (!inputs.empty()) {
      VELOX_CHECK(
          inputs.size() == 1, "Field reference can have a max of 1 input.")

      // Check that field exists in the input row type.
      auto rowType = asRowType(inputs[0]->type());
      VELOX_CHECK(rowType, "Input of field reference must be a row type.")
      VELOX_CHECK(
          rowType->getChildIdxIfExists(field_).has_value(),
          fmt::format(
              "Field {} not found in type {}.", field_, rowType->toString()));
    }
  }

  const std::string& field() const {
    return field_;
  }

  bool isConstant() const override {
    return SpecialForm::isConstant() && !inputs_.empty();
  }

  int32_t index(const EvalCtx& context) {
    if (index_ != -1) {
      return index_;
    }
    auto* rowType = dynamic_cast<const RowType*>(context.row()->type().get());
    VELOX_CHECK(rowType, "The context has no row");
    index_ = rowType->getChildIdx(field_);
    return index_;
  }

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

  void evalSpecialFormSimplified(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

  std::string toString(bool recursive = true) const override;

  std::string toSql(
      std::vector<VectorPtr>* complexConstants = nullptr) const override;

 private:
  const std::string field_;
  int32_t index_ = -1;
};
} // namespace facebook::velox::exec

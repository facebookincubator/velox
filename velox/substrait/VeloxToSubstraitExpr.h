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

#include "velox/core/PlanNode.h"

#include "velox/substrait/VeloxToSubstraitFunc.h"
#include "velox/substrait/VeloxToSubstraitType.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

using namespace facebook::velox::core;

namespace facebook::velox::substrait {

class VeloxToSubstraitExprConvertor {
 public:
  // Convert Velox Expression to Substrait Expression.
  void toSubstraitExpr(
      google::protobuf::Arena& arena,
      const std::shared_ptr<const ITypedExpr>& vExpr,
      RowTypePtr vPreNodeOutPut,
      ::substrait::Expression* sExpr);

  // Convert Velox Constant Expression to Substrait
  // Literal Expression.
  void toSubstraitLiteral(
      google::protobuf::Arena& arena,
      const velox::variant& vConstExpr,
      ::substrait::Expression_Literal* sLiteralExpr);

  // Convert Velox vector to Substrait literal.
  void toSubstraitLiteral(
      google::protobuf::Arena& arena,
      const velox::VectorPtr& vVectorValue,
      ::substrait::Expression_Literal_Struct* sLitValue,
      ::substrait::Expression_Literal* sField);

 private:
  // Convert Velox null literal to Substrait null literal.
  void toSubstraitNullLiteral(
      google::protobuf::Arena& arena,
      const velox::TypePtr& vValueType,
      ::substrait::Expression_Literal* sField);

  VeloxToSubstraitTypeConvertor v2STypeConvertor_;
  VeloxToSubstraitFuncConvertor v2SFuncConvertor_;
};

} // namespace facebook::velox::substrait

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

#include "VeloxToSubstraitCallConverter.h"
#include "VeloxToSubstraitExpr.h"
#include "iostream"
#include "unordered_map"
#include "velox/expression/Expr.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/substrait/VeloxToSubstraitExpr.h"
#include "velox/substrait/VeloxToSubstraitFunctionCollector.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

class VeloxToSubstraitFunctionConvertor {
 public:
  void setExprConverter(VeloxToSubstraitExprConvertorPtr exprConvertorPtr) {
    exprConvertorPtr_ = std::move(exprConvertorPtr);
  }

 protected:
  VeloxToSubstraitFunctionConvertor(
      const FunctionSignatureMap& functionSignatureMap)
      : functionSignatureMap_(functionSignatureMap) {}

  const FunctionSignatureMap functionSignatureMap_;
  const VeloxToSubstraitTypeConvertorPtr typeConvertor_;
  VeloxToSubstraitFunctionCollectorPtr functionCollector_;
  VeloxToSubstraitExprConvertorPtr exprConvertorPtr_;
};

class VeloxToSubstraitScalarFunctionConverter
    : public VeloxToSubstraitFunctionConvertor,
      public VeloxToSubstraitCallConverter {
 public:
  VeloxToSubstraitScalarFunctionConverter(
      const FunctionSignatureMap& functionSignatureMap)
      : VeloxToSubstraitFunctionConvertor(functionSignatureMap) {}

 public:
  const std::optional<std::shared_ptr<::substrait::Expression>> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      const RowTypePtr& inputType) const override;
};

} // namespace facebook::velox::substrait
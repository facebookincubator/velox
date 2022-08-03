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
#include "velox/substrait/SubstraitFunction.h"
#include "velox/substrait/SubstraitFunctionCollector.h"
#include "velox/substrait/SubstraitFunctionMappings.h"
#include "velox/substrait/VeloxToSubstraitExpr.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

class SubstraitFunctionLookup {
 private:
  class SubstraitFunctionFinder {
   public:
    const std::optional<SubstraitFunctionPtr> lookupFunction(
        const core::TypedExprPtr& exprPtr) const;
  };

  using FunctionSignatures = std::unordered_map<
      std::string,
      std::shared_ptr<const SubstraitFunctionFinder>&>;

 public:
  SubstraitFunctionLookup(
      const std::vector<SubstraitFunctionPtr>& functions,
      const SubstraitFunctionCollectorPtr& functionCollector);

  const std::optional<SubstraitFunctionPtr> lookupFunction(
      const core::CallTypedExprPtr& callTypeExpr) const;

 protected:
  virtual const FunctionMappingMap& getFunctionMappings() const = 0;

  VeloxToSubstraitTypeConvertorPtr typeConvertor_;
  SubstraitFunctionCollectorPtr functionCollector_;

 private:
  const FunctionSignatures functionSignatures_;
};

class SubstraitScalarFunctionConverter : public SubstraitFunctionLookup,
                                         public VeloxToSubstraitCallConverter {
 public:
  explicit SubstraitScalarFunctionConverter(
      const std::vector<SubstraitFunctionPtr>& functions,
      const SubstraitFunctionCollectorPtr& functionCollector)
      : SubstraitFunctionLookup(functions, functionCollector) {}

  const std::optional<::substrait::Expression> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      const RowTypePtr& inputType,
      SubstraitExprConverter& topLevelConverter) const override;

 protected:
  const FunctionMappingMap& getFunctionMappings() const override {
    return FunctionMappings::scalarMappings();
  }
};

using SubstraitScalarFunctionConverterPtr =
    std::shared_ptr<const SubstraitScalarFunctionConverter>;

class SubstraitAggregateFunctionLookup : public SubstraitFunctionLookup {
 public:
  SubstraitAggregateFunctionLookup(
      const std::vector<SubstraitFunctionPtr>& functions,
      const SubstraitFunctionCollectorPtr& functionCollector)
      : SubstraitFunctionLookup(functions, functionCollector) {}

 protected:
  const FunctionMappingMap& getFunctionMappings() const override {
    return FunctionMappings::aggregateMappings();
  }
};

using SubstraitAggregateFunctionLookupPtr =
    std::shared_ptr<const SubstraitAggregateFunctionLookup>;

} // namespace facebook::velox::substrait

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

#include "velox/common/base/Exceptions.h"
#include "velox/core/Expressions.h"
#include "velox/substrait/SubstraitExtension.h"
#include "velox/substrait/SubstraitParser.h"
#include "velox/substrait/VeloxToSubstraitType.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {

/// A map store function names in difference between Velox and Substrait.
using FunctionMappingMap = std::unordered_map<std::string, std::string>;

struct SubstraitFunctionMappings {
  /// scalar function names in difference between velox and Substrait.
  static FunctionMappingMap& scalarMappings();

  /// aggregate function names in difference between velox and Substrait.
  static FunctionMappingMap& aggregateMappings();

  /// window function names in difference between velox and Substrait.
  static FunctionMappingMap& windowMappings();
};

class SubstraitFunctionLookup {
 public:
  SubstraitFunctionLookup(
      const std::vector<SubstraitFunctionVariantPtr>& functions);

  /// lookup function variant by given CallTyped Expression
  std::optional<std::shared_ptr<SubstraitFunctionVariant>> lookupFunction(
      google::protobuf::Arena& arena,
      const core::CallTypedExprPtr& callTypeExpr) const;

 protected:
  /// get the map which store the function names in difference between velox and
  /// substrait.
  virtual const FunctionMappingMap& getFunctionMappings() const = 0;

 private:
  class SubstraitFunctionFinder {
   public:
    SubstraitFunctionFinder(
        const std::string& name,
        const std::vector<SubstraitFunctionVariantPtr>& functions);

    std::optional<std::shared_ptr<SubstraitFunctionVariant>> lookupFunction(
        const std::string& substraitFuncName,
        google::protobuf::Arena& arena,
        const core::CallTypedExprPtr& exprPtr) const;

   private:
    const std::string& name_;
    std::unordered_map<std::string, SubstraitFunctionVariantPtr> directMap_;
    std::optional<SubstraitFunctionVariantPtr> anyTypeOption_;
    VeloxToSubstraitTypeConvertorPtr typeConvertor_;
  };

  using SubstraitFunctionFinderPtr =
      std::shared_ptr<const SubstraitFunctionFinder>;

  std::unordered_map<std::string, SubstraitFunctionFinderPtr>
      functionSignatures_;
};

class SubstraitScalarFunctionLookup : public SubstraitFunctionLookup {
 public:
  SubstraitScalarFunctionLookup(
      const std::vector<SubstraitFunctionVariantPtr>& functions)
      : SubstraitFunctionLookup(functions) {}

 protected:
  /// A  map store the difference of scalar function names between velox
  /// and substrait.
  const FunctionMappingMap& getFunctionMappings() const override {
    return SubstraitFunctionMappings::scalarMappings();
  }
};

using SubstraitScalarFunctionLookupPtr =
    std::shared_ptr<const SubstraitScalarFunctionLookup>;

class SubstraitAggregateFunctionLookup : public SubstraitFunctionLookup {
 public:
  SubstraitAggregateFunctionLookup(
      const std::vector<SubstraitFunctionVariantPtr>& functions)
      : SubstraitFunctionLookup(functions) {}

 protected:
  /// A  map store the difference of aggregate function names between velox
  /// and substrait.
  const FunctionMappingMap& getFunctionMappings() const override {
    return SubstraitFunctionMappings::aggregateMappings();
  }
};

using SubstraitAggregateFunctionLookupPtr =
    std::shared_ptr<const SubstraitAggregateFunctionLookup>;

} // namespace facebook::velox::substrait

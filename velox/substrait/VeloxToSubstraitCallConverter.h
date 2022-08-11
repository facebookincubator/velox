/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include "optional"
#include "velox/expression/Expr.h"
#include "velox/substrait/SubstraitExtensionCollector.h"
#include "velox/substrait/SubstraitFunctionLookup.h"
#include "velox/substrait/VeloxToSubstraitType.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

using SubstraitExprConverter =
    std::function<::substrait::Expression(const core::TypedExprPtr&)>;

// This class is used to convert the velox CallTypedExpr into substrait scalar
// function expression.
class VeloxToSubstraitCallConverter {
 public:
  /// convert callTypedExpr to substrait Expression.
  virtual const std::optional<::substrait::Expression*> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      SubstraitExprConverter& topLevelConverter) const = 0;
};

using VeloxToSubstraitCallConverterPtr =
    std::shared_ptr<const VeloxToSubstraitCallConverter>;


/// convert 'if' CallTypedExpr to substrait ifThen expression.
class VeloxToSubstraitIfThenConverter : public VeloxToSubstraitCallConverter {
 public:
  const std::optional<::substrait::Expression*> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      SubstraitExprConverter& topLevelConverter) const override;
};

/// convert 'switch' CallTypedExpr to substrait switch expression.
class VeloxToSubstraitSwitchConverter : public VeloxToSubstraitCallConverter {
 public:
  const std::optional<::substrait::Expression*> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      SubstraitExprConverter& topLevelConverter) const override;
};

/// convert callTypedExpr to substrait expression except 'if/switch'
class VeloxToSubstraitScalarFunctionConverter
    : public VeloxToSubstraitCallConverter {
 public:
  VeloxToSubstraitScalarFunctionConverter(
      const SubstraitScalarFunctionLookupPtr functionLookup,
      const SubstraitExtensionCollectorPtr extensionCollector)
      : functionLookup_(functionLookup),
        extensionCollector_(extensionCollector) {}

  const std::optional<::substrait::Expression*> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      SubstraitExprConverter& topLevelConverter) const override;

 private:
  const SubstraitExtensionCollectorPtr extensionCollector_;
  const SubstraitScalarFunctionLookupPtr functionLookup_;
  VeloxToSubstraitTypeConvertorPtr typeConvertor_;
};

} // namespace facebook::velox::substrait

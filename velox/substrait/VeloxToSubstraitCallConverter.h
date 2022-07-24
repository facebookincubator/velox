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
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

// This class is used to convert the velox CallTypedExpr into substrait scalar
// function expression.
class VeloxToSubstraitCallConverter {
 public:
  /**
   * convert callTypedExpr to substrait Expression
   * @param callTypeExpr
   * @param arena
   * @param inputType
   * @return an optional of substrait expression
   */
  virtual const std::optional<std::shared_ptr<::substrait::Expression>> convert(
      const core::CallTypedExprPtr& callTypeExpr,
      google::protobuf::Arena& arena,
      const RowTypePtr& inputType) const = 0;
};

using VeloxToSubstraitCallConverterPtr =
    std::shared_ptr<const VeloxToSubstraitCallConverter>;

} // namespace facebook::velox::substrait
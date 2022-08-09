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

#include <substrait/proto/substrait/algebra.pb.h>
#include <substrait/proto/substrait/plan.pb.h>
#include <optional>
#include "velox/core/Expressions.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/substrait/SubstraitFunction.h"

namespace facebook::velox::substrait {

/// Maintains a mapping between function and function references.
class SubstraitFunctionCollector {
 private:
  /// A bi-direction hash map to keep the relation between reference number and
  /// function
  class BidiMap {
   public:
    void put(const int& reference, const SubstraitFunctionVariantPtr& function);
    std::unordered_map<int, SubstraitFunctionAnchor> forwardMap_;
    std::unordered_map<SubstraitFunctionAnchor, int> reverseMap_;
  };

 public:
  /**
   * get function reference number by given Substrait function
   * @param function substrait extension function
   * @return reference number of a Substrait extension function
   */
  int getFunctionReference(const SubstraitFunctionVariantPtr& function);

  /**
   * add extension functions referenced in a Substrait plan
   * @param plan Substrait plan
   */
  void addFunctionToPlan(::substrait::Plan& plan) const;

 private:
  int counter_ = -1;
  std::shared_ptr<BidiMap> bidiMap_;
};

using SubstraitFunctionCollectorPtr =
    std::shared_ptr< SubstraitFunctionCollector>;

} // namespace facebook::velox::substrait

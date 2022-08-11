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
#include "velox/substrait/SubstraitExtension.h"

namespace facebook::velox::substrait {

/// Maintains a mapping between function and function references.
class SubstraitExtensionCollector {
 public:
  SubstraitExtensionCollector();
  /// get function reference by given Substrait function.
  /// @param function substrait extension function
  /// @return reference number of a Substrait extension function
  int getFunctionReference(const SubstraitFunctionVariantPtr& function);

  /// get type reference by given Substrait type.
  /// @param type substrait extension type
  /// @return reference number of a Substrait extension type
  int getTypeReference(const SubstraitTypeAnchorPtr& type);

  /// add extension functions and types to Substrait plan.
  void addExtensionToPlan(::substrait::Plan* plan) const;

 private:
  /// A bi-direction hash map to keep the relation between reference number and
  /// either function or type.
  /// @T either SubstraitFunctionAnchor or std::string
  template <class T>
  class BidiMap {
   public:
    void put(const int& key, const T& value);
    std::unordered_map<int, T> forwardMap_;
    std::unordered_map<T, int> reverseMap_;
  };

  /// add extension functions to Substrait plan.
  void addFunctionToPlan(::substrait::Plan* plan) const;

  /// add extension functions to Substrait plan.
  void addTypeToPlan(::substrait::Plan* plan) const;

  int functionReference_ = -1;
  int typeReference_ = -1;
  std::shared_ptr<BidiMap<SubstraitFunctionAnchor>> functions_;
  std::shared_ptr<BidiMap<SubstraitTypeAnchor>> types_;
};

using SubstraitExtensionCollectorPtr =
    std::shared_ptr<SubstraitExtensionCollector>;

} // namespace facebook::velox::substrait

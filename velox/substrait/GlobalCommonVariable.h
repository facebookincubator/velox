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

#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/substrait/proto/substrait/type.pb.h"

namespace facebook::velox::substrait {

/// A Singleton class that used to store the common variables that will
/// make them visible for both Velox2SubstraitConvertor and
/// substrait2VeloxConvertor
class GlobalCommonVarSingleton {
 public:
  static GlobalCommonVarSingleton& getInstance();

  GlobalCommonVarSingleton(GlobalCommonVarSingleton const&) = delete;

  GlobalCommonVarSingleton& operator=(GlobalCommonVarSingleton const&) = delete;

  ~GlobalCommonVarSingleton(){};

  std::shared_ptr<::substrait::Plan> getSPlan() const;

  void setSPlan(std::shared_ptr<::substrait::Plan> sPlan);

  uint64_t getPreFunctionReference() const;

  void setPreFunctionReference();

 protected:
  /// An intermediate variable to help us get the corresponding function mapping
  /// relationship when convert from substrait to velox
  std::shared_ptr<::substrait::Plan> sPlan_;

  /// The function id in the extension function mapping
  uint64_t preFunctionReference_;

 private:
  GlobalCommonVarSingleton()
      : sPlan_(std::make_shared<::substrait::Plan>()),
        preFunctionReference_(0){};
};
} // namespace facebook::velox::substrait

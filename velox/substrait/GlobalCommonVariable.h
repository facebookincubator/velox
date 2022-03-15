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
class GlobalCommonVarSingleton {
 public:
  static GlobalCommonVarSingleton& getInstance();
  GlobalCommonVarSingleton(GlobalCommonVarSingleton const&) = delete;
  GlobalCommonVarSingleton& operator=(GlobalCommonVarSingleton const&) = delete;
  ~GlobalCommonVarSingleton(){};

  ::substrait::Plan* getSPlan() const;
  void setSPlan(::substrait::Plan* s_plan);
  const std::unordered_map<uint64_t, std::string>& getFunctionsMap() const;
  void setFunctionsMap(
      const std::unordered_map<uint64_t, std::string>& functions_map);

 protected:
  // An intermediate variable to help us get the corresponding function mapping
  // relationship when convert from substrait to velox
  ::substrait::Plan* sPlan_;

  // parse the function mapping from substrait plan.
  std::unordered_map<uint64_t, std::string> functions_map_;

 private:
  GlobalCommonVarSingleton()
      : sPlan_(new ::substrait::Plan), functions_map_({{0, ""}}){};
};
} // namespace facebook::velox::substrait

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

#include "velox/substrait/VeloxToSubstraitFunc.h"

#include "velox/substrait/GlobalCommonVariable.h"

namespace facebook::velox::substrait {

uint64_t VeloxToSubstraitFuncConvertor::registerSubstraitFunction(
    std::string name) {
  GlobalCommonVarSingleton& sGlobSingleton =
      GlobalCommonVarSingleton::getInstance();

  ::substrait::Plan* sPlanSingleton = sGlobSingleton.getSPlan();

  if (function_map_.find(name) == function_map_.end()) {
    auto function_id = last_function_id++;
    auto sFun = sPlanSingleton->add_extensions()->mutable_extension_function();
    sFun->set_function_anchor(function_id);
    sFun->set_name(name);
    sFun->set_extension_uri_reference(44);

    function_map_[name] = function_id;
  }
  sGlobSingleton.setSPlan(sPlanSingleton);
  return function_map_[name];
}

} // namespace facebook::velox::substrait

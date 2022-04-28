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

  uint64_t lastFunctionId = sGlobSingleton.getPreFunctionReference();

  if (functionMap_.find(name) == functionMap_.end()) {
    auto functionId = lastFunctionId;

    functionMap_[name] = functionId;
    sGlobSingleton.setPreFunctionReference();
  }
  return functionMap_[name];
}

} // namespace facebook::velox::substrait

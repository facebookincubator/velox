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

#include "velox/substrait/GlobalCommonVariable.h"

namespace facebook::velox::substrait {

GlobalCommonVarSingleton& GlobalCommonVarSingleton::getInstance() {
  static GlobalCommonVarSingleton instance;
  return instance;
}

::substrait::Plan* GlobalCommonVarSingleton::getSPlan() const {
  return sPlan_;
}

void GlobalCommonVarSingleton::setSPlan(::substrait::Plan* s_plan) {
  sPlan_ = s_plan;
}

} // namespace facebook::velox::substrait

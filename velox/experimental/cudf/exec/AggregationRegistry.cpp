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

#include "velox/experimental/cudf/exec/AggregationRegistry.h"

namespace facebook::velox::cudf_velox {

StepAwareAggregationRegistry& getStepAwareAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

void appendRegisterAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  auto& registry = getStepAwareAggregationRegistry();
  registry[name][step].push_back(signature);
}

void unregisterAggregateFunctions() {
  auto& registry = getStepAwareAggregationRegistry();
  registry.clear();
}

void unregisterAggregateFunctionsWithPrefix(const std::string& prefix) {
  auto& registry = getStepAwareAggregationRegistry();
  for (auto it = registry.begin(); it != registry.end();) {
    if (it->first.rfind(prefix, 0) == 0) { // starts with prefix
      it = registry.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace facebook::velox::cudf_velox

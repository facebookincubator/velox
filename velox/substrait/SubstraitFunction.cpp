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

#include "SubstraitFunction.h"

namespace facebook::velox::substrait {
std::string SubstraitFunctionVariant::constructKey(
    const std::string& name,
    const std::vector<SubstraitFunctionArgumentPtr>& arguments) {
  std::stringstream ss;
  ss << name;
  if (!arguments.empty()) {
    ss << ":";
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
      const auto& typeSign = (*it)->toTypeString();
      if (it == arguments.end() - 1) {
        ss << typeSign;
      } else {
        ss << typeSign << "_";
      }
    }
  }

  return ss.str();
}

std::vector<SubstraitFunctionArgumentPtr>
SubstraitFunctionVariant::requireArguments() const {
  std::vector<SubstraitFunctionArgumentPtr> res;
  for (auto& arg : arguments) {
    if (arg->isRequired()) {
      res.push_back(arg);
    }
  }
  return res;
}

} // namespace facebook::velox::substrait

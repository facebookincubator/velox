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

namespace facebook::velox::substrait {

class VeloxToSubstraitFuncConvertor {
 public:
  uint64_t registerSubstraitFunction(std::string name);

  // the function mapping get from velox node
  std::unordered_map<std::string, uint64_t> function_map_;
  // the function id in the function mapping
  uint64_t last_function_id = 0;
};

} // namespace facebook::velox::substrait

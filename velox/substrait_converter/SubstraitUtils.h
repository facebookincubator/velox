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

#include <folly/executors/IOThreadPoolExecutor.h>

#include "substrait/algebra.pb.h"
#include "substrait/capabilities.pb.h"
#include "substrait/extensions/extensions.pb.h"
#include "substrait/function.pb.h"
#include "substrait/parameterized_types.pb.h"
#include "substrait/plan.pb.h"
#include "substrait/type.pb.h"
#include "substrait/type_expressions.pb.h"

namespace facebook::velox::substraitconverter {

// This class contains some common funcitons used to parse Substrait components,
// and convert it to recognizable representations.
class SubstraitParser {
 public:
  SubstraitParser();
  struct SubstraitType {
    std::string type;
    bool nullable;
    SubstraitType(const std::string& sub_type, const bool& sub_nullable) {
      type = sub_type;
      nullable = sub_nullable;
    }
  };
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>> parseNamedStruct(
      const substrait::NamedStruct& named_struct);
  std::shared_ptr<SubstraitType> parseType(const substrait::Type& stype);
  std::vector<std::string> makeNames(const std::string& prefix, int size);
  std::string makeNodeName(int node_id, int col_idx);
  std::string findFunction(
      const std::unordered_map<uint64_t, std::string>& functions_map,
      const uint64_t& id) const;
  // Used for mapping Substrait function key words into Velox functions.
  std::unordered_map<std::string, std::string> substrait_velox_function_map = {
      {"MULTIPLY", "multiply"},
      {"SUM", "sum"}};
};

} // namespace facebook::velox::substraitconverter

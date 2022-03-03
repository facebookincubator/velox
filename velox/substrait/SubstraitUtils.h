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

#include "substrait/algebra.pb.h"
#include "substrait/capabilities.pb.h"
#include "substrait/extensions/extensions.pb.h"
#include "substrait/function.pb.h"
#include "substrait/parameterized_types.pb.h"
#include "substrait/plan.pb.h"
#include "substrait/type.pb.h"
#include "substrait/type_expressions.pb.h"

namespace facebook::velox::substrait {

/// This class contains some common funcitons used to parse Substrait
/// components, and convert them into recognizable representations.
class SubstraitParser {
 public:
  /// Used to store the type name and nullability.
  struct SubstraitType {
    std::string type;
    bool nullable;
    SubstraitType(const std::string& subType, const bool& subNullable) {
      type = subType;
      nullable = subNullable;
    }
  };

  /// Used to parse Substrait NamedStruct.
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>> parseNamedStruct(
      const ::substrait::NamedStruct& namedStruct);

  /// Used to parse Substrait Type.
  std::shared_ptr<SubstraitType> parseType(const ::substrait::Type& sType);

  /// Used to parse Substrait ReferenceSegment.
  int32_t parseReferenceSegment(
      const ::substrait::Expression::ReferenceSegment& sRef);

  /// Used to make names in the format of {prefix}_{index}.
  std::vector<std::string> makeNames(const std::string& prefix, int size);

  /// Used to make node name in the format of n{nodeId}_{colIdx}.
  std::string makeNodeName(int nodeId, int colIdx);

  /// Used to find the Substrait function name according to the function id
  /// from a pre-constructed function map.
  std::string findSubstraitFunction(
      const std::unordered_map<uint64_t, std::string>& functionMap,
      const uint64_t& id) const;

  /// Used to find the Velox function name according to the function id
  /// from a pre-constructed function map.
  std::string findVeloxFunction(
      const std::unordered_map<uint64_t, std::string>& functionMap,
      const uint64_t& id) const;

  /// Used to map the Substrait function key word into Velox function key word.
  std::string mapToVeloxFunction(const std::string& subFunc) const;

 private:
  /// Used for mapping Substrait function key words into Velox functions' key
  /// words. Key: the Substrait function key word, Value: the Velox function key
  /// word.
  const std::unordered_map<std::string, std::string> substraitVeloxFunctionMap =
      {{"MULTIPLY", "multiply"}, {"SUM", "sum"}};
};

} // namespace facebook::velox::substrait

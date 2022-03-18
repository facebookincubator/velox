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

#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/capabilities.pb.h"
#include "velox/substrait/proto/substrait/extensions/extensions.pb.h"
#include "velox/substrait/proto/substrait/function.pb.h"
#include "velox/substrait/proto/substrait/parameterized_types.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/substrait/proto/substrait/type.pb.h"
#include "velox/substrait/proto/substrait/type_expressions.pb.h"

namespace facebook::velox::substrait {

/// This class contains some common functions used to parse Substrait
/// components, and convert them into recognizable representations.
class SubstraitParser {
 public:
  /// Used to store the type name and nullability.
  struct SubstraitType {
    std::string type;
    bool nullable;
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
  /// from a pre-constructed function map. The function specification can be
  /// a simple name or a compound name. The compound name format is:
  /// <function name>:<short_arg_type0>_<short_arg_type1>_..._<short_arg_typeN>.
  /// Currently, the input types in the function specification are not used. But
  /// in the future, they should be used for the validation according the
  /// specifications in Substrait yaml files.
  std::string findSubstraitFuncSpec(
      const std::unordered_map<uint64_t, std::string>& functionMap,
      uint64_t id) const;

  /// This function is used to get the function name from the compound name.
  /// When the input is a simple name, it will be returned.
  std::string getSubFunctionName(const std::string& subFuncSpec) const;

  /// Used to find the Velox function name according to the function id
  /// from a pre-constructed function map.
  std::string findVeloxFunction(
      const std::unordered_map<uint64_t, std::string>& functionMap,
      uint64_t id) const;

  /// Used to map the Substrait function key word into Velox function key word.
  std::string mapToVeloxFunction(const std::string& subFunc) const;

 private:
  /// Used for mapping Substrait function key words into Velox functions' key
  /// words. Key: the Substrait function key word, Value: the Velox function key
  /// word. For those functions with different names in Substrait and Velox,
  /// a mapping relation should be added here.
  std::unordered_map<std::string, std::string> substraitVeloxFunctionMap;
};

} // namespace facebook::velox::substrait

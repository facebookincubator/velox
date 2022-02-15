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

#include "substrait_utils.h"

namespace facebook::velox::substrait {

SubstraitParser::SubstraitParser() {}

std::shared_ptr<SubstraitParser::SubstraitType> SubstraitParser::parseType(
    const substrait::Type& stype) {
  std::string type_name;
  substrait::Type_Nullability nullability;
  switch (stype.kind_case()) {
    case substrait::Type::KindCase::kBool: {
      type_name = "BOOL";
      nullability = stype.bool_().nullability();
      break;
    }
    case substrait::Type::KindCase::kFp64: {
      type_name = "FP64";
      nullability = stype.fp64().nullability();
      break;
    }
    case substrait::Type::KindCase::kStruct: {
      // TODO
      auto sstruct = stype.struct_();
      auto stypes = sstruct.types();
      for (auto& type : stypes) {
        parseType(type);
      }
      break;
    }
    case substrait::Type::KindCase::kString: {
      type_name = "STRING";
      nullability = stype.string().nullability();
      break;
    }
    default:
      std::cout << "Type not supported" << std::endl;
      break;
  }
  bool nullable;
  switch (nullability) {
    case substrait::Type_Nullability::Type_Nullability_NULLABILITY_UNSPECIFIED:
      nullable = true;
      break;
    case substrait::Type_Nullability::Type_Nullability_NULLABILITY_NULLABLE:
      nullable = true;
      break;
    case substrait::Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED:
      nullable = false;
      break;
    default:
      throw new std::runtime_error("Unrecognized NULLABILITY.");
      break;
  }
  std::shared_ptr<SubstraitType> substrait_type =
      std::make_shared<SubstraitType>(type_name, nullable);
  return substrait_type;
}

std::vector<std::shared_ptr<SubstraitParser::SubstraitType>>
SubstraitParser::parseNamedStruct(const substrait::NamedStruct& named_struct) {
  auto& snames = named_struct.names();
  std::vector<std::string> name_list;
  for (auto& sname : snames) {
    name_list.push_back(sname);
  }
  // Parse Struct
  auto& sstruct = named_struct.struct_();
  auto& stypes = sstruct.types();
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>> substrait_type_list;
  for (auto& type : stypes) {
    auto substrait_type = parseType(type);
    substrait_type_list.push_back(substrait_type);
  }
  return substrait_type_list;
}

std::vector<std::string> SubstraitParser::makeNames(const std::string& prefix, int size) {
  std::vector<std::string> names;
  for (int i = 0; i < size; i++) {
    names.push_back(fmt::format("{}_{}", prefix, i));
  }
  return names;
}

std::string SubstraitParser::makeNodeName(int node_id, int col_idx) {
  return fmt::format("n{}_{}", node_id, col_idx);
}

std::string SubstraitParser::findFunction(
    const std::unordered_map<uint64_t, std::string>& functions_map,
    const uint64_t& id) const {
  if (functions_map.find(id) == functions_map.end()) {
    throw std::runtime_error("Could not find function " + std::to_string(id));
  }
  std::unordered_map<uint64_t, std::string>& map =
      const_cast<std::unordered_map<uint64_t, std::string>&>(functions_map);
  return map[id];
}

} // namespace facebook::velox::substrait

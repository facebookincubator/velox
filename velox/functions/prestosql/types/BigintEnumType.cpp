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

#include "BigintEnumType.h"

#include <stdexcept>

namespace facebook::velox {

// Enum Type (enum name and key-value mapping) is passed from the coordinator in
// the following format as a string:
// e.g., "test.enum.mood:BigintEnum(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0})".
// This function checks and extracts the following:
// 1. Check that the type is correct (BigintEnum).
// 2. Extract the name of the enum type ("test.enum.mood").
// 3. Construct a map of the enum mapping ({"CURIOUS": -2, "HAPPY": 0}).
std::pair<std::string, std::unordered_map<std::string, int64_t>>
BigintEnumType::parseTypeInfo(const std::string& enumTypeString) {
  std::string enumName;
  std::unordered_map<std::string, int64_t> enumMap;
  std::string inputStr = enumTypeString;
  boost::replace_all(inputStr, "“", "\"");
  boost::replace_all(inputStr, "”", "\"");

  size_t colonPos = inputStr.find(':');
  size_t parenthesisPos = inputStr.find('(', colonPos);
  size_t braceOpen = inputStr.find('{', parenthesisPos);
  size_t braceClose = inputStr.find('}', braceOpen);

  if (colonPos == std::string::npos || parenthesisPos == std::string::npos ||
      braceOpen == std::string::npos || braceClose == std::string::npos) {
    throw std::invalid_argument("malformed enum type");
  }

  std::string type =
      inputStr.substr(colonPos + 1, parenthesisPos - colonPos - 1);
  boost::algorithm::trim(type);
  if (type != "BigintEnum") {
    throw std::invalid_argument("invalid type: " + type);
  }

  std::string name =
      inputStr.substr(parenthesisPos + 1, braceOpen - parenthesisPos - 1);
  boost::algorithm::trim(name);
  enumName = boost::algorithm::to_upper_copy(name);

  std::string mapBody =
      inputStr.substr(braceOpen + 1, braceClose - braceOpen - 1);
  boost::algorithm::trim(mapBody);

  std::vector<std::string> entries;
  boost::split(entries, mapBody, boost::is_any_of(","));

  for (auto& entry : entries) {
    boost::algorithm::trim(entry);
    size_t quote1 = entry.find('"');
    size_t quote2 = entry.find('"', quote1 + 1);
    size_t colon = entry.find(':', quote2);

    if (quote1 == std::string::npos || quote2 == std::string::npos ||
        colon == std::string::npos) {
      LOG(WARNING) << "invalid map entry: " << entry << "\n";
      continue;
    }

    std::string key = entry.substr(quote1 + 1, quote2 - quote1 - 1);
    std::string valStr = entry.substr(colon + 1);
    boost::algorithm::trim(valStr);

    try {
      int64_t value = std::stoi(valStr);
      enumMap[key] = value;
    } catch (...) {
      throw std::invalid_argument("invalid value " + valStr);
    }
  }
  return {enumName, enumMap};
}

std::unordered_map<std::string, std::shared_ptr<const BigintEnumType>>&
BigintEnumType::getInstances() {
  static std::unordered_map<std::string, std::shared_ptr<const BigintEnumType>>
      instances;
  return instances;
}

// A new BigintEnumType is created for each enum type.
// i.e. "test.enum.mood" is not the same type as "test.enum.languages"
// Each instance of a BigintEnumType is stored in a static unordered map with
// the enum names as keys, so if an instance for a given enum name already
// exists, that instance will be returned. Otherwise, a new instance is created,
// added to the map, then returned.
std::shared_ptr<const BigintEnumType> BigintEnumType::create(
    const std::string& enumName,
    const std::unordered_map<std::string, int64_t>& enumMap) {
  auto instance = BigintEnumType::get(enumName);
  if (instance) {
    return instance.value();
  }

  auto& instances = getInstances();
  auto newInstance = std::make_shared<const BigintEnumType>(enumName, enumMap);
  instances[enumName] = newInstance;
  return instances[enumName];
}

std::optional<std::shared_ptr<const BigintEnumType>> BigintEnumType::get(
    const std::string& enumName) {
  auto& instances = getInstances();
  auto it = instances.find(enumName);
  if (it != instances.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace facebook::velox

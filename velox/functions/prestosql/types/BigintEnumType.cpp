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
#include <boost/algorithm/string.hpp>
#include <stdexcept>

namespace facebook::velox {

// Type info (enum name and key-value mapping) is passed from the coordinator in
// the following format as a string:
// e.g., "test.enum.mood:BigintEnum(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0})".
// This function checks and extracts the following:
// 1. Check that the type is correct (BigintEnum).
// 2. Extract the name of the enum type ("test.enum.mood").
// 3. Construct a map of the enum mapping ({"CURIOUS": -2, "HAPPY": 0}).
void BigintEnumType::parseTypeInfo(const std::string& typeInfoString) {
  std::string inputStr = typeInfoString;
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
  enumName_ = boost::algorithm::to_upper_copy(name);

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
      enumMap_[key] = value;
    } catch (...) {
      throw std::invalid_argument("invalid value " + valStr);
    }
  }
}

// A new BigintEnumType is created for each enum type.
// i.e. "test.enum.mood" is not the same type as "test.enum.languages"
// Each instance of a BigintEnumType is stored in a static unordered map,
// so if an instance for a given typeInfoString already exists, that instance
// will be returned. Otherwise, a new instance is created, added to the map,
// then returned.
const std::shared_ptr<const BigintEnumType>& BigintEnumType::get(
    const std::string& typeInfoString) {
  static std::unordered_map<std::string, std::shared_ptr<const BigintEnumType>>
      instances;
  auto it = instances.find(typeInfoString);
  if (it != instances.end()) {
    return it->second;
  } else {
    auto instance = std::make_shared<const BigintEnumType>(typeInfoString);
    instances[typeInfoString] = instance;
    return instances[typeInfoString];
  }
}

} // namespace facebook::velox

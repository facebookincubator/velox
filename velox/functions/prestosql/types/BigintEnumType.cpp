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

#include <folly/Synchronized.h>

#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {

BigintEnumType::BigintEnumType(const std::vector<TypeParameter>& typeParameters)
    : parameters_{typeParameters} {
  name_ = typeParameters[0].longEnumLiteral.value().name;
  for (const auto& [key, value] :
       typeParameters[0].longEnumLiteral.value().valuesMap) {
    VELOX_USER_CHECK(
        flippedMap_.count(value) == 0,
        "Invalid enum type {}, contains duplicate value {}",
        name_,
        value);
    flippedMap_[value] = key;
  }
}

std::string BigintEnumType::flippedMapToString(
    const std::unordered_map<int64_t, std::string>& flippedMap) {
  std::map<std::string, int64_t> sortedMap;
  for (const auto& [key, value] : flippedMap) {
    sortedMap[value] = key;
  }
  return orderedMapToString(sortedMap);
}

std::string BigintEnumType::mapToString(
    const std::unordered_map<std::string, int64_t>& unorderedMap) {
  std::map<std::string, int64_t> sortedMap(
      unorderedMap.begin(), unorderedMap.end());
  for (const auto& [key, value] : unorderedMap) {
    sortedMap[key] = value;
  }
  return orderedMapToString(sortedMap);
}

std::string BigintEnumType::orderedMapToString(
    const std::map<std::string, int64_t>& valuesMap) {
  // The values map should be sorted when converting to string for 2 reasons:
  // 1. Printing the type will be consistent.
  // 2. The enum name + values map is used as the key in the cache for get(), so
  // the string should be consistent.
  std::map<std::string, int64_t> sortedMap(valuesMap.begin(), valuesMap.end());
  for (const auto& [key, value] : valuesMap) {
    sortedMap[key] = value;
  }
  std::ostringstream oss;
  oss << "{";
  for (auto it = sortedMap.begin(); it != sortedMap.end(); ++it) {
    if (it != sortedMap.begin()) {
      oss << ", ";
    }
    oss << "\"" << it->first << "\"" << ": " << it->second;
  }
  oss << "}";
  return oss.str();
}

// A thread-safe LRU cache to store instances of BigintEnumType.
using Cache = folly::Synchronized<
    folly::EvictingCacheMap<std::string, BigintEnumTypePtr>>;

BigintEnumTypePtr BigintEnumType::get(
    const std::vector<TypeParameter>& typeParameters) {
  VELOX_CHECK_EQ(
      typeParameters.size(),
      1,
      "Expected exactly one type parameters for BigintEnumType");
  VELOX_CHECK(
      typeParameters[0].longEnumLiteral.has_value(),
      "BigintEnumType parameter must be longEnumLiteral");

  const auto& mapKey = typeParameters[0].longEnumLiteral.value().name +
      mapToString(typeParameters[0].longEnumLiteral.value().valuesMap);

  const int maxCacheSize = 1000;
  static Cache kCache{
      folly::EvictingCacheMap<std::string, BigintEnumTypePtr>(maxCacheSize)};

  return kCache.withWLock([&](auto& cache) -> BigintEnumTypePtr {
    auto it = cache.find(mapKey);
    if (it != cache.end()) {
      return it->second;
    }

    auto instance = std::shared_ptr<const BigintEnumType>(
        new BigintEnumType(typeParameters));
    cache.insert(mapKey, instance);
    return instance;
  });
}

folly::dynamic BigintEnumType::serializeEnumParameter() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["enumName"] = name_;
  folly::dynamic valuesMap = folly::dynamic::object;
  for (const auto& [key, value] : flippedMap_) {
    valuesMap[value] = key;
  }
  obj["valuesMap"] = valuesMap;
  return obj;
}

folly::dynamic BigintEnumType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  obj["kLongEnumParam"] = serializeEnumParameter();
  return obj;
}

} // namespace facebook::velox

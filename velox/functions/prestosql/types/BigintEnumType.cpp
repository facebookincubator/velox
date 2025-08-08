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
#include <folly/container/EvictingCacheMap.h>

#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {

BigintEnumType::BigintEnumType(const std::vector<TypeParameter>& typeParameters)
    : parameters_{typeParameters} {
  VELOX_CHECK_EQ(
      typeParameters.size(),
      1,
      "Expected exactly one type parameters for BigintEnumType");
  VELOX_CHECK(
      typeParameters[0].longEnumLiteral.has_value(),
      "First parameter of BigintEnumType must be a long enum literal");

  name_ = typeParameters[0].longEnumLiteral.value().name;
  map_ = typeParameters[0].longEnumLiteral.value().valuesMap;
  for (const auto& [key, value] : map_) {
    flippedMap_[value] = key;
  }
}

std::string BigintEnumType::mapToString(
    std::unordered_map<std::string, int64_t> valuesMap) {
  // The values map should be sorted when converting to string for 2 reasons:
  // 1. Printing the type will be consistent.
  // 2. The enum name + values map is used as the key in the cache for get(), so
  // the string should be consistent.
  std::map<std::string, int64_t> sortedMap(valuesMap.begin(), valuesMap.end());
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

const std::string& BigintEnumType::keyAt(
    int64_t value,
    const std::unordered_map<int64_t, std::string>& flippedLongEnumMap) {
  auto it = flippedLongEnumMap.find(value);
  if (it != flippedLongEnumMap.end()) {
    return it->second;
  }
  VELOX_USER_FAIL("Value \'{}\' not in enum 'BigintEnum'", value);
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

    auto instance = std::make_shared<BigintEnumType>(typeParameters);
    cache.insert(mapKey, instance);
    return instance;
  });
}

folly::dynamic BigintEnumType::serializeEnumParameter(
    const std::string& name,
    const std::unordered_map<std::string, int64_t>& map) const {
  folly::dynamic obj = folly::dynamic::object;
  obj["enumName"] = name;
  folly::dynamic valuesMap = folly::dynamic::array;
  for (const auto& [key, value] : map) {
    folly::dynamic pair = folly::dynamic::array(key, value);
    valuesMap.push_back(std::move(pair));
  }
  obj["valuesMap"] = valuesMap;
  return obj;
}

folly::dynamic BigintEnumType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  obj["kLongEnumParam"] = serializeEnumParameter(name_, map_);
  return obj;
}

} // namespace facebook::velox

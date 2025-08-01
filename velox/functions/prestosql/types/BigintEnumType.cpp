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
  VELOX_CHECK_EQ(
      typeParameters.size(),
      2,
      "Expected exactly two type parameters for BigintEnumType");
  VELOX_CHECK(
      typeParameters[0].stringLiteral.has_value(),
      "First parameter of BigintEnumType must be a string literal");
  VELOX_CHECK(
      typeParameters[1].longEnumMapLiteral.has_value(),
      "Second parameter of BigintEnumType must be a long enum map literal");
  name_ = typeParameters[0].stringLiteral.value();
  longEnumMap_ = typeParameters[1].longEnumMapLiteral.value();
  for (const auto& [key, value] : longEnumMap_) {
    flippedLongEnumMap_[value] = key;
  }
}

std::string BigintEnumType::mapToString(
    std::map<std::string, int64_t> longEnumMap) {
  std::ostringstream oss;
  oss << "{";
  for (auto it = longEnumMap.begin(); it != longEnumMap.end(); ++it) {
    if (it != longEnumMap.begin()) {
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
      2,
      "Expected exactly two type parameters for BigintEnumType");
  VELOX_CHECK(
      typeParameters[0].stringLiteral.has_value(),
      "First parameter of BigintEnumType must be a string literal");
  VELOX_CHECK(
      typeParameters[1].longEnumMapLiteral.has_value(),
      "Second parameter of BigintEnumType must be a long enum map literal");
  const auto& mapKey = typeParameters[0].stringLiteral.value() +
      mapToString(typeParameters[1].longEnumMapLiteral.value());

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

folly::dynamic serializeMapParameter(
    const std::map<std::string, int64_t>& map) {
  folly::dynamic obj = folly::dynamic::array;
  for (const auto& [key, value] : map) {
    folly::dynamic pair = folly::dynamic::array(key, value);
    obj.push_back(std::move(pair));
  }
  return obj;
}

folly::dynamic BigintEnumType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  obj["stringParam"] = name_;
  obj["kLongEnumMapParam"] = serializeMapParameter(longEnumMap_);

  return obj;
}

} // namespace facebook::velox

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

#include <mutex>

#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {
const std::shared_ptr<const BigintEnumType>& BigintEnumType::get(
    const std::vector<TypeParameter>& typeParameters) {
  VELOX_CHECK_EQ(typeParameters.size(), 2);
  VELOX_CHECK(typeParameters[0].stringLiteral.has_value());
  VELOX_CHECK(typeParameters[1].stringLiteral.has_value());

  auto name = typeParameters[0].stringLiteral.value();
  auto mapString = typeParameters[1].stringLiteral.value();
  auto mapKey = name + mapString;

  const int maxCacheSize = 1000;
  static folly::
      EvictingCacheMap<std::string, std::shared_ptr<const BigintEnumType>>
          typeCache(maxCacheSize);
  static std::mutex cacheMutex;
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (typeCache.exists(mapKey)) {
      return typeCache.get(mapKey);
    }
  }
  static std::shared_ptr<const BigintEnumType> bigintInstance;
  bigintInstance =
      std::shared_ptr<const BigintEnumType>(new BigintEnumType(typeParameters));
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    typeCache.insert(mapKey, bigintInstance);
  }
  return bigintInstance;
}

} // namespace facebook::velox

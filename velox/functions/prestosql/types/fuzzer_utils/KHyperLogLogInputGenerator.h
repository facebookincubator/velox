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

#include "velox/common/fuzzer/Utils.h"
#include "velox/common/hyperloglog/KHyperLogLog.h"
#include "velox/type/Type.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class KHyperLogLogInputGenerator : public AbstractInputGenerator {
 public:
  KHyperLogLogInputGenerator(
      const size_t seed,
      const double nullRatio,
      memory::MemoryPool* pool,
      int32_t minNumValues = 1);

  variant generate() override;

 private:
  template <typename T>
  variant generateTyped() {
    common::hll::KHyperLogLog<memory::MemoryPool> khll{
        maxSize_, hllBuckets_, pool_};

    auto numPairs = rand<int32_t>(rng_, minNumValues_, 100);
    for (auto i = 0; i < numPairs; ++i) {
      int64_t joinKey;
      int64_t uii;

      if constexpr (std::is_same_v<T, int64_t>) {
        // For BIGINT: use both random joinKey and UII.
        joinKey = rand<int64_t>(rng_);
        uii = rand<int64_t>(rng_);
      } else if constexpr (
          std::is_same_v<T, std::string> || std::is_same_v<T, StringView>) {
        // For VARCHAR: generate random string and hash to int64_t for
        // joinKey.
        auto size = rand<int32_t>(rng_, 0, 100);
        std::string str;
        str.reserve(size);
        for (int j = 0; j < size; ++j) {
          char c = static_cast<char>(rand<int32_t>(rng_, 32, 126));
          str.push_back(c);
        }
        joinKey = std::hash<std::string>{}(str);
        uii = rand<int64_t>(rng_);
      } else if constexpr (std::is_same_v<T, UnknownValue>) {
        // No-op since KHyperLogLog ignores input nulls.
        continue;
      } else {
        // For other numeric types, hash to int64_t.
        auto value = rand<T>(rng_);
        joinKey = std::hash<T>{}(value);
        uii = rand<int64_t>(rng_);
      }

      khll.add(joinKey, uii);
    }

    auto size = khll.estimatedSerializedSize();
    std::string buff(size, '\0');
    khll.serialize(buff.data());
    return variant::create<TypeKind::VARBINARY>(buff);
  }

  TypePtr baseType_;
  int32_t maxSize_;
  int32_t hllBuckets_;
  int32_t minNumValues_;
  memory::MemoryPool* pool_;
};
#pragma GCC diagnostic pop

} // namespace facebook::velox::fuzzer

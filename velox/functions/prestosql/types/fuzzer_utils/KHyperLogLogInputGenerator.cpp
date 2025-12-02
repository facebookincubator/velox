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

#include "velox/functions/prestosql/types/fuzzer_utils/KHyperLogLogInputGenerator.h"

#include "velox/common/fuzzer/Utils.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer {

KHyperLogLogInputGenerator::KHyperLogLogInputGenerator(
    const size_t seed,
    const double nullRatio,
    memory::MemoryPool* pool,
    int32_t minNumValues)
    : AbstractInputGenerator{seed, KHYPERLOGLOG(), nullptr, nullRatio},
      minNumValues_{minNumValues},
      pool_{pool} {
  static const std::vector<TypePtr> kBaseTypes{
      BIGINT(), VARCHAR(), DOUBLE(), UNKNOWN()};
  baseType_ = kBaseTypes[rand<int32_t>(rng_, 0, kBaseTypes.size() - 1)];

  // Randomly select maxSize and hllBuckets.
  // maxSize: number of distinct join keys to track exactly (default 4096).
  maxSize_ = rand<int32_t>(rng_, 100, 5000);

  // hllBuckets: must be power of 2 (default 256).
  static const std::vector<int32_t> kValidBuckets{64, 128, 256, 512, 1024};
  hllBuckets_ = kValidBuckets[rand<int32_t>(rng_, 0, kValidBuckets.size() - 1)];
}

variant KHyperLogLogInputGenerator::generate() {
  if (coinToss(rng_, nullRatio_)) {
    return variant::null(type_->kind());
  }

  if (baseType_->isBigint()) {
    return generateTyped<int64_t>();
  } else if (baseType_->isVarchar()) {
    return generateTyped<std::string>();
  } else if (baseType_->isDouble()) {
    return generateTyped<double>();
  } else {
    return generateTyped<UnknownValue>();
  }
}

} // namespace facebook::velox::fuzzer

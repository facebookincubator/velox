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

#include "velox/functions/sparksql/fuzzer/BloomFilterInputGenerator.h"

namespace facebook::velox::exec::test {

std::vector<VectorPtr> BloomFilterInputGenerator::generate(
    const std::vector<TypePtr>& types,
    VectorFuzzer& fuzzer,
    FuzzerGenerator&,
    memory::MemoryPool*) {
  VELOX_CHECK_EQ(types.size(), 1);
  VELOX_CHECK(types.front()->isBigint());
  return {fuzzer.fuzzNotNull(types.front())};
}

std::vector<VectorPtr> BloomFilterInputGenerator::generateConstants(
    const std::vector<TypePtr>& types,
    VectorFuzzer&,
    FuzzerGenerator& rng,
    memory::MemoryPool* pool) {
  VELOX_CHECK_LE(types.size(), 2);
  for (const auto& type : types) {
    VELOX_CHECK(type->isBigint());
  }

  std::vector<VectorPtr> constants;
  if (!types.empty()) {
    const int64_t expectedNumItems = 1 + rng() % 128;
    constants.push_back(
        BaseVector::createConstant(BIGINT(), expectedNumItems, 1, pool));
  }
  if (types.size() > 1) {
    const auto numBits = int64_t{64} << (rng() % 6);
    constants.push_back(BaseVector::createConstant(BIGINT(), numBits, 1, pool));
  }
  return constants;
}

} // namespace facebook::velox::exec::test

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

#include "velox/exec/fuzzer/InputGenerator.h"

namespace facebook::velox::exec::test {

/// Supplies non-null values and positive size literals for bloom_filter_agg.
/// Bounds expected items to [1, 128] and bit counts to powers of two in
/// [64, 2'048] to limit aggregation memory use.
class BloomFilterInputGenerator : public InputGenerator {
 public:
  /// Generates non-null BIGINT input columns.
  std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) override;

  /// Generates the expected-item and optional bit-count literals.
  std::vector<VectorPtr> generateConstants(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) override;

  void reset() override {}
};

} // namespace facebook::velox::exec::test

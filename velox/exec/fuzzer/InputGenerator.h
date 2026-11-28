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

#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::exec::test {

/// Generates constrained input columns and literal arguments for a function.
class InputGenerator {
 public:
  virtual ~InputGenerator() = default;

  /// Generates input columns of the specified types. The aggregation fuzzer
  /// excludes arguments declared constant. May return an empty list to let the
  /// fuzzer use randomly-generated inputs.
  virtual std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) = 0;

  /// Generates one-row values for arguments declared constant, in argument
  /// order. Called once per aggregation-fuzzer iteration, before input columns
  /// are generated. May return an empty list to use random constants.
  virtual std::vector<VectorPtr> generateConstants(
      const std::vector<TypePtr>&,
      VectorFuzzer&,
      FuzzerGenerator&,
      memory::MemoryPool*) {
    return {};
  }

  /// Called after generating all inputs for a given fuzzer iteration.
  virtual void reset() = 0;
};

} // namespace facebook::velox::exec::test

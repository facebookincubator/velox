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

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <vector>
#include "velox/exec/fuzzer/InputGenerator.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::exec::test {

class NoisyCountIfInputGenerator : public InputGenerator {
 public:
  std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) override {
    vector_size_t size = static_cast<int32_t>(fuzzer.getOptions().vectorSize);
    std::vector<VectorPtr> result;

    // Make sure to use the same value of 'noiseScale' for all batches inputs,
    // so we only set it once.
    if (!noiseScale_.has_value()) {
      noiseScale_ =
          boost::random::uniform_real_distribution<double>(0.0, 10.0)(rng);
    }

    // Process each type in the input.
    // Types of parameters in noisy_count_if(col, noiseScale, randomSeed)
    VELOX_CHECK(types.size() >= 2);
    for (size_t i = 0; i < types.size(); ++i) {
      const auto& type = types[i];

      // For the first boolean argument(col)
      if (i == 0 && type->isBoolean()) {
        // Create a simple boolean vector with alternating true/false values
        auto flatVector = std::static_pointer_cast<BaseVector>(
            BaseVector::create<FlatVector<bool>>(BOOLEAN(), size, pool));

        // Add some nulls
        for (vector_size_t j = 0; j < size; j += 10) {
          if (j < size) {
            flatVector->setNull(j, true);
          }
        }

        result.push_back(flatVector);
      }
      // For the second argument (noise scale)
      else if (i == 1) {
        if (type->isDouble()) {
          result.push_back(
              BaseVector::createConstant(DOUBLE(), *noiseScale_, size, pool));
        } else if (type->isBigint()) {
          // Create a variant with the correct integer value
          variant intValue = static_cast<int64_t>(*noiseScale_);
          result.push_back(
              BaseVector::createConstant(BIGINT(), intValue, size, pool));
        }
      }
    }

    return result;
  }
  void reset() override {
    noiseScale_.reset();
  }

 private:
  std::optional<double> noiseScale_;
};

} // namespace facebook::velox::exec::test

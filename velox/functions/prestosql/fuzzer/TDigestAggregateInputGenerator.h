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

#include <boost/random/uniform_int_distribution.hpp>
#include <string>

#include "velox/exec/fuzzer/InputGenerator.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/GeneratorSpec.h"

namespace facebook::velox::exec::test {

class TDigestAggregateInputGenerator : public InputGenerator {
 public:
  std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) override {
    VELOX_CHECK_GE(types.size(), 1);
    VELOX_CHECK_LE(types.size(), 3);
    VELOX_CHECK(types[0]->isDouble());
    if (types.size() > 1) {
      VELOX_CHECK(types[1]->isBigint());
    }
    if (types.size() > 2) {
      VELOX_CHECK(types[2]->isDouble());
    }
    std::vector<VectorPtr> inputs;
    const auto size = fuzzer.getOptions().vectorSize;
    inputs.reserve(types.size());
    velox::test::VectorMaker vectorMaker{pool};
    auto values = vectorMaker.flatVector<double>(size, [&](auto /*row*/) {
      return boost::random::uniform_real_distribution<double>(1, 1'000)(rng);
    });
    inputs.push_back(values);
    // Weight is optional
    if (types.size() > 1 && types[1]->isBigint()) {
      auto weights = vectorMaker.flatVector<int64_t>(size, [&](auto /*row*/) {
        return boost::random::uniform_int_distribution<int64_t>(1, 1'000)(rng);
      });
      inputs.push_back(weights);
    }
    // Compression is optional
    if (types.size() > 2 && types[2]->isDouble()) {
      // Make sure to use the same value of 'compression' for all batches in a
      // given Fuzzer iteration.
      if (!compression_.has_value()) {
        boost::random::uniform_real_distribution<double> dist(1.0, 1000.0);
        compression_ = dist(rng);
      }
      inputs.push_back(BaseVector::createConstant(
          DOUBLE(), compression_.value(), size, pool));
    }
    return inputs;
  }

  void reset() override {
    compression_.reset();
  }

 private:
  std::optional<double> compression_;
};

} // namespace facebook::velox::exec::test

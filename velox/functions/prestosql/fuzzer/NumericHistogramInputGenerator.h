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
#include <boost/random/uniform_real_distribution.hpp>

#include "velox/exec/fuzzer/InputGenerator.h"
#include "velox/type/Type.h"
#include "velox/type/Variant.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec::test {

class PositiveNumberGenerator : public AbstractInputGenerator {
 public:
  PositiveNumberGenerator(size_t seed, const TypePtr& type)
      : AbstractInputGenerator(seed, type, nullptr, 0.0) {}

  Variant generate() override {
    if (type_->isReal()) {
      float value =
          boost::random::uniform_real_distribution<float>(0.1f, 1000.0f)(rng_);
      return variant(value);
    } else if (type_->isDouble()) {
      double value =
          boost::random::uniform_real_distribution<double>(0.1, 1000.0)(rng_);
      return variant(value);
    }

    // Should not reach here given the type checks in
    // NumericHistogramInputGenerator
    VELOX_UNREACHABLE("Unsupported type for PositiveNumberGenerator");
  }
};

class NumericHistogramInputGenerator : public InputGenerator {
 public:
  std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& types,
      VectorFuzzer& fuzzer,
      FuzzerGenerator& rng,
      memory::MemoryPool* pool) override {
    VELOX_CHECK_GE(types.size(), 2);
    VELOX_CHECK_LE(types.size(), 3);

    std::vector<VectorPtr> inputs;
    inputs.reserve(types.size());
    const auto size = fuzzer.getOptions().vectorSize;
    if (!bucketsSize_.has_value()) {
      bucketsSize_ =
          boost::random::uniform_int_distribution<int64_t>(0, 9'999)(rng);
    }
    inputs.push_back(
        BaseVector::createConstant(BIGINT(), bucketsSize_.value(), size, pool));

    if (types.size() > 1) {
      VELOX_CHECK(types[1]->isDouble() || types[1]->isReal());
      auto positiveGenerator =
          std::make_shared<PositiveNumberGenerator>(rng(), types[1]);
      auto valuesVector = fuzzer.fuzz(types[1], positiveGenerator);
      inputs.push_back(valuesVector);
    }

    if (types.size() > 2) {
      VELOX_CHECK(types[2]->isDouble() || types[2]->isReal());
      auto positiveGenerator =
          std::make_shared<PositiveNumberGenerator>(rng(), types[2]);
      auto weightsVector = fuzzer.fuzz(types[2], positiveGenerator);
      inputs.push_back(weightsVector);
    }
    return inputs;
  }

  void reset() override {
    bucketsSize_.reset();
  }

 private:
  std::optional<int64_t> bucketsSize_;
};

} // namespace facebook::velox::exec::test

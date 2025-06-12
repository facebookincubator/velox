// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <random>
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::function::aggregate {

class SecureRandomizationStrategy : public RandomizationStrategy {
 public:
  SecureRandomizationStrategy() = default;

  bool nextBoolean(double probability) override {
    return dist_(gen_) <= probability;
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

} // namespace facebook::velox::function::aggregate

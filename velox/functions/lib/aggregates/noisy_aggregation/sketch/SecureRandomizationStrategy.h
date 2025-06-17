// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Random.h>
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::function::aggregate {

class SecureRandomizationStrategy : public RandomizationStrategy {
 public:
  SecureRandomizationStrategy() = default;

  bool nextBoolean(double probability) override {
    return folly::Random::secureRandDouble01() <= probability;
  }
};

} // namespace facebook::velox::function::aggregate

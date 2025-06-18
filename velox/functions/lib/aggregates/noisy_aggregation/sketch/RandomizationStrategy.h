// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace facebook::velox::function::aggregate {

class RandomizationStrategy {
 public:
  virtual bool nextBoolean(double probability) = 0;

  virtual ~RandomizationStrategy() = default;
};

} // namespace facebook::velox::function::aggregate

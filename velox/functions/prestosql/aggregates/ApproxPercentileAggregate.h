// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace facebook::velox::aggregate::prestosql {

enum ApproxPercentileIntermediateTypeChildIndex {
  kPercentiles = 0,
  kPercentilesIsArray = 1,
  kAccuracy = 2,
  kK = 3,
  kN = 4,
  kMinValue = 5,
  kMaxValue = 6,
  kItems = 7,
  kLevels = 8,
};

} // namespace facebook::velox::aggregate::prestosql

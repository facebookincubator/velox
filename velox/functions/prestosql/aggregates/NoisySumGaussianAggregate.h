// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>

namespace facebook::velox::aggregate::prestosql {

void registerNoisySumGaussianAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::aggregate::prestosql

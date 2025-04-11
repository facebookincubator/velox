#pragma once

#include <string>

#include "velox/exec/Aggregate.h"

namespace facebook::velox::functions::aggregate::sparksql {

void registerCovarianceAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
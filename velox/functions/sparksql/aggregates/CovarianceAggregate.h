#pragma once

#include <string>

namespace facebook::velox::functions::aggregate::sparksql {

void registerCovarianceAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
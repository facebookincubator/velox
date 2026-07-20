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

#include <string>
#include <vector>

namespace facebook::velox::aggregate::prestosql {

void registerCovarPopAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerCovarSampAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerCorrAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrInterceptAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrSlopeAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrCountAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrAvgyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrAvgxAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrSxyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrSxxAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrSyyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

void registerRegrR2Aggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
} // namespace facebook::velox::aggregate::prestosql

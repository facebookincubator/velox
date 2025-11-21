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

#include "velox/functions/prestosql/KHyperLogLogFunctions.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"

namespace facebook::velox::functions {

void registerKHyperLogLogFunctions(const std::string& prefix) {
  registerKHyperLogLogType();

  // These scalar functions are registered as vector functions, rather than
  // using the simple function api, so that MemoryPool can be used as the
  // allocator.
  exec::registerVectorFunction(
      prefix + "cardinality",
      KHyperLogLogCardinalityFunction::signatures(),
      std::make_unique<KHyperLogLogCardinalityFunction>());

  exec::registerVectorFunction(
      prefix + "intersection_cardinality",
      KHyperLogLogIntersectionCardinalityFunction::signatures(),
      std::make_unique<KHyperLogLogIntersectionCardinalityFunction>());

  exec::registerVectorFunction(
      prefix + "jaccard_index",
      KHyperLogLogJaccardIndexFunction::signatures(),
      std::make_unique<KHyperLogLogJaccardIndexFunction>());

  exec::registerVectorFunction(
      prefix + "reidentification_potential",
      KHyperLogLogReidentificationPotentialFunction::signatures(),
      std::make_unique<KHyperLogLogReidentificationPotentialFunction>());

  exec::registerVectorFunction(
      prefix + "uniqueness_distribution",
      KHyperLogLogUniquenessDistributionFunction::signatures(),
      std::make_unique<KHyperLogLogUniquenessDistributionFunction>());

  exec::registerVectorFunction(
      prefix + "merge_khll",
      MergeKHyperLogLogFunction::signatures(),
      std::make_unique<MergeKHyperLogLogFunction>());
}

} // namespace facebook::velox::functions

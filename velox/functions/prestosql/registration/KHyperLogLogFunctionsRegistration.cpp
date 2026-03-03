/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/KHyperLogLogFunctions.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"

namespace facebook::velox::functions {

void registerKHyperLogLogFunctions(const std::string& prefix) {
  registerKHyperLogLogType();

  registerFunction<KHyperLogLogCardinalityFunction, int64_t, KHyperLogLog>(
      {prefix + "cardinality"});

  registerFunction<
      KHyperLogLogIntersectionCardinalityFunction,
      int64_t,
      KHyperLogLog,
      KHyperLogLog>({prefix + "intersection_cardinality"});

  registerFunction<
      KHyperLogLogJaccardIndexFunction,
      double,
      KHyperLogLog,
      KHyperLogLog>({prefix + "jaccard_index"});

  registerFunction<
      KHyperLogLogReidentificationPotentialFunction,
      double,
      KHyperLogLog,
      int64_t>({prefix + "reidentification_potential"});

  registerFunction<
      KHyperLogLogUniquenessDistributionFunction,
      Map<int64_t, double>,
      KHyperLogLog>({prefix + "uniqueness_distribution"});

  registerFunction<
      KHyperLogLogUniquenessDistributionFunction,
      Map<int64_t, double>,
      KHyperLogLog,
      int64_t>({prefix + "uniqueness_distribution"});

  registerFunction<
      MergeKHyperLogLogFunction,
      KHyperLogLog,
      Array<KHyperLogLog>>({prefix + "merge_khll"});
}

} // namespace facebook::velox::functions

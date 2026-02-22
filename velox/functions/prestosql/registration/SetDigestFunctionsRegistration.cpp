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
#include "velox/functions/prestosql/SetDigestFunctions.h"
#include "velox/functions/prestosql/types/SetDigestRegistration.h"
#include "velox/functions/prestosql/types/SetDigestType.h"

namespace facebook::velox::functions {

void registerSetDigestFunctions(const std::string& prefix) {
  facebook::velox::registerSetDigestType();

  // Register cardinality(setdigest) -> bigint
  registerFunction<
      CardinalitySetDigestFunction,
      int64_t,
      facebook::velox::SetDigest>({prefix + "cardinality"});

  // Register intersection_cardinality(setdigest, setdigest) -> bigint
  registerFunction<
      IntersectionCardinalityFunction,
      int64_t,
      facebook::velox::SetDigest,
      facebook::velox::SetDigest>({prefix + "intersection_cardinality"});

  // Register jaccard_index(setdigest, setdigest) -> double
  registerFunction<
      JaccardIndexFunction,
      double,
      facebook::velox::SetDigest,
      facebook::velox::SetDigest>({prefix + "jaccard_index"});

  // Register hash_counts(setdigest) -> map(bigint, smallint)
  registerFunction<
      HashCountsFunction,
      Map<int64_t, int16_t>,
      facebook::velox::SetDigest>({prefix + "hash_counts"});
}

} // namespace facebook::velox::functions

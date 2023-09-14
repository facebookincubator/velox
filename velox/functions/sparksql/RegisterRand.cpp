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
#include "velox/functions/Registerer.h"
#include "velox/functions/sparksql/Rand.h"

namespace facebook::velox::functions::sparksql {

void registerRandFunctions(const std::string& prefix) {
  // No input.
  registerFunction<RandFunction, double>({prefix + "rand", prefix + "random"});
  // Has seed & partition index as input.
  registerFunction<
      RandFunction,
      double,
      int32_t /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
  // Has seed & partition index as input.
  registerFunction<
      RandFunction,
      double,
      int64_t /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
  // NULL constant as seed in unknown type.
  registerFunction<
      RandFunction,
      double,
      UnknownValue /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
}
} // namespace facebook::velox::functions::sparksql

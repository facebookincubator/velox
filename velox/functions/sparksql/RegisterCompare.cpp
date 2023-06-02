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
#include "velox/functions/sparksql/RegisterCompare.h"

#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::functions::sparksql {

void registerCompareFunctions(const std::string& prefix) {
  registerFunction<BetweenFunction, bool, int8_t, int8_t, int8_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int16_t, int16_t, int16_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int32_t, int32_t, int32_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int64_t, int64_t, int64_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, double, double, double>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, float, float, float>(
      {prefix + "between"});

  registerFunction<BetweenFunction, bool, int64_t, int64_t, int64_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int128_t, int128_t, int128_t>(
      {prefix + "between"});
  registerFunction<GtFunction, bool, int64_t, int64_t>(
      {prefix + "greaterthan"});
  registerFunction<GtFunction, bool, int128_t, int128_t>(
      {prefix + "greaterthan"});
  registerFunction<LtFunction, bool, int64_t, int64_t>({prefix + "lessthan"});
  registerFunction<LtFunction, bool, int128_t, int128_t>({prefix + "lessthan"});
  registerFunction<GteFunction, bool, int64_t, int64_t>(
      {prefix + "greaterthanorequal"});
  registerFunction<GteFunction, bool, int128_t, int128_t>(
      {prefix + "greaterthanorequal"});
  registerFunction<LteFunction, bool, int64_t, int64_t>(
      {prefix + "lessthanorequal"});
  registerFunction<LteFunction, bool, int128_t, int128_t>(
      {prefix + "lessthanorequal"});
  registerFunction<EqFunction, bool, int64_t, int64_t>({prefix + "equalto"});
  registerFunction<EqFunction, bool, int128_t, int128_t>({prefix + "equalto"});
}

} // namespace facebook::velox::functions::sparksql

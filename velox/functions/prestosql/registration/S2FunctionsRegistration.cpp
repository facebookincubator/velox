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
#include "velox/functions/prestosql/S2Functions.h"
#include "velox/type/SimpleFunctionApi.h"

namespace facebook::velox::functions {

using namespace facebook::velox;

void registerS2Functions(const std::string& prefix) {
  registerFunction<S2CellAreaSqKmFunction, double, int64_t>(
      {prefix + "s2_cell_area_sq_km"});
  registerFunction<S2CellContainsFunction, bool, int64_t, int64_t>(
      {prefix + "s2_cell_contains"});
  registerFunction<S2CellFromTokenFunction, int64_t, Varchar>(
      {prefix + "s2_cell_from_token"});
  registerFunction<S2CellLevelFunction, int32_t, int64_t>(
      {prefix + "s2_cell_level"});
  registerFunction<S2CellParentFunction, int64_t, int64_t, int32_t>(
      {prefix + "s2_cell_parent"});
  registerFunction<S2CellToTokenFunction, Varchar, int64_t>(
      {prefix + "s2_cell_to_token"});
  // Two overloads of s2_cells: fixed-level and dissolved (mixed-level).
  registerFunction<S2CellsFunction, Array<int64_t>, Geometry, int32_t>(
      {prefix + "s2_cells"});
  registerFunction<
      S2CellsDissolvedFunction,
      Array<int64_t>,
      Geometry,
      int32_t,
      int32_t,
      int32_t>({prefix + "s2_cells"});
}

} // namespace facebook::velox::functions

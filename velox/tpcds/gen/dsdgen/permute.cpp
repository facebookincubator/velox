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
/*
 * Copyright owned by the Transaction Processing Performance Council.
 *
 * A copy of the license is included under tpcds/gen/dsdgen/LICENSE
 * in this repository.
 *
 * You may not use this file except in compliance with the License.
 *
 * THE TPC SOFTWARE IS AVAILABLE WITHOUT CHARGE FROM TPC.
 */

#include "velox/tpcds/gen/dsdgen/include/config.h"
#include "velox/tpcds/gen/dsdgen/include/porting.h"
#include <stdio.h>
#include "velox/tpcds/gen/dsdgen/include/genrand.h"

/*
 * Routine: MakePermutation(int nSize)
 * Purpose: Permute the integers in [1..nSize]
 * Algorithm:
 * Data Structures:
 *
 * Params:
 * Returns:
 * Called By:
 * Calls:
 * Assumptions:
 * Side Effects:
 * TODO: None
 */
std::vector<int32_t>
makePermutation(int nSize, int nStream, DSDGenContext& dsdGenContext) {
  if (nSize <= 0)
    return {};

  std::vector<int32_t> perm(nSize);
  for (int i = 0; i < nSize; i++)
    perm[i] = i;

  for (int i = 0; i < nSize; i++) {
    int nIndex = genrand_integer(
        NULL, DIST_UNIFORM, 0, nSize - 1, 0, nStream, dsdGenContext);
    std::swap(perm[i], perm[nIndex]);
  }

  return perm;
}

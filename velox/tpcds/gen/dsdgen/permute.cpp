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

#include "config.h"
#include "porting.h"
#ifndef USE_STDLIB_H
#include <malloc.h>
#endif
#include <stdio.h>
#include "genrand.h"

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
int* makePermutation(
    int* nNumberSet,
    int nSize,
    int nStream,
    DSDGenContext& dsdGenContext) {
  int i, nTemp, nIndex, *pInt;

  if (nSize <= 0)
    return (NULL);

  if (!nNumberSet) {
    nNumberSet = static_cast<int*>(malloc(nSize * sizeof(int)));
    MALLOC_CHECK(nNumberSet);
    pInt = nNumberSet;
    for (i = 0; i < nSize; i++)
      *pInt++ = i;
  }

  for (i = 0; i < nSize; i++) {
    nIndex = genrand_integer(
        NULL, DIST_UNIFORM, 0, nSize - 1, 0, nStream, dsdGenContext);
    nTemp = nNumberSet[i];
    nNumberSet[i] = nNumberSet[nIndex];
    nNumberSet[nIndex] = nTemp;
  }

  return (nNumberSet);
}

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
ds_key_t* makeKeyPermutation(
    ds_key_t* nNumberSet,
    ds_key_t nSize,
    int nStream,
    DSDGenContext& dsdGenContext) {
  ds_key_t i, nTemp, nIndex, *pInt;
  if (nSize <= 0)
    return (NULL);

  if (!nNumberSet) {
    nNumberSet = static_cast<ds_key_t*>(malloc(nSize * sizeof(ds_key_t)));
    MALLOC_CHECK(nNumberSet);
    pInt = nNumberSet;
    for (i = 0; i < nSize; i++)
      *pInt++ = i;
  }

  for (i = 0; i < nSize; i++) {
    nIndex = genrand_key(
        NULL, DIST_UNIFORM, 0, nSize - 1, 0, nStream, dsdGenContext);
    nTemp = nNumberSet[i];
    nNumberSet[i] = nNumberSet[nIndex];
    nNumberSet[nIndex] = nTemp;
  }

  return (nNumberSet);
}

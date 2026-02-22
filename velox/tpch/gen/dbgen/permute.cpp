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
/*
 * Copyright owned by the Transaction Processing Performance Council.
 *
 * A copy of the license is included under extension/tpch/dbgen/LICENSE
 * in this repository.
 *
 * You may not use this file except in compliance with the License.
 *
 * THE TPC SOFTWARE IS AVAILABLE WITHOUT CHARGE FROM TPC.
 *//*
 * permute.c -- a permutation generator for the query
 *              sequences in TPC-H and TPC-R
 */

#include "dbgen/config.h" // @manual
#include "dbgen/dss.h" // @manual

namespace facebook::velox::tpch::dbgen {

DSS_HUGE NextRand(DSS_HUGE seed);
void permute(long* set, int cnt, seed_t* seed);
void permute_dist(distribution* d, seed_t* seed, DBGenContext* ctx);
long seed;
char* eol[2] = {" ", "},"};

#define MAX_QUERY 22
#define ITERATIONS 1000
#define UNSET 0

void permute(long* a, int c, seed_t* seed) {
  int i;
  DSS_HUGE source;
  long temp;

  if (a != reinterpret_cast<long*>(NULL)) {
    for (i = 0; i < c; i++) {
      RANDOM(source, static_cast<long>(i), static_cast<long>(c - 1), seed);
      temp = *(a + source);
      *(a + source) = *(a + i);
      *(a + i) = temp;
    }
  }

  return;
}

void permute_dist(distribution* d, seed_t* seed, DBGenContext* ctx) {
  int i;

  if (d != NULL) {
    if (ctx->permute == reinterpret_cast<long*>(NULL)) {
      ctx->permute =
          reinterpret_cast<long*>(malloc(sizeof(long) * DIST_SIZE(d)));
      MALLOC_CHECK(ctx->permute);
    }
    for (i = 0; i < DIST_SIZE(d); i++)
      *(ctx->permute + i) = i;
    permute(ctx->permute, DIST_SIZE(d), seed);
  } else
    INTERNAL_ERROR("Bad call to permute_dist");

  return;
}

} // namespace facebook::velox::tpch::dbgen

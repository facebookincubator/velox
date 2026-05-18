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
#include "velox/functions/prestosql/coercion/PrestoCoercions.h"

namespace facebook::velox::functions::prestosql {

namespace {

// Complete Presto coercion rule set. Self-contained; intentionally does not
// reference TypeCoercer::defaults() so changes to Velox defaults cannot
// silently shift Presto semantics. Lower cost = preferred during overload
// resolution.
//
// TODO: transcribe the full Presto Java `TypeCoercion` table. Today's set
// matches Velox's defaults plus BIGINT -> REAL (Presto allows this lossy
// conversion; Velox defaults do not).
std::vector<CoercionEntry> prestoRules() {
  std::vector<CoercionEntry> rules;

  auto add = [&](const TypePtr& from, const std::vector<TypePtr>& to) {
    int32_t cost = 0;
    for (const auto& toType : to) {
      rules.push_back({from, toType, ++cost});
    }
  };

  add(TINYINT(),
      {SMALLINT(), INTEGER(), BIGINT(), DECIMAL(3, 0), REAL(), DOUBLE()});
  add(SMALLINT(), {INTEGER(), BIGINT(), DECIMAL(5, 0), REAL(), DOUBLE()});
  add(INTEGER(), {BIGINT(), DECIMAL(10, 0), REAL(), DOUBLE()});
  // Presto-specific: BIGINT -> REAL is added between DECIMAL and DOUBLE,
  // mirroring the INTEGER row's ordering. This makes divide(real, bigint)
  // resolve to divide(real, real) via BIGINT -> REAL (cost 2) instead of
  // divide(double, double) via REAL -> DOUBLE + BIGINT -> DOUBLE (cost 1 +
  // 3 = 4), matching Presto's overload resolution.
  add(BIGINT(), {DECIMAL(19, 0), REAL(), DOUBLE()});
  add(REAL(), {DOUBLE()});
  add(DECIMAL(1, 0), {REAL(), DOUBLE()});
  add(DATE(), {TIMESTAMP()});
  add(UNKNOWN(),
      {TINYINT(),
       BOOLEAN(),
       SMALLINT(),
       INTEGER(),
       BIGINT(),
       REAL(),
       DOUBLE(),
       VARCHAR(),
       VARBINARY()});

  return rules;
}

} // namespace

const TypeCoercer& typeCoercer() {
  static const TypeCoercer instance{prestoRules()};
  return instance;
}

} // namespace facebook::velox::functions::prestosql

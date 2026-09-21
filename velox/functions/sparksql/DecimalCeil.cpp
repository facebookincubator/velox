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
#include "velox/functions/sparksql/DecimalCeilFunction.h"

namespace facebook::velox::functions::sparksql {

void registerDecimalCeil(const std::string& prefix) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P2::name(),
          fmt::format(
              "min(38, {p} - {s} + min({s}, 1))",
              fmt::arg("p", P1::name()),
              fmt::arg("s", S1::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S2::name(), "0", exec::ParameterType::kIntegerParameter),
  };

  registerFunction<
      DecimalCeilFunction,
      LongDecimal<P2, S2>,
      LongDecimal<P1, S1>>({prefix + "ceil"}, constraints);

  registerFunction<
      DecimalCeilFunction,
      ShortDecimal<P2, S2>,
      LongDecimal<P1, S1>>({prefix + "ceil"}, constraints);

  registerFunction<
      DecimalCeilFunction,
      ShortDecimal<P2, S2>,
      ShortDecimal<P1, S1>>({prefix + "ceil"}, constraints);
}
} // namespace facebook::velox::functions::sparksql

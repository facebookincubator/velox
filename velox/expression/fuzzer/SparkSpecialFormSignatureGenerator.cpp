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
#include "velox/expression/fuzzer/SparkSpecialFormSignatureGenerator.h"

namespace facebook::velox::fuzzer {

std::vector<exec::FunctionSignaturePtr>
SparkSpecialFormSignatureGenerator::getSignaturesForCast() const {
  std::vector<exec::FunctionSignaturePtr> signatures =
      getCommonSignaturesForCast();

  // Cast tinyint/smallint/integer/bigint as varbinary is supported in Spark.
  for (auto fromType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(makeCastSignature(fromType, "varbinary"));
  }

  // Cast tinyint/smallint/integer/bigint as timestamp is supported in Spark.
  for (auto fromType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(makeCastSignature(fromType, "timestamp"));
  }

  // For each supported translation pair T --> U, add signatures of array(T) -->
  // array(U), map(varchar, T) --> map(varchar, U), row(T) --> row(U).
  auto size = signatures.size();
  for (auto i = 0; i < size; ++i) {
    auto from = signatures[i]->argumentTypes()[0].baseName();
    auto to = signatures[i]->returnType().baseName();

    signatures.push_back(makeCastSignature(
        fmt::format("array({})", from), fmt::format("array({})", to)));

    signatures.push_back(makeCastSignature(
        fmt::format("map(varchar, {})", from),
        fmt::format("map(varchar, {})", to)));

    signatures.push_back(makeCastSignature(
        fmt::format("row({})", from), fmt::format("row({})", to)));
  }
  return signatures;
}

const std::unordered_map<std::string, std::vector<exec::FunctionSignaturePtr>>&
SparkSpecialFormSignatureGenerator::getSignatures() const {
  const static std::
      unordered_map<std::string, std::vector<exec::FunctionSignaturePtr>>
          kSpecialForms{
              {"and", getSignaturesForAnd()},
              {"or", getSignaturesForOr()},
              {"coalesce", getSignaturesForCoalesce()},
              {"if", getSignaturesForIf()},
              {"switch", getSignaturesForSwitch()},
              {"cast", getSignaturesForCast()},
              {"concat_ws", getSignaturesForConcatWs()}};
  return kSpecialForms;
}

std::vector<exec::FunctionSignaturePtr>
SparkSpecialFormSignatureGenerator::getSignaturesForConcatWs() const {
  // Signature: concat_ws (separator, input, ...) -> output:
  // varchar, varchar, varchar, ... -> varchar
  return {facebook::velox::exec::FunctionSignatureBuilder()
              .argumentType("varchar")
              .variableArity("varchar")
              .returnType("varchar")
              .build()};
}

} // namespace facebook::velox::fuzzer

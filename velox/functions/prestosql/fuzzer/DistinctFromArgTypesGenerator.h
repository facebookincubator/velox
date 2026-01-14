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
#pragma once

#include "velox/expression/fuzzer/ArgTypesGenerator.h"
#include "velox/expression/fuzzer/ArgumentTypeFuzzer.h"
#include "velox/functions/prestosql/types/IPAddressType.h"

namespace facebook::velox::exec::test {

/// Custom argument type generator for distinct_from function that blocks
/// array(IPADDRESS) combinations which fail in Presto due to missing
/// compareTo() implementation in Int128ArrayBlock.
/// See: https://github.com/prestodb/presto/issues/26836
class DistinctFromArgTypesGenerator : public fuzzer::ArgTypesGenerator {
 public:
  std::vector<TypePtr> generateArgs(
      const exec::FunctionSignature& signature,
      const TypePtr& returnType,
      fuzzer::FuzzerGenerator& rng) override {
    // Use default ArgumentTypeFuzzer to generate types.
    fuzzer::ArgumentTypeFuzzer fuzzer(signature, returnType, rng);

    if (!fuzzer.fuzzArgumentTypes(0 /* maxVariadicArgs */)) {
      // Cannot generate valid argument types.
      return {};
    }

    auto argTypes = fuzzer.argumentTypes();

    // Block array(IPADDRESS) combinations as they fail in Presto.
    // Presto throws: UnsupportedOperationException in
    // Int128ArrayBlock.compareTo()
    for (const auto& argType : argTypes) {
      if (argType->isArray() &&
          isIPAddressType(argType->asArray().elementType())) {
        // Return empty vector to indicate this type combination is not
        // supported.
        return {};
      }
    }

    return argTypes;
  }
};

} // namespace facebook::velox::exec::test

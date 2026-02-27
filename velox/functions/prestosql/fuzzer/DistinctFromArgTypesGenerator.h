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
/// types containing IPADDRESS in array-based value positions which fail in
/// Presto due to missing compareTo() implementation in Int128ArrayBlock.
/// Specifically blocks: ARRAY<IPADDRESS>, MAP<K, IPADDRESS>, and any nested
/// structures containing these patterns.
/// Note: MAP<IPADDRESS, V> (IPADDRESS as key) and ROW<IPADDRESS> work fine.
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

    // Block types containing IPADDRESS in array-based value positions.
    // Presto throws: UnsupportedOperationException in
    // Int128ArrayBlock.compareTo()
    for (const auto& argType : argTypes) {
      if (containsIPAddressInArrayOrMapValue(argType)) {
        // Return empty vector to indicate this type combination is not
        // supported.
        return {};
      }
    }

    return argTypes;
  }

 private:
  /// Recursively checks if a type contains IPADDRESS in an ARRAY or as a MAP
  /// value (not key). These combinations fail in Presto's Int128ArrayBlock.
  /// Note: IPADDRESS wrapped in ROW is safe, and MAP keys are safe.
  bool containsIPAddressInArrayOrMapValue(const TypePtr& type) const {
    if (type->isArray()) {
      const auto& elementType = type->asArray().elementType();
      // Direct ARRAY<IPADDRESS> fails.
      if (isIPAddressType(elementType)) {
        return true;
      }
      // For ROW elements, check if they contain problematic nested structures.
      return containsIPAddressInArrayOrMapValue(elementType);
    }

    if (type->isMap()) {
      const auto& valueType = type->asMap().valueType();
      // MAP<K, IPADDRESS> fails, but MAP<IPADDRESS, V> works.
      // We only check the value type, not the key type - keys are safe.
      if (isIPAddressType(valueType)) {
        return true;
      }
      // Only recursively check the value type, not the key.
      // MAP<ARRAY<IPADDRESS>, V> and MAP<MAP<K, IPADDRESS>, V> are fine.
      return containsIPAddressInArrayOrMapValue(valueType);
    }

    if (type->isRow()) {
      // ROW<IPADDRESS> works fine, but check for nested arrays/maps
      // that might contain IPADDRESS in problematic positions.
      const auto& rowType = type->asRow();
      for (size_t i = 0; i < rowType.size(); ++i) {
        if (containsIPAddressInArrayOrMapValue(rowType.childAt(i))) {
          return true;
        }
      }
    }

    return false;
  }
};

} // namespace facebook::velox::exec::test

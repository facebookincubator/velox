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
#pragma once

#include <functional>
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::functions::test {

/// Configuration options for fuzzer tests.
struct FuzzerTestOptions {
  vector_size_t vectorSize = 100;
  double nullRatio = 0.1;
  size_t containerLength = 5;
  bool containerHasNulls = false;
  bool containerVariableLength = false;
  size_t stringLength = 20;
  bool stringVariableLength = true;
  int iterations = 1;
};

/// Helper class for running fuzzer tests on map functions.
/// Eliminates boilerplate code for VectorFuzzer setup across test files.
class FuzzerTestHelper {
 public:
  explicit FuzzerTestHelper(memory::MemoryPool* pool) : pool_(pool) {}

  /// Creates a VectorFuzzer with the given options.
  VectorFuzzer createFuzzer(const FuzzerTestOptions& opts) {
    VectorFuzzer::Options fuzzerOpts;
    fuzzerOpts.vectorSize = opts.vectorSize;
    fuzzerOpts.nullRatio = opts.nullRatio;
    fuzzerOpts.containerLength = opts.containerLength;
    fuzzerOpts.containerHasNulls = opts.containerHasNulls;
    fuzzerOpts.containerVariableLength = opts.containerVariableLength;
    fuzzerOpts.stringLength = opts.stringLength;
    fuzzerOpts.stringVariableLength = opts.stringVariableLength;
    return VectorFuzzer(fuzzerOpts, pool_);
  }

  /// Runs a fuzzer test for map functions that take (map, array) as input.
  /// Used by: map_intersect, map_except
  /// @param keyType The type for map keys and array elements
  /// @param valueType The type for map values
  /// @param testFn The test function to call with the generated vectors
  ///               (inputMap, keysArray)
  /// @param opts Optional configuration for the fuzzer
  void runMapArrayTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      std::function<void(const VectorPtr&, const VectorPtr&)> testFn,
      const FuzzerTestOptions& opts = {}) {
    auto fuzzer = createFuzzer(opts);
    auto mapType = MAP(keyType, valueType);
    auto arrayType = ARRAY(keyType);
    for (int i = 0; i < opts.iterations; ++i) {
      auto inputMap = fuzzer.fuzz(mapType);
      auto keys = fuzzer.fuzz(arrayType);
      testFn(inputMap, keys);
    }
  }

  /// Runs a fuzzer test for map functions that take (map, array, array) as
  /// input. Used by: map_append
  /// @param keyType The type for map keys and key array elements
  /// @param valueType The type for map values and value array elements
  /// @param testFn The test function to call with the generated vectors
  ///               (inputMap, keysArray, valuesArray)
  /// @param opts Optional configuration for the fuzzer
  void runMapArrayArrayTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      std::function<void(const VectorPtr&, const VectorPtr&, const VectorPtr&)>
          testFn,
      const FuzzerTestOptions& opts = {}) {
    auto fuzzer = createFuzzer(opts);
    auto mapType = MAP(keyType, valueType);
    auto keyArrayType = ARRAY(keyType);
    auto valueArrayType = ARRAY(valueType);
    for (int i = 0; i < opts.iterations; ++i) {
      auto inputMap = fuzzer.fuzz(mapType);
      auto keys = fuzzer.fuzz(keyArrayType);
      auto values = fuzzer.fuzz(valueArrayType);
      testFn(inputMap, keys, values);
    }
  }

  /// Runs a fuzzer test for map functions with same key and value types.
  /// Convenience overload for functions like map_append where key == value
  /// type.
  /// @param type The type for both keys and values
  /// @param testFn The test function to call with the generated vectors
  /// @param opts Optional configuration for the fuzzer
  void runMapArrayArrayTestSameType(
      const TypePtr& type,
      std::function<void(const VectorPtr&, const VectorPtr&, const VectorPtr&)>
          testFn,
      const FuzzerTestOptions& opts = {}) {
    runMapArrayArrayTest(type, type, testFn, opts);
  }

  /// Runs a fuzzer test for map functions that take (map, map) as input.
  /// @param keyType The type for map keys
  /// @param valueType The type for map values
  /// @param testFn The test function to call with the generated vectors
  /// @param opts Optional configuration for the fuzzer
  void runMapMapTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      std::function<void(const VectorPtr&, const VectorPtr&)> testFn,
      const FuzzerTestOptions& opts = {}) {
    auto fuzzer = createFuzzer(opts);
    auto mapType = MAP(keyType, valueType);
    for (int i = 0; i < opts.iterations; ++i) {
      auto inputMap1 = fuzzer.fuzz(mapType);
      auto inputMap2 = fuzzer.fuzz(mapType);
      testFn(inputMap1, inputMap2);
    }
  }

 private:
  memory::MemoryPool* pool_;
};

} // namespace facebook::velox::functions::test

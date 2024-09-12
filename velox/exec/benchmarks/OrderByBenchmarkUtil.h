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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

class OrderByBenchmarkUtil {
 public:
  /// Add the benchmarks with the parameter.
  /// @param benchmark benchmark generator.
  static void addBenchmarks(std::function<void(
                                const std::string& testName,
                                size_t numRows,
                                const RowTypePtr& rowType,
                                int iterations,
                                int numKeys)> benchmark);

  /// Generate RowVector by VectorFuzzer according to rowType, for front keys
  /// (column 0 to numKeys -2) use high
  /// nullRatio to enforce all columns to be compared.
  /// @param numKeys 0 to numKeys - 2 is high null ratio column, other
  /// columns do not have null values.
  static RowVectorPtr fuzzRows(
      const RowTypePtr& rowType,
      size_t numRows,
      int numKeys,
      memory::MemoryPool* pool);
};
} // namespace facebook::velox

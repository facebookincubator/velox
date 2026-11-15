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

#include <bit>
#include <cstdint>
#include <vector>

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql::test {

using facebook::velox::functions::test::FunctionBaseTest;

class SparkFunctionBaseTest : public FunctionBaseTest {
 protected:
  // Ensure Spark functions are registered; don't register the "common"
  // (CoreSQL) functions.
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void setSparkPartitionId(int32_t partitionId) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kPartitionId),
          std::to_string(partitionId)}});
  }

  // Sets the Spark 'spark.sql.mapKeyDedupPolicy' equivalent query config.
  void setThrowExceptionOnDuplicateMapKeys(bool value) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kThrowExceptionOnDuplicateMapKeys,
         value ? "true" : "false"},
    });
  }

  // Returns the bit pattern of each element of 'arrays', an ARRAY(DOUBLE)
  // vector, row by row. assertEqualVectors treats 0.0 and -0.0 as equal, and
  // their bit patterns differ.
  static std::vector<uint64_t> elementBitPatterns(const VectorPtr& arrays) {
    DecodedVector decodedArrays(*arrays);
    const auto* baseArrays = decodedArrays.base()->as<ArrayVector>();
    DecodedVector decodedElements(*baseArrays->elements());
    std::vector<uint64_t> bitPatterns;
    for (vector_size_t row = 0; row < arrays->size(); ++row) {
      const auto index = decodedArrays.index(row);
      const auto offset = baseArrays->offsetAt(index);
      for (vector_size_t i = 0; i < baseArrays->sizeAt(index); ++i) {
        bitPatterns.push_back(
            std::bit_cast<uint64_t>(
                decodedElements.valueAt<double>(offset + i)));
      }
    }
    return bitPatterns;
  }
};

} // namespace facebook::velox::functions::sparksql::test

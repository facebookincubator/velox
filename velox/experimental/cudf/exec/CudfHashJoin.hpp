/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <string>

namespace facebook::velox::cudf_velox {

/*
class CudfHashJoinDemo : public facebook::velox::test::VectorTestBase {
 public:
  using vector_size_t = facebook::velox::vector_size_t;
  using RowVectorPtr = facebook::velox::RowVectorPtr;
  using TaskCursor = facebook::velox::exec::test::TaskCursor;
  CudfHashJoinDemo();

  facebook::velox::memory::MemoryPool* get_pool() const {
    return pool();
  }

  RowVectorPtr makeSimpleRowVector(
      vector_size_t size,
      vector_size_t init = 0,
      std::string name_prefix = "c");

  using result_type =
      std::pair<std::unique_ptr<TaskCursor>, std::vector<RowVectorPtr>>;
  result_type testVeloxHashJoin(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery);

  result_type testCudfHashJoin(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery);

  bool CompareResults(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery);
};

*/

} // namespace facebook::velox::cudf_velox

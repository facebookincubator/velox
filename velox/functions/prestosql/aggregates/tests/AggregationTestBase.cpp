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

#include "AggregationTestBase.h"
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"

namespace facebook::velox::aggregate::test {

std::vector<RowVectorPtr> AggregationTestBase::makeVectors(
    const std::shared_ptr<const RowType>& rowType,
    vector_size_t size,
    int numVectors) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < numVectors; ++i) {
    auto vector = std::dynamic_pointer_cast<RowVector>(
        velox::test::BatchMaker::createBatch(rowType, size, *pool_));
    vectors.push_back(vector);
  }
  return vectors;
}

} // namespace facebook::velox::aggregate::test

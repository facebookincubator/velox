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

#include "velox/core/PlanNode.h"
#include "velox/vector/PartitionedVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

class PartitionedVectorTestBase : public VectorTestBase {
 protected:
  std::vector<VectorPtr> partitionVectorByWrapping(
      VectorPtr vector,
      const std::vector<uint32_t>& partitions,
      uint32_t numPartitions);

  std::vector<VectorPtr> partitionRowVectors(
      const std::vector<RowVectorPtr>& rowVectors,
      int32_t numPartitions,
      core::PartitionFunction* partitionFunction);

  VectorPtr canonicalize(VectorPtr vector);

  VectorPtr mergeVectors(const std::vector<VectorPtr>& vectors);
};

} // namespace facebook::velox::test

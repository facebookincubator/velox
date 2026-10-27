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
#include "velox/experimental/cudf/exec/CudfPlanNode.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"

namespace facebook::velox::cudf_velox::detail {

core::PlanNodePtr rebuildCudfLocalPartition(
    const core::LocalPartitionNode& node,
    std::vector<core::PlanNodePtr> sources) {
  VELOX_USER_CHECK(
      !node.scaleWriter(), "GPU local exchange cannot scale writers");
  auto rebuilt = core::LocalPartitionNode::Builder(node)
                     .sources(std::move(sources))
                     .build();
  return std::shared_ptr<const CudfLocalPartitionNode>(
      new CudfLocalPartitionNode(*rebuilt));
}

} // namespace facebook::velox::cudf_velox::detail

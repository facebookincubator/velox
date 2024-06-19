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

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::cudf_velox::test {

/// A builder class inheriting from PlanBuilder
class CudfPlanBuilder : public facebook::velox::exec::test::PlanBuilder {
 public:
  explicit CudfPlanBuilder(
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      memory::MemoryPool* pool = nullptr);

  /// Add a CudfHashJoinNode to join two inputs using one or more join keys and an
  /// optional filter.
  ///
  /// @param leftKeys Join keys from the probe side, the preceding plan node.
  /// Cannot be empty.
  /// @param rightKeys Join keys from the build side, the plan node specified in
  /// 'build' parameter. The number and types of left and right keys must be the
  /// same.
  /// @param build Plan node for the build side. Typically, to reduce memory
  /// usage, the smaller input is placed on the build-side.
  /// @param filter Optional SQL expression for the additional join filter. Can
  /// use columns from both probe and build sides of the join.
  /// @param outputLayout Output layout consisting of columns from probe and
  /// build sides.
  /// @param joinType Type of the join: inner, left, right, full, semi, or anti.
  /// @param nullAware Applies to semi and anti joins. Indicates whether the
  /// join follows IN (null-aware) or EXISTS (regular) semantic.
  CudfPlanBuilder& hashJoin(
      const std::vector<std::string>& leftKeys,
      const std::vector<std::string>& rightKeys,
      const core::PlanNodePtr& build,
      const std::string& filter,
      const std::vector<std::string>& outputLayout,
      core::JoinType joinType = core::JoinType::kInner,
      bool nullAware = false);

};

} // namespace facebook::velox::cudf_velox::test
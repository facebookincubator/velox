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
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox::test_utils {

/// Builds TIMESTAMP WITH TIME ZONE columns that hold one instant under two
/// zone encodings, which Velox compares as one key and cuDF as two, and checks
/// that a plan over such a column gives the CPU answer. Allows CPU fallback
/// while alive, since a GPU plan can only match by declining the operator.
class CustomComparisonKeys {
 public:
  explicit CustomComparisonKeys(memory::MemoryPool* pool);

  ~CustomComparisonKeys();

  /// Returns the columns {g, k, id}, where 'k' holds each of the
  /// 'numInstants' instants once per zone, 'g' is the instant number and 'id'
  /// the row position. Odd instants hold the Los Angeles encoding first and
  /// even ones the UTC encoding, so that a reduction over the packed bits
  /// diverges from the first encoding seen for a minimum on the former and
  /// for a maximum on the latter.
  RowVectorPtr makeRows(int32_t numInstants);

  /// Returns the two columns 'names', where the first holds the instants
  /// 'seconds' under 'zone' and the second the matching 'values'.
  RowVectorPtr makeRowsInZone(
      const std::vector<std::string>& names,
      std::string_view zone,
      const std::vector<int64_t>& seconds,
      const std::vector<int64_t>& values);

  /// Runs 'plan' on the GPU over 'maxDrivers' drivers and expects
  /// 'cpuOperator' and no cuDF operator in the task, and the rows of the plan
  /// run with cuDF disabled, in any order.
  void assertFallsBackToCpu(
      const core::PlanNodePtr& plan,
      std::string_view cpuOperator,
      int32_t maxDrivers = 1);

  /// Like assertFallsBackToCpu, and expects the rows in the same order.
  void assertFallsBackToCpuInOrder(
      const core::PlanNodePtr& plan,
      std::string_view cpuOperator);

  /// Runs 'plan' on the GPU and expects an operator whose type starts with
  /// 'cudfOperator' in the task, and the rows of the plan run with cuDF
  /// disabled, in any order.
  void assertRunsOnGpu(
      const core::PlanNodePtr& plan,
      std::string_view cudfOperator);

 private:
  // Holds the inputs and the copied results.
  memory::MemoryPool* const pool_;
  velox::test::VectorMaker maker_;
  // Fallback setting to restore when this object goes away.
  const bool previousAllowCpuFallback_;
};

} // namespace facebook::velox::cudf_velox::test_utils

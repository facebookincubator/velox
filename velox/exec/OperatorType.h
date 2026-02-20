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

#include <string_view>

namespace facebook::velox::exec {

/// Centralized constants for operator type strings used in operator
/// constructors and comparison sites throughout the execution engine.
struct OperatorType {
  static constexpr std::string_view kAggregation = "Aggregation";
  static constexpr std::string_view kArrowStream = "ArrowStream";
  static constexpr std::string_view kAssignUniqueId = "AssignUniqueId";
  static constexpr std::string_view kBlockedOperator = "BlockedOperator";
  static constexpr std::string_view kCallbackSink = "CallbackSink";
  static constexpr std::string_view kEnforceDistinct = "EnforceDistinct";
  static constexpr std::string_view kEnforceSingleRow = "EnforceSingleRow";
  static constexpr std::string_view kExchange = "Exchange";
  static constexpr std::string_view kExpand = "Expand";
  static constexpr std::string_view kFilterProject = "FilterProject";
  static constexpr std::string_view kGroupId = "GroupId";
  static constexpr std::string_view kHashBuild = "HashBuild";
  static constexpr std::string_view kHashProbe = "HashProbe";
  static constexpr std::string_view kIndexLookupJoin = "IndexLookupJoin";
  static constexpr std::string_view kLimit = "Limit";
  static constexpr std::string_view kLocalExchange = "LocalExchange";
  static constexpr std::string_view kLocalMerge = "LocalMerge";
  static constexpr std::string_view kLocalPartition = "LocalPartition";
  static constexpr std::string_view kMarkDistinct = "MarkDistinct";
  static constexpr std::string_view kMergeExchange = "MergeExchange";
  static constexpr std::string_view kMergeJoin = "MergeJoin";
  static constexpr std::string_view kMixedUnion = "MixedUnion";
  static constexpr std::string_view kNestedLoopJoinBuild =
      "NestedLoopJoinBuild";
  static constexpr std::string_view kNestedLoopJoinProbe =
      "NestedLoopJoinProbe";
  static constexpr std::string_view kOperatorTraceScan = "OperatorTraceScan";
  static constexpr std::string_view kOrderBy = "OrderBy";
  static constexpr std::string_view kParallelProject = "ParallelProject";
  static constexpr std::string_view kPartialAggregation = "PartialAggregation";
  static constexpr std::string_view kPartitionedOutput = "PartitionedOutput";
  static constexpr std::string_view kRowNumber = "RowNumber";
  static constexpr std::string_view kSpatialJoinBuild = "SpatialJoinBuild";
  static constexpr std::string_view kSpatialJoinProbe = "SpatialJoinProbe";
  static constexpr std::string_view kStreamingEnforceDistinct =
      "StreamingEnforceDistinct";
  static constexpr std::string_view kTableScan = "TableScan";
  static constexpr std::string_view kTableWrite = "TableWrite";
  static constexpr std::string_view kTableWriteMerge = "TableWriteMerge";
  static constexpr std::string_view kTopN = "TopN";
  static constexpr std::string_view kTopNRowNumber = "TopNRowNumber";
  static constexpr std::string_view kUnnest = "Unnest";
  static constexpr std::string_view kValues = "Values";
  static constexpr std::string_view kWindow = "Window";
};

} // namespace facebook::velox::exec

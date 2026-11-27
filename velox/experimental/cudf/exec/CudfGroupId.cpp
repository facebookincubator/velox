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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfGroupId.h"
#include "velox/experimental/cudf/exec/CudfProjectionOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

namespace {

// Marks a grouping key that is not part of the grouping set.
constexpr column_index_t kMissingGroupingKey =
    std::numeric_limits<column_index_t>::max();

// Builds one projection per grouping set. Output columns are the grouping
// keys, then the aggregation inputs, then the group id.
std::vector<std::vector<ProjectionColumn>> makeProjections(
    const std::shared_ptr<const core::GroupIdNode>& groupIdNode) {
  const auto& inputType = groupIdNode->sources()[0]->outputType();
  const auto& outputType = groupIdNode->outputType();

  // Build output-to-input mapping for grouping keys
  std::unordered_map<column_index_t, column_index_t>
      outputToInputGroupingKeyMapping;
  for (const auto& groupingKeyInfo : groupIdNode->groupingKeyInfos()) {
    outputToInputGroupingKeyMapping[outputType->getChildIdx(
        groupingKeyInfo.output)] =
        inputType->getChildIdx(groupingKeyInfo.input->name());
  }

  const auto numGroupingKeys = groupIdNode->numGroupingKeys();
  const auto numGroupingSets = groupIdNode->groupingSets().size();

  // Build groupingKeyMappings - one vector per grouping set
  std::vector<std::vector<column_index_t>> groupingKeyMappings;
  groupingKeyMappings.reserve(numGroupingSets);
  for (const auto& groupingSet : groupIdNode->groupingSets()) {
    std::vector<column_index_t> mappings(numGroupingKeys, kMissingGroupingKey);
    for (const auto& groupingKey : groupingSet) {
      auto outputChannel = outputType->getChildIdx(groupingKey);
      VELOX_USER_CHECK_NE(
          outputToInputGroupingKeyMapping.count(outputChannel),
          0,
          "GroupIdNode didn't map grouping key {} to input channel",
          groupingKey);
      VELOX_CHECK_LT(
          outputChannel,
          outputToInputGroupingKeyMapping.size(),
          "outputChannel out of bounds in outputToInputGroupingKeyMapping");
      auto inputChannel = outputToInputGroupingKeyMapping.at(outputChannel);
      VELOX_CHECK_LT(
          outputChannel,
          mappings.size(),
          "outputChannel out of bounds in mappings");
      mappings[outputChannel] = inputChannel;
    }
    groupingKeyMappings.emplace_back(std::move(mappings));
  }

  // Build aggregationInputs - indices of aggregation input columns
  const auto& aggregationInputs = groupIdNode->aggregationInputs();
  std::vector<column_index_t> aggregationInputChannels;
  aggregationInputChannels.reserve(aggregationInputs.size());
  for (const auto& input : aggregationInputs) {
    aggregationInputChannels.push_back(inputType->getChildIdx(input->name()));
  }

  // Precompute cudf data types for grouping key columns (used to create
  // all-null columns for keys not in a grouping set).
  std::vector<cudf::data_type> groupingKeyCudfTypes;
  groupingKeyCudfTypes.reserve(numGroupingKeys);
  for (size_t i = 0; i < numGroupingKeys; ++i) {
    groupingKeyCudfTypes.push_back(veloxToCudfDataType(outputType->childAt(i)));
  }

  std::vector<std::vector<ProjectionColumn>> projections;
  projections.reserve(numGroupingSets);
  for (size_t groupingSetIndex = 0; groupingSetIndex < numGroupingSets;
       ++groupingSetIndex) {
    const auto& mapping = groupingKeyMappings[groupingSetIndex];
    auto& columns = projections.emplace_back();
    for (size_t i = 0; i < numGroupingKeys; ++i) {
      if (mapping[i] == kMissingGroupingKey) {
        // Create an all-null column for keys not in this grouping set
        columns.push_back(
            ProjectionColumn::constant(
                [type = groupingKeyCudfTypes[i]](auto stream, auto* /*pool*/) {
                  return cudf::make_default_constructed_scalar(
                      type, stream, get_temp_mr());
                }));
      } else {
        columns.push_back(ProjectionColumn::field(mapping[i]));
      }
    }
    for (auto inputIdx : aggregationInputChannels) {
      columns.push_back(ProjectionColumn::field(inputIdx));
    }
    // Add group_id column (constant BIGINT with current grouping set index)
    columns.push_back(
        ProjectionColumn::constant(
            [groupingSetIndex](auto stream, auto* /*pool*/) {
              return std::make_unique<cudf::numeric_scalar<int64_t>>(
                  static_cast<int64_t>(groupingSetIndex),
                  true,
                  stream,
                  get_temp_mr());
            }));
  }
  return projections;
}

} // namespace

CudfGroupId::CudfGroupId(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::GroupIdNode>& groupIdNode)
    : CudfProjectionOperator(
          operatorId,
          driverCtx,
          groupIdNode->outputType(),
          groupIdNode->id(),
          "CudfGroupId",
          nvtx3::rgb{128, 0, 128}, // Purple
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          groupIdNode,
          makeProjections(groupIdNode),
          groupIdNode->sources()[0]->outputType()->size()) {}

} // namespace facebook::velox::cudf_velox

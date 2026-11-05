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

namespace facebook::velox::cudf_velox {

namespace {

// Builds one projection per grouping set. Output columns are the grouping
// keys, then the aggregation inputs, then the group id.
std::vector<std::vector<ProjectionColumn>> makeProjections(
    const core::GroupIdNode& groupIdNode) {
  const auto& inputType = groupIdNode.sources()[0]->outputType();
  const auto& outputType = groupIdNode.outputType();

  std::unordered_map<column_index_t, column_index_t> outputToInputKey;
  for (const auto& groupingKeyInfo : groupIdNode.groupingKeyInfos()) {
    outputToInputKey[outputType->getChildIdx(groupingKeyInfo.output)] =
        inputType->getChildIdx(groupingKeyInfo.input->name());
  }

  // Keys outside a grouping set are all-null columns of the key's cuDF type.
  const auto numGroupingKeys = groupIdNode.numGroupingKeys();
  std::vector<cudf::data_type> keyTypes;
  keyTypes.reserve(numGroupingKeys);
  for (size_t i = 0; i < numGroupingKeys; ++i) {
    keyTypes.push_back(veloxToCudfDataType(outputType->childAt(i)));
  }

  std::vector<std::vector<ProjectionColumn>> projections;
  projections.reserve(groupIdNode.groupingSets().size());
  for (const auto& groupingSet : groupIdNode.groupingSets()) {
    std::vector<std::optional<column_index_t>> keyChannels(numGroupingKeys);
    for (const auto& groupingKey : groupingSet) {
      const auto outputChannel = outputType->getChildIdx(groupingKey);
      VELOX_USER_CHECK_NE(
          outputToInputKey.count(outputChannel),
          0,
          "GroupIdNode didn't map grouping key {} to input channel",
          groupingKey);
      VELOX_CHECK_LT(outputChannel, numGroupingKeys);
      keyChannels[outputChannel] = outputToInputKey.at(outputChannel);
    }

    auto& columns = projections.emplace_back();
    for (size_t i = 0; i < numGroupingKeys; ++i) {
      if (keyChannels[i].has_value()) {
        columns.push_back(ProjectionColumn::field(*keyChannels[i]));
      } else {
        columns.push_back(
            ProjectionColumn::constant(
                [type = keyTypes[i]](auto stream, auto* /*pool*/) {
                  return cudf::make_default_constructed_scalar(
                      type, stream, get_temp_mr());
                }));
      }
    }
    for (const auto& input : groupIdNode.aggregationInputs()) {
      columns.push_back(
          ProjectionColumn::field(inputType->getChildIdx(input->name())));
    }
    const auto groupId = static_cast<int64_t>(projections.size() - 1);
    columns.push_back(
        ProjectionColumn::constant([groupId](auto stream, auto* /*pool*/) {
          return std::make_unique<cudf::numeric_scalar<int64_t>>(
              groupId, true, stream, get_temp_mr());
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
          makeProjections(*groupIdNode),
          groupIdNode->sources()[0]->outputType()->size()) {}

} // namespace facebook::velox::cudf_velox

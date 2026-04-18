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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

CudfGroupId::CudfGroupId(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::GroupIdNode>& groupIdNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          groupIdNode->outputType(),
          groupIdNode->id(),
          "CudfGroupId",
          nvtx3::rgb{128, 0, 128}, // Purple
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          groupIdNode) {
  const auto& inputType = groupIdNode->sources()[0]->outputType();

  // Build output-to-input mapping for grouping keys
  std::unordered_map<column_index_t, column_index_t>
      outputToInputGroupingKeyMapping;
  for (const auto& groupingKeyInfo : groupIdNode->groupingKeyInfos()) {
    outputToInputGroupingKeyMapping[outputType_->getChildIdx(
        groupingKeyInfo.output)] =
        inputType->getChildIdx(groupingKeyInfo.input->name());
  }

  numGroupingKeys_ = groupIdNode->numGroupingKeys();
  numGroupingSets_ = groupIdNode->groupingSets().size();

  // Build groupingKeyMappings_ - one vector per grouping set
  groupingKeyMappings_.reserve(numGroupingSets_);
  for (const auto& groupingSet : groupIdNode->groupingSets()) {
    std::vector<column_index_t> mappings(numGroupingKeys_, kMissingGroupingKey);
    for (const auto& groupingKey : groupingSet) {
      auto outputChannel = outputType_->getChildIdx(groupingKey);
      VELOX_USER_CHECK_NE(
          outputToInputGroupingKeyMapping.count(outputChannel),
          0,
          "GroupIdNode didn't map grouping key {} to input channel",
          groupingKey);
      auto inputChannel = outputToInputGroupingKeyMapping.at(outputChannel);
      mappings[outputChannel] = inputChannel;
    }
    groupingKeyMappings_.emplace_back(std::move(mappings));
  }

  // Build aggregationInputs_ - indices of aggregation input columns
  const auto& aggregationInputs = groupIdNode->aggregationInputs();
  aggregationInputs_.reserve(aggregationInputs.size());
  for (const auto& input : aggregationInputs) {
    aggregationInputs_.push_back(inputType->getChildIdx(input->name()));
  }
}

bool CudfGroupId::needsInput() const {
  return !noMoreInput_ && inputColumns_.empty();
}

void CudfGroupId::doAddInput(RowVectorPtr input) {
  VELOX_CHECK(inputColumns_.empty(), "Previous input not fully consumed");

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfGroupId expects CudfVector input");

  inputStream_ = cudfInput->stream();
  inputSize_ = static_cast<cudf::size_type>(cudfInput->size());

  // Store input columns for cycling through grouping sets
  auto table = cudfInput->release();
  inputColumns_ = table->release();
  groupingSetIndex_ = 0;
}

RowVectorPtr CudfGroupId::doGetOutput() {
  if (inputColumns_.empty()) {
    return nullptr;
  }

  auto stream = inputStream_;
  auto outputMr = get_output_mr();
  auto tempMr = get_temp_mr();
  auto numRows = inputSize_;

  const auto& mapping = groupingKeyMappings_[groupingSetIndex_];
  std::vector<std::unique_ptr<cudf::column>> outputColumns(outputType_->size());

  // Track how many more times each input column will be used
  // This helps us decide whether to move or copy
  std::unordered_map<column_index_t, int> inputColumnUsageCount;

  // Count usage for remaining grouping sets (including current one)
  for (size_t setIdx = groupingSetIndex_; setIdx < numGroupingSets_; ++setIdx) {
    const auto& setMapping = groupingKeyMappings_[setIdx];
    for (size_t i = 0; i < numGroupingKeys_; ++i) {
      if (setMapping[i] != kMissingGroupingKey) {
        inputColumnUsageCount[setMapping[i]]++;
      }
    }
  }

  // Count usage for aggregation inputs (used in all remaining sets)
  size_t remainingSets = numGroupingSets_ - groupingSetIndex_;
  for (auto inputIdx : aggregationInputs_) {
    inputColumnUsageCount[inputIdx] += remainingSets;
  }

  // Fill in grouping keys
  for (size_t i = 0; i < numGroupingKeys_; ++i) {
    if (mapping[i] == kMissingGroupingKey) {
      // Create an all-null column for keys not in this grouping set
      auto cudfType = veloxToCudfDataType(outputType_->childAt(i));
      auto nullScalar =
          cudf::make_default_constructed_scalar(cudfType, stream, tempMr);
      outputColumns[i] =
          cudf::make_column_from_scalar(*nullScalar, numRows, stream, outputMr);
    } else {
      auto inputIdx = mapping[i];
      inputColumnUsageCount[inputIdx]--;
      if (inputColumnUsageCount[inputIdx] == 0) {
        // Last use - move the column
        outputColumns[i] = std::move(inputColumns_[inputIdx]);
      } else {
        // Will be needed again - copy the column
        outputColumns[i] = std::make_unique<cudf::column>(
            *inputColumns_[inputIdx], stream, outputMr);
      }
    }
  }

  // Fill in aggregation inputs
  for (size_t i = 0; i < aggregationInputs_.size(); ++i) {
    auto inputIdx = aggregationInputs_[i];
    auto outputIdx = numGroupingKeys_ + i;
    inputColumnUsageCount[inputIdx]--;
    if (inputColumnUsageCount[inputIdx] == 0) {
      // Last use - move the column
      outputColumns[outputIdx] = std::move(inputColumns_[inputIdx]);
    } else {
      // Will be needed again - copy the column
      outputColumns[outputIdx] = std::make_unique<cudf::column>(
          *inputColumns_[inputIdx], stream, outputMr);
    }
  }

  // Add group_id column (constant BIGINT with current grouping set index)
  auto groupIdScalar = cudf::numeric_scalar<int64_t>(
      static_cast<int64_t>(groupingSetIndex_), true, stream, tempMr);
  outputColumns[outputType_->size() - 1] =
      cudf::make_column_from_scalar(groupIdScalar, numRows, stream, outputMr);

  // Build output table
  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));

  // Advance to next grouping set
  ++groupingSetIndex_;
  if (groupingSetIndex_ == numGroupingSets_) {
    // All grouping sets processed for this input
    inputColumns_.clear();
    groupingSetIndex_ = 0;
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, numRows, std::move(outputTable), stream);
}

} // namespace facebook::velox::cudf_velox

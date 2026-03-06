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

#include "velox/exec/MarkSorted.h"
#include "velox/common/base/CompareFlags.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

MarkSorted::MarkSorted(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MarkSortedNode>& planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "MarkSorted"),
      markerName_(planNode->markerName()) {
  const auto& inputType = planNode->sources()[0]->outputType();
  const auto& sortingKeys = planNode->sortingKeys();
  const auto& sortingOrders = planNode->sortingOrders();

  // Set all input columns as identity projection.
  for (auto i = 0; i < inputType->size(); ++i) {
    identityProjections_.emplace_back(i, i);
  }

  // Map the marker result (results_[0]) to the last column position in output,
  // immediately after all input columns.
  resultProjections_.emplace_back(0, inputType->size());

  // Extract channel indices for sorting keys.
  sortingKeyChannels_.reserve(sortingKeys.size());
  compareFlags_.reserve(sortingKeys.size());

  for (auto i = 0; i < sortingKeys.size(); ++i) {
    const auto& key = sortingKeys[i];
    auto channel = inputType->getChildIdx(key->name());
    sortingKeyChannels_.push_back(channel);

    // Build CompareFlags from SortOrder.
    const auto& order = sortingOrders[i];
    compareFlags_.push_back(
        {order.isNullsFirst(),
         order.isAscending(),
         false, // equalsOnly
         CompareFlags::NullHandlingMode::kNullAsValue});
  }

  results_.resize(1);
}

void MarkSorted::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

bool MarkSorted::isSortedRelativeTo(
    const RowVectorPtr& currentData,
    vector_size_t currentIndex,
    const RowVectorPtr& prevData,
    vector_size_t prevIndex) {
  // Compare each sorting key column.
  // For sorted data, each row should be >= previous (ascending) or <= previous
  // (descending). The compare function respects ascending/descending flags,
  // so we just check if compare() returns >= 0 for each key.
  for (auto i = 0; i < sortingKeyChannels_.size(); ++i) {
    auto channel = sortingKeyChannels_[i];
    const auto& currentColumn = currentData->childAt(channel);
    const auto& prevColumn = prevData->childAt(channel);

    auto result = currentColumn->compare(
        prevColumn.get(), currentIndex, prevIndex, compareFlags_[i]);

    if (result.has_value()) {
      if (result.value() < 0) {
        // Current row is less than previous (in sort order), NOT sorted.
        return false;
      } else if (result.value() > 0) {
        // Current row is greater than previous, sorted (no need to check more
        // keys).
        return true;
      }
      // Equal on this key, continue to next key.
    }
  }

  // All keys are equal - this is still considered sorted.
  return true;
}

RowVectorPtr MarkSorted::getOutput() {
  if (isFinished() || !input_) {
    return nullptr;
  }

  auto outputSize = input_->size();

  // Handle empty batches.
  if (outputSize == 0) {
    input_ = nullptr;
    return nullptr;
  }

  // Re-use memory for the marker vector if possible.
  VectorPtr& result = results_[0];
  if (result && result.use_count() == 1) {
    BaseVector::prepareForReuse(result, outputSize);
  } else {
    result = BaseVector::create(BOOLEAN(), outputSize, operatorCtx_->pool());
  }

  auto resultBits =
      results_[0]->as<FlatVector<bool>>()->mutableRawValues<uint64_t>();

  // Initialize all bits to true (sorted), then set false for violations.
  bits::fillBits(resultBits, 0, outputSize, true);

  // First row of first batch is always marked true (already initialized).
  // First row of subsequent batches is compared with lastRow_.
  if (lastRow_) {
    // Subsequent batch: compare first row with stored lastRow_.
    // Use last row index of previous batch.
    bool sorted = isSortedRelativeTo(input_, 0, lastRow_, lastRow_->size() - 1);
    if (!sorted) {
      bits::setBit(resultBits, 0, false);
    }
  }

  // Process remaining rows in batch.
  for (auto i = 1; i < outputSize; ++i) {
    bool sorted = isSortedRelativeTo(input_, i, input_, i - 1);
    if (!sorted) {
      bits::setBit(resultBits, i, false);
    }
  }

  // Store the entire input for next batch comparison.
  // Future optimization (D3): copy only key columns of last row.
  lastRow_ = input_;

  auto output = fillOutput(outputSize, nullptr);

  // Drop reference to input_ to make it singly-referenced at the producer and
  // allow for memory reuse.
  input_ = nullptr;

  return output;
}

void MarkSorted::noMoreInput() {
  Operator::noMoreInput();
  lastRow_.reset();
}

bool MarkSorted::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

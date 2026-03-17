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

  // Precompute type for lastRow_ (contains only sorting key columns).
  std::vector<std::string> keyNames;
  std::vector<TypePtr> keyTypes;
  keyNames.reserve(sortingKeys.size());
  keyTypes.reserve(sortingKeys.size());
  for (auto i = 0; i < sortingKeys.size(); ++i) {
    auto channel = sortingKeyChannels_[i];
    keyNames.push_back(inputType->nameOf(channel));
    keyTypes.push_back(inputType->childAt(channel));
  }
  lastRowType_ = ROW(std::move(keyNames), std::move(keyTypes));

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
    // Compare first row of current batch with stored last row.
    // lastRow_ has key columns at sequential indices (0, 1, 2, ...) so we
    // cannot use isSortedRelativeTo which expects the same schema on both
    // sides.
    bool sorted = true;
    for (auto i = 0; i < sortingKeyChannels_.size(); ++i) {
      auto channel = sortingKeyChannels_[i];
      const auto& currentColumn = input_->childAt(channel);
      const auto& prevColumn = lastRow_->childAt(i);

      auto result =
          currentColumn->compare(prevColumn.get(), 0, 0, compareFlags_[i]);

      if (result.has_value()) {
        if (result.value() < 0) {
          sorted = false;
          break;
        } else if (result.value() > 0) {
          break;
        }
      }
    }
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

  // Copy only sorting key columns of the last row for cross-batch comparison.
  // Avoids holding a reference to the entire batch, preventing OOM on wide
  // schemas and allowing the upstream producer to reuse memory.
  copyLastRowKeyColumns();

  auto output = fillOutput(outputSize, nullptr);

  // Drop reference to input_ to make it singly-referenced at the producer and
  // allow for memory reuse.
  input_ = nullptr;

  return output;
}

void MarkSorted::copyLastRowKeyColumns() {
  auto lastIndex = input_->size() - 1;
  auto numKeys = sortingKeyChannels_.size();

  std::vector<VectorPtr> keyChildren(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    auto channel = sortingKeyChannels_[i];
    const auto& sourceColumn = input_->childAt(channel);
    keyChildren[i] =
        BaseVector::create(sourceColumn->type(), 1, operatorCtx_->pool());
    keyChildren[i]->copy(sourceColumn.get(), 0, lastIndex, 1);
  }

  lastRow_ = std::make_shared<RowVector>(
      operatorCtx_->pool(),
      lastRowType_,
      nullptr, // no nulls
      1, // single row
      std::move(keyChildren));
}

void MarkSorted::noMoreInput() {
  Operator::noMoreInput();
  lastRow_.reset();
}

bool MarkSorted::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

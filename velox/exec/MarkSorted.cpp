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
#include "velox/core/QueryConfig.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

namespace {

/// Check if type is a primitive integer type suitable for the fast comparison
/// path. Excludes floating-point types because IEEE 754 NaN comparison
/// semantics differ from Velox's compare() which treats NaN as greater than
/// all non-NaN values.
bool isPrimitiveTypeForFastPath(TypeKind kind) {
  switch (kind) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return true;
    default:
      return false;
  }
}

} // namespace

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

  // Get zero-copy threshold from config.
  zeroCopyThreshold_ = driverCtx->queryConfig().markSortedZeroCopyThreshold();

  // Check if the fast comparison path can be used:
  // - Single sorting key only
  // - Integer type (TINYINT, SMALLINT, INTEGER, BIGINT)
  // - Nulls last (so we don't need special null handling)
  if (sortingKeys.size() == 1) {
    auto keyType = inputType->childAt(sortingKeyChannels_[0]);
    if (isPrimitiveTypeForFastPath(keyType->kind()) &&
        !sortingOrders[0].isNullsFirst()) {
      canUseFastPath_ = true;
      fastPathKeyTypeKind_ = keyType->kind();
    }
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

void MarkSorted::copyLastRowKeyColumns() {
  // Create a RowVector with only the key columns from the last row.
  auto lastIndex = input_->size() - 1;

  // Build type and vectors for the copy.
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<VectorPtr> keyColumns;
  names.reserve(sortingKeyChannels_.size());
  types.reserve(sortingKeyChannels_.size());
  keyColumns.reserve(sortingKeyChannels_.size());

  for (size_t i = 0; i < sortingKeyChannels_.size(); ++i) {
    auto channel = sortingKeyChannels_[i];
    const auto& sourceColumn = input_->childAt(channel);

    names.push_back(std::to_string(i)); // Use index as name.
    types.push_back(sourceColumn->type());

    // Create a single-row vector with the last row's value.
    auto singleRowVector =
        BaseVector::create(sourceColumn->type(), 1, operatorCtx_->pool());
    singleRowVector->copy(sourceColumn.get(), 0, lastIndex, 1);
    keyColumns.push_back(std::move(singleRowVector));
  }

  auto copyType = ROW(std::move(names), std::move(types));
  lastRowCopy_ = std::make_shared<RowVector>(
      operatorCtx_->pool(), copyType, nullptr, 1, std::move(keyColumns));
}

bool MarkSorted::canApplyFastPathToInput() const {
  if (!canUseFastPath_) {
    return false;
  }

  auto channel = sortingKeyChannels_[0];
  const auto& keyColumn = input_->childAt(channel);

  if (keyColumn->encoding() != VectorEncoding::Simple::FLAT) {
    return false;
  }

  if (keyColumn->mayHaveNulls()) {
    return false;
  }

  // Require minimum batch size for the fast path to be worthwhile.
  static constexpr vector_size_t kMinFastPathBatchSize = 16;
  if (input_->size() < kMinFastPathBatchSize) {
    return false;
  }

  return true;
}

template <typename T>
void MarkSorted::applyFastPathComparison(
    const T* data,
    vector_size_t size,
    bool ascending,
    uint64_t* resultBits) {
  // First row is already marked true (initialized before this call).
  // Compare data[i] with data[i-1] for i = 1..size-1.
  for (vector_size_t i = 1; i < size; ++i) {
    bool sorted;
    if (ascending) {
      sorted = data[i] >= data[i - 1];
    } else {
      sorted = data[i] <= data[i - 1];
    }
    if (!sorted) {
      bits::setBit(resultBits, i, false);
    }
  }
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

  // Handle cross-batch comparison for the first row.
  // Use prevInput_ (zero-copy mode) or lastRowCopy_ (copy mode).
  RowVectorPtr prevBatchData;
  vector_size_t prevBatchIndex = 0;
  if (prevInput_) {
    prevBatchData = prevInput_;
    prevBatchIndex = prevInput_->size() - 1;
  } else if (lastRowCopy_) {
    // In copy mode, lastRowCopy_ has only key columns indexed 0..N-1.
    // We need to use a different comparison that maps channels.
    // For simplicity, use the generic comparison with the full batch.
    // But lastRowCopy_ only has key columns, so we compare directly.
    for (auto i = 0; i < sortingKeyChannels_.size(); ++i) {
      const auto& currentColumn = input_->childAt(sortingKeyChannels_[i]);
      const auto& prevColumn = lastRowCopy_->childAt(i);

      auto compareResult =
          currentColumn->compare(prevColumn.get(), 0, 0, compareFlags_[i]);

      if (compareResult.has_value()) {
        if (compareResult.value() < 0) {
          bits::setBit(resultBits, 0, false);
          break;
        } else if (compareResult.value() > 0) {
          break; // Sorted, no need to check more keys.
        }
      }
    }
    prevBatchData = nullptr; // Already handled.
  }

  if (prevBatchData) {
    bool sorted = isSortedRelativeTo(input_, 0, prevBatchData, prevBatchIndex);
    if (!sorted) {
      bits::setBit(resultBits, 0, false);
    }
  }

  // Process remaining rows in batch using fast path or generic path.
  if (canApplyFastPathToInput()) {
    auto channel = sortingKeyChannels_[0];
    const auto& keyColumn = input_->childAt(channel);
    bool ascending = compareFlags_[0].ascending;

    switch (fastPathKeyTypeKind_) {
      case TypeKind::TINYINT: {
        auto* data = keyColumn->as<FlatVector<int8_t>>()->rawValues();
        applyFastPathComparison(data, outputSize, ascending, resultBits);
        break;
      }
      case TypeKind::SMALLINT: {
        auto* data = keyColumn->as<FlatVector<int16_t>>()->rawValues();
        applyFastPathComparison(data, outputSize, ascending, resultBits);
        break;
      }
      case TypeKind::INTEGER: {
        auto* data = keyColumn->as<FlatVector<int32_t>>()->rawValues();
        applyFastPathComparison(data, outputSize, ascending, resultBits);
        break;
      }
      case TypeKind::BIGINT: {
        auto* data = keyColumn->as<FlatVector<int64_t>>()->rawValues();
        applyFastPathComparison(data, outputSize, ascending, resultBits);
        break;
      }
      default:
        VELOX_UNREACHABLE();
    }
  } else {
    // Generic path for multi-key, non-primitive types, or small batches.
    for (auto i = 1; i < outputSize; ++i) {
      bool sorted = isSortedRelativeTo(input_, i, input_, i - 1);
      if (!sorted) {
        bits::setBit(resultBits, i, false);
      }
    }
  }

  // Store reference for next batch comparison using zero-copy or copy mode.
  if (outputSize < zeroCopyThreshold_) {
    // Zero-copy mode: hold entire batch.
    prevInput_ = input_;
    lastRowCopy_.reset();
  } else {
    // Copy mode: copy only key columns of last row.
    copyLastRowKeyColumns();
    prevInput_.reset();
  }

  auto output = fillOutput(outputSize, nullptr);

  // Drop reference to input_ to make it singly-referenced at the producer and
  // allow for memory reuse.
  input_ = nullptr;

  return output;
}

void MarkSorted::noMoreInput() {
  Operator::noMoreInput();
  prevInput_.reset();
  lastRowCopy_.reset();
}

bool MarkSorted::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

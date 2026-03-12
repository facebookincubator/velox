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
#include "velox/functions/lib/SIMDComparisonUtil.h"
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
  sortingOrders_.reserve(sortingKeys.size());

  for (auto i = 0; i < sortingKeys.size(); ++i) {
    const auto& key = sortingKeys[i];
    auto channel = inputType->getChildIdx(key->name());
    sortingKeyChannels_.push_back(channel);

    const auto& order = sortingOrders[i];
    sortingOrders_.push_back(order);

    // Build CompareFlags from SortOrder.
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

  zeroCopyThreshold_ = driverCtx->queryConfig().markSortedZeroCopyThreshold();

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
  for (auto i = 0; i < sortingKeyChannels_.size(); ++i) {
    auto channel = sortingKeyChannels_[i];
    const auto& currentColumn = currentData->childAt(channel);
    const auto& prevColumn = prevData->childAt(channel);

    auto result = currentColumn->compare(
        prevColumn.get(), currentIndex, prevIndex, compareFlags_[i]);

    if (result.has_value()) {
      if (result.value() < 0) {
        return false;
      } else if (result.value() > 0) {
        return true;
      }
    }
  }

  // All keys are equal - this is still considered sorted.
  return true;
}

bool MarkSorted::allKeysConstant() const {
  for (auto channel : sortingKeyChannels_) {
    auto& keyCol = input_->childAt(channel);
    if (!keyCol->isConstantEncoding() || keyCol->isNullAt(0)) {
      return false;
    }
  }
  return true;
}

bool MarkSorted::isSimdEligibleType(TypeKind kind) {
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

bool MarkSorted::canApplySimdPath() const {
  if (sortingKeyChannels_.size() != 1) {
    return false;
  }
  auto& keyCol = input_->childAt(sortingKeyChannels_[0]);
  if (!keyCol->isFlatEncoding() || keyCol->mayHaveNulls()) {
    return false;
  }
  return isSimdEligibleType(keyCol->typeKind());
}

void MarkSorted::applySimdComparison(
    uint64_t* resultBits,
    vector_size_t numRows) {
  auto channel = sortingKeyChannels_[0];
  auto& keyCol = input_->childAt(channel);
  bool ascending = sortingOrders_[0].isAscending();
  const auto numCompares = numRows - 1;

  // Allocate bit-packed result buffer for SIMD output.
  const auto bufferBytes = bits::nbytes(numCompares);
  auto simdBuffer = AlignedBuffer::allocate<uint8_t>(bufferBytes, pool());
  auto simdResult = simdBuffer->asMutable<uint8_t>();
  memset(simdResult, 0, bufferBytes);

  // Dispatch by type to call applySimdComparison with typed raw data.
  // Consecutive-row trick: rawData+1 as lhs, rawData as rhs.
  // For ascending: row[i+1] >= row[i] means sorted.
  // For descending: row[i+1] <= row[i] means sorted.
  auto dispatchSimd = [&](auto dummy) {
    using T = decltype(dummy);
    const T* rawData = keyCol->asFlatVector<T>()->rawValues();
    if (ascending) {
      functions::applySimdComparison<T, false, false, std::greater_equal<>>(
          0, numCompares, rawData + 1, rawData, simdResult);
    } else {
      functions::applySimdComparison<T, false, false, std::less_equal<>>(
          0, numCompares, rawData + 1, rawData, simdResult);
    }
  };

  switch (keyCol->typeKind()) {
    case TypeKind::TINYINT:
      dispatchSimd(int8_t{});
      break;
    case TypeKind::SMALLINT:
      dispatchSimd(int16_t{});
      break;
    case TypeKind::INTEGER:
      dispatchSimd(int32_t{});
      break;
    case TypeKind::BIGINT:
      dispatchSimd(int64_t{});
      break;
    default:
      VELOX_UNREACHABLE();
  }

  // Merge bit-packed SIMD results into resultBits with +1 offset.
  // simdResult bit i = comparison result for (data[i+1] vs data[i]).
  // Row 0 is handled by cross-batch logic, so we write to bit i+1.
  for (vector_size_t i = 0; i < numCompares; ++i) {
    if (!bits::isBitSet(simdResult, i)) {
      bits::clearBit(resultBits, i + 1);
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

  // Initialize all bits to true (sorted), then clear for violations.
  bits::fillBits(resultBits, 0, outputSize, true);

  // Cross-batch comparison: compare first row of current batch with last row
  // of previous batch. Single if/else if chain (U3).
  if (prevInput_) {
    // Zero-copy mode: prevInput_ has same schema as current input.
    for (column_index_t k = 0; k < sortingKeyChannels_.size(); ++k) {
      auto channel = sortingKeyChannels_[k];
      auto cmp = prevInput_->childAt(channel)->compare(
          input_->childAt(channel).get(),
          prevInput_->size() - 1,
          0,
          compareFlags_[k]);
      if (cmp.has_value() && cmp.value() != 0) {
        if (cmp.value() > 0) {
          // Previous > current in sort order means NOT sorted.
          bits::clearBit(resultBits, 0);
        }
        break;
      }
    }
    prevInput_.reset();
  } else if (lastRow_) {
    // Copy mode: lastRow_ has key columns at sequential indices.
    for (column_index_t k = 0; k < sortingKeyChannels_.size(); ++k) {
      auto channel = sortingKeyChannels_[k];
      auto cmp = lastRow_->childAt(k)->compare(
          input_->childAt(channel).get(), 0, 0, compareFlags_[k]);
      if (cmp.has_value() && cmp.value() != 0) {
        if (cmp.value() > 0) {
          bits::clearBit(resultBits, 0);
        }
        break;
      }
    }
  }

  // Within-batch comparison.
  if (outputSize > 1) {
    if (allKeysConstant()) {
      // ConstantVector fast path (U6): all key columns are constant non-null,
      // so all rows are trivially sorted. Bits are already true.
    } else if (canApplySimdPath()) {
      // SIMD fast path (U5): single flat non-null primitive key.
      applySimdComparison(resultBits, outputSize);
    } else {
      // Generic path: compare each row with its predecessor.
      for (auto i = 1; i < outputSize; ++i) {
        if (!isSortedRelativeTo(input_, i, input_, i - 1)) {
          bits::setBit(resultBits, i, false);
        }
      }
    }
  }

  // Store last row for next batch's cross-batch comparison.
  if (input_->size() < zeroCopyThreshold_) {
    // Zero-copy: hold reference to entire batch.
    prevInput_ = input_;
    lastRow_.reset();
  } else {
    // Copy: deep-copy key columns only.
    copyLastRowKeyColumns();
    prevInput_.reset();
  }

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
  prevInput_.reset();
}

bool MarkSorted::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

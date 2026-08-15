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

#include "velox/connectors/hive/iceberg/IcebergMergeProcessor.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Returns the merge_row child as a RowVector. The merge_row column is
// expected to be a flat ROW after the caller has decoded any
// dictionary/lazy/constant wrappers. `transform` flattens its input
// merge_row child via `BaseVector::flattenVector` before calling this; this
// helper is a final defensive type-check that the result really is a
// `RowVector` and not, for example, a flat top-level Row whose children
// are still encoded.
const RowVector* asMergeRowVector(const VectorPtr& mergeRowChild) {
  VELOX_CHECK_NOT_NULL(mergeRowChild, "merge_row column is null.");
  const auto loaded = mergeRowChild->loadedVector();
  const auto* rowVector = loaded->as<RowVector>();
  VELOX_CHECK_NOT_NULL(
      rowVector,
      "merge_row column must be a flat RowVector, got encoding: {}",
      static_cast<int>(loaded->encoding()));
  return rowVector;
}

// Returns the operation TINYINT child of merge_row. Position is
// `numMergeFields - 2` per the documented merge_row layout
// (`ConnectorMergeSink.storeMergedRows` Javadoc): the last two fields are
// operation and case_number, in that order.
const FlatVector<int8_t>* asOperationVector(const RowVector& mergeRow) {
  const auto numMergeFields = mergeRow.childrenSize();
  VELOX_CHECK_GE(
      numMergeFields,
      2,
      "merge_row must have at least operation + case_number fields, got {}",
      numMergeFields);
  const auto& operationChild =
      mergeRow.childAt(static_cast<column_index_t>(numMergeFields - 2));
  VELOX_CHECK_NOT_NULL(operationChild, "operation field of merge_row is null.");
  const auto* operationVector =
      operationChild->loadedVector()->asFlatVector<int8_t>();
  VELOX_CHECK_NOT_NULL(
      operationVector,
      "operation field of merge_row must be a flat TINYINT vector.");
  return operationVector;
}

} // namespace

IcebergMergeProcessor::IcebergMergeProcessor(
    std::vector<TypePtr> targetColumnTypes,
    std::vector<std::string> outputColumnNames,
    TypePtr rowIdType,
    column_index_t targetRowIdChannel,
    column_index_t mergeRowChannel)
    : targetColumnTypes_(std::move(targetColumnTypes)),
      outputColumnNames_(std::move(outputColumnNames)),
      rowIdType_(std::move(rowIdType)),
      targetRowIdChannel_(targetRowIdChannel),
      mergeRowChannel_(mergeRowChannel),
      numTargetColumns_(targetColumnTypes_.size()),
      outputType_(
          buildOutputType(targetColumnTypes_, outputColumnNames_, rowIdType_)) {
  VELOX_CHECK_NOT_NULL(rowIdType_, "rowIdType is null.");
  VELOX_CHECK_EQ(
      outputColumnNames_.size(),
      targetColumnTypes_.size() + 3,
      "outputColumnNames size must be targetColumnTypes.size() + 3 "
      "(target cols + operation + row_id + insert_from_update).");
  VELOX_CHECK_NE(
      targetRowIdChannel_,
      mergeRowChannel_,
      "targetRowIdChannel and mergeRowChannel must differ.");
}

RowTypePtr IcebergMergeProcessor::buildOutputType(
    const std::vector<TypePtr>& targetColumnTypes,
    const std::vector<std::string>& outputColumnNames,
    const TypePtr& rowIdType) {
  VELOX_CHECK_EQ(
      outputColumnNames.size(),
      targetColumnTypes.size() + 3,
      "outputColumnNames size must be targetColumnTypes.size() + 3.");
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  const auto numTargetColumns = targetColumnTypes.size();
  names.reserve(numTargetColumns + 3);
  types.reserve(numTargetColumns + 3);
  for (size_t i = 0; i < numTargetColumns; ++i) {
    VELOX_CHECK_NOT_NULL(
        targetColumnTypes[i], "targetColumnTypes[{}] is null.", i);
    // Output names come from the planner-declared output variables so
    // downstream nodes (TableWriter::setTypeMappings) bind by name to
    // iceberg/planner-correct identifiers, not synthetic placeholders.
    names.push_back(outputColumnNames[i]);
    types.push_back(targetColumnTypes[i]);
  }
  // Trailing three columns: operation TINYINT, rowId, insert_from_update
  // TINYINT — names from the planner output list, types fixed by the
  // IcebergMergeProcessor contract.
  names.push_back(outputColumnNames[numTargetColumns]);
  types.push_back(TINYINT());
  names.push_back(outputColumnNames[numTargetColumns + 1]);
  types.push_back(rowIdType);
  names.push_back(outputColumnNames[numTargetColumns + 2]);
  types.push_back(TINYINT());
  return ROW(std::move(names), std::move(types));
}

IcebergMergeProcessor::OperationCounts IcebergMergeProcessor::countOperations(
    const FlatVector<int8_t>& operationVector,
    vector_size_t numRows) const {
  OperationCounts counts;
  for (vector_size_t i = 0; i < numRows; ++i) {
    VELOX_USER_CHECK(
        !operationVector.isNullAt(i),
        "merge_row operation field is null at position {}.",
        i);
    const int8_t operation = operationVector.valueAt(i);
    switch (operation) {
      case kDefaultCaseOperationNumber:
        break;
      case kInsertOperationNumber:
        ++counts.numInsert;
        break;
      case kDeleteOperationNumber:
        ++counts.numDelete;
        break;
      case kUpdateOperationNumber:
        ++counts.numUpdate;
        break;
      default:
        VELOX_USER_FAIL(
            "Unknown merge operation byte: {}", static_cast<int>(operation));
    }
  }
  return counts;
}

std::vector<VectorPtr> IcebergMergeProcessor::allocateOutputChildren(
    vector_size_t numRows,
    memory::MemoryPool* pool) const {
  // BaseVector::create gives each child the correct null-vector and value
  // storage to be mutated by setNull / copy / set by the caller.
  std::vector<VectorPtr> outputChildren;
  outputChildren.reserve(numTargetColumns_ + 3);
  for (size_t c = 0; c < numTargetColumns_; ++c) {
    outputChildren.push_back(
        BaseVector::create(targetColumnTypes_[c], numRows, pool));
  }
  outputChildren.push_back(BaseVector::create(TINYINT(), numRows, pool));
  outputChildren.push_back(BaseVector::create(rowIdType_, numRows, pool));
  outputChildren.push_back(BaseVector::create(TINYINT(), numRows, pool));
  return outputChildren;
}

RowVectorPtr IcebergMergeProcessor::transform(
    const RowVectorPtr& input,
    memory::MemoryPool* pool) const {
  VELOX_CHECK_NOT_NULL(input, "Input row vector is null.");
  VELOX_CHECK_NOT_NULL(pool, "Memory pool is null.");

  const vector_size_t inputPositions = input->size();
  if (inputPositions == 0) {
    return BaseVector::create<RowVector>(outputType_, 0, pool);
  }

  VELOX_CHECK_LT(
      targetRowIdChannel_,
      input->childrenSize(),
      "targetRowIdChannel out of range.");
  VELOX_CHECK_LT(
      mergeRowChannel_, input->childrenSize(), "mergeRowChannel out of range.");

  const auto& rowIdInput = input->childAt(targetRowIdChannel_);
  VELOX_CHECK_NOT_NULL(rowIdInput, "targetRowId column is null.");

  // The merge_row column may arrive dictionary- or constant-encoded when
  // upstream operators wrap it in a Project rather than materializing a flat
  // ROW. The per-row fan-out below indexes children positionally, which
  // requires a flat RowVector with flat children. Flatten the column (and
  // its children) in place before passing it to asMergeRowVector so the
  // downstream encoding-strict check passes.
  auto mergeRowChild = input->childAt(mergeRowChannel_);
  BaseVector::flattenVector(mergeRowChild);
  const auto* mergeRow = asMergeRowVector(mergeRowChild);
  const auto* operationVector = asOperationVector(*mergeRow);

  VELOX_CHECK_GE(
      mergeRow->childrenSize(),
      numTargetColumns_ + 2,
      "merge_row must have at least {} fields (target + operation + "
      "case_number), got {}",
      numTargetColumns_ + 2,
      mergeRow->childrenSize());

  const OperationCounts counts =
      countOperations(*operationVector, inputPositions);
  const vector_size_t totalOutput = static_cast<vector_size_t>(
      counts.numInsert + counts.numDelete + 2 * counts.numUpdate);

  // Allocate output children with capacity for `totalOutput` positions.
  std::vector<VectorPtr> outputChildren =
      allocateOutputChildren(totalOutput, pool);

  const size_t operationOutIndex = numTargetColumns_;
  const size_t rowIdOutIndex = numTargetColumns_ + 1;
  const size_t insertFromUpdateOutIndex = numTargetColumns_ + 2;

  auto* operationOut =
      outputChildren[operationOutIndex]->asFlatVector<int8_t>();
  auto& rowIdOut = outputChildren[rowIdOutIndex];
  auto* insertFromUpdateOut =
      outputChildren[insertFromUpdateOutIndex]->asFlatVector<int8_t>();
  VELOX_CHECK_NOT_NULL(operationOut, "operation output is not flat.");
  VELOX_CHECK_NOT_NULL(
      insertFromUpdateOut, "insert_from_update output is not flat.");

  vector_size_t outIdx = 0;

  // Emits a DELETE output row sourced from the target-row-id column. Target
  // columns are nulled because they are not meaningful on a DELETE row (the
  // output schema is shared with INSERT rows).
  auto emitDeleteRow = [&](vector_size_t inputIdx) {
    for (size_t c = 0; c < numTargetColumns_; ++c) {
      outputChildren[c]->setNull(outIdx, true);
    }
    operationOut->set(outIdx, kDeleteOperationNumber);
    rowIdOut->copy(rowIdInput.get(), outIdx, inputIdx, 1);
    insertFromUpdateOut->set(outIdx, 0);
    ++outIdx;
  };

  // Emits an INSERT output row whose target column values come from the
  // leading fields of merge_row, in the same order as targetColumnTypes_.
  // 'fromUpdate' is 1 when this INSERT is the second half of an UPDATE, 0 for
  // a plain INSERT (downstream sinks ignore it today; kept for OSS parity).
  auto emitInsertRow = [&](vector_size_t inputIdx, bool fromUpdate) {
    for (size_t c = 0; c < numTargetColumns_; ++c) {
      outputChildren[c]->copy(
          mergeRow->childAt(static_cast<column_index_t>(c)).get(),
          outIdx,
          inputIdx,
          1);
    }
    operationOut->set(outIdx, kInsertOperationNumber);
    rowIdOut->setNull(outIdx, true);
    insertFromUpdateOut->set(outIdx, fromUpdate ? 1 : 0);
    ++outIdx;
  };

  for (vector_size_t i = 0; i < inputPositions; ++i) {
    const int8_t operation = operationVector->valueAt(i);
    if (operation == kDefaultCaseOperationNumber) {
      continue;
    }
    // DELETE and UPDATE both emit a DELETE row.
    if (operation == kDeleteOperationNumber ||
        operation == kUpdateOperationNumber) {
      emitDeleteRow(i);
    }
    // INSERT and UPDATE both emit an INSERT row.
    if (operation == kInsertOperationNumber ||
        operation == kUpdateOperationNumber) {
      emitInsertRow(i, operation == kUpdateOperationNumber);
    }
  }

  VELOX_CHECK_EQ(
      outIdx,
      totalOutput,
      "Emitted output positions {} do not match precomputed total {}.",
      outIdx,
      totalOutput);

  return std::make_shared<RowVector>(
      pool,
      outputType_,
      /*nulls=*/nullptr,
      totalOutput,
      std::move(outputChildren));
}

} // namespace facebook::velox::connector::hive::iceberg

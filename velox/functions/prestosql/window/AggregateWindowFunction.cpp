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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RawVector.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window {

namespace {

class AggregateWindowFunction : public exec::WindowFunction {
 public:
  explicit AggregateWindowFunction(
      const std::string& name,
      const std::vector<exec::RowColumn>& argColumns,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool), argColumns_(argColumns) {
    aggregate_ = exec::Aggregate::create(
        name, core::AggregationNode::Step::kSingle, argTypes, resultType);

    // Aggregate initialization.
    // Construct and set HashStringAllocator stringAllocator_ from MappedMemory.
    // aggregate_->setAllocator(&stringAllocator_);
    int32_t rowSizeOffset = bits::nbytes(1);
    int32_t singleGroupRowSize_ = rowSizeOffset + sizeof(int32_t);
    int32_t nullOffset = 0;
    aggregate_->setOffsets(
        singleGroupRowSize_,
        exec::RowContainer::nullByte(nullOffset),
        exec::RowContainer::nullMask(nullOffset),
        /* TODO needed for out of line allocations */ rowSizeOffset);
    singleGroupRowSize_ += aggregate_->accumulatorFixedWidthSize();

    auto singleGroup = std::vector<vector_size_t>{0};
    // Constructing the single row in the MemoryPool for now.
    singleGroupRow_ = (char*)(pool_->allocate(singleGroupRowSize_));
    aggregate_->initializeNewGroups(&singleGroupRow_, singleGroup);

    argVectors_.reserve(argTypes.size());
    for (const auto& argType : argTypes) {
      argVectors_.emplace_back(BaseVector::create(argType, 0, pool));
    }
  }

  ~AggregateWindowFunction() {
    std::vector<char*> singleGroupRowVector = {singleGroupRow_};
    aggregate_->destroy(folly::Range(singleGroupRowVector.data(), 1));
    pool()->free(singleGroupRow_, singleGroupRowSize_);
  }

  void resetPartition(const folly::Range<char**>& rows) {
    numPartitionRows_ = rows.size();
    // Setup the argument vectors for the rows in the partition.
    for (int i = 0; i < argColumns_.size(); i++) {
      argVectors_[i]->resize(numPartitionRows_);
      exec::RowContainer::extractColumn(
          rows.data(), numPartitionRows_, argColumns_[i], argVectors_[i]);
    }
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    int numRows = frameStarts->size();
    auto frameStartsVector = frameStarts->as<vector_size_t>();
    auto frameEndsVector = frameEnds->as<vector_size_t>();
    auto resultVector = BaseVector::create(resultType(), numRows, pool());

    for (int i = 0; i < numRows; i++) {
      // Very naive algorithm.
      // Set input rows from frameStart to frameEnd in the SelectivityVector
      // and evaluate the results.
      aggregate_->clear();

      // TODO : Check if we have to offset 1 here for correct accounting.
      rows_.resize(numPartitionRows_);
      rows_.setValidRange(frameStartsVector[i], frameEndsVector[i], true);
      rows_.updateBounds();

      aggregate_->addSingleGroupRawInput(
          singleGroupRow_, rows_, argVectors_, false);
      aggregate_->extractValues(&singleGroupRow_, 1, &resultVector);
      // TODO : Figure how to copy the result.
    }
  }

 private:
  // Aggregate function object required for this window function evaluation.
  // TODO : This simple implementation does a 1 aggregate per window function
  // mapping. But we can do multiple aggregates together. Change the
  // implementation for it.
  std::unique_ptr<exec::Aggregate> aggregate_;

  // This needs to be a copy of the argColumns passed to the function
  // as we need to retain them across function calls.
  std::vector<exec::RowColumn> argColumns_;
  // This is a vector of all the argument values that have to be
  // obtained from the partition and passed to the aggregate evaluation.
  std::vector<VectorPtr> argVectors_;
  vector_size_t numPartitionRows_;

  // This is a single aggregate row needed by the aggregate function for its
  // computation.
  char* singleGroupRow_;
  vector_size_t singleGroupRowSize_;

  // Used for per-row aggregate computations.
  SelectivityVector rows_;
};

} // namespace

void registerAggregateWindowFunction(const std::string& name) {
  auto aggregateFunctionSignatures = exec::getAggregateFunctionSignatures(name);
  if (aggregateFunctionSignatures.has_value()) {
    // This copy is needed to obtain a vector of the base FunctionSignaturePtr
    // from the AggregateFunctionSignaturePtr.
    std::vector<exec::FunctionSignaturePtr> signatures(
        aggregateFunctionSignatures.value().begin(),
        aggregateFunctionSignatures.value().end());

    exec::registerWindowFunction(
        name,
        std::move(signatures),
        [name](
            const std::vector<exec::RowColumn>& argColumns,
            const std::vector<TypePtr>& argTypes,
            const TypePtr& resultType,
            velox::memory::MemoryPool* pool)
            -> std::unique_ptr<exec::WindowFunction> {
          return std::make_unique<AggregateWindowFunction>(
              name, argColumns, argTypes, resultType, pool);
        });
  }
}
} // namespace facebook::velox::window

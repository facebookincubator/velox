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

#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/WindowFunction.h"

namespace facebook::velox::exec {

class Window : public Operator {
 public:
  Window(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  void close() override {
    Operator::close();
  }

 private:
  class Comparator {
   public:
    Comparator(
        const std::shared_ptr<const RowType>& outputType,
        const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
            sortingKeys,
        const std::vector<core::SortOrder>& sortingOrders,
        RowContainer* rowContainer);

    // Returns true if lhs < rhs, false otherwise.
    bool operator()(const char* lhs, const char* rhs) {
      if (lhs == rhs) {
        return false;
      }
      for (auto& key : keyInfo_) {
        if (auto result = rowContainer_->compare(
                lhs,
                rhs,
                key.first,
                {key.second.isNullsFirst(), key.second.isAscending(), false})) {
          return result < 0;
        }
      }
      return false;
    }

    // Returns true if lhs < decodeVectors[index], false otherwise.
    bool operator()(
        const char* lhs,
        const std::vector<DecodedVector>& decodedVectors,
        vector_size_t index) {
      for (auto& key : keyInfo_) {
        if (auto result = rowContainer_->compare(
                lhs,
                rowContainer_->columnAt(key.first),
                decodedVectors[key.first],
                index,
                {key.second.isNullsFirst(), key.second.isAscending(), false})) {
          return result < 0;
        }
      }
      return false;
    }

   private:
    std::vector<std::pair<ChannelIndex, core::SortOrder>> keyInfo_;
    RowContainer* rowContainer_;
  };

  bool finished_ = false;

  // As the inputs are added to Window operator, we use windowPartitionsQueue_
  // (a priority queue) to keep track of the pointers to rows stored in the
  // RowContainer (data_).
  // Once all inputs are available, we can separate into partitions the rows
  // with the same partition key in the sort order specified. This is easily
  // available using the windowPartitionsQueue.
  // This WindowPartition is sent to the Window function to compute its output
  // values.
  const int inputColumnsSize_;
  std::unique_ptr<RowContainer> data_;
  std::vector<DecodedVector> decodedInputVectors_;

  Comparator allKeysComparator_;
  Comparator partitionKeysComparator_;
  std::priority_queue<char*, std::vector<char*>, Comparator>
      windowPartitionsQueue_;
  std::vector<char*> rows_;

  std::vector<std::unique_ptr<exec::WindowFunction>> windowFunctions_;
};

} // namespace facebook::velox::exec

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

  bool isFinished() override {
    return finished_;
  }

  void close() override {
    Operator::close();
  }

 private:
  static const int32_t kBatchSizeInBytes{2 * 1024 * 1024};

  void initKeyInfo(
      const std::shared_ptr<const RowType>& type,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      std::vector<std::pair<ChannelIndex, core::SortOrder>>& keyInfo);

  bool finished_ = false;

  // As the inputs are added to Window operator, we store the rows in the
  // RowContainer (data_).
  // Once all inputs are available, we can separate the input rows into Window
  // partitions by sorting all rows with a combination of Window partitionKeys
  // followed by sortKeys.
  // This WindowFunction is invoked with these partition rows to compute its
  // output values.
  const int inputColumnsSize_;
  std::unique_ptr<RowContainer> data_;
  std::vector<DecodedVector> decodedInputVectors_;

  std::vector<std::pair<ChannelIndex, core::SortOrder>> allKeyInfo_;
  std::vector<std::pair<ChannelIndex, core::SortOrder>> partitionKeyInfo_;

  size_t numRows_ = 0;
  size_t q = 0;
  std::vector<char*> rows_;
  std::vector<char*> returningRows_;

  std::vector<std::unique_ptr<exec::WindowFunction>> windowFunctions_;
};

} // namespace facebook::velox::exec

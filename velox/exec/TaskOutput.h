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

#include <folly/Random.h>
#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {
namespace detail {
struct Split {
  IndexRange range;
  vector_size_t estimatedSize;
};
} // namespace detail

// In a distributed query engine data needs to be broadcast to workers. Either
// so that each worker has a copy of the data, e.g. a join where one side is
// small, or to distribute work arbitrarily to roughly balance the number of
// rows. TaskOutput operator is responsible for this process: it takes a stream
// of data that is destined for a single OutputBuffer to be broadcast from, and
// breaks the stream into a series of batches which are written to the
// OutputBuffer. This operator is also capable of re-ordering and dropping
// columns from its input.
class TaskOutput : public Operator {
 public:
  TaskOutput(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  void addInput(RowVectorPtr input) override;

  // Always returns nullptr. The action is to further process
  // unprocessed input. If all input has been processed, 'this' is in
  // a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  // Always true but the caller will check isBlocked before adding input, hence
  // the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    if (blockingReason_ != BlockingReason::kNotBlocked) {
      *future = std::move(future_);
      blockingReason_ = BlockingReason::kNotBlocked;
      return BlockingReason::kWaitForConsumer;
    }
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  void initializeOutput(const RowVectorPtr& input);

  std::unique_ptr<BatchVectorSerializer> serializer_;
  // Empty if column order in the output is exactly the same as in input.
  const std::vector<column_index_t> outputChannels_;
  const std::weak_ptr<exec::OutputBufferManager> bufferManager_;
  const std::function<void()> bufferReleaseFn_;
  const std::string taskId_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};
  RowVectorPtr output_;

  // Specifies ranges of the input vector that will be written as batches to the
  // output buffer.
  std::vector<detail::Split> splits_;
  vector_size_t splitIdx_{0};

  // Generator for varying target batch size. Randomly seeded at construction.
  folly::Random::DefaultGenerator rng_;

  // Reusable memory.
  Scratch scratch_;
};

} // namespace facebook::velox::exec

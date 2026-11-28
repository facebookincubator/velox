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

#include <deque>

#include "velox/exec/Operator.h"
#include "velox/exec/PythonCallStatNames.h"
#include "velox/exec/python/PythonWorker.h"

namespace facebook::velox::exec {

/// Evaluates the Python calls of a PythonCallNode in a worker process, one per
/// driver. Sends each input batch's argument columns to the worker and keeps
/// the batch until its results return, so the worker computes one batch while
/// the next is sent. A row whose call fails fails the query with the call's
/// error, or gets NULL for a call under TRY.
class PythonCall : public Operator {
 public:
  PythonCall(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::PythonCallNodePtr& planNode);

  void initialize() override;

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  // A column sent to the worker: an input column, or a constant when
  // 'constant' is set.
  struct WorkerColumn {
    // The input channel of an input column.
    column_index_t channel{0};
    // The constant of a constant column.
    core::ConstantTypedExprPtr constant;
    // The constant as a single-row vector, made in initialize().
    VectorPtr constantValue;
  };

  // A call's output column. Its index in 'callColumns_' is its index in
  // 'results_'.
  struct CallColumn {
    // Index of the call's function among the worker's output functions.
    size_t resultIndex;
    // The called function's own name, for error messages.
    std::string functionName;
    // Whether the call is under TRY.
    bool nullOnError;
    // Whether the result type contains a MAP, whose keys need checking.
    bool containsMap;
  };

  // Receives the results for the oldest pending batch into 'workerResults_'
  // if they arrived. Otherwise sets 'future' unless it is null.
  void tryReceive(ContinueFuture* future);

  // Returns the results of a call's column, with failed rows handled.
  VectorPtr callResults(const CallColumn& column, vector_size_t numRows);

  // Records the worker's usage and reserves its memory in the pool, so the
  // query's memory limit covers the worker.
  void accountForWorker();

  // The most input batches sent ahead of their results.
  const uint32_t maxPending_;
  // What the worker runs, built from the plan node; moved into the worker
  // when it starts.
  python::PythonWorker::Options workerOptions_;
  // The columns sent to the worker, in the order sent.
  std::vector<WorkerColumn> workerColumns_;
  // The type of the batches sent to the worker: the argument columns.
  RowTypePtr workerInputType_;
  // The calls' output columns, in the order of 'results_'.
  std::vector<CallColumn> callColumns_;
  // Started on the first batch; stopped in close().
  std::unique_ptr<python::PythonWorker> worker_;
  // Input batches sent to the worker and awaiting results, oldest first: at
  // most 'maxPending_'.
  std::deque<RowVectorPtr> pending_;
  // The worker's results for the oldest pending batch, once received.
  RowVectorPtr workerResults_;
  // Whether the unpickling time is in the runtime stats yet.
  bool unpickleRecorded_{false};
  // A pool that holds the worker's memory as a reservation, outside the
  // operator's own allocations.
  std::shared_ptr<memory::MemoryPool> workerPool_;
  // The bytes reserved in 'workerPool_'.
  uint64_t reservedWorkerBytes_{0};
};

} // namespace facebook::velox::exec

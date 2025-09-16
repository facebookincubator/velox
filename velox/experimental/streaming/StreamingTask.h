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
#include "velox/exec/Task.h"
#include "velox/experimental/streaming/StreamingOperator.h"
#include "velox/experimental/streaming/StreamElement.h"

namespace facebook::velox::streaming {

/**
 * StreamingTask is used to support streaming engines such as Flink.
 * It only runs in a streaming mode, which call next and get a result once,
 * also, it needs to handle state operations  
 * Make it based on Task to reuse context releated classes.
 */
class StreamingTask : public exec::Task {
 public:

  /// Creates a streaming task to execute a plan fragment, but doesn't start execution
  /// until StreamingTask::next() method is called.
  /// @param taskId Unique task identifier.
  /// @param planFragment Plan fragment.
  /// @param queryCtx Query context containing MemoryPool and MemoryAllocator
  /// instances to use for memory allocations during execution, executor to
  /// schedule operators on, and session properties.
  /// execution fails.
  static std::shared_ptr<StreamingTask> create(
      const std::string& taskId,
      core::PlanFragment planFragment,
      std::shared_ptr<core::QueryCtx> queryCtx);

  ~StreamingTask();

  /// Single-threaded execution API. Runs the query and returns results one
  /// batch at a time. Returns nullptr and retCode 1 if query evaluation is finished and no
  /// more data will be produced, return nullptt and retCode 0 is no data produced for this batch.
  ///  Throws an exception if query execution failed.
  ///
  /// This API is available for streaming plans such as Flink.
  ///
  /// The caller is required to add all the necessary splits, and signal
  /// no-more-splits before calling 'next' for the first time.
  StreamElementPtr next(int32_t& retCode);

  void notifyWatermark(long watermark, int index);

  void snapshotState(long checkpointId);

  void notifyCheckpointComplete(long checkpointId);

  void notifyCheckpointAborted(long checkpointId);

  void init();

  // The task is finished, close all operators and reset driver
  void finish();

  void addOutput(StreamElementPtr element);

 private:
 
  StreamingTask(
      const std::string& taskId,
      core::PlanFragment planFragment,
      std::shared_ptr<core::QueryCtx> queryCtx);

  void initOperators();

  void initStateBackend();

  StreamElementPtr popOutput();

  std::unique_ptr<StreamingOperator> operatorChain_;

  // A task may have multi outputs once run,
  // store them and return one by one.
  std::list<StreamElementPtr> pendings_;

  // hold the driver only to avoid it be released.
  std::shared_ptr<exec::Driver> driver;
};

} // namespace facebook::velox::streaming

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
#include "velox/experimental/streaming/CombinedWatermarkStatus.h"

namespace facebook::velox::streaming {

/**
 * This class is to support streaming engines. Unlike batch jobs,
 * operators in streaming jobs may have several outputs,
 * so the pull mode of velox that downstream call its upstream to get output
 * will not fit for it. We need to use push mode that sources
 * an output, and then push to its downstreams and so on.
 *
 * StreamingOperator is designed to hold the state and an velox operator,
 * it gives the state and input to velox operator to calculate the result.
 * For example, Streaming Top1 node gets the previous result from state,
 * and then call TopN operator to get new result, and at last update the state.
 */
class StreamingOperator {
 public:
  StreamingOperator(
      std::unique_ptr<exec::Operator> op,
      std::vector<std::unique_ptr<StreamingOperator>> targets)
      : operator_(std::move(op)),
        targets_(std::move(targets)) {
    isSink_ = operator_->operatorType() == "TableWrite";
  }

  virtual void initialize();

  virtual bool isFinished();

  virtual void addInput(RowVectorPtr input);

  virtual void getOutput();

  bool sourceEmpty();

  virtual void close();

  void processWatermark(long timestamp, int index);

  void snapshotState(long checkpointId);

  void notifyCheckpointComplete(long checkpointId);

  void notifyCheckpointAborted(long checkpointId);

  virtual std::string name() const {
    return operator_->operatorType();
  }

 protected:
  void pushOutput(RowVectorPtr output);
  void pushWatermark(long timestamp, int index);
  virtual void processWatermarkInternal(long timestamp) {}

  virtual int numInputs() const {
    return 1;
  }

  std::unique_ptr<exec::Operator>& op() {
    return operator_;
  }

 private:
  bool isSink() {
    return isSink_;
  }

  std::unique_ptr<exec::Operator> operator_;
  std::vector<std::unique_ptr<StreamingOperator>> targets_;
  bool isSink_;
  bool sourceEmpty_ = true;
  std::shared_ptr<CombinedWatermarkStatus> combinedWatermarkStatus_;
};

using StreamingOperatorPtr = std::unique_ptr<StreamingOperator>;

} // namespace facebook::velox::streaming

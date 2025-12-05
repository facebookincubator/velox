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
#include "velox/experimental/streaming/StreamingOperator.h"
#include "velox/experimental/streaming/StreamingTask.h"
#include "velox/experimental/streaming/StreamElement.h"

#include <iostream>

namespace facebook::velox::streaming {

void StreamingOperator::initialize() {
  operator_->initialize();
  for (auto& target : targets_) {
    target->initialize();
  }
  combinedWatermarkStatus_ = std::make_shared<CombinedWatermarkStatus>(numInputs());
}
  
bool StreamingOperator::isFinished() {
  return operator_->isFinished();
}

void StreamingOperator::addInput(RowVectorPtr input) {
  operator_->traceInput(input);
  operator_->addInput(std::move(input));
}

bool StreamingOperator::sourceEmpty() {
  return sourceEmpty_;
}

void StreamingOperator::close() {
  operator_->close();
  for (auto& target : targets_) {
    target->close();
  }
  operator_.reset();
  targets_.clear();
}

void StreamingOperator::getOutput() {
  sourceEmpty_ = true;
  auto intermediateResult = operator_->getOutput();
  if (!intermediateResult) {
    return;
  }
  sourceEmpty_ = false;
  pushOutput(std::move(intermediateResult));
}

void StreamingOperator::pushOutput(RowVectorPtr output) {
  if (targets_.empty()) {
    auto outNodeId = operator_->planNodeId();
    auto task = std::static_pointer_cast<StreamingTask>(operator_->operatorCtx()->driverCtx()->task);
    task->addOutput(std::make_shared<StreamRecord>(outNodeId, std::move(output)));
    return;
  }
  for (int i = 0; i < targets_.size(); i++) {
    targets_[i]->addInput(output);
    targets_[i]->getOutput();
  }
}

void StreamingOperator::pushWatermark(long timestamp, int index) {
  // watermark need not to be written to sink.
  if (isSink())
    return;
  if (targets_.empty()) {
    auto outNodeId = operator_->planNodeId();
    auto task = std::static_pointer_cast<StreamingTask>(operator_->operatorCtx()->driverCtx()->task);
    task->addOutput(std::make_shared<Watermark>(outNodeId, timestamp));
    return;
  }
  for (int i = 0; i < targets_.size(); i++) {
    targets_[i]->processWatermark(timestamp, index);
  }
}

void StreamingOperator::processWatermark(long timestamp, int index) {
  if (combinedWatermarkStatus_->updateWatermark(index - 1, timestamp)) {
    long combinedWatermark = combinedWatermarkStatus_->getCombinedWatermark();
    // If the watermark is updated, we need to notify the operator,
    // it may need to do something, e.g. advance the timer service or tirgger the window.
    processWatermarkInternal(combinedWatermark);
    pushWatermark(combinedWatermark, 1);
  }
}

void StreamingOperator::snapshotState(long checkpointId) {
  // TODO: not impltemented yet
}

void StreamingOperator::notifyCheckpointComplete(long checkpointId) {
  // TODO: not impltemented yet
}

void StreamingOperator::notifyCheckpointAborted(long checkpointId) {
  // TODO: not impltemented yet
}

} // namespace facebook::velox::streaming

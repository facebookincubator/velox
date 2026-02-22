/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/StreamingEnforceDistinct.h"

#include "velox/exec/OperatorType.h"

namespace facebook::velox::exec {

namespace {
// Compares two rows in the same or different vectors and returns true if they
// match in all key columns.
bool equalKeys(
    const std::vector<column_index_t>& keyChannels,
    const RowVectorPtr& batch,
    vector_size_t index,
    const RowVectorPtr& otherBatch,
    vector_size_t otherIndex) {
  for (auto channel : keyChannels) {
    if (!batch->childAt(channel)->equalValueAt(
            otherBatch->childAt(channel).get(), index, otherIndex)) {
      return false;
    }
  }
  return true;
}
} // namespace

StreamingEnforceDistinct::StreamingEnforceDistinct(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::EnforceDistinctNode>& planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          OperatorType::kStreamingEnforceDistinct),
      inputType_{planNode->sources()[0]->outputType()},
      keyChannels_{toChannels(
          inputType_,
          std::vector<core::TypedExprPtr>{
              planNode->distinctKeys().begin(),
              planNode->distinctKeys().end()})},
      errorMessage_{planNode->errorMessage()} {
  for (auto i = 0; i < inputType_->size(); ++i) {
    identityProjections_.emplace_back(i, i);
  }
}

void StreamingEnforceDistinct::addInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  // Check first row against previous batch's last row.
  if (prevKeyValues_ != nullptr &&
      equalKeys(keyChannels_, input, 0, prevKeyValues_, 0)) {
    VELOX_USER_FAIL("{}", errorMessage_);
  }

  // Check consecutive rows within this batch.
  for (vector_size_t i = 1; i < input->size(); ++i) {
    if (equalKeys(keyChannels_, input, i, input, i - 1)) {
      VELOX_USER_FAIL("{}", errorMessage_);
    }
  }

  // Save key values from the last row for comparison with next batch.

  if (prevKeyValues_ == nullptr) {
    std::vector<VectorPtr> keyVectors(inputType_->size());
    for (auto channel : keyChannels_) {
      keyVectors[channel] =
          BaseVector::create(inputType_->childAt(channel), 1, pool());
    }
    prevKeyValues_ = std::make_shared<RowVector>(
        pool(), inputType_, nullptr, 1, std::move(keyVectors));
  }

  const auto lastRow = input->size() - 1;
  for (auto channel : keyChannels_) {
    prevKeyValues_->childAt(channel)->copy(
        input->childAt(channel).get(), 0, lastRow, 1);
  }

  input_ = std::move(input);
}

RowVectorPtr StreamingEnforceDistinct::getOutput() {
  if (isFinished() || !input_) {
    return nullptr;
  }

  auto output = fillOutput(input_->size(), nullptr);
  input_ = nullptr;
  return output;
}

bool StreamingEnforceDistinct::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

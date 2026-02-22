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

#include "velox/exec/EnforceDistinct.h"

#include "velox/exec/OperatorType.h"

namespace facebook::velox::exec {

EnforceDistinct::EnforceDistinct(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::EnforceDistinctNode>& planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          OperatorType::kEnforceDistinct),
      errorMessage_{planNode->errorMessage()} {
  const auto& inputType = planNode->sources()[0]->outputType();

  for (auto i = 0; i < inputType->size(); ++i) {
    identityProjections_.emplace_back(i, i);
  }

  groupingSet_ = GroupingSet::createForDistinct(
      inputType,
      createVectorHashers(inputType, planNode->distinctKeys()),
      toChannels(
          inputType,
          std::vector<core::TypedExprPtr>{
              planNode->preGroupedKeys().begin(),
              planNode->preGroupedKeys().end()}),
      operatorCtx_.get(),
      &nonReclaimableSection_);
}

void EnforceDistinct::addInput(RowVectorPtr input) {
  groupingSet_->addInput(input, /*mayPushdown=*/false);

  const auto& newGroups = groupingSet_->hashLookup().newGroups;
  if (newGroups.size() != input->size()) {
    VELOX_USER_FAIL("{}", errorMessage_);
  }

  input_ = std::move(input);
}

RowVectorPtr EnforceDistinct::getOutput() {
  if (isFinished() || !input_) {
    return nullptr;
  }

  auto output = fillOutput(input_->size(), nullptr);

  input_ = nullptr;

  return output;
}

bool EnforceDistinct::isFinished() {
  return noMoreInput_ && !input_;
}

} // namespace facebook::velox::exec

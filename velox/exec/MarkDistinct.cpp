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

#include "MarkDistinct.h"
#include "velox/common/base/Range.h"
#include "velox/vector/FlatVector.h"

#include <algorithm>
#include <utility>

namespace facebook::velox::exec {

MarkDistinct::MarkDistinct(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MarkDistinctNode>& planNode,
    std::vector<uint32_t> keyChannels)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "MarkDistinct"),
      keyChannels_(std::move(keyChannels)) {
  results_.resize(1);
  results_[0] = BaseVector::create(BOOLEAN(), 0, operatorCtx_->pool());
  const auto numColumns = planNode->outputType()->size();
  identityProjections_.reserve(numColumns - 1);
  for (uint32_t i = 0; i < numColumns - 1; ++i) {
    identityProjections_.emplace_back(i, i);
  }
  resultProjections_.emplace_back(0, numColumns - 1);
  decodedVectors_.resize(keyChannels_.size());
  columnExists_.resize(keyChannels_.size());
}

void MarkDistinct::addInput(RowVectorPtr input) {
  allRows_.resize(input->size());
  // Decode the channels.
  std::fill(columnExists_.begin(), columnExists_.end(), false);
  for (auto idx = 0; idx < keyChannels_.size(); ++idx) {
    if (input->childAt(keyChannels_[idx])) {
      columnExists_[idx] = true;
      decodedVectors_[idx].decode(*input->children()[idx], allRows_);
    }
  }

  VectorPtr& result = results_[0];
  result->resize(input->size());
  auto resultBits =
      result->as<FlatVector<bool>>()->mutableRawValues<uint64_t>();

  for (int row = 0; row < input->size(); ++row) {
    uint64_t hash = BaseVector::kNullHash;
    bool isFirst = true;

    if (!input->isNullAt(row)) {
      for (auto idx = 0; idx < columnExists_.size(); ++idx) {
        if (columnExists_[idx]) {
          auto& decodedChild = decodedVectors_[idx];
          auto cellHash =
              decodedChild.base()->hashValueAt(decodedChild.indices()[row]);
          hash = isFirst ? cellHash : bits::hashMix(hash, cellHash);
          isFirst = false;
        }
      }
    }

    if (hashSet_.find(hash) == hashSet_.end()) {
      hashSet_.emplace(hash);
      bits::setBit(resultBits, row, true);
    } else {
      bits::setBit(resultBits, row, false);
    }
  }

  input_ = std::move(input);
}

RowVectorPtr MarkDistinct::getOutput() {
  if (input_ == nullptr || input_->size() == 0) {
    return nullptr;
  }
  auto output = fillOutput(input_->size(), nullptr);
  input_ = nullptr;
  return output;
}

bool MarkDistinct::isFinished() {
  return noMoreInput_ && input_ == nullptr;
}

} // namespace facebook::velox::exec
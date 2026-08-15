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
#include "velox/exec/window/SingleRowValues.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::window {

SingleRowValues::SingleRowValues(
    std::vector<column_index_t> channels,
    memory::MemoryPool* pool)
    : channels_(std::move(channels)), pool_(pool) {
  VELOX_CHECK_NOT_NULL(pool_);
}

void SingleRowValues::capture(const RowVectorPtr& input, vector_size_t row) {
  VELOX_CHECK_NOT_NULL(input);

  values_.clear();
  values_.reserve(channels_.size());
  for (const auto channel : channels_) {
    auto value = BaseVector::create(input->childAt(channel)->type(), 1, pool_);
    value->copy(input->childAt(channel).get(), 0, row, 1);
    values_.push_back(std::move(value));
  }
  hasValue_ = true;
}

void SingleRowValues::reset() {
  hasValue_ = false;
  values_.clear();
}

bool SingleRowValues::equals(const RowVectorPtr& input, vector_size_t row)
    const {
  VELOX_CHECK(hasValue_);
  for (auto i = 0; i < channels_.size(); ++i) {
    const auto channel = channels_[i];
    if (!values_[i]->equalValueAt(input->childAt(channel).get(), 0, row)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::exec::window

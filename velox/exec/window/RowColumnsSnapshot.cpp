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
#include "velox/exec/window/RowColumnsSnapshot.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::window {

void RowColumnsSnapshot::capture(
    const RowVectorPtr& input,
    vector_size_t row,
    const std::vector<column_index_t>& channels,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(pool);

  clear();
  valid_ = true;
  channels_.reserve(channels.size());
  values_.reserve(channels.size());
  for (const auto channel : channels) {
    auto value = BaseVector::create(input->childAt(channel)->type(), 1, pool);
    value->copy(input->childAt(channel).get(), 0, row, 1);
    channels_.push_back(channel);
    values_.push_back(std::move(value));
  }
}

void RowColumnsSnapshot::clear() {
  valid_ = false;
  channels_.clear();
  values_.clear();
}

const VectorPtr& RowColumnsSnapshot::valueAt(column_index_t channel) const {
  for (auto i = 0; i < channels_.size(); ++i) {
    if (channels_[i] == channel) {
      return values_[i];
    }
  }
  VELOX_FAIL("Missing captured key channel: {}", channel);
}

bool RowColumnsSnapshot::rowsEqual(
    const RowVectorPtr& input,
    vector_size_t row,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const std::vector<column_index_t>& inputChannels) const {
  VELOX_CHECK(isValid());
  for (const auto& key : keyInfo) {
    const auto inputColumn = inputChannels[key.first];
    if (!valueAt(inputColumn)
             ->equalValueAt(input->childAt(inputColumn).get(), 0, row)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::exec::window

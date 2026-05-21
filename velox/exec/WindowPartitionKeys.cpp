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
#include "velox/exec/WindowPartitionKeys.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>

namespace facebook::velox::exec::detail {

void WindowPartitionKeyRowSnapshot::capture(
    const RowVectorPtr& input,
    vector_size_t row,
    const std::vector<column_index_t>& keyChannels,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(pool);

  clear();
  valid_ = true;
  channels_.reserve(keyChannels.size());
  values_.reserve(keyChannels.size());
  for (const auto channel : keyChannels) {
    auto value = BaseVector::create(input->childAt(channel)->type(), 1, pool);
    value->copy(input->childAt(channel).get(), 0, row, 1);
    channels_.push_back(channel);
    values_.push_back(std::move(value));
  }
}

void WindowPartitionKeyRowSnapshot::capture(
    const WindowPartitionRowReference& row,
    const std::vector<column_index_t>& keyChannels,
    memory::MemoryPool* pool) {
  capture(row.input, row.row, keyChannels, pool);
}

void WindowPartitionKeyRowSnapshot::clear() {
  valid_ = false;
  channels_.clear();
  values_.clear();
}

const VectorPtr& WindowPartitionKeyRowSnapshot::valueAt(
    column_index_t channel) const {
  for (auto i = 0; i < channels_.size(); ++i) {
    if (channels_[i] == channel) {
      return values_[i];
    }
  }
  VELOX_FAIL("Missing previous-row key channel: {}", channel);
}

bool WindowPartitionKeyRowSnapshot::rowsEqual(
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

bool WindowPartitionKeyRowSnapshot::rowsEqual(
    const WindowPartitionRowReference& row,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const std::vector<column_index_t>& inputChannels) const {
  return rowsEqual(row.input, row.row, keyInfo, inputChannels);
}

std::vector<column_index_t> WindowPartitionKeyChannels::create(
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(keyInfo.size());
  for (const auto& key : keyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  return channels;
}

std::vector<column_index_t> WindowPartitionKeyChannels::create(
    const std::vector<std::pair<column_index_t, core::SortOrder>>& firstKeyInfo,
    const std::vector<std::pair<column_index_t, core::SortOrder>>&
        secondKeyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(firstKeyInfo.size() + secondKeyInfo.size());
  for (const auto& key : firstKeyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  for (const auto& key : secondKeyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  return channels;
}

void WindowPartitionKeyChannels::appendUnique(
    std::vector<column_index_t>& channels,
    column_index_t channel) {
  if (std::find(channels.begin(), channels.end(), channel) == channels.end()) {
    channels.push_back(channel);
  }
}

} // namespace facebook::velox::exec::detail

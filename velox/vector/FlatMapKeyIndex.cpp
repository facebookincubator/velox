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

#include "velox/vector/FlatMapKeyIndex.h"

namespace facebook::velox::detail {

FlatMapKeyIndex::FlatMapKeyIndex(const BaseVector& distinctKeys) {
  for (vector_size_t i = 0; i < distinctKeys.size(); ++i) {
    add(distinctKeys.hashValueAt(i));
  }
}

void FlatMapKeyIndex::add(uint64_t hash) {
  const auto channel = static_cast<column_index_t>(keyToChannel_.size());
  keyToChannel_.insert({hash, channel});
}

} // namespace facebook::velox::detail

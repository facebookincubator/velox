/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/exec/VectorSearchJoinTable.h"

namespace facebook::velox::exec {

VectorSearchJoinTable::VectorSearchJoinTable(
    std::unique_ptr<BaseHashTable> hashTable,
    std::vector<std::unique_ptr<RowContainer>> rows,
    int32_t dimension)
    : hashTable_{std::move(hashTable)},
      ownedRows_{std::move(rows)},
      dimension_{dimension} {
  VELOX_CHECK(hashTable_ == nullptr || ownedRows_.empty());
  if (hashTable_ != nullptr) {
    rows_ = hashTable_->allRows();
  }
  for (const auto& container : ownedRows_) {
    rows_.push_back(container.get());
  }
}

} // namespace facebook::velox::exec

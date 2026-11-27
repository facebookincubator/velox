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

#include "velox/dwio/nimble/selective/FlatMapKeyNode.h"

namespace facebook::nimble {

bool FlatMapKeyNodePool::take(uint32_t inMapStream, FlatMapKeyNodeBase& node) {
  auto it = nodes_.find(inMapStream);
  if (it == nodes_.end()) {
    return false;
  }
  node = std::move(it->second);
  nodes_.erase(it);
  return true;
}

void FlatMapKeyNodePool::put(uint32_t inMapStream, FlatMapKeyNodeBase& node) {
  NIMBLE_CHECK_NOT_NULL(node.reader);
  NIMBLE_CHECK(
      node.reader->rebindable(), "Keeping a key node that cannot rebind.");
  auto [it, inserted] = nodes_.try_emplace(inMapStream);
  if (!inserted) {
    return;
  }
  node.reader->unbind();
  if (node.inMap != nullptr) {
    node.inMap->unbindStream();
  }
  it->second = std::move(node);
}

} // namespace facebook::nimble

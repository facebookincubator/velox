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

#include "velox/buffer/BufferPool.h"

namespace facebook::velox {

BufferPtr BufferPool::get(uint64_t minBytes) {
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i]->capacity() >= minBytes) {
      auto result = std::move(buffers_[i]);
      buffers_[i] = std::move(buffers_.back());
      buffers_.pop_back();
      return result;
    }
  }
  return nullptr;
}

BufferPtr BufferPool::get() {
  if (buffers_.empty()) {
    return nullptr;
  }
  auto result = std::move(buffers_.back());
  buffers_.pop_back();
  return result;
}

void BufferPool::release(BufferPtr buffer) {
  if (buffer != nullptr && buffers_.size() < kMaxCached) {
    buffers_.push_back(std::move(buffer));
  }
}

size_t BufferPool::size() const {
  return buffers_.size();
}

} // namespace facebook::velox

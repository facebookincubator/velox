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
#pragma once
#include "velox/common/memory/MemoryPool.h"

namespace facebook::velox {

struct ByteRange;

/// An abstract class that holds memory for serialized vector content. A single
/// repartitioning target is one use case: The bytes held are released as a unit
/// when the destination acknowledges receipt. Another use case is a hash table
/// partition that holds complex types as serialized rows.
class StreamArena {
 public:
  explicit StreamArena(memory::MemoryPool* pool) : pool_(pool) {}
  virtual ~StreamArena() = default;

  virtual void
  newRange(int32_t bytes, ByteRange* lastRange, ByteRange* range) = 0;

  virtual void
  newTinyRange(int32_t bytes, ByteRange* lastRange, ByteRange* range) = 0;

  virtual void clear() = 0;

  virtual size_t size() const = 0;

  memory::MemoryPool* pool() const {
    return pool_;
  }

 protected:
  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox

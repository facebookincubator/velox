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

#include <velox/vector/ComplexVector.h>

#include "velox4j/query/Query.h"

namespace facebook::velox4j {

/// An up-iterator is the opposite of down-iterator. It transmits data
/// that is output from Velox pipeline from C++ to Java.
class UpIterator {
 public:
  enum class State { AVAILABLE = 0, BLOCKED = 1, FINISHED = 2 };

  UpIterator() = default;

  // Delete copy/move CTORs.
  UpIterator(UpIterator&&) = delete;
  UpIterator(const UpIterator&) = delete;
  UpIterator& operator=(const UpIterator&) = delete;
  UpIterator& operator=(UpIterator&&) = delete;

  virtual ~UpIterator() = default;

  // Gets the next state.
  virtual State advance() = 0;

  /// Called once `advance` returns `BLOCKED` state to wait until
  /// the state gets refreshed, either by the next row-vector
  /// is ready for reading or by end of stream.
  virtual void wait() = 0;

  /// Called to close the iterator.
  virtual facebook::velox::RowVectorPtr get() = 0;
};
} // namespace facebook::velox4j

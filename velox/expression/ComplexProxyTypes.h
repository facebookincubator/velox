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
#include <iterator>
#include <optional>
#include "velox/common/base/Exceptions.h"
#include "velox/core/CoreTypeSystem.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::exec {

template <typename T>
struct VectorReader;

template <typename T, typename B>
struct VectorWriter;

// The object passed to the simple function interface that represent a single
// array entry.
template <typename V>
class ArrayProxy {
  using child_writer_t = VectorWriter<V, void>;
  using element_t = typename child_writer_t::exec_out_t;

 public:
  ArrayProxy<V>(const ArrayProxy<V>&) = delete;

  element_t& addItem() {
    commitMostRecentChildItem();
    prepareNextChildItem();
    needCommit_ = true;
    return childWriter_->current();
  }

  typename std::enable_if<
      CppToType<V>::isPrimitiveType && !std::is_same<Varchar, V>::value>::type
  push_back(typename CppToType<V>::NativeType value) {
    auto& ref = addItem();
    ref = value;
  }

  typename std::enable_if<
      CppToType<V>::isPrimitiveType && !std::is_same<Varchar, V>::value>::type
  push_back(std::nullopt_t) {
    addNull();
  }

  typename std::enable_if<
      CppToType<V>::isPrimitiveType && !std::is_same<Varchar, V>::value>::type
  push_back(const std::optional<typename CppToType<V>::NativeType>& value) {
    if (value) {
      push_back(*value);
    } else {
      addNull();
    }
  }

  void addNull() {
    prepareNextChildItem();
    childWriter_->commitNull();
    needCommit_ = false;
  }

  void prepareNextChildItem() {
    // TODO: make sure its growing exponentially or manage size and avoid keep
    // calling it.
    childWriter_->ensureSize(valuesOffset_ + length_ + 1);
    childWriter_->setOffset(valuesOffset_ + length_);
    length_++;
  }

  void commitMostRecentChildItem() {
    if (LIKELY(needCommit_)) {
      childWriter_->commit(true);
    }
  }

  // Should be called by the user(VectorWriter) when writing is done to commit
  // last item if needed.
  void finalize() {
    commitMostRecentChildItem();
  }

  vector_size_t getLength() {
    return length_;
  }

 private:
  // Prepare the proxy for a new element.
  // TODO: childWriter does not change.
  void setUp(child_writer_t* childWriter, vector_size_t valuesOffset) {
    childWriter_ = childWriter;
    valuesOffset_ = valuesOffset;
    length_ = 0;
    needCommit_ = false;
  }

  ArrayProxy<V>() {}

  template <typename A, typename B>
  friend class VectorWriter;

  // Pointer to child vector writer.
  child_writer_t* childWriter_;

  // Indicate if commit need to be called on the childWriter_ before addign a
  // new element or when finalize is called.
  bool needCommit_ = false;

  // Length of the array.
  vector_size_t length_;

  // The offset within the child vector at which this array starts.
  vector_size_t valuesOffset_;
};
} // namespace facebook::velox::exec

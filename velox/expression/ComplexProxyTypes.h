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
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {

template <typename T>
struct VectorReader;

template <typename T, typename B>
struct VectorWriter;

// Lightweight object that can be used as a proxy for array primitive elements.
// It can writes both value and nullability, returned by operator[].
template <typename T>
struct PrimitiveNullableWriterProxy {
  using vector_t = typename TypeToFlatVector<T>::type;
  using element_t = typename CppToType<T>::NativeType;

  PrimitiveNullableWriterProxy(vector_t* flatVector, vector_size_t index)
      : flatVector_(flatVector), index_(index) {}
  void operator=(std::nullopt_t) {
    flatVector_->setNull(index_, true);
  }

  void operator=(element_t value) {
    flatVector_->set(index_, value);
  }

  void operator=(const std::optional<element_t>& value) {
    if (value.has_value()) {
      flatVector_->set(index_, value);
    } else {
      flatVector_->setNull(index_, true);
    }
  }

 private:
  vector_t* flatVector_;
  vector_size_t index_;
};

// The object passed to the simple function interface that represent a single
// array entry.
// 1. General Interface:
// - add_item()  : Add not null item and return proxy to the value to be
// written.
// - add_null()  : Add null item.
// - size()     : return the size of the array.

// 2. Special std::like interfaces when V is primitive:
// - resize(n): create x items, nullability not decided yet.
// - operator[](index): returns PrimitiveNullableWriterProxy which can be used
// to write value or nullability at index.
// - push_back(std::optional<v> value) : increase size by 1, adding a value
// or null.
template <typename V>
class ArrayProxy {
  using child_writer_t = VectorWriter<V, void>;
  using element_t = typename child_writer_t::exec_out_t;

 public:
  ArrayProxy<V>(const ArrayProxy<V>&) = delete;

  // String and bool not yet supported, this probably wont work for string.
  static bool constexpr provide_std_interface = CppToType<V>::isPrimitiveType &&
      !std::is_same<Varchar, V>::value && !std::is_same<bool, V>::value;

  // Add a new not null item to the array, increasing its size by 1.
  FOLLY_ALWAYS_INLINE element_t& add_item() {
    commitMostRecentChildItem();
    auto index = valuesOffset_ + length_;
    reserve(index + 1);
    length_++;

    if constexpr (!provide_std_interface) {
      childWriter_->setOffset(index);
      needCommit_ = true;
      return childWriter_->current();
    } else {
      childWriter_->vector().setNull(index, false);
      return childWriter_->data_[index];
    }
  }

  // Add a new null item to the array.
  FOLLY_ALWAYS_INLINE void add_null() {
    commitMostRecentChildItem();
    reserve(valuesOffset_ + length_ + 1);
    childWriter_->vector().setNull(valuesOffset_ + length_, true);
    length_++;
    // Note: no need to commit the null item.
  }

  vector_size_t size() {
    return length_;
  }

  // Functions below provides an std::like interface, and are enabled only when
  // the array element is primitive that is not string or bool.

  // Note: size is with respect to the current size of this array being written.
  FOLLY_ALWAYS_INLINE typename std::enable_if<provide_std_interface>::type
  resize(vector_size_t size) {
    commitMostRecentChildItem();
    reserve(size);
    length_ = size;
  }

  typename std::enable_if<provide_std_interface>::type FOLLY_ALWAYS_INLINE
  push_back(element_t value) {
    auto& item = add_item();
    item = value;
  }

  typename std::enable_if<provide_std_interface>::type FOLLY_ALWAYS_INLINE
  push_back(std::nullopt_t) {
    add_null();
  }

  typename std::enable_if<provide_std_interface>::type FOLLY_ALWAYS_INLINE
  push_back(const std::optional<element_t>& value) {
    if (value.has_value()) {
      push_back(*value);
    } else {
      add_null();
    }
  }

  typename std::
      enable_if<provide_std_interface, PrimitiveNullableWriterProxy<V>>::type
      operator[](vector_size_t index_) {
    return PrimitiveNullableWriterProxy<V>{&childWriter_->vector(),
                                           valuesOffset_ + index_};
  }

 private:
  ArrayProxy<V>() {}

  // Note: size is with respect to the current size of this array being written.
  FOLLY_ALWAYS_INLINE void reserve(vector_size_t size) {
    if (UNLIKELY(size > capacity_)) {
      while (capacity_ < size) {
        capacity_ = 2 * capacity_ + 1;
      }
      childWriter_->ensureSize(valuesOffset_ + capacity_);
    }
  }

  FOLLY_ALWAYS_INLINE void commitMostRecentChildItem() {
    if constexpr (!provide_std_interface) {
      if (needCommit_) {
        childWriter_->commit(true);
        needCommit_ = false;
      }
    }
  }

  // Should be called by the user (VectorWriter) when writing is done to commit
  // last item if needed.
  void finalize() {
    commitMostRecentChildItem();
    // Downsize to the actual size used in the underlying vector.
    // Some writer's logic depend on the previous size to append data.
    childWriter_->vector().resize(valuesOffset_ + length_);
  }

  // Prepare the proxy for a new element.
  // TODO: childWriter does not change.
  FOLLY_ALWAYS_INLINE void setUp(
      child_writer_t* childWriter,
      vector_size_t valuesOffset) {
    childWriter_ = childWriter;
    valuesOffset_ = valuesOffset;
    length_ = 0;
    capacity_ = 0;
    needCommit_ = false;
  }

  // Pointer to child vector writer.
  child_writer_t* childWriter_;

  // Indicate if commit need to be called on the childWriter_ before adding a
  // new element or when finalize is called.
  bool needCommit_ = false;

  // Length of the array.
  vector_size_t length_;

  // The offset within the child vector at which this array starts.
  vector_size_t valuesOffset_;

  // Virtual capacity for the current array.
  // childWriter guaranteed to be safely writable at indices [valuesOffset_,
  // valuesOffset_ + capacity_].
  vector_size_t capacity_ = 0;

  template <typename A, typename B>
  friend class VectorWriter;
};
} // namespace facebook::velox::exec

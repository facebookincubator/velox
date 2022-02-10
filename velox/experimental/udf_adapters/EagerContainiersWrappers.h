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
#include <optional>
#include <vector>
// Provides utilities that wrap std::vector and F14FastMap with Velox UDF writer
// interface. This allows running VLEOX udf with such inputs in the
// UDFBackwardAdapter.

namespace facebook::velox::utils {

template <typename T>
class VectorWrapper;

// Only map, vector and row needs wrapping in
// VectorWrapper, MapWrapper and RowWrapper to expose Velox interface.
template <typename T>
struct WrapperTypeResolver {
  using type = T;
  static constexpr bool isWrapped = false;
};

template <typename T>
struct WrapperTypeResolver<std::vector<T>> {
  using type = VectorWrapper<T>;
  static constexpr bool isWrapped = true;
  using innerType = std::vector<T>;
};

template <typename T>
class VectorWrapper {
 public:
  // Implicit constructor.
  VectorWrapper(std::vector<T>& vector) : vector_(vector) {}

  typename WrapperTypeResolver<T>::type& add_item() {
    vector_.push_back(T());
    auto& item = vector_.back();
    if constexpr (WrapperTypeResolver<T>::isWrapped) {
      return VectorWrapper<T>(item);
    } else {
      return item;
    }
  }

  // enabled only if T is primitive.
  void push_back(T value) {
    static_assert(!WrapperTypeResolver<T>::isWrapped && "T is not primitive");
    vector_.push_back(value);
  }

  void resize(size_t size) {
    vector_.resize(size);
  }

  T& operator[](size_t index) {
    static_assert(!WrapperTypeResolver<T>::isWrapped && "T is not primitive");
    return vector_[index];
  }

  std::vector<T>& get() {
    return vector_;
  }

  // When passed as input to another function, it is extracted trivially.
  operator const std::vector<T>&() {
    return vector_;
  }

 private:
  std::vector<T>& vector_;
};
}; // namespace facebook::velox::utils
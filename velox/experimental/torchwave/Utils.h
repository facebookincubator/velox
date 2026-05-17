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

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

/// Reusable pool of unique_ptr<T>. Thread-safe. Returns an existing item if
/// available, otherwise creates one via the make function passed at
/// construction.
template <typename T>
class Pool {
 public:
  using MakeFunc = std::function<std::unique_ptr<T>()>;

  explicit Pool(MakeFunc make) : make_(std::move(make)) {}

  std::unique_ptr<T> get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!items_.empty()) {
      auto item = std::move(items_.back());
      items_.pop_back();
      return item;
    }
    return make_();
  }

  void put(std::unique_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    items_.push_back(std::move(item));
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<T>> items_;
  MakeFunc make_;
};

using StreamPool = Pool<facebook::velox::wave::Stream>;
using EventPool = Pool<facebook::velox::wave::Event>;

} // namespace torch::wave

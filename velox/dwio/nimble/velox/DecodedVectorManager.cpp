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

#include "velox/dwio/nimble/velox/DecodedVectorManager.h"

namespace facebook::nimble {

std::unique_ptr<velox::DecodedVector>
DecodedVectorManager::getLocalDecodedVector() {
  if (decodedVectorPool_.empty()) {
    return std::make_unique<velox::DecodedVector>();
  }
  auto vector = std::move(decodedVectorPool_.back());
  decodedVectorPool_.pop_back();
  return vector;
}

void DecodedVectorManager::releaseDecodedVector(
    std::unique_ptr<velox::DecodedVector>&& vector) {
  decodedVectorPool_.push_back(std::move(vector));
}

DecodedVectorManager::LocalDecodedVector::LocalDecodedVector(
    DecodedVectorManager& manager)
    : manager_(manager), vector_(manager_.getLocalDecodedVector()) {}

DecodedVectorManager::LocalDecodedVector::LocalDecodedVector(
    LocalDecodedVector&& other) noexcept
    : manager_{other.manager_}, vector_{std::move(other.vector_)} {}

DecodedVectorManager::LocalDecodedVector::~LocalDecodedVector() {
  if (vector_) {
    manager_.releaseDecodedVector(std::move(vector_));
  }
}

velox::DecodedVector& DecodedVectorManager::LocalDecodedVector::get() {
  return *vector_;
}

} // namespace facebook::nimble

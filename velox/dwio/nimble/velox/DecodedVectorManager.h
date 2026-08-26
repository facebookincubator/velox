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

#pragma once

#include "velox/vector/DecodedVector.h"

namespace facebook::nimble {

class DecodedVectorManager {
 public:
  DecodedVectorManager() = default;

  class LocalDecodedVector {
   public:
    explicit LocalDecodedVector(DecodedVectorManager& manager);
    LocalDecodedVector(LocalDecodedVector&& other) noexcept;
    LocalDecodedVector& operator=(LocalDecodedVector&& other) = delete;
    ~LocalDecodedVector();

    // Lint suggestion to delete copy constructor and assignment operator.
    // Preventing these operations to avoid confusion about ownership.
    // Ownership is managed by DecodedVectorManager.
    LocalDecodedVector(const LocalDecodedVector&) = delete;
    LocalDecodedVector& operator=(const LocalDecodedVector&) = delete;

    velox::DecodedVector& get();

   private:
    DecodedVectorManager& manager_;
    std::unique_ptr<velox::DecodedVector> vector_;
  };

 private:
  std::unique_ptr<velox::DecodedVector> getLocalDecodedVector();
  void releaseDecodedVector(std::unique_ptr<velox::DecodedVector>&& vector);

  std::vector<std::unique_ptr<velox::DecodedVector>> decodedVectorPool_;
};

} // namespace facebook::nimble

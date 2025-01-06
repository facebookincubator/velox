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

#include "velox/vector/BaseVector.h"
#include "velox/vector/fuzzer/Utils.h"

namespace facebook::velox {
/// An interface for fuzzing Vectors of a custom type. This is intended for use
/// when a custom type does not support the full range of values of the backing
/// physical type.
///
/// Implementations of this interface need to be registered in an instance of
/// VectorFuzzer via registerCustomVectorFuzzer in order for them to be used in
/// general purpose fuzzing.
class CustomVectorFuzzer {
 public:
  virtual ~CustomVectorFuzzer() = default;

  /// Should return a flat Vector of the given size without nulls.
  virtual const VectorPtr fuzzFlat(
      memory::MemoryPool* pool,
      const TypePtr& type,
      vector_size_t size,
      FuzzerGenerator& rng) = 0;

  /// Should return a ConstantVector of the given size backed by a single
  /// non-null scalar value (complex types do not need to implement this).
  virtual const VectorPtr fuzzConstant(
      memory::MemoryPool* pool,
      const TypePtr& type,
      vector_size_t size,
      FuzzerGenerator& rng) = 0;
};
} // namespace facebook::velox

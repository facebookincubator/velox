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

#include "velox/type/Type.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer {

// Suppress warnings about deprecated declarations for AbstractInputGenerator
// interface and std::hash usage in type conversions.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class KHyperLogLogInputGenerator : public AbstractInputGenerator {
 public:
  KHyperLogLogInputGenerator(
      const size_t seed,
      const double nullRatio,
      memory::MemoryPool* pool,
      int32_t minNumValues = 1);

  variant generate() override;

 private:
  template <typename T>
  variant generateTyped();

  TypePtr baseType_;
  int32_t maxSize_;
  int32_t hllBuckets_;
  int32_t minNumValues_;
  memory::MemoryPool* pool_;
};
#pragma GCC diagnostic pop

} // namespace facebook::velox::fuzzer

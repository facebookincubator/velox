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
#include "velox/common/base/CompareFlags.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble {

class DeduplicationUtils {
 public:
  // Compares elements of two maps at the given index.
  static bool CompareMapsAtIndex(
      const velox::MapVector& leftMap,
      velox::vector_size_t leftIdx,
      const velox::MapVector& rightMap,
      velox::vector_size_t rightIdx);
};

} // namespace facebook::nimble

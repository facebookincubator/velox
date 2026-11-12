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

#include <cstdint>

namespace facebook::velox {

/// Represents a column copied from an input channel to an output channel,
/// possibly with rows removed or duplicated.
struct IdentityProjection {
  IdentityProjection() = default;

  // Clang 15 does not support initializing an aggregate with parentheses, e.g.
  // emplace_back(input, output), so the type needs a constructor.
  IdentityProjection(uint32_t inputChannel, uint32_t outputChannel)
      : inputChannel{inputChannel}, outputChannel{outputChannel} {}

  /// Identifies the source column.
  uint32_t inputChannel{0};

  /// Identifies the destination column.
  uint32_t outputChannel{0};
};

} // namespace facebook::velox

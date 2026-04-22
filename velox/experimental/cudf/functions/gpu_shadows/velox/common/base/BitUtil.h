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

// GPU shadow: minimal stubs so Bitwise.h compiles under NVCC.
// Only BitCountFunction references these; we don't register it on GPU.
#pragma once

#include <cstdint>

namespace facebook::velox::bits {

inline int32_t countBits(
    const uint64_t* /*bits*/,
    int32_t /*begin*/,
    int32_t /*end*/) {
  return 0;
}

} // namespace facebook::velox::bits

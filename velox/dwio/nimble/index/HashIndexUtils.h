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

#include <cstdint>
#include <string_view>

#include "folly/hash/Hash.h"

namespace facebook::nimble::index {

/// Computes the bucket index for an encoded key using FNV-1a hash.
/// Used by both HashIndexWriter (write path) and HashIndex (read path).
inline uint32_t bucketIndex(std::string_view key, uint32_t bucketMask) {
  return static_cast<uint32_t>(
             folly::hash::fnva64_buf(key.data(), key.size())) &
      bucketMask;
}

} // namespace facebook::nimble::index

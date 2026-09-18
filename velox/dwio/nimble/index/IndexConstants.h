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

namespace facebook::nimble {

// Placeholder stream ID for the key stream used in cluster index.
// This is a reserved stream ID that should not be used by regular data streams.
// The key stream contains encoded keys for efficient range-based filtering.
constexpr uint32_t kKeyStreamId = UINT32_MAX;

} // namespace facebook::nimble

namespace facebook::nimble::index {

inline constexpr std::string_view kClusterIndexName{"nimble.cluster.v1"};
inline constexpr std::string_view kDenseHashIndexName{"nimble.dense.hash.v1"};
inline constexpr std::string_view kDenseSortedIndexName{
    "nimble.dense.sorted.v1"};

} // namespace facebook::nimble::index

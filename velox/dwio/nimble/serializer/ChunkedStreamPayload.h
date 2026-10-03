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

#include <string_view>

#include "velox/dwio/nimble/common/Buffer.h"

namespace facebook::nimble::serde {

/// Removes the header and decompresses a stream containing exactly one chunk.
/// Returns a zero-copy input view for an uncompressed chunk; compressed output
/// is backed by `outputBuffer` and remains valid for its lifetime.
/// Fails if the input does not contain exactly one complete chunk.
std::string_view stripSingleChunkHeader(
    std::string_view streamData,
    Buffer& outputBuffer);

/// Removes chunk headers and decompresses chunk payloads. Returns a zero-copy
/// input view for one uncompressed chunk; other views are backed by
/// `outputBuffer` and remain valid for its lifetime.
///
/// TODO: Change stripChunkHeaders() to return a vector of stripped payload
/// views so all-uncompressed multi-chunk streams can avoid copying.
std::string_view stripChunkHeaders(
    std::string_view streamData,
    Buffer& outputBuffer);

} // namespace facebook::nimble::serde

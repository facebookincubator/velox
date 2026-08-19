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

#include <memory>
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

namespace detail {

template <typename T>
std::unique_ptr<TypedEncodingView<T>> createTypedEncodingView(
    std::string_view data,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options);

} // namespace detail

/// Returns whether createEncodingView() can wrap a stream of this encoding,
/// which is what lets a caller read one row without decoding the whole stream.
/// Data type restrictions still apply on top of this; createEncodingView()
/// throws when an encoding has no view for the stream's type.
bool supportsEncodingView(EncodingType encodingType);

/// Creates a random-access view over a supported uncompressed encoding stream.
std::unique_ptr<EncodingView> createEncodingView(
    std::string_view data,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options = {});

} // namespace facebook::nimble

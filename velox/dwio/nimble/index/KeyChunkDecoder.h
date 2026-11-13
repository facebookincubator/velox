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

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/KeyReader.h"
#include "velox/dwio/nimble/tablet/DataInput.h"

namespace facebook::nimble::index {

/// Decoded key chunk with ownership of backing buffers.
struct DecodedKeyChunk {
  // Zero-copy path: handle kept alive so the bytes it owns back the encoding,
  // such as a load buffer or a pinned cache entry.
  DataInput::Handle handle;

  // Thread-safe random-access reader over the decoded key payload.
  std::unique_ptr<KeyReader> reader;

  std::vector<velox::BufferPtr> stringBuffers;
};

using KeyReaderFactory = std::function<std::unique_ptr<KeyReader>(
    std::string_view,
    std::function<void*(uint32_t)>,
    velox::memory::MemoryPool*)>;

/// Decodes the key chunk in 'chunk', which holds the chunk header and payload.
/// Reads the chunk header, validates compression and payload layout, and
/// creates the reader with 'readerFactory'. 'handle' keeps the bytes of 'chunk'
/// alive. The returned DecodedKeyChunk holds 'handle' when it reads an
/// uncompressed payload in place.
///
/// Returns a shared_ptr so the reader's string allocator lambda (which
/// captures a pointer to the DecodedKeyChunk) remains valid when the caller
/// moves or reassigns the shared_ptr.
std::shared_ptr<DecodedKeyChunk> decodeKeyChunk(
    std::string_view chunk,
    DataInput::Handle handle,
    const KeyReaderFactory& readerFactory,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index

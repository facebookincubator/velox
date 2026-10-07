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
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/index/KeyReader.h"

namespace facebook::nimble::index {

/// Decoded key chunk with ownership of backing buffers.
struct DecodedKeyChunk {
  // Zero-copy path: stream kept alive so its buffer backs the encoding.
  std::unique_ptr<velox::dwio::common::SeekableInputStream> dataStream;

  // Thread-safe random-access reader over the decoded key payload.
  std::unique_ptr<KeyReader> reader;

  std::vector<velox::BufferPtr> stringBuffers;
};

using KeyReaderFactory = std::function<std::unique_ptr<KeyReader>(
    std::string_view,
    std::function<void*(uint32_t)>,
    velox::memory::MemoryPool*)>;

/// Decodes a key chunk from an input stream. Reads the chunk header, validates
/// compression and payload layout, and creates the reader. Handles both
/// contiguous (zero-copy) and multi-buffer reads.
///
/// Returns a shared_ptr so the reader's string allocator lambda (which
/// captures a pointer to the DecodedKeyChunk) remains valid when the caller
/// moves or reassigns the shared_ptr.
///
/// @param dataBuffer Caller-owned reuse slot for multi-buffer reads. When
///        the existing buffer has sufficient capacity it is reused;
///        otherwise a fresh buffer is allocated and written back into the
///        slot. The destination buffer (reused or freshly allocated) is
///        always appended to DecodedKeyChunk::stringBuffers, so the
///        returned DecodedKeyChunk holds its own reference and the caller
///        may independently reassign or destroy dataBuffer without
///        affecting the returned DecodedKeyChunk. Callers that don't want
///        cross-call reuse can pass a local BufferPtr that goes out of
///        scope after the call.
/// Decodes a key chunk and constructs its reader with `readerFactory`.
std::shared_ptr<DecodedKeyChunk> decodeKeyChunk(
    std::unique_ptr<velox::dwio::common::SeekableInputStream> inputStream,
    const KeyReaderFactory& readerFactory,
    velox::BufferPtr& dataBuffer,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index

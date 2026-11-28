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

#include <cstddef>
#include <span>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"

namespace facebook::nimble {

/// Gives the encoding views that share it room for the strings they decode,
/// in buffers that they share. The views of the chunks of one stream that
/// share an instance decode their strings into a few large buffers instead of
/// at least one buffer each, so that a caller that keeps the strings it reads
/// by holding their buffers, instead of copying them, holds and frees a few
/// buffers. A buffer stays valid while a view or a caller holds it. The views
/// do not own the instance, so it must outlive them. clear() lets one instance
/// serve the views of one stream's stripes in turn.
///
/// Not thread-safe, like Encoding::Options::bufferPool: the caller that gives
/// an instance to views must not read them at once, so that one decode at a
/// time writes into it. A reserve() before the room that the last one returned
/// finishes fails. StreamReader gives each stream its own instance, and
/// decodes a stream on one thread at a time.
///
///   DecodedStringBuffers buffers{pool};
///   Encoding::Options options{.cacheEncodingView = true};
///   options.decodedStringBuffers = &buffers;
///   const auto first = createEncodingView(firstChunk, pool, options);
///   const auto second = createEncodingView(secondChunk, pool, options);
///
/// A decode reserves room, writes its strings into it, and finishes it with
/// the bytes that they take, and the next reserve() starts right after them.
/// The room is the rest of the last buffer, or a new buffer when the rest is
/// too small. A decode that needs more room than it reserved finishes the room
/// and reserves again, so that its strings can lie in several buffers and none
/// is copied. Each new buffer takes twice the bytes of the one before, from
/// 32 KB up to 1 MB, unless the reservation needs more. A reservation larger
/// than 1 MB takes a buffer of its own, which no later room shares, so that a
/// caller that holds a later room for a few strings does not keep that buffer
/// alive.
class DecodedStringBuffers {
 public:
  /// Allocates the buffers from 'pool'.
  explicit DecodedStringBuffers(velox::memory::MemoryPool* pool);

  ~DecodedStringBuffers() = default;

  DecodedStringBuffers(const DecodedStringBuffers&) = delete;

  DecodedStringBuffers& operator=(const DecodedStringBuffers&) = delete;

  DecodedStringBuffers(DecodedStringBuffers&&) = delete;

  DecodedStringBuffers& operator=(DecodedStringBuffers&&) = delete;

  /// Starts over for the views of other chunks, such as those of the next
  /// stripe of a stream: drops the last buffer, so that the next reserve()
  /// takes a new buffer of 32 KB. The buffers that earlier decodes took stay
  /// valid while their holders hold them. No room may be reserved meanwhile.
  void clear();

  /// Returns room of at least 'numReservedBytes' bytes for strings: the rest
  /// of the last buffer, right after the strings of the rooms before it, if it
  /// has that many bytes, or else a new buffer, which is a buffer of its own
  /// when 'numReservedBytes' exceeds the 1 MB of the largest buffer. Fails
  /// until the room that the last call returned finishes.
  std::span<char> reserve(size_t numReservedBytes);

  /// Finishes the room that the last reserve() returned, whose strings take
  /// its first 'numUsedBytes' bytes, and returns the buffer that holds them. A
  /// decode that fails passes zero.
  velox::BufferPtr finish(size_t numUsedBytes);

  /// The bytes that the largest buffer takes from the pool, with the header of
  /// the buffer: 1 MB, a power of two, which fits a size class of the
  /// allocators exactly. It bounds what a buffer that a caller holds for a few
  /// strings keeps alive, since a larger reservation takes a buffer of its
  /// own. A decoder that would reserve more for the most bytes that its
  /// strings can decode to counts the bytes that they decode to instead.
  static constexpr size_t kMaxBufferBytes{1 << 20};

 private:
  // The bytes that the first buffer takes from the pool, with the header of
  // the buffer, a power of two like kMaxBufferBytes.
  static constexpr size_t kMinBufferBytes{32 << 10};

  // Allocates the buffers.
  velox::memory::MemoryPool* const pool_;

  // The buffer that rooms come from. Null before the first reserve() and
  // after clear().
  velox::BufferPtr buffer_;

  // The bytes of 'buffer_' that the finished rooms took, from its start. The
  // next room starts right after them.
  size_t usedBytes_{0};

  // The bytes that the next buffer takes from the pool, unless the
  // reservation needs more.
  size_t nextBufferBytes_{kMinBufferBytes};

  // The buffer of a room larger than kMaxBufferBytes, which no other room
  // shares, until the room finishes. Null otherwise.
  velox::BufferPtr oversizeBuffer_;

  // Whether reserve() returned room that has not finished.
  bool decoding_{false};
};

} // namespace facebook::nimble

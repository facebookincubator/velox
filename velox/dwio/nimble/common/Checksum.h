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

#include "velox/dwio/nimble/common/Types.h"

#include <memory>
#include <string>

namespace facebook::nimble {

/// Accumulating checksum over a byte sequence.
///
/// Instances are stateful and not thread-safe: each thread must use its own.
class Checksum {
 public:
  virtual ~Checksum() = default;

  /// Discards accumulated state, returning the instance to its initial
  /// condition.
  virtual void reset() = 0;

  virtual void update(std::string_view data) = 0;

  /// Returns the checksum of everything accumulated since the last reset. When
  /// `reset` is true, state is discarded afterwards so the next `update` starts
  /// a new sequence.
  virtual uint64_t getChecksum64(bool reset = false) = 0;

  /// Narrows getChecksum64() to 32 bits. Implementations own the narrowing
  /// because it is only sound for a hash whose output bits are evenly
  /// distributed; a natively 32-bit algorithm narrows differently than a
  /// 64-bit one.
  ///
  /// The narrowing is persisted on disk, so an implementation must never
  /// change it once files exist.
  virtual uint32_t getChecksum32(bool reset = false) = 0;

  /// Returns the checksum of `data` alone, independent of anything previously
  /// accumulated. Discards accumulated state.
  virtual uint64_t computeChecksum64(std::string_view data) = 0;

  /// Narrows computeChecksum64() to 32 bits, under the same constraints as
  /// getChecksum32().
  virtual uint32_t computeChecksum32(std::string_view data) = 0;

  virtual ChecksumType getType() const = 0;
};

class ChecksumFactory {
 public:
  static std::unique_ptr<Checksum> create(ChecksumType type);
};

} // namespace facebook::nimble

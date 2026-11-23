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
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"

namespace facebook::nimble::index {

/// Identifies the layout of a serialized bloom filter.
///
/// Every serialized filter records its type, so a value that has been written
/// to a file must never be reassigned to a different layout. Introduce a new
/// value whenever the block geometry, the probe scheme, or the hash function
/// changes. Reading a filter whose value this build does not recognize fails
/// rather than guessing at the bytes.
///
/// Values are allocated here even for implementations that live outside the
/// open source build. The number is not the secret, the implementation is: a
/// build with no factory registered for a value cannot read such a filter, and
/// says so.
enum class BloomFilterType : uint8_t {
  /// Split-block layout: 256-bit blocks, eight probes per block, xxHash64.
  kBlocked = 1,
};

/// Parameters controlling how a bloom filter is built. Readers ignore these
/// and follow whatever the filter itself records.
///
/// Implementations needing their own knobs derive from this and are reached
/// through checkedBloomFilterConfig. Bits per key stays in the base because
/// every layout has to trade space against false positives somehow.
struct BloomFilterConfig {
  BloomFilterConfig(BloomFilterType type, float bitsPerKey)
      : type{type}, bitsPerKey{bitsPerKey} {}

  virtual ~BloomFilterConfig() = default;

  /// Selects the registered factory that builds the filter.
  BloomFilterType type;

  /// Filter size per distinct key. Larger values trade memory for a lower
  /// false positive rate; 10 bits per key gives roughly 1%.
  float bitsPerKey;
};

/// Casts 'config' to the concrete type a factory expects, checking the cast.
template <typename T>
const T& checkedBloomFilterConfig(const BloomFilterConfig& config) {
  static_assert(std::is_base_of_v<BloomFilterConfig, T>);
  return *velox::checkedPointerCast<const T>(&config);
}

/// Accumulates keys and serializes them into a self-describing filter. Not
/// safe for concurrent use.
class BloomFilterBuilder {
 public:
  virtual ~BloomFilterBuilder() = default;

  /// Adds 'key' to the filter. Repeated keys are allowed and leave the filter
  /// unchanged after the first insert.
  virtual void insert(std::string_view key) = 0;

  /// Finalizes the filter and returns its serialized bytes, including the
  /// trailer recording the layout. The returned view points into memory this
  /// builder owns, so the caller must keep the builder alive for as long as it
  /// uses the view. Call at most once, and do not insert afterwards.
  virtual std::string_view finish() = 0;
};

/// Tests keys against a serialized filter. Immutable once created, so one
/// reader can serve concurrent lookups.
class BloomFilterReader {
 public:
  virtual ~BloomFilterReader() = default;

  /// Returns false only if 'key' was definitely never inserted. A true result
  /// may be a false positive, so the caller must still verify the key.
  virtual bool maybeContains(std::string_view key) const = 0;

  /// Tests every key in 'keys', writing the result for 'keys[i]' into
  /// 'out[i]'. 'out' must be at least as long as 'keys'. Each probe is a
  /// random access into a filter that is usually larger than the last-level
  /// cache, so an implementation may hash a run of keys up front and prefetch
  /// what they probe, overlapping the cache misses instead of paying them one
  /// at a time. The default implementation tests the keys one by one.
  virtual void maybeContains(
      std::span<const std::string_view> keys,
      std::span<bool> out) const;
};

/// Fixed-size record appended to every serialized filter so that a reader can
/// pick the matching implementation with no help from the enclosing metadata.
/// Keeping the discriminator inside the payload lets a caller store a filter as
/// an opaque byte range, which matters where one filter per key chunk would
/// make a per-filter metadata table more expensive than the filter itself.
/// Normal callers go through the factories below rather than using this
/// directly.
///
/// Layout: one byte of BloomFilterType followed by three reserved bytes that
/// must be zero. The reserved bytes leave room for fields that later revisions
/// may need without spending a new type value.
struct BloomFilterTrailer {
  static constexpr size_t kSize{4};

  /// Writes the trailer for 'type' into the 'kSize' bytes at 'destination'.
  static void write(char* destination, BloomFilterType type);

  /// Returns the trailer occupying the last 'kSize' bytes of 'serialized', or
  /// nullopt when 'serialized' is too short to hold one or its reserved bytes
  /// are not zero. The filter body is everything preceding it.
  static std::optional<BloomFilterTrailer> read(std::string_view serialized);

  BloomFilterType type;
};

/// Builds and reads one bloom filter layout.
class BloomFilterFactory {
 public:
  virtual ~BloomFilterFactory() = default;

  /// Layout this factory handles, as recorded in the filter trailer.
  virtual BloomFilterType type() const = 0;

  /// Creates a builder sized for 'numKeys' expected distinct keys. 'config'
  /// must be the concrete type this factory expects.
  virtual std::unique_ptr<BloomFilterBuilder> createBuilder(
      const BloomFilterConfig& config,
      uint64_t numKeys,
      velox::memory::MemoryPool* pool) const = 0;

  /// Creates a reader over 'payload', the filter body with the trailer already
  /// stripped.
  virtual std::unique_ptr<BloomFilterReader> createReader(
      std::string_view payload,
      velox::memory::MemoryPool* pool) const = 0;
};

/// Registers 'factory' under the type it reports. Throws if that type already
/// has one. Implementations outside the open source build register from their
/// own library rather than being named here.
void registerBloomFilterFactory(
    std::shared_ptr<const BloomFilterFactory> factory);

/// Returns the factory registered for 'type', or nullptr when this build has
/// none. A returned factory stays alive for the rest of the process, since
/// registration is permanent.
const BloomFilterFactory* bloomFilterFactory(BloomFilterType type);

/// Creates a builder for 'config', sized for 'numKeys' expected distinct keys.
/// The count is a sizing hint: inserting more keys raises the false positive
/// rate but stays correct. Throws if no factory is registered for the
/// configured type, because a writer that cannot honor its own config should
/// not silently produce a filter of some other shape.
std::unique_ptr<BloomFilterBuilder> createBloomFilterBuilder(
    const BloomFilterConfig& config,
    uint64_t numKeys,
    velox::memory::MemoryPool* pool);

/// Creates a reader over 'serialized', which must come from
/// BloomFilterBuilder::finish(). Copies the filter into 'pool', so the reader
/// does not depend on 'serialized' outliving it.
///
/// Throws if the trailer is unreadable or names a layout this build has no
/// factory for. A filter that cannot be interpreted is an error rather than
/// something to read past: silently treating it as matching every key would
/// turn a corrupt or unsupported file into a slow but plausible-looking
/// lookup.
std::unique_ptr<BloomFilterReader> createBloomFilterReader(
    std::string_view serialized,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index

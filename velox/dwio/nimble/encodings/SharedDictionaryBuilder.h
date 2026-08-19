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
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "absl/container/flat_hash_map.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"

#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"

namespace facebook::nimble {

inline constexpr uint64_t kMaxDictionaryEntryCount =
    std::numeric_limits<uint32_t>::max();

/// Maps dictionary values to their encoded shared dictionary indices.
template <typename T>
using DictionaryIndexType = absl::flat_hash_map<T, uint32_t>;

template <typename T>
DictionaryIndexType<T> buildDictionaryIndex(std::span<const T> alphabet) {
  NIMBLE_USER_CHECK_LE(
      alphabet.size(),
      kMaxDictionaryEntryCount,
      "Shared dictionary size exceeds maximum.");
  DictionaryIndexType<T> dictionaryIndex;
  dictionaryIndex.reserve(alphabet.size());
  for (uint32_t i = 0; i < alphabet.size(); ++i) {
    const auto [_, inserted] = dictionaryIndex.emplace(alphabet[i], i);
    NIMBLE_USER_CHECK(inserted, "Shared dictionary has duplicate values.");
  }
  return dictionaryIndex;
}

template <typename T>
class StreamingSharedDictionaryBuilder;

template <typename T>
class FixedSharedDictionaryBuilder;

template <typename T>
class ExternalSharedDictionaryBuilder;

/// Builds stable indices for one shared dictionary.
template <typename T>
class SharedDictionaryBuilder {
 public:
  /// Identifies how a writer-side shared dictionary builder owns or resolves
  /// its alphabet.
  enum class Kind : uint8_t {
    /// Builds an alphabet by inserting missing values during lookup.
    Streaming = 0,
    /// Uses a fixed alphabet supplied at construction time.
    Fixed = 1,
    /// Uses an externally supplied value-to-index map without file alphabet
    /// data.
    External = 2,
  };

  virtual ~SharedDictionaryBuilder() = default;

  /// Mapping produced by lookup().
  /// Streaming builders insert missing entries during lookup and report those
  /// entries here so callers can estimate alphabet cost.
  class Mapping {
   public:
    explicit Mapping(velox::memory::MemoryPool* pool) : indices_{pool} {}

    /// Returns one dictionary index for each input value passed to lookup().
    std::span<const uint32_t> indices() const {
      return indices_;
    }

    /// Returns how many distinct values lookup() inserted into a streaming
    /// dictionary. Streaming builders append them, so they are the last
    /// entries of alphabet(); fixed and external builders never insert.
    uint32_t newEntryCount() const {
      return newEntryCount_;
    }

   private:
    friend class StreamingSharedDictionaryBuilder<T>;
    friend class FixedSharedDictionaryBuilder<T>;
    friend class ExternalSharedDictionaryBuilder<T>;

    // One dictionary index per input value.
    Vector<uint32_t> indices_;
    // Number of distinct values inserted into a streaming dictionary by
    // lookup(), tracked as a count so the values are not copied twice.
    uint32_t newEntryCount_{0};
  };

  virtual Kind kind() const = 0;

  /// Returns a stable debug string for a builder kind.
  static std::string kindString(Kind kind) {
    switch (kind) {
      case Kind::Streaming:
        return "Streaming";
      case Kind::Fixed:
        return "Fixed";
      case Kind::External:
        return "External";
    }
    return fmt::format("Unknown: {}", static_cast<int>(kind));
  }

  /// Returns the lifetime and lookup namespace for this dictionary.
  SharedDictionaryScope scope() const {
    return scope_;
  }

  /// Returns the dictionary id within scope().
  uint32_t dictionaryId() const {
    return dictionaryId_;
  }

  /// Maps values to dictionary indices. Streaming builders insert missing
  /// values immediately; fixed and external builders throw because their
  /// alphabet is prebuilt and cannot grow.
  Mapping lookup(std::span<const T> values) {
    return lookupImpl(values);
  }

  /// Clears stripe-owned alphabet state before values from the next stripe are
  /// looked up. Fixed file/external builders do not reset because their
  /// dictionaries are prebuilt.
  void reset() {
    resetImpl();
  }

  /// Returns the alphabet entries when the writer owns and serializes them.
  virtual std::span<const T> alphabet() const = 0;

 protected:
  SharedDictionaryBuilder(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      velox::memory::MemoryPool* pool)
      : scope_{scope},
        dictionaryId_{dictionaryId},
        pool_{velox::checkedNotNull(pool)} {}

  velox::memory::MemoryPool* pool() const {
    return pool_;
  }

  virtual Mapping lookupImpl(std::span<const T> values) = 0;

  virtual void resetImpl() = 0;

 private:
  const SharedDictionaryScope scope_;
  const uint32_t dictionaryId_;
  velox::memory::MemoryPool* const pool_;
};

/// Streaming dictionary builder for stripe/file-owned alphabets.
template <typename T>
class StreamingSharedDictionaryBuilder final
    : public SharedDictionaryBuilder<T> {
 public:
  using Mapping = typename SharedDictionaryBuilder<T>::Mapping;
  using Kind = typename SharedDictionaryBuilder<T>::Kind;

  StreamingSharedDictionaryBuilder(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      velox::memory::MemoryPool* pool)
      : SharedDictionaryBuilder<T>{scope, dictionaryId, pool} {
    NIMBLE_CHECK_NE(
        scope,
        SharedDictionaryScope::External,
        "Streaming shared dictionary builder cannot use external scope.");
  }

  Kind kind() const final {
    return Kind::Streaming;
  }

  std::span<const T> alphabet() const final {
    return {alphabet_.data(), alphabet_.size()};
  }

 protected:
  Mapping lookupImpl(std::span<const T> values) final {
    Mapping mapping{this->pool()};
    mapping.indices_.reserve(values.size());

    for (const auto& value : values) {
      const auto it = alphabetMapping_.find(value);
      if (it != alphabetMapping_.end()) {
        mapping.indices_.push_back(it->second);
        continue;
      }

      const auto index = alphabet_.size();
      NIMBLE_USER_CHECK_LT(
          index,
          kMaxDictionaryEntryCount,
          "Shared dictionary size exceeds maximum.");
      const auto dictionaryIndex = static_cast<uint32_t>(index);
      const auto [_, inserted] =
          alphabetMapping_.emplace(value, dictionaryIndex);
      NIMBLE_CHECK(inserted, "Shared dictionary mapping insertion failed.");
      alphabet_.push_back(value);
      ++mapping.newEntryCount_;
      mapping.indices_.push_back(dictionaryIndex);
    }
    return mapping;
  }

  void resetImpl() final {
    alphabet_.clear();
    alphabetMapping_.clear();
  }

 private:
  std::vector<T> alphabet_;
  DictionaryIndexType<T> alphabetMapping_;
};

/// Fixed dictionary builder backed by a prebuilt alphabet. The caller keeps the
/// alphabet alive and unchanged for the builder's lifetime; the builder only
/// stores a view so the prebuilt alphabet is not copied again.
template <typename T>
class FixedSharedDictionaryBuilder final : public SharedDictionaryBuilder<T> {
 public:
  using Mapping = typename SharedDictionaryBuilder<T>::Mapping;
  using Kind = typename SharedDictionaryBuilder<T>::Kind;

  FixedSharedDictionaryBuilder(
      std::span<const T> alphabet,
      DictionaryIndexType<T> dictionaryIndex,
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      velox::memory::MemoryPool* pool)
      : SharedDictionaryBuilder<T>{scope, dictionaryId, pool},
        alphabet_{alphabet},
        dictionaryIndex_{std::move(dictionaryIndex)} {
    NIMBLE_CHECK_NE(
        scope,
        SharedDictionaryScope::External,
        "Fixed shared dictionary builder cannot use external scope.");
  }

  Kind kind() const final {
    return Kind::Fixed;
  }

  std::span<const T> alphabet() const final {
    return alphabet_;
  }

 protected:
  Mapping lookupImpl(std::span<const T> values) final {
    Mapping mapping{this->pool()};
    mapping.indices_.reserve(values.size());

    for (const auto& value : values) {
      const auto it = dictionaryIndex_.find(value);
      NIMBLE_USER_CHECK(
          it != dictionaryIndex_.end(),
          "{} shared dictionary {} does not contain value {}.",
          this->scope(),
          this->dictionaryId(),
          value);
      mapping.indices_.push_back(it->second);
    }
    return mapping;
  }

  void resetImpl() final {
    NIMBLE_UNSUPPORTED(
        "{} shared dictionary builder does not support reset().",
        SharedDictionaryBuilder<T>::kindString(this->kind()));
  }

 private:
  const std::span<const T> alphabet_;
  const DictionaryIndexType<T> dictionaryIndex_;
};

/// Lookup-only builder for externally owned dictionaries. It maps values to
/// indices, but does not expose dictionary entries because external alphabets
/// are not serialized into the Nimble file.
template <typename T>
class ExternalSharedDictionaryBuilder final
    : public SharedDictionaryBuilder<T> {
 public:
  using Mapping = typename SharedDictionaryBuilder<T>::Mapping;
  using Kind = typename SharedDictionaryBuilder<T>::Kind;

  ExternalSharedDictionaryBuilder(
      DictionaryIndexType<T> dictionaryIndex,
      uint32_t dictionaryId,
      velox::memory::MemoryPool* pool)
      : SharedDictionaryBuilder<
            T>{SharedDictionaryScope::External, dictionaryId, pool},
        dictionaryIndex_{std::move(dictionaryIndex)} {}

  Kind kind() const final {
    return Kind::External;
  }

  std::span<const T> alphabet() const final {
    NIMBLE_UNSUPPORTED(
        "{} shared dictionary builder does not expose an alphabet.",
        SharedDictionaryBuilder<T>::kindString(this->kind()));
  }

 protected:
  Mapping lookupImpl(std::span<const T> values) final {
    Mapping mapping{this->pool()};
    mapping.indices_.reserve(values.size());

    for (const auto& value : values) {
      const auto it = dictionaryIndex_.find(value);
      NIMBLE_USER_CHECK(
          it != dictionaryIndex_.end(),
          "{} shared dictionary {} does not contain value {}.",
          this->scope(),
          this->dictionaryId(),
          value);
      mapping.indices_.push_back(it->second);
    }
    return mapping;
  }

  void resetImpl() final {
    NIMBLE_UNSUPPORTED(
        "{} shared dictionary builder does not support reset().",
        SharedDictionaryBuilder<T>::kindString(this->kind()));
  }

 private:
  const DictionaryIndexType<T> dictionaryIndex_;
};

} // namespace facebook::nimble

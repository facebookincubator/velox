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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "absl/container/flat_hash_map.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"

namespace facebook::nimble {

/// Maps dictionary values to their encoded shared dictionary indices.
template <typename T>
using DictionaryIndexType = absl::flat_hash_map<T, uint32_t>;

template <typename T>
DictionaryIndexType<T> buildDictionaryIndex(std::span<const T> alphabet) {
  NIMBLE_USER_CHECK_LE(
      alphabet.size(),
      kMaxSharedDictionarySize,
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
class ExternalSharedDictionaryBuilder;

/// Builds stable indices for one shared dictionary.
template <typename T>
class SharedDictionaryBuilder {
  static_assert(
      !std::is_same_v<T, std::string>,
      "Shared dictionary string values must use std::string_view.");

 public:
  /// Identifies how a writer-side shared dictionary builder owns or resolves
  /// its alphabet.
  enum class Kind : uint8_t {
    /// Builds an alphabet by inserting missing values during lookup.
    Streaming = 0,
    /// Uses an externally supplied alphabet and value-to-index map.
    External = 1,
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
    /// entries of alphabet(); external builders never insert.
    uint32_t newEntryCount() const {
      return newEntryCount_;
    }

   private:
    friend class StreamingSharedDictionaryBuilder<T>;
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
  /// values immediately; external builders throw because their alphabet is
  /// prebuilt and cannot grow.
  Mapping lookup(std::span<const T> values) {
    return lookupImpl(values);
  }

  /// Clears stripe-owned alphabet state before values from the next stripe are
  /// looked up. External builders do not reset because their dictionaries are
  /// prebuilt.
  void reset() {
    resetImpl();
  }

  /// Returns the current alphabet entries.
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
      : SharedDictionaryBuilder<T>{scope, dictionaryId, pool}, alphabet_{pool} {
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
      const auto it = alphabetIndex_.find(value);
      if (it != alphabetIndex_.end()) {
        mapping.indices_.push_back(it->second);
        continue;
      }

      const auto index = alphabet_.size();
      NIMBLE_USER_CHECK_LT(
          index,
          kMaxSharedDictionarySize,
          "Shared dictionary size exceeds maximum.");
      const auto dictionaryIndex = static_cast<uint32_t>(index);
      const auto [alphabetIndexIt, inserted] =
          alphabetIndex_.emplace(value, dictionaryIndex);
      NIMBLE_CHECK(
          inserted,
          "Shared dictionary mapping insertion failed because the value already exists: value={}, existingIndex={}, newIndex={}.",
          value,
          alphabetIndexIt->second,
          dictionaryIndex);
      alphabet_.push_back(value);
      ++mapping.newEntryCount_;
      mapping.indices_.push_back(dictionaryIndex);
    }
    return mapping;
  }

  void resetImpl() final {
    alphabet_.clear();
    alphabetIndex_.clear();
  }

 private:
  Vector<T> alphabet_;
  DictionaryIndexType<T> alphabetIndex_;
};

/// Builder for externally supplied dictionaries.
template <typename T>
class ExternalSharedDictionaryBuilder final
    : public SharedDictionaryBuilder<T> {
 public:
  using Mapping = typename SharedDictionaryBuilder<T>::Mapping;
  using Kind = typename SharedDictionaryBuilder<T>::Kind;

  ExternalSharedDictionaryBuilder(
      SharedDictionaryScope scope,
      std::span<const T> alphabet,
      uint32_t dictionaryId,
      velox::memory::MemoryPool* pool)
      : SharedDictionaryBuilder<T>{checkScope(scope), dictionaryId, pool},
        alphabet_{alphabet},
        dictionaryIndex_{buildDictionaryIndex(alphabet)} {}

  Kind kind() const final {
    return Kind::External;
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
        "{} shared dictionary builder does not support reset().", this->kind());
  }

 private:
  static SharedDictionaryScope checkScope(SharedDictionaryScope scope) {
    NIMBLE_CHECK(
        scope == SharedDictionaryScope::File ||
            scope == SharedDictionaryScope::External,
        "External shared dictionary builder requires file or external scope.");
    return scope;
  }

  const std::span<const T> alphabet_;
  const DictionaryIndexType<T> dictionaryIndex_;
};

} // namespace facebook::nimble

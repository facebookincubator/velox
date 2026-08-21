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

#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

// ===========================================================================
// Leaf encoding structs (no children)
// ===========================================================================

struct TrivialEnc {
  operator EncodingLayout() const;
};

// TrivialEncoding<std::string_view> has a nested Lengths sub-encoding
// (EncodingIdentifiers::Trivial::Lengths = 0). Use this instead of TrivialEnc
// when encoding string types.
struct StringTrivialEnc {
  EncodingLayout lengths = TrivialEnc{};
  operator EncodingLayout() const;
};

struct ConstantEnc {
  operator EncodingLayout() const;
};

struct FixedBitWidthEnc {
  operator EncodingLayout() const;
};

struct VarintEnc {
  operator EncodingLayout() const;
};

struct SparseBoolEnc {
  operator EncodingLayout() const;
};

struct BlockBitPackingEnc {
  operator EncodingLayout() const;
};

struct DeltaEnc {
  EncodingLayout deltas = TrivialEnc{};
  EncodingLayout restatements = TrivialEnc{};
  EncodingLayout isRestatements = TrivialEnc{};
  operator EncodingLayout() const;
};

// ===========================================================================
// Compound encoding structs (with named children, defaulting to TrivialEnc{})
// ===========================================================================

struct RLEEnc {
  EncodingLayout runLengths = TrivialEnc{};
  EncodingLayout runValues = TrivialEnc{};
  operator EncodingLayout() const;
};

struct DictionaryEnc {
  EncodingLayout alphabet = TrivialEnc{};
  EncodingLayout indices = TrivialEnc{};
  operator EncodingLayout() const;
};

struct MainlyConstantEnc {
  EncodingLayout isCommon = TrivialEnc{};
  EncodingLayout otherValues = TrivialEnc{};
  operator EncodingLayout() const;
};

struct NullableEnc {
  EncodingLayout data = TrivialEnc{};
  EncodingLayout nulls = TrivialEnc{};
  operator EncodingLayout() const;
};

// ===========================================================================
// TrivialNestedPolicy — always selects Trivial for nested sub-streams.
// ===========================================================================

template <typename T>
class TrivialNestedPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  EncodingSelectionResult select(
      std::span<const physicalType>,
      const Statistics<physicalType>&,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Trivial};
  }
  EncodingSelectionResult selectNullable(
      std::span<const physicalType>,
      std::span<const bool>,
      const Statistics<physicalType>&,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }
  std::unique_ptr<EncodingSelectionPolicyBase>
  createImpl(EncodingType, NestedEncodingIdentifier, DataType type) override {
    UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
  }
};

// ===========================================================================
// createFromCustomLayout — encode data using an EncodingLayout tree.
// ===========================================================================

template <typename T>
std::unique_ptr<Encoding> createFromCustomLayout(
    const EncodingLayout& layout,
    const std::vector<T>& data,
    velox::memory::MemoryPool& pool,
    Buffer& buffer) {
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      layout,
      CompressionOptions{},
      [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
      });
  auto encoded = EncodingFactory::encode<T>(
      std::move(policy), std::span<const T>(data.data(), data.size()), buffer);
  return EncodingFactory().create(pool, encoded, nullptr);
}

// ===========================================================================
// createNullableFromCustomLayout — encode nullable data using a layout for
// the non-null values child.
// ===========================================================================

template <typename T>
std::unique_ptr<Encoding> createNullableFromCustomLayout(
    const EncodingLayout& dataLayout,
    const std::vector<std::optional<T>>& data,
    velox::memory::MemoryPool& pool,
    Buffer& buffer) {
  using physicalType = typename TypeTraits<T>::physicalType;

  std::vector<physicalType> nonNullValues;
  Vector<bool> nulls(&pool);
  nulls.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    nulls[i] = data[i].has_value();
    if (data[i].has_value()) {
      nonNullValues.push_back(
          reinterpret_cast<const physicalType&>(data[i].value()));
    }
  }

  std::span<const physicalType> valuesSpan(
      nonNullValues.data(), nonNullValues.size());
  std::span<const bool> nullsSpan(nulls.data(), nulls.size());

  NullableEnc nullableEnc{.data = dataLayout};
  EncodingLayout nullableLayout = nullableEnc;

  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      nullableLayout,
      CompressionOptions{},
      [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
      });

  auto encoded = EncodingFactory::encodeNullable<T>(
      std::move(policy), valuesSpan, nullsSpan, buffer);
  return EncodingFactory().create(pool, encoded, nullptr);
}

} // namespace facebook::nimble

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
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {

/// Encodes values and returns an alphabet reading them back. The returned
/// alphabet owns the buffer holding its encoded bytes, so callers do not have
/// to keep one alive themselves.
template <typename T>
std::shared_ptr<const SharedDictionaryAlphabet> createSharedDictionaryAlphabet(
    std::span<const T> values,
    std::span<const EncodingType> candidateEncodings,
    velox::memory::MemoryPool* pool) {
  struct OwnedAlphabet {
    Buffer buffer;
    std::optional<SharedDictionaryAlphabet> alphabet;

    explicit OwnedAlphabet(velox::memory::MemoryPool* pool) : buffer{*pool} {}
  };

  auto owned = std::make_shared<OwnedAlphabet>(pool);
  const auto encoded = SharedDictionaryAlphabet::encode<T>(
      values, candidateEncodings, owned->buffer);
  owned->alphabet.emplace(encoded, Encoding::Options{}, pool);
  return {owned, &owned->alphabet.value()};
}

class TestSharedDictionarySelectionPolicy final
    : public EncodingSelectionPolicy<int32_t> {
 public:
  using physicalType = typename TypeTraits<int32_t>::physicalType;

  TestSharedDictionarySelectionPolicy(
      SharedDictionaryEncodingInput sharedDictionary,
      EncodingSelectionPolicyCreator nestedPolicyCreator);

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) final;

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> values,
      std::span<const bool> nulls,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) final;

 private:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) final;

  const SharedDictionaryEncodingInput sharedDictionary_;
  const EncodingSelectionPolicyCreator nestedPolicyCreator_;
};

std::string_view encodeSharedDictionary(
    Buffer& buffer,
    const std::vector<uint32_t>& indices);

} // namespace facebook::nimble::test

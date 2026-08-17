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

#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

namespace facebook::nimble::test {

TestSharedDictionarySelectionPolicy::TestSharedDictionarySelectionPolicy(
    SharedDictionaryEncodingInput sharedDictionary,
    EncodingSelectionPolicyCreator nestedPolicyCreator)
    : sharedDictionary_{sharedDictionary},
      nestedPolicyCreator_{std::move(nestedPolicyCreator)} {
  NIMBLE_CHECK_NOT_NULL(nestedPolicyCreator_);
}

EncodingSelectionResult TestSharedDictionarySelectionPolicy::select(
    std::span<const physicalType> /*values*/,
    const Statistics<physicalType>& /*statistics*/,
    const Encoding::Options& /*options*/) {
  return {
      .encodingType = EncodingType::SharedDictionary,
      .sharedDictionaryInput = sharedDictionary_};
}

EncodingSelectionResult TestSharedDictionarySelectionPolicy::selectNullable(
    std::span<const physicalType> /*values*/,
    std::span<const bool> /*nulls*/,
    const Statistics<physicalType>& /*statistics*/,
    const Encoding::Options& /*options*/) {
  return {.encodingType = EncodingType::Nullable};
}

std::unique_ptr<EncodingSelectionPolicyBase>
TestSharedDictionarySelectionPolicy::createImpl(
    EncodingType /*parentEncodingType*/,
    NestedEncodingIdentifier /*nestedEncodingIdentifier*/,
    DataType nestedDataType) {
  auto policy = nestedPolicyCreator_(nestedDataType);
  NIMBLE_CHECK_NOT_NULL(policy);
  return policy;
}

std::string_view encodeSharedDictionary(
    Buffer& buffer,
    const std::vector<uint32_t>& indices) {
  std::vector<int32_t> values;
  values.reserve(indices.size());
  for (const auto index : indices) {
    values.push_back(static_cast<int32_t>(index));
  }
  auto options = Encoding::Options{};
  auto nestedPolicyCreator =
      [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    const auto encodingType = dataType == DataType::Uint32
        ? EncodingType::FixedBitWidth
        : EncodingType::Trivial;
    ManualEncodingSelectionPolicyFactory factory{
        {{encodingType, 1.0}}, std::nullopt};
    return factory.createPolicy(dataType);
  };
  return EncodingFactory::encode<int32_t>(
      std::make_unique<TestSharedDictionarySelectionPolicy>(
          SharedDictionaryEncodingInput{
              .scope = SharedDictionaryScope::Stripe,
              .dictionaryId = 7,
              .indices = std::span<const uint32_t>{indices}},
          nestedPolicyCreator),
      values,
      buffer,
      options);
}

} // namespace facebook::nimble::test

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

#include <optional>

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {

std::string_view encodeSharedDictionary(
    Buffer& buffer,
    const std::vector<uint32_t>& indices) {
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
  return SharedDictionaryEncoding<int32_t>::encode(
      indices, nestedPolicyCreator, buffer, options);
}

} // namespace facebook::nimble::test

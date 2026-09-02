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
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

/* static */ std::vector<std::pair<EncodingType, float>>
ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors() {
  return {
      {EncodingType::Constant, 1.0},
      {EncodingType::Trivial, 0.7},
      {EncodingType::FixedBitWidth, 0.9},
      {EncodingType::MainlyConstant, 1.0},
      {EncodingType::SparseBool, 1.0},
      {EncodingType::Dictionary, 1.0},
      {EncodingType::RLE, 1.0},
      {EncodingType::Varint, 1.0},
  };
}

/* static */ std::vector<std::pair<nimble::EncodingType, float>>
ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
    const std::string& readFactorsConfig) {
  std::vector<std::pair<nimble::EncodingType, float>> encodingReadFactors;
  std::vector<std::string> parts;
  folly::split(';', folly::trimWhitespace(readFactorsConfig), parts);
  encodingReadFactors.reserve(parts.size());

  const auto possibleEncodings =
      ManualEncodingSelectionPolicyFactory::possibleEncodings();
  std::vector<std::string> possibleEncodingStrs;
  possibleEncodingStrs.reserve(possibleEncodings.size());
  std::transform(
      possibleEncodings.cbegin(),
      possibleEncodings.cend(),
      std::back_inserter(possibleEncodingStrs),
      [](const auto& encoding) { return toString(encoding); });
  for (const auto& part : parts) {
    std::vector<std::string> kv;
    folly::split('=', part, kv);
    NIMBLE_USER_CHECK_EQ(
        kv.size(),
        2,
        "Invalid read factor format. "
        "Expected format is <EncodingType>=<factor>;<EncodingType>=<factor>. "
        "Unable to parse '{}'.",
        part);
    const auto value = folly::tryTo<float>(kv[1]);
    NIMBLE_USER_CHECK(
        value.hasValue(),
        "Unable to parse read factor value '{}' in '{}'. Expected valid float value.",
        kv[1],
        part);
    bool found = false;
    const auto key = folly::trimWhitespace(kv[0]);
    NIMBLE_USER_CHECK(
        !isReadOnlyEncoding(key),
        "Encoding is read-only and cannot be used for new writes: {}",
        key);
    for (auto i = 0; i < possibleEncodings.size(); ++i) {
      auto encoding = possibleEncodings[i];
      // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
      if (key == possibleEncodingStrs[i]) {
        found = true;
        encodingReadFactors.emplace_back(encoding, value.value());
        break;
      }
    }
    NIMBLE_USER_CHECK(
        found,
        "Unknown or unexpected read factor encoding '{}'. Allowed values: {}",
        key,
        folly::join(",", possibleEncodingStrs));
  }
  return encodingReadFactors;
}

/* static */ std::optional<ManualEncodingSelectionPolicyFactory>
ManualEncodingSelectionPolicyFactory::create(
    std::string_view configStr,
    std::optional<CompressionOptions> compressionOptions) {
  std::optional<std::string_view> readFactors;
  std::vector<std::string_view> entries;
  folly::split(',', configStr, entries);
  for (const auto entry : entries) {
    const auto colonPos = entry.find(':');
    NIMBLE_USER_CHECK(
        colonPos != std::string_view::npos,
        "Malformed nimble.encoding_selection_config entry '{}'; want key:value.",
        entry);
    const auto key = entry.substr(0, colonPos);
    const auto value = entry.substr(colonPos + 1);
    if (key == "type") {
      // The dispatcher (createEncodingSelectionPolicyFactory) routes here only
      // for type 'default'; reject a mismatched type reaching this factory.
      NIMBLE_USER_CHECK_EQ(
          value,
          "default",
          "nimble.encoding_selection_config type is not valid for the default encoding selection policy.");
      continue;
    } else if (key == "read_factors") {
      // A repeated read_factors key takes the last value (last-one-wins).
      readFactors = value;
    } else {
      NIMBLE_USER_FAIL(
          "Unknown nimble.encoding_selection_config key '{}' for type 'default'.",
          key);
    }
  }
  if (!readFactors.has_value()) {
    return std::nullopt;
  }
  return ManualEncodingSelectionPolicyFactory{
      parseEncodingReadFactors(std::string(*readFactors)),
      std::move(compressionOptions)};
}

ManualEncodingSelectionPolicyFactory::ManualEncodingSelectionPolicyFactory(
    std::vector<std::pair<EncodingType, float>> encodingReadFactors,
    std::optional<CompressionOptions> compressionOptions,
    std::optional<std::vector<std::pair<EncodingType, float>>>
        nestedEncodingReadFactors)
    : encodingReadFactors_{std::move(encodingReadFactors)},
      compressionOptions_{std::move(compressionOptions)},
      nestedEncodingReadFactors_{std::move(nestedEncodingReadFactors)} {}

std::unique_ptr<EncodingSelectionPolicyBase>
ManualEncodingSelectionPolicyFactory::createPolicy(DataType dataType) const {
  UNIQUE_PTR_FACTORY(
      dataType,
      ManualEncodingSelectionPolicy,
      encodingReadFactors_,
      compressionOptions_,
      std::nullopt,
      nestedEncodingReadFactors_);
}

/* static */ std::vector<EncodingType>
ManualEncodingSelectionPolicyFactory::possibleEncodings() {
  return {
      EncodingType::Constant,
      EncodingType::Trivial,
      EncodingType::FixedBitWidth,
      EncodingType::MainlyConstant,
      EncodingType::SparseBool,
      EncodingType::Dictionary,
      EncodingType::RLE,
      EncodingType::Varint,
      // EXPERIMENTAL: The following encodings are not production-ready. Do not
      // enable for production tables without consulting the Nimble team
      // (oncall: dwios).
      EncodingType::ALP,
      EncodingType::PFOR,
      EncodingType::SimdForBitpack,
      EncodingType::SubIntSplit,
      EncodingType::BlockBitPacking,
      EncodingType::DeltaBlock,
      EncodingType::Fsst,
      EncodingType::Huffman,
  };
}

} // namespace facebook::nimble

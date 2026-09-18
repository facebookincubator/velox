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
#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"

#include <folly/Conv.h>
#include <folly/String.h>
#include <algorithm>

namespace facebook::nimble::testing {

namespace {

// Resolves a ';'-separated list of encoding names to EncodingTypes, matching
// against the full parseable set. Surfaces a NimbleUserError on an empty list,
// an unknown name, or a duplicate.
std::vector<EncodingType> parseEncodings(std::string_view value) {
  const auto candidates =
      ManualEncodingSelectionPolicyFactory::possibleEncodings();
  std::vector<EncodingType> choices;
  std::vector<std::string_view> names;
  folly::split(';', value, names);
  for (const auto name : names) {
    NIMBLE_USER_CHECK(
        !name.empty(),
        "Empty encoding name in nimble.encoding_selection_config.");
    bool found = false;
    for (const auto candidate : candidates) {
      if (name == toString(candidate)) {
        NIMBLE_USER_CHECK(
            std::find(choices.begin(), choices.end(), candidate) ==
                choices.end(),
            "Duplicate encoding '{}' in nimble.encoding_selection_config.",
            name);
        choices.push_back(candidate);
        found = true;
        break;
      }
    }
    NIMBLE_USER_CHECK(
        found,
        "Unknown encoding '{}' in nimble.encoding_selection_config.",
        name);
  }
  NIMBLE_USER_CHECK(
      !choices.empty(),
      "nimble.encoding_selection_config 'encodings' must list at least one encoding.");
  return choices;
}

} // namespace

/* static */ std::vector<EncodingType>
RandomEncodingSelectionPolicyFactory::defaultEncodingChoices() {
  return {
      EncodingType::Constant,
      EncodingType::Trivial,
      EncodingType::FixedBitWidth,
      EncodingType::MainlyConstant,
      EncodingType::SparseBool,
      EncodingType::Dictionary,
      EncodingType::RLE,
      EncodingType::Varint,
      EncodingType::ALP,
      EncodingType::BlockBitPacking,
      EncodingType::Fsst,
  };
}

/* static */ RandomEncodingSelectionPolicyFactory
RandomEncodingSelectionPolicyFactory::create(
    std::string_view configStr,
    std::optional<CompressionOptions> compressionOptions) {
  std::optional<uint64_t> seed;
  std::vector<EncodingType> candidateEncodingTypes = defaultEncodingChoices();
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
      // Selected by createEncodingSelectionPolicyFactory; ignore it here.
      continue;
    } else if (key == "seed") {
      const auto parsed = folly::tryTo<uint64_t>(folly::StringPiece(value));
      NIMBLE_USER_CHECK(
          parsed.hasValue(),
          "Invalid seed '{}' in nimble.encoding_selection_config.",
          value);
      seed = parsed.value();
    } else if (key == "encodings") {
      candidateEncodingTypes = parseEncodings(value);
    } else {
      NIMBLE_USER_FAIL(
          "Unknown nimble.encoding_selection_config key '{}' for type 'random'.",
          key);
    }
  }
  NIMBLE_USER_CHECK(
      seed.has_value(),
      "nimble.encoding_selection_config type 'random' requires a 'seed'.");
  return RandomEncodingSelectionPolicyFactory{
      seed.value(),
      std::move(candidateEncodingTypes),
      std::move(compressionOptions)};
}

RandomEncodingSelectionPolicyFactory::RandomEncodingSelectionPolicyFactory(
    uint64_t seed,
    std::vector<EncodingType> candidateEncodingTypes,
    std::optional<CompressionOptions> compressionOptions)
    : seed_{seed},
      candidateEncodingTypes_{std::move(candidateEncodingTypes)},
      compressionOptions_{std::move(compressionOptions)} {}

std::unique_ptr<EncodingSelectionPolicyBase>
RandomEncodingSelectionPolicyFactory::createPolicy(DataType dataType) const {
  // Derive the root policy's seed from the base seed and the column's data type
  // so top-level columns of different types diverge; createImpl then folds in
  // each nested slot.
  const uint64_t rootSeed = folly::hash::hash_combine(
      seed_, static_cast<std::underlying_type_t<DataType>>(dataType));
  UNIQUE_PTR_FACTORY(
      dataType,
      RandomEncodingSelectionPolicy,
      rootSeed,
      candidateEncodingTypes_,
      compressionOptions_);
}
} // namespace facebook::nimble::testing

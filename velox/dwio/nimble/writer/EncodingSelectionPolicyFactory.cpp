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
#include "velox/dwio/nimble/writer/EncodingSelectionPolicyFactory.h"

#include <folly/String.h>
#include <glog/logging.h>
#include <vector>

#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"

namespace facebook::nimble {

namespace {

// Reads the "type" from a "type:...,key:value,..." config without consuming it;
// the selected factory re-reads the whole config itself. Returns an empty view
// when there is no "type" entry.
std::string_view peekEncodingSelectionType(std::string_view configStr) {
  std::vector<std::string_view> entries;
  folly::split(',', configStr, entries);
  for (const auto entry : entries) {
    const auto colonPos = entry.find(':');
    if (colonPos != std::string_view::npos &&
        entry.substr(0, colonPos) == "type") {
      return entry.substr(colonPos + 1);
    }
  }
  return {};
}

} // namespace

std::optional<EncodingSelectionPolicyCreator>
createEncodingSelectionPolicyFactory(
    std::string_view configStr,
    std::optional<CompressionOptions> compressionOptions) {
  if (configStr.empty()) {
    return std::nullopt;
  }
  // Peek (do not consume) "type" to pick the factory; the factory parses the
  // whole config itself. A non-empty config with no "type" is malformed (an
  // empty config was handled above).
  const auto type = peekEncodingSelectionType(configStr);
  NIMBLE_USER_CHECK(
      !type.empty(),
      "nimble.encoding_selection_config '{}' is missing a 'type'.",
      configStr);
  if (type == "default") {
    // The manual factory parses its own keys; nullopt (no read_factors) keeps
    // the nimble.manual_encoding_selection_read_factors default.
    auto manualFactory = ManualEncodingSelectionPolicyFactory::create(
        configStr, std::move(compressionOptions));
    if (!manualFactory.has_value()) {
      return std::nullopt;
    }
    return
        [factory = std::move(*manualFactory)](
            DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
          return factory.createPolicy(dataType);
        };
  }
  if (type == "random") {
    LOG(WARNING)
        << "Using test-only Nimble random encoding selection; not for production use.";
    return
        [factory = testing::RandomEncodingSelectionPolicyFactory::create(
             configStr, std::move(compressionOptions))](
            DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
          return factory.createPolicy(dataType);
        };
  }
  NIMBLE_USER_FAIL(
      "Invalid nimble.encoding_selection_config type '{}'. Valid: 'default', 'random'.",
      type);
}

} // namespace facebook::nimble

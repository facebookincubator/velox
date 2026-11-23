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

#include <optional>
#include <string_view>

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

/// Builds an EncodingSelectionPolicyCreator from a
/// nimble.encoding_selection_config string of the form
/// "type:<t>,key:value,...". The "type" selects which encoding-selection
/// factory to build; that factory then parses the whole config itself.
/// Currently supported:
///   type:random,seed:<n>[,encodings:<E1>;<E2>;...] ->
///       RandomEncodingSelectionPolicy (test/fuzz only).
/// An absent config or absent type returns nullopt, so the caller keeps the
/// default (manual) encoding selection. A malformed entry, unknown key,
/// unparseable value, or unknown type is a NimbleUserError.
std::optional<EncodingSelectionPolicyCreator>
createEncodingSelectionPolicyFactory(
    std::string_view configStr,
    std::optional<CompressionOptions> compressionOptions =
        CompressionOptions{});

} // namespace facebook::nimble

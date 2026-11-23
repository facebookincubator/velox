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

// The ways the options-off tests leave the typed-lanes options off. Unlike
// OptionsOffFixtures.h, this file names those options, so only this writer
// builds it.

#include <functional>
#include <string>
#include <vector>

#include "velox/dwio/nimble/writer/WriterOptions.h"

namespace facebook::nimble::test::options_off {

/// One way of leaving the typed-lanes options off.
struct OffForm {
  /// Name for test traces.
  std::string name;

  /// Applies the form to writer options.
  std::function<void(WriterOptions&)> apply;
};

/// Returns the forms that must leave everything as it was before the
/// typed-lanes options: the file, the published statistics and the compress
/// calls. They are: options untouched (always first); every option set to its
/// off value; and lanes off with the row floor at 1 and at its maximum, where
/// the floor has no effect.
std::vector<OffForm> offForms();

/// Returns lanes off with a typed compression level for FixedBitWidth. Without
/// lanes no FixedBitWidth payload is typed, so the level reaches nothing: the
/// compress calls must not change. Valid with MetaInternal compression only.
OffForm fixedBitWidthLevelWithoutLanes();

} // namespace facebook::nimble::test::options_off

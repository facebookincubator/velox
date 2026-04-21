/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

// GPU shadow for velox/common/base/CompareFlags.h
// Provides the CompareFlags struct without fmt dependency.
#pragma once

#include <optional>

namespace facebook::velox {

constexpr auto kIndeterminate = std::nullopt;

struct CompareFlags {
  bool nullsFirst = true;
  bool ascending = true;
  bool equalsOnly = false;

  enum class NullHandlingMode {
    kNullAsValue,
    kNullAsIndeterminate,
  };

  NullHandlingMode nullHandlingMode = NullHandlingMode::kNullAsValue;

  bool nullAsValue() const {
    return nullHandlingMode == CompareFlags::NullHandlingMode::kNullAsValue;
  }

  static constexpr CompareFlags equality(NullHandlingMode nullHandlingMode) {
    return CompareFlags{
        .equalsOnly = true, .nullHandlingMode = nullHandlingMode};
  }
};

} // namespace facebook::velox

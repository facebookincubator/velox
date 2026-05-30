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
#pragma once

#include <gtest/gtest.h>

#include "velox/external/regex_compat/Pcre2Regex.h"
#include "velox/external/regex_compat/Re2Regex.h"

#if VELOX_REGEX_COMPAT_HAS_JAVA
#include "velox/external/regex_compat/JavaRegex.h"
#endif

namespace facebook::velox::regex_compat::test {

/// GTest TYPED_TEST type list, instantiated once per backend at compile time.
/// Tests written as `TYPED_TEST_SUITE_P(MySuite, AllBackends)` automatically
/// run for every backend type that is enabled in this build.
#if VELOX_REGEX_COMPAT_HAS_JAVA
using AllBackends =
    ::testing::Types<Re2Regex, Pcre2Regex, JavaRegex>;
#else
using AllBackends = ::testing::Types<Re2Regex, Pcre2Regex>;
#endif

/// Base fixture for tests that should run against every backend.
template <typename R>
class BackendTest : public ::testing::Test {};

} // namespace facebook::velox::regex_compat::test

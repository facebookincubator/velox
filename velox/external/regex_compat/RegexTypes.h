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

namespace facebook::velox::regex_compat {

/// Mirrors `re2::RE2::Anchor`.
enum class Anchor { kUnanchored, kAnchorStart, kAnchorBoth };

/// Subset of `re2::RE2::Options` exposed to the regex-compat test suite.
/// Each backend (Re2Regex / Pcre2Regex / JavaRegex) maps fields to its native
/// option type.
struct Options {
  bool caseSensitive = true;
  bool dotNl = false;
  bool oneLine = true;
  bool logErrors = false;
  int maxMem = 8 << 20;
};

} // namespace facebook::velox::regex_compat

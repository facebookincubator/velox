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

// This header is only meaningful when the Java backend is enabled.  Clang-tidy
// scans changed headers in isolation and cannot find <jni.h> on hosts without
// a JDK, so guard the entire content rather than relying on every consumer to
// gate the include.
#if VELOX_REGEX_COMPAT_HAS_JAVA

#include <map>
#include <string>
#include <string_view>

#include <jni.h>

#include "velox/external/regex_compat/RegexTypes.h"

namespace facebook::velox::regex_compat {

/// `java.util.regex` backend in the regex-compat test suite, via an embedded
/// JVM (see JvmFixture).  Public method names and signatures mirror
/// `re2::RE2`'s subset that Velox uses.
///
/// Internally each `Match` / `GlobalReplace` call creates a fresh
/// `java.util.regex.Matcher` via the cached `jobject pattern_` and invokes
/// the JDK's regex engine.  Pattern + replacement input is the canonical
/// Java syntax (this is the native source of truth for the other two
/// backends' translation correctness).
class JavaRegex {
 public:
  explicit JavaRegex(std::string_view javaPattern, Options opt = {});
  ~JavaRegex();

  JavaRegex(const JavaRegex&) = delete;
  JavaRegex& operator=(const JavaRegex&) = delete;

  bool ok() const;
  const std::string& error() const;
  int NumberOfCapturingGroups() const;
  const std::map<std::string, int>& NamedCapturingGroups() const;

  bool Match(
      std::string_view input,
      std::size_t startpos,
      std::size_t endpos,
      Anchor anchor,
      std::string_view* submatch,
      int nsubmatch) const;

  static bool FullMatch(std::string_view input, const JavaRegex& re);
  static bool PartialMatch(std::string_view input, const JavaRegex& re);

  static int GlobalReplace(
      std::string* str,
      const JavaRegex& re,
      std::string_view javaReplacement);

 private:
  // Pinned global reference to java.util.regex.Pattern instance.
  jobject pattern_ = nullptr;
  std::string error_;
  int captureCount_ = 0;
  std::map<std::string, int> named_;
};

} // namespace facebook::velox::regex_compat

#endif // VELOX_REGEX_COMPAT_HAS_JAVA

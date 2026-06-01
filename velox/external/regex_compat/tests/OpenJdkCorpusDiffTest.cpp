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
//
// Runs the OpenJDK 17 `java/util/regex/TestCases.txt` corpus (~299 cases)
// against each backend and reports per-backend pass rate.
//
// File format (per OpenJDK header):
//   line 1: pattern
//   line 2: input
//   line 3: "true|false <match> <groupCount> <g1> <g2> <g3> <g4>"
//           — match-string and groups present only when first token is true.
// Empty lines and `//` comments are skipped.
//
// The corpus is fetched at CMake configure time and its path is injected
// via OPENJDK_CORPUS_PATH.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef OPENJDK_CORPUS_PATH
#error "OPENJDK_CORPUS_PATH must be defined by the build system"
#endif

namespace facebook::velox::regex_compat::test {
namespace {

struct CorpusCase {
  std::string pattern;
  std::string input;
  std::string expectedResult; // verbatim "true ..." / "false 0" / "error"
};

// Mirror OpenJDK 17 RegExTest.grabLine: handles only `\n` (→ U+000A) and
// `\uXXXX` (→ that code point); everything else passes through verbatim.
// Surrogate-pair `\uD8##\uDC##` sequences are combined into the proper
// supplementary code point so that we end up with a valid UTF-8 4-byte
// encoding (which both RE2/PCRE2 require and our Java JNI bridge
// re-splits to a surrogate pair).
static std::string processEscapes(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') {
      out.push_back('\n');
      i += 2;
      continue;
    }
    if (s[i] == '\\' && i + 5 < s.size() && s[i + 1] == 'u') {
      std::uint32_t cp = 0;
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        char c = s[i + 2 + k];
        cp <<= 4;
        if (c >= '0' && c <= '9') cp |= (c - '0');
        else if (c >= 'a' && c <= 'f') cp |= (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') cp |= (c - 'A' + 10);
        else { ok = false; break; }
      }
      if (!ok) {
        out.push_back(s[i++]);
        continue;
      }
      i += 6;
      // Combine surrogate pair if a low surrogate follows.
      if (cp >= 0xD800 && cp <= 0xDBFF && i + 5 < s.size() && s[i] == '\\'
          && s[i + 1] == 'u') {
        std::uint32_t lo = 0;
        bool ok2 = true;
        for (int k = 0; k < 4; ++k) {
          char c = s[i + 2 + k];
          lo <<= 4;
          if (c >= '0' && c <= '9') lo |= (c - '0');
          else if (c >= 'a' && c <= 'f') lo |= (c - 'a' + 10);
          else if (c >= 'A' && c <= 'F') lo |= (c - 'A' + 10);
          else { ok2 = false; break; }
        }
        if (ok2 && lo >= 0xDC00 && lo <= 0xDFFF) {
          cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
          i += 6;
        }
      }
      if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      continue;
    }
    out.push_back(s[i++]);
  }
  return out;
}
// OpenJDK format uses spaces both as field separators and inside captured
// group text.  We don't need to split — the OpenJDK runner emits the
// expected line via plain StringBuilder concatenation; we rebuild the
// actual result the same way and compare strings.

static std::vector<CorpusCase> loadCorpus(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  // Replicate OpenJDK's grabLine: skip blank and `//` lines.
  auto grab = [&](std::string& out) -> bool {
    while (std::getline(in, out)) {
      if (out.empty()) continue;
      if (out.size() >= 2 && out[0] == '/' && out[1] == '/') continue;
      return true;
    }
    return false;
  };
  std::vector<CorpusCase> cases;
  std::string pattern, input, expected;
  while (grab(pattern) && grab(input) && grab(expected)) {
    CorpusCase c;
    c.pattern = processEscapes(pattern);
    c.input = processEscapes(input);
    c.expectedResult = processEscapes(expected);
    cases.push_back(std::move(c));
  }
  return cases;
}

// Per-backend tally.  Printed by the global Environment at TearDown.
struct CorpusStats {
  int passed = 0;
  int failed = 0;
  int compileErrors = 0;
};

CorpusStats& statsFor(const std::string& backend) {
  static std::map<std::string, CorpusStats> s;
  return s[backend];
}

// Tear-down printer.  Registered as a global Environment so it runs after
// the typed tests.
class CorpusReporter : public ::testing::Environment {
 public:
  void TearDown() override {
    extern std::map<std::string, CorpusStats>& allStats();
    auto& m = allStats();
    if (m.empty()) {
      return;
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "========== OpenJDK corpus compat rate ==========\n");
    for (const auto& [name, st] : m) {
      int total = st.passed + st.failed + st.compileErrors;
      double pct = total > 0 ? 100.0 * st.passed / total : 0.0;
      std::fprintf(
          stderr,
          "  %-12s %4d / %4d  (%.2f%%)   [compile-err: %d]\n",
          name.c_str(),
          st.passed,
          total,
          pct,
          st.compileErrors);
    }
    std::fprintf(stderr, "================================================\n");
  }
};

// Definition for the extern declaration above.
std::map<std::string, CorpusStats>& allStats() {
  static std::map<std::string, CorpusStats> s;
  return s;
}

// Register the reporter exactly once.
[[maybe_unused]] static auto* kReporter =
    ::testing::AddGlobalTestEnvironment(new CorpusReporter);

template <typename R>
const char* backendName() {
  if constexpr (std::is_same_v<R, Re2Regex>) {
    return "Re2";
  } else if constexpr (std::is_same_v<R, Pcre2Regex>) {
    return "Pcre2";
  } else {
    return "Java";
  }
}

template <typename R>
using OpenJdkCorpusDiffTest = BackendTest<R>;
TYPED_TEST_SUITE(OpenJdkCorpusDiffTest, AllBackends);

TYPED_TEST(OpenJdkCorpusDiffTest, runCorpus) {
  static const std::vector<CorpusCase> kCorpus = loadCorpus(OPENJDK_CORPUS_PATH);
  ASSERT_FALSE(kCorpus.empty())
      << "Corpus is empty — failed to load " << OPENJDK_CORPUS_PATH;

  const std::string name = backendName<TypeParam>();
  auto& st = allStats()[name];

  for (const auto& c : kCorpus) {
    TypeParam re(c.pattern);
    if (!re.ok()) {
      // OpenJDK convention: expected-line starting with "error" means
      // the pattern is intentionally invalid; a compile-error here is
      // the correct outcome.
      if (c.expectedResult.rfind("error", 0) == 0) {
        ++st.passed;
      } else {
        ++st.compileErrors;
        if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
          std::fprintf(
              stderr,
              "[OpenJDK corpus] Java compile-err: pattern=[%s] err=[%s]\n",
              c.pattern.c_str(),
              re.error().c_str());
        }
      }
      continue;
    }
    JavaMatcherAdapter<TypeParam> m(&re, c.input);
    const bool found = m.find();
    // Rebuild the result string in exactly the same shape OpenJDK does
    // in RegExTest.processFile: "true <g0> <gc> <g1> <g2> ..."
    // (each captured non-null group appended) or "false 0".
    std::string actual;
    if (found) {
      actual.append("true ");
      actual.append(std::string(m.group(0).value()));
      actual.push_back(' ');
      actual.append(std::to_string(m.groupCount()));
      for (int i = 1; i <= m.groupCount(); ++i) {
        auto gi = m.group(i);
        if (gi) {
          actual.push_back(' ');
          actual.append(std::string(*gi));
        }
      }
    } else {
      actual.append("false ");
      actual.append(std::to_string(m.groupCount()));
    }
    if (actual == c.expectedResult) {
      ++st.passed;
    } else {
      ++st.failed;
      if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
        std::fprintf(
            stderr,
            "[OpenJDK corpus] Java mismatch:\n  pattern=[%s]\n  input=[%s]\n  expected=[%s]\n  actual=  [%s]\n",
            c.pattern.c_str(),
            c.input.c_str(),
            c.expectedResult.c_str(),
            actual.c_str());
      }
    }
  }

  // Java backend must round-trip the entire corpus (it IS java.util.regex).
  if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
    EXPECT_EQ(0, st.failed)
        << "Java backend should match every case in OpenJDK's own corpus";
    EXPECT_EQ(0, st.compileErrors)
        << "Java backend should compile every pattern in OpenJDK's own corpus";
  }
}

} // namespace
} // namespace facebook::velox::regex_compat::test

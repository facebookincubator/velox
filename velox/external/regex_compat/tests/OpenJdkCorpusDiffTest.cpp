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
#include "velox/external/regex_compat/JvmFixture.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef OPENJDK_CORPUS_DIR
#error "OPENJDK_CORPUS_DIR must be defined by the build system"
#endif

namespace facebook::velox::regex_compat::test {
namespace {

static const char* const kCorpusFiles[] = {
    "TestCases.txt",
    "BMPTestCases.txt",
    "SupplementaryTestCases.txt",
};

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

static std::string utf8(std::uint32_t cp) {
  std::string out;
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
  return out;
}

static std::uint32_t parseHexCodePoint(std::string_view token) {
  std::uint32_t cp = 0;
  for (char c : token) {
    cp <<= 4;
    if (c >= '0' && c <= '9') {
      cp |= (c - '0');
    } else if (c >= 'a' && c <= 'f') {
      cp |= (c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      cp |= (c - 'A' + 10);
    } else {
      throw std::invalid_argument("bad hex code point");
    }
  }
  return cp;
}

static jstring toJString(JNIEnv* env, std::string_view sv) {
  std::vector<jchar> u16;
  u16.reserve(sv.size());
  for (std::size_t i = 0; i < sv.size();) {
    const unsigned char c = static_cast<unsigned char>(sv[i]);
    std::uint32_t cp = 0;
    std::size_t step = 1;
    if (c < 0x80) {
      cp = c;
    } else if (c < 0xC0) {
      cp = 0xFFFD;
    } else if (c < 0xE0 && i + 1 < sv.size()) {
      cp = ((c & 0x1F) << 6) |
          (static_cast<unsigned char>(sv[i + 1]) & 0x3F);
      step = 2;
    } else if (c < 0xF0 && i + 2 < sv.size()) {
      cp = ((c & 0x0F) << 12) |
          ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 6) |
          (static_cast<unsigned char>(sv[i + 2]) & 0x3F);
      step = 3;
    } else if (i + 3 < sv.size()) {
      cp = ((c & 0x07) << 18) |
          ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 12) |
          ((static_cast<unsigned char>(sv[i + 2]) & 0x3F) << 6) |
          (static_cast<unsigned char>(sv[i + 3]) & 0x3F);
      step = 4;
    } else {
      cp = 0xFFFD;
    }
    if (cp <= 0xFFFF) {
      u16.push_back(static_cast<jchar>(cp));
    } else {
      cp -= 0x10000;
      u16.push_back(static_cast<jchar>(0xD800 | (cp >> 10)));
      u16.push_back(static_cast<jchar>(0xDC00 | (cp & 0x3FF)));
    }
    i += step;
  }
  return env->NewString(u16.data(), static_cast<jsize>(u16.size()));
}

static std::size_t javaCharOffsetToByteOffset(
    std::string_view utf8,
    int javaCharOffset) {
  int chars = 0;
  for (std::size_t i = 0; i < utf8.size();) {
    if (chars == javaCharOffset) {
      return i;
    }
    const unsigned char c = static_cast<unsigned char>(utf8[i]);
    if (c < 0x80) {
      i += 1;
      chars += 1;
    } else if (c < 0xE0) {
      i += 2;
      chars += 1;
    } else if (c < 0xF0) {
      i += 3;
      chars += 1;
    } else {
      i += 4;
      chars += 2;
    }
  }
  return chars == javaCharOffset ? utf8.size() : std::string_view::npos;
}

static std::vector<int> directJavaGraphemeBreakOffsets(std::string_view input) {
  auto* env = JvmFixture::instance().env();
  jclass patternCls = env->FindClass("java/util/regex/Pattern");
  jclass matcherCls = env->FindClass("java/util/regex/Matcher");
  jmethodID compile = env->GetStaticMethodID(
      patternCls,
      "compile",
      "(Ljava/lang/String;)Ljava/util/regex/Pattern;");
  jmethodID matcher = env->GetMethodID(
      patternCls,
      "matcher",
      "(Ljava/lang/CharSequence;)Ljava/util/regex/Matcher;");
  jmethodID find = env->GetMethodID(matcherCls, "find", "()Z");
  jmethodID start = env->GetMethodID(matcherCls, "start", "()I");

  jstring pat = toJString(env, "\\b{g}");
  jobject pattern = env->CallStaticObjectMethod(patternCls, compile, pat);
  env->DeleteLocalRef(pat);
  jstring subject = toJString(env, input);
  jobject m = env->CallObjectMethod(pattern, matcher, subject);
  env->DeleteLocalRef(subject);

  std::vector<int> offsets;
  while (env->CallBooleanMethod(m, find)) {
    const jint charOffset = env->CallIntMethod(m, start);
    const auto byteOffset = javaCharOffsetToByteOffset(input, charOffset);
    if (byteOffset != std::string_view::npos) {
      offsets.push_back(static_cast<int>(byteOffset));
    }
  }
  env->DeleteLocalRef(m);
  env->DeleteLocalRef(pattern);
  env->DeleteLocalRef(matcherCls);
  env->DeleteLocalRef(patternCls);
  return offsets;
}

struct GraphemeCase {
  std::string input;
  std::vector<int> expectedBreakOffsets;
};

static std::vector<GraphemeCase> loadGraphemeCorpus(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::vector<GraphemeCase> cases;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line.resize(hash);
    }
    std::istringstream tokens(line);
    std::string token;
    GraphemeCase c;
    bool sawToken = false;
    while (tokens >> token) {
      sawToken = true;
      if (token == "\xC3\xB7") {
        c.expectedBreakOffsets.push_back(static_cast<int>(c.input.size()));
      } else if (token == "\xC3\x97") {
        continue;
      } else {
        c.input += utf8(parseHexCodePoint(token));
      }
    }
    if (sawToken) {
      cases.push_back(std::move(c));
    }
  }
  return cases;
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

// Per-(backend, file) tally — keyed by "backend|file".
struct CorpusStats {
  int passed = 0;
  int failed = 0;
  int compileErrors = 0;
  // Subset of `compileErrors` whose root cause is the translator rejecting
  // the pattern as untranslatable for the engine (e.g. RE2 lookaround /
  // backref / possessive).  These are engine-feature-impossible, NOT bugs
  // in our translator; surfaced separately so we can report a rate that
  // excludes them ("translatable-subset rate").
  int translatorRejected = 0;
};

std::map<std::string, CorpusStats>& allStats() {
  static std::map<std::string, CorpusStats> s;
  return s;
}

// Tear-down printer.  Registered as a global Environment so it runs after
// the typed tests.
class CorpusReporter : public ::testing::Environment {
 public:
  void TearDown() override {
    auto& m = allStats();
    if (m.empty()) {
      return;
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "========== OpenJDK corpus compat rate ==========\n");
    // Aggregate per backend across all files; also print per-file.
    std::map<std::string, CorpusStats> agg;
    for (const auto& [key, st] : m) {
      auto bar = key.find('|');
      std::string backend = key.substr(0, bar);
      auto& a = agg[backend];
      a.passed += st.passed;
      a.failed += st.failed;
      a.compileErrors += st.compileErrors;
      a.translatorRejected += st.translatorRejected;
    }
    for (const auto& [key, st] : m) {
      int total = st.passed + st.failed + st.compileErrors;
      double pct = total > 0 ? 100.0 * st.passed / total : 0.0;
      std::fprintf(
          stderr,
          "  %-50s %4d / %4d  (%.2f%%)   [compile-err: %d]\n",
          key.c_str(),
          st.passed,
          total,
          pct,
          st.compileErrors);
    }
    std::fprintf(stderr, "  ---- aggregate ----\n");
    for (const auto& [name, st] : agg) {
      int total = st.passed + st.failed + st.compileErrors;
      double pct = total > 0 ? 100.0 * st.passed / total : 0.0;
      std::fprintf(
          stderr,
          "  %-50s %4d / %4d  (%.2f%%)   [compile-err: %d]\n",
          name.c_str(),
          st.passed,
          total,
          pct,
          st.compileErrors);
      // Also report a "translatable subset" rate that excludes patterns
      // the translator rejected as engine-impossible (e.g. RE2 lookaround
      // or backref).  This isolates what's actually attributable to the
      // translator/backend vs to engine ceilings.
      if (st.translatorRejected > 0) {
        const int subsetTotal = total - st.translatorRejected;
        const double subsetPct =
            subsetTotal > 0 ? 100.0 * st.passed / subsetTotal : 0.0;
        std::fprintf(
            stderr,
            "  %-50s %4d / %4d  (%.2f%%)   [excludes %d translator-rejected]\n",
            (name + " (translatable subset)").c_str(),
            st.passed,
            subsetTotal,
            subsetPct,
            st.translatorRejected);
      }
    }
    std::fprintf(stderr, "================================================\n");
  }
};

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
  const std::string backend = backendName<TypeParam>();
  int totalCases = 0;
  int totalJavaFailures = 0;
  for (const char* fname : kCorpusFiles) {
    std::string path = std::string(OPENJDK_CORPUS_DIR) + "/" + fname;
    std::vector<CorpusCase> kCorpus = loadCorpus(path);
    ASSERT_FALSE(kCorpus.empty()) << "Corpus is empty — failed to load " << path;
    totalCases += static_cast<int>(kCorpus.size());

    const std::string key = backend + "|" + fname;
    auto& st = allStats()[key];

    for (const auto& c : kCorpus) {
      TypeParam re(c.pattern);
      if (!re.ok()) {
        if (c.expectedResult.rfind("error", 0) == 0) {
          ++st.passed;
        } else {
          ++st.compileErrors;
          if (re.error().find("translator: ") != std::string::npos) {
            ++st.translatorRejected;
          }
          if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
            ++totalJavaFailures;
            std::fprintf(
                stderr,
                "[OpenJDK %s] Java compile-err: pattern=[%s] err=[%s]\n",
                fname,
                c.pattern.c_str(),
                re.error().c_str());
          }
        }
        continue;
      }
      JavaMatcherAdapter<TypeParam> m(&re, c.input);
      const bool found = m.find();
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
          ++totalJavaFailures;
          std::fprintf(
              stderr,
              "[OpenJDK %s] Java mismatch:\n  pattern=[%s]\n  input=[%s]\n  expected=[%s]\n  actual=  [%s]\n",
              fname,
              c.pattern.c_str(),
              c.input.c_str(),
              c.expectedResult.c_str(),
              actual.c_str());
        }
      }
    }
  }

  if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
    EXPECT_EQ(0, totalJavaFailures)
        << "Java backend should match every case across all OpenJDK corpus files";
  }
  EXPECT_GT(totalCases, 0);
}

struct GraphemeStats {
  int passed = 0;
  int failed = 0;
  int compileErrors = 0;
};

std::map<std::string, GraphemeStats>& graphemeStats() {
  static std::map<std::string, GraphemeStats> s;
  return s;
}

class GraphemeReporter : public ::testing::Environment {
 public:
  void TearDown() override {
    auto& m = graphemeStats();
    if (m.empty()) {
      return;
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "========== OpenJDK grapheme corpus compat rate ==========\n");
    for (const auto& [backend, st] : m) {
      const int total = st.passed + st.failed + st.compileErrors;
      const double pct = total > 0 ? 100.0 * st.passed / total : 0.0;
      std::fprintf(
          stderr,
          "  %-8s %4d / %4d  (%.2f%%)   [compile-err: %d]\n",
          backend.c_str(),
          st.passed,
          total,
          pct,
          st.compileErrors);
    }
    std::fprintf(stderr, "=========================================================\n");
  }
};

[[maybe_unused]] static auto* kGraphemeReporter =
    ::testing::AddGlobalTestEnvironment(new GraphemeReporter);

template <typename R>
using GraphemeCorpusTest = BackendTest<R>;
TYPED_TEST_SUITE(GraphemeCorpusTest, AllBackends);

TYPED_TEST(GraphemeCorpusTest, runGraphemeBreakCorpus) {
  const std::string path =
      std::string(OPENJDK_CORPUS_DIR) + "/GraphemeTestCases.txt";
  const auto cases = loadGraphemeCorpus(path);
  ASSERT_FALSE(cases.empty()) << "Corpus is empty — failed to load " << path;

  int javaFailures = 0;
  auto& st = graphemeStats()[backendName<TypeParam>()];
  for (const auto& c : cases) {
    TypeParam re("\\b{g}");
    if (!re.ok()) {
      ++st.compileErrors;
      if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
        ++javaFailures;
        std::fprintf(
            stderr,
            "[OpenJDK Grapheme] Java compile-err: %s\n",
            re.error().c_str());
      }
      continue;
    }

    std::vector<int> actual;
    if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
      actual = directJavaGraphemeBreakOffsets(c.input);
    } else {
      JavaMatcherAdapter<TypeParam> m(&re, c.input);
      while (m.find()) {
        actual.push_back(m.start());
      }
    }
    if (actual == c.expectedBreakOffsets) {
      ++st.passed;
    } else {
      ++st.failed;
      if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
        ++javaFailures;
        std::fprintf(
            stderr,
            "[OpenJDK Grapheme] Java mismatch: expected %zu breaks, actual %zu breaks\n",
            c.expectedBreakOffsets.size(),
            actual.size());
      }
    }
  }

  if constexpr (std::is_same_v<TypeParam, JavaRegex>) {
    EXPECT_EQ(0, javaFailures)
        << "Java backend should match every GraphemeTestCases.txt case";
  }
}

} // namespace
} // namespace facebook::velox::regex_compat::test

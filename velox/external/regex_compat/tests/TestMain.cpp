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
#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <string>

#if VELOX_REGEX_COMPAT_HAS_JAVA
#include "velox/external/regex_compat/JvmFixture.h"
#endif

namespace {

// Per-backend tally listener.  Counts test pass/fail by extracting the
// backend label from typed-test suite names like "MatchingPortedTest/0"
// (TypeParam = Re2Regex), "/1" = Pcre2Regex, "/2" = JavaRegex.  Aggregates
// across all typed tests so we can print a per-backend compatibility rate
// at the end of the run.
class PerBackendTallyListener : public ::testing::EmptyTestEventListener {
 public:
  void OnTestEnd(const ::testing::TestInfo& info) override {
    const std::string suite(info.test_suite_name());
    const std::string backend = extractBackend(suite);
    auto& t = tally_[backend];
    // Skipped tests are excluded from both numerator and denominator so
    // that "Java-API-only" GTEST_SKIP entries do not show up as Java
    // failures in the per-backend rate.
    if (info.result()->Skipped()) {
      ++t.skipped;
      return;
    }
    ++t.total;
    if (info.result()->Passed()) {
      ++t.passed;
    }
  }

  void OnTestProgramEnd(const ::testing::UnitTest& /*ut*/) override {
    std::cout << "\n========== Per-backend compatibility rate ==========\n";
    for (const auto& [name, t] : tally_) {
      const double pct = 100.0 * t.passed / std::max(t.total, 1);
      std::cout << "  " << name << "  " << t.passed << " / " << t.total
                << "  (" << pct << "%)";
      if (t.skipped > 0) {
        std::cout << "   [skipped: " << t.skipped << "]";
      }
      std::cout << "\n";
    }
    std::cout << "====================================================\n";

    // JavaRegex IS the ground truth — any failure means our port or JNI
    // bridge is wrong, not a real engine difference.  Loud-warn so it does
    // not get silently buried in the per-suite tally above.
    for (const auto& [name, t] : tally_) {
      if (name.find("Java") == std::string::npos) {
        continue;
      }
      if (t.passed != t.total) {
        std::cerr
            << "*** JavaRegex backend has " << (t.total - t.passed)
            << " failing test(s) in '" << name
            << "' — Java IS the canonical reference; failures here are"
            << " bugs in our port/JNI bridge, NOT real engine differences."
            << " Investigate or, after 5 unsuccessful fix attempts, mark"
            << " them as TODO for human review.\n";
      }
    }
  }

 private:
  struct Tally {
    int total = 0;
    int passed = 0;
    int skipped = 0;
  };
  std::map<std::string, Tally> tally_;

  static std::string extractBackend(const std::string& suite) {
    // Typed suites name themselves as "<Base>/0", "<Base>/1", "<Base>/2".
    // Anything without `/N` is a non-typed (backend-specific) suite — pass
    // its name through so it shows up explicitly in the report.
    const auto slash = suite.rfind('/');
    if (slash == std::string::npos) {
      return suite;
    }
    const std::string idx = suite.substr(slash + 1);
    if (idx == "0") return "Re2Regex  (typed)";
    if (idx == "1") return "Pcre2Regex (typed)";
    if (idx == "2") return "JavaRegex (typed)";
    return suite;
  }
};

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
#if VELOX_REGEX_COMPAT_HAS_JAVA
  facebook::velox::regex_compat::JvmFixture::Register();
#endif
  ::testing::UnitTest::GetInstance()->listeners().Append(
      new PerBackendTallyListener);
  return RUN_ALL_TESTS();
}

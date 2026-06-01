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
// Phase 1 (scaffolding) smoke test: verifies the library links and the
// identity passthrough is wired up.  Real test coverage arrives with
// Phases 2-5 (per-module unit tests) and Phase 5 (end-to-end pipeline
// tests ported from pcre4j's JavaRegexTranslatorTest.java).
//
#include "velox/functions/lib/java_pcre2_translator/JavaRegexTranslator.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

TEST(JavaRegexTranslator, scaffoldingIdentityPassthrough) {
  EXPECT_EQ("", toPcre2Pattern(""));
  EXPECT_EQ("abc", toPcre2Pattern("abc"));
  EXPECT_EQ("\\d+", toPcre2Pattern("\\d+"));
  EXPECT_EQ("(?i)hello", toPcre2Pattern("(?i)hello"));
}

TEST(JavaRegexTranslator, evaluationFailedExceptionIsConstructible) {
  EvaluationFailedException ex("nope");
  EXPECT_STREQ("nope", ex.what());
}

} // namespace facebook::velox::functions::java_pcre2_translator::test

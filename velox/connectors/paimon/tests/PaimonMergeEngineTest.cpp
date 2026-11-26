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
#include "velox/connectors/paimon/PaimonMergeEngine.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::connector::paimon;
using namespace facebook::velox;

TEST(PaimonMergeEngineTest, mergeEngineString) {
  EXPECT_EQ(
      paimonMergeEngineString(PaimonMergeEngine::kDeduplicate), "deduplicate");
  EXPECT_EQ(
      paimonMergeEngineString(PaimonMergeEngine::kPartialUpdate),
      "partial-update");
  EXPECT_EQ(
      paimonMergeEngineString(PaimonMergeEngine::kAggregation), "aggregation");
  EXPECT_EQ(paimonMergeEngineString(PaimonMergeEngine::kFirstRow), "first-row");
  EXPECT_EQ(fmt::format("{}", PaimonMergeEngine::kAggregation), "aggregation");
  VELOX_ASSERT_THROW(
      paimonMergeEngineString(static_cast<PaimonMergeEngine>(10)),
      "Unknown PaimonMergeEngine: 10");
}

TEST(PaimonMergeEngineTest, mergeEngineFromString) {
  for (auto mergeEngine :
       {PaimonMergeEngine::kDeduplicate,
        PaimonMergeEngine::kPartialUpdate,
        PaimonMergeEngine::kAggregation,
        PaimonMergeEngine::kFirstRow}) {
    SCOPED_TRACE(paimonMergeEngineString(mergeEngine));
    EXPECT_EQ(
        paimonMergeEngineFromString(paimonMergeEngineString(mergeEngine)),
        mergeEngine);
  }
  // Ignores capitalization.
  EXPECT_EQ(
      paimonMergeEngineFromString("First-Row"), PaimonMergeEngine::kFirstRow);
  EXPECT_EQ(
      paimonMergeEngineFromString("DEDUPLICATE"),
      PaimonMergeEngine::kDeduplicate);
  VELOX_ASSERT_USER_THROW(
      paimonMergeEngineFromString("first_row"),
      "Unknown Paimon merge engine: first_row");
  VELOX_ASSERT_USER_THROW(
      paimonMergeEngineFromString(""), "Unknown Paimon merge engine: ");
}

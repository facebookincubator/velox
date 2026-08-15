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
#include "velox/connectors/hive/paimon/PaimonRowKind.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <sstream>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::connector::hive::paimon;
using namespace facebook::velox;

TEST(PaimonRowKindTest, rowKindString) {
  EXPECT_EQ(paimonRowKindString(PaimonRowKind::kInsert), "+I");
  EXPECT_EQ(paimonRowKindString(PaimonRowKind::kUpdateBefore), "-U");
  EXPECT_EQ(paimonRowKindString(PaimonRowKind::kUpdateAfter), "+U");
  EXPECT_EQ(paimonRowKindString(PaimonRowKind::kDelete), "-D");
}

TEST(PaimonRowKindTest, rowKindFromValue) {
  EXPECT_EQ(paimonRowKindFromValue(0), PaimonRowKind::kInsert);
  EXPECT_EQ(paimonRowKindFromValue(1), PaimonRowKind::kUpdateBefore);
  EXPECT_EQ(paimonRowKindFromValue(2), PaimonRowKind::kUpdateAfter);
  EXPECT_EQ(paimonRowKindFromValue(3), PaimonRowKind::kDelete);

  VELOX_ASSERT_THROW(
      paimonRowKindFromValue(4), "Unknown PaimonRowKind value: 4");
  VELOX_ASSERT_THROW(
      paimonRowKindFromValue(-1), "Unknown PaimonRowKind value: -1");
}

TEST(PaimonRowKindTest, rowKindStreamAndFormat) {
  {
    std::ostringstream os;
    os << PaimonRowKind::kInsert;
    EXPECT_EQ(os.str(), "+I");
  }
  {
    std::ostringstream os;
    os << PaimonRowKind::kDelete;
    EXPECT_EQ(os.str(), "-D");
  }

  EXPECT_EQ(fmt::format("{}", PaimonRowKind::kInsert), "+I");
  EXPECT_EQ(fmt::format("{}", PaimonRowKind::kUpdateBefore), "-U");
  EXPECT_EQ(fmt::format("{}", PaimonRowKind::kUpdateAfter), "+U");
  EXPECT_EQ(fmt::format("{}", PaimonRowKind::kDelete), "-D");
}

TEST(PaimonRowKindTest, rowKindValues) {
  EXPECT_EQ(static_cast<int8_t>(PaimonRowKind::kInsert), 0);
  EXPECT_EQ(static_cast<int8_t>(PaimonRowKind::kUpdateBefore), 1);
  EXPECT_EQ(static_cast<int8_t>(PaimonRowKind::kUpdateAfter), 2);
  EXPECT_EQ(static_cast<int8_t>(PaimonRowKind::kDelete), 3);
}

TEST(PaimonRowKindTest, changelogModeStringAndParse) {
  EXPECT_EQ(paimonChangelogModeString(PaimonChangelogMode::kNone), "NONE");
  EXPECT_EQ(paimonChangelogModeString(PaimonChangelogMode::kInput), "INPUT");
  EXPECT_EQ(paimonChangelogModeString(PaimonChangelogMode::kLookup), "LOOKUP");
  EXPECT_EQ(
      paimonChangelogModeString(PaimonChangelogMode::kFullCompaction),
      "FULL_COMPACTION");

  EXPECT_EQ(paimonChangelogModeFromString("NONE"), PaimonChangelogMode::kNone);
  EXPECT_EQ(
      paimonChangelogModeFromString("INPUT"), PaimonChangelogMode::kInput);
  EXPECT_EQ(
      paimonChangelogModeFromString("LOOKUP"), PaimonChangelogMode::kLookup);
  EXPECT_EQ(
      paimonChangelogModeFromString("FULL_COMPACTION"),
      PaimonChangelogMode::kFullCompaction);

  VELOX_ASSERT_THROW(
      paimonChangelogModeFromString("UNKNOWN"),
      "Unknown PaimonChangelogMode: UNKNOWN");
}

TEST(PaimonRowKindTest, changelogModeStreamAndFormat) {
  {
    std::ostringstream os;
    os << PaimonChangelogMode::kNone;
    EXPECT_EQ(os.str(), "NONE");
  }
  {
    std::ostringstream os;
    os << PaimonChangelogMode::kLookup;
    EXPECT_EQ(os.str(), "LOOKUP");
  }

  EXPECT_EQ(fmt::format("{}", PaimonChangelogMode::kNone), "NONE");
  EXPECT_EQ(fmt::format("{}", PaimonChangelogMode::kInput), "INPUT");
  EXPECT_EQ(fmt::format("{}", PaimonChangelogMode::kLookup), "LOOKUP");
  EXPECT_EQ(
      fmt::format("{}", PaimonChangelogMode::kFullCompaction),
      "FULL_COMPACTION");
}

TEST(PaimonRowKindTest, rowKindColumnName) {
  EXPECT_EQ(kRowKindColumn, "_rowkind");
}

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

#include "velox/connectors/hive/iceberg/IcebergSessionFileReadOps.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/iceberg/IcebergSessionCredentials.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

std::shared_ptr<config::ConfigBase> makeConfig(
    std::unordered_map<std::string, std::string> values) {
  return std::make_shared<config::ConfigBase>(std::move(values));
}

TEST(IcebergSessionFileReadOpsTest, nullArgumentsReturnEmpty) {
  const auto session = makeConfig({{"tags", "team:a"}});
  const auto connector =
      makeConfig({{std::string(kSessionFileReadOpKeysConfig), "tags"}});
  EXPECT_THAT(sessionFileReadOps(nullptr, session.get()), IsEmpty());
  EXPECT_THAT(sessionFileReadOps(connector.get(), nullptr), IsEmpty());
}

TEST(IcebergSessionFileReadOpsTest, noKeyListReturnsEmpty) {
  const auto connector = makeConfig({});
  const auto session = makeConfig({{"tags", "team:a"}});
  EXPECT_THAT(sessionFileReadOps(connector.get(), session.get()), IsEmpty());
}

TEST(IcebergSessionFileReadOpsTest, emptyKeyListReturnsEmpty) {
  const auto connector =
      makeConfig({{std::string(kSessionFileReadOpKeysConfig), ""}});
  const auto session = makeConfig({{"tags", "team:a"}});
  EXPECT_THAT(sessionFileReadOps(connector.get(), session.get()), IsEmpty());
}

TEST(IcebergSessionFileReadOpsTest, collectsListedKeysWithValues) {
  const auto connector =
      makeConfig({{std::string(kSessionFileReadOpKeysConfig), "tags,trace"}});
  const auto session = makeConfig({{"tags", "team:a"}, {"trace", "t-1"}});
  EXPECT_THAT(
      sessionFileReadOps(connector.get(), session.get()),
      UnorderedElementsAre(Pair("tags", "team:a"), Pair("trace", "t-1")));
}

TEST(IcebergSessionFileReadOpsTest, skipsMissingAndEmptySessionValues) {
  const auto connector = makeConfig(
      {{std::string(kSessionFileReadOpKeysConfig), "present,missing,blank"}});
  // 'missing' has no session value; 'blank' has an empty one.
  const auto session = makeConfig({{"present", "team:a"}, {"blank", ""}});
  EXPECT_THAT(
      sessionFileReadOps(connector.get(), session.get()),
      UnorderedElementsAre(Pair("present", "team:a")));
}

TEST(IcebergSessionFileReadOpsTest, trimsKeysAndSkipsEmptyEntries) {
  const auto connector = makeConfig(
      {{std::string(kSessionFileReadOpKeysConfig), " tags , , trace "}});
  const auto session = makeConfig({{"tags", "team:a"}, {"trace", "t-1"}});
  EXPECT_THAT(
      sessionFileReadOps(connector.get(), session.get()),
      UnorderedElementsAre(Pair("tags", "team:a"), Pair("trace", "t-1")));
}

TEST(IcebergSessionFileReadOpsTest, ignoresCredentialKeyList) {
  const auto connector =
      makeConfig({{std::string(kSessionCredentialKeysConfig), "cat"}});
  const auto session = makeConfig({{"cat", "token"}});
  EXPECT_THAT(sessionFileReadOps(connector.get(), session.get()), IsEmpty());
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg

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

#include "velox/connectors/hive/iceberg/IcebergSessionCredentials.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

std::shared_ptr<config::ConfigBase> makeConfig(
    std::unordered_map<std::string, std::string> values) {
  return std::make_shared<config::ConfigBase>(std::move(values));
}

TEST(IcebergSessionCredentialsTest, nullArgumentsReturnEmpty) {
  const auto session = makeConfig({{"cat", "token"}});
  const auto connector =
      makeConfig({{std::string(kSessionCredentialKeysConfig), "cat"}});
  EXPECT_THAT(sessionCredentials(nullptr, session.get()), IsEmpty());
  EXPECT_THAT(sessionCredentials(connector.get(), nullptr), IsEmpty());
}

TEST(IcebergSessionCredentialsTest, noKeyListReturnsEmpty) {
  const auto connector = makeConfig({});
  const auto session = makeConfig({{"cat", "token"}});
  EXPECT_THAT(sessionCredentials(connector.get(), session.get()), IsEmpty());
}

TEST(IcebergSessionCredentialsTest, emptyKeyListReturnsEmpty) {
  const auto connector =
      makeConfig({{std::string(kSessionCredentialKeysConfig), ""}});
  const auto session = makeConfig({{"cat", "token"}});
  EXPECT_THAT(sessionCredentials(connector.get(), session.get()), IsEmpty());
}

TEST(IcebergSessionCredentialsTest, collectsListedKeysWithValues) {
  const auto connector =
      makeConfig({{std::string(kSessionCredentialKeysConfig), "cat,dcat"}});
  const auto session = makeConfig({{"cat", "token1"}, {"dcat", "token2"}});
  EXPECT_THAT(
      sessionCredentials(connector.get(), session.get()),
      UnorderedElementsAre(Pair("cat", "token1"), Pair("dcat", "token2")));
}

TEST(IcebergSessionCredentialsTest, skipsMissingAndEmptySessionValues) {
  const auto connector = makeConfig(
      {{std::string(kSessionCredentialKeysConfig), "present,missing,blank"}});
  // 'missing' has no session value; 'blank' has an empty one.
  const auto session = makeConfig({{"present", "token"}, {"blank", ""}});
  EXPECT_THAT(
      sessionCredentials(connector.get(), session.get()),
      UnorderedElementsAre(Pair("present", "token")));
}

TEST(IcebergSessionCredentialsTest, trimsKeysAndSkipsEmptyEntries) {
  const auto connector = makeConfig(
      {{std::string(kSessionCredentialKeysConfig), " cat , , dcat "}});
  const auto session = makeConfig({{"cat", "token1"}, {"dcat", "token2"}});
  EXPECT_THAT(
      sessionCredentials(connector.get(), session.get()),
      UnorderedElementsAre(Pair("cat", "token1"), Pair("dcat", "token2")));
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg

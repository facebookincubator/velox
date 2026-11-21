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

#include "velox/core/QueryConfigProvider.h"
#include <algorithm>

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"

using namespace facebook::velox::config;
using namespace facebook::velox::core;

namespace {

// Finds a property by name in a provider's materialized properties.
const ConfigProperty* findProperty(
    const std::vector<ConfigProperty>& properties,
    std::string_view name) {
  auto it = std::ranges::find(properties, name, &ConfigProperty::name);
  return it == properties.end() ? nullptr : &*it;
}

} // namespace

class QueryConfigProviderTest : public ::testing::Test {
 protected:
  QueryConfigProvider provider_;
};

TEST_F(QueryConfigProviderTest, propertiesNotEmpty) {
  auto props = provider_.properties();
  EXPECT_GT(props.size(), 130);
}

TEST_F(QueryConfigProviderTest, allNamesNonEmpty) {
  for (const auto& prop : provider_.properties()) {
    EXPECT_FALSE(prop.name.empty()) << "Found property with empty name";
    EXPECT_FALSE(prop.description.empty())
        << "Property " << prop.name << " has empty description";
  }
}

TEST_F(QueryConfigProviderTest, noDuplicateNames) {
  auto props = provider_.properties();
  std::set<std::string> names;
  for (const auto& prop : props) {
    EXPECT_TRUE(names.insert(prop.name).second)
        << "Duplicate property name: " << prop.name;
  }
}

TEST_F(QueryConfigProviderTest, knownProperties) {
  auto props = provider_.properties();

  // Check a boolean property.
  auto spillEnabled = findProperty(props, QueryConfig::kSpillEnabled);
  ASSERT_NE(spillEnabled, nullptr);
  EXPECT_EQ(spillEnabled->type, ConfigPropertyType::kBoolean);
  EXPECT_EQ(spillEnabled->defaultValue, "false");

  // Check a string property.
  auto sessionTz = findProperty(props, QueryConfig::kSessionTimezone);
  ASSERT_NE(sessionTz, nullptr);
  EXPECT_EQ(sessionTz->type, ConfigPropertyType::kString);
  EXPECT_EQ(sessionTz->defaultValue, "");

  // Check an integer property (macro-registered).
  auto startTime = findProperty(props, QueryConfig::kSessionStartTime);
  ASSERT_NE(startTime, nullptr);
  EXPECT_EQ(startTime->type, ConfigPropertyType::kInteger);
  EXPECT_EQ(startTime->defaultValue, "0");

  // Check a double property (macro-registered).
  auto cpuOverhead =
      findProperty(props, QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct);
  ASSERT_NE(cpuOverhead, nullptr);
  EXPECT_EQ(cpuOverhead->type, ConfigPropertyType::kDouble);
  EXPECT_EQ(cpuOverhead->defaultValue, "1");

  auto rpcRateLimiterMinLimit =
      findProperty(props, QueryConfig::kRpcRateLimiterMinLimit);
  ASSERT_NE(rpcRateLimiterMinLimit, nullptr);
  EXPECT_EQ(rpcRateLimiterMinLimit->type, ConfigPropertyType::kInteger);
  EXPECT_EQ(rpcRateLimiterMinLimit->defaultValue, "50");

  auto rpcRateLimiterMaxLimit =
      findProperty(props, QueryConfig::kRpcRateLimiterMaxLimit);
  ASSERT_NE(rpcRateLimiterMaxLimit, nullptr);
  EXPECT_EQ(rpcRateLimiterMaxLimit->type, ConfigPropertyType::kInteger);
  EXPECT_EQ(rpcRateLimiterMaxLimit->defaultValue, "200");

  auto rpcRateLimiterHardLimit =
      findProperty(props, QueryConfig::kRpcRateLimiterHardLimit);
  ASSERT_NE(rpcRateLimiterHardLimit, nullptr);
  EXPECT_EQ(rpcRateLimiterHardLimit->type, ConfigPropertyType::kInteger);
  EXPECT_EQ(rpcRateLimiterHardLimit->defaultValue, "-1");
}

TEST(QueryConfigProviderConfigOverridesTest, overridesApplied) {
  QueryConfigProvider provider(
      {{QueryConfig::kSpillEnabled, "true"},
       {QueryConfig::kSessionTimezone, "UTC"}});
  auto props = provider.properties();

  // Overridden properties reflect config-file values.
  auto spillEnabled = findProperty(props, QueryConfig::kSpillEnabled);
  ASSERT_NE(spillEnabled, nullptr);
  EXPECT_EQ(spillEnabled->defaultValue, "true");

  auto sessionTz = findProperty(props, QueryConfig::kSessionTimezone);
  ASSERT_NE(sessionTz, nullptr);
  EXPECT_EQ(sessionTz->defaultValue, "UTC");

  // Non-overridden properties retain code defaults.
  auto startTime = findProperty(props, QueryConfig::kSessionStartTime);
  ASSERT_NE(startTime, nullptr);
  EXPECT_EQ(startTime->defaultValue, "0");
}

TEST(QueryConfigProviderConfigOverridesTest, fixedValues) {
  // A config override equal to the fixed value is accepted.
  const auto provider = QueryConfigProvider::withFixedValues(
      /*configOverrides=*/
      {{QueryConfig::kSpillEnabled, "true"},
       {QueryConfig::kIgnoreTimeZoneLabel, "true"}},
      /*fixedValues=*/{{QueryConfig::kIgnoreTimeZoneLabel, "true"}});

  const auto properties = provider.properties();
  const auto fixed =
      findProperty(properties, QueryConfig::kIgnoreTimeZoneLabel);
  ASSERT_NE(fixed, nullptr);
  EXPECT_EQ(fixed->defaultValue, "true");
  EXPECT_FALSE(fixed->sessionOverridable);

  const auto sessionOverridable =
      findProperty(properties, QueryConfig::kSpillEnabled);
  ASSERT_NE(sessionOverridable, nullptr);
  EXPECT_EQ(sessionOverridable->defaultValue, "true");
  EXPECT_TRUE(sessionOverridable->sessionOverridable);
}

TEST(QueryConfigProviderConfigOverridesTest, fixedValueConflict) {
  VELOX_ASSERT_THROW(
      QueryConfigProvider::withFixedValues(
          /*configOverrides=*/{{QueryConfig::kIgnoreTimeZoneLabel, "false"}},
          /*fixedValues=*/{{QueryConfig::kIgnoreTimeZoneLabel, "true"}})
          .properties(),
      "Fixed query config value must match config override: "
      "ignore_time_zone_label");
}

TEST(QueryConfigProviderConfigOverridesTest, unknownFixedValue) {
  VELOX_ASSERT_THROW(
      QueryConfigProvider::withFixedValues(
          /*configOverrides=*/{}, /*fixedValues=*/{{"unknown", "true"}})
          .properties(),
      "Unknown fixed query config property: unknown");
}

TEST_F(QueryConfigProviderTest, normalizePassthrough) {
  EXPECT_EQ(provider_.normalize("spill_enabled", "true"), "true");
  EXPECT_EQ(provider_.normalize("session_timezone", "UTC"), "UTC");
}

TEST_F(QueryConfigProviderTest, configPropertyTypeNames) {
  EXPECT_EQ(
      ConfigPropertyTypeName::toName(ConfigPropertyType::kBoolean), "BOOLEAN");
  EXPECT_EQ(
      ConfigPropertyTypeName::toName(ConfigPropertyType::kInteger), "INTEGER");
  EXPECT_EQ(
      ConfigPropertyTypeName::toName(ConfigPropertyType::kDouble), "DOUBLE");
  EXPECT_EQ(
      ConfigPropertyTypeName::toName(ConfigPropertyType::kString), "STRING");

  EXPECT_EQ(
      ConfigPropertyTypeName::toConfigPropertyType("BOOLEAN"),
      ConfigPropertyType::kBoolean);
  EXPECT_EQ(
      ConfigPropertyTypeName::toConfigPropertyType("INTEGER"),
      ConfigPropertyType::kInteger);
}

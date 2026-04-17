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
#include <gtest/gtest.h>
#include "velox/core/QueryConfig.h"

using namespace facebook::velox::config;
using namespace facebook::velox::core;

class QueryConfigProviderTest : public ::testing::Test {
 protected:
  QueryConfigProvider provider_;
};

TEST_F(QueryConfigProviderTest, propertiesNotEmpty) {
  auto props = provider_.properties();
  EXPECT_GT(props.size(), 140);
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

  auto findProp =
      [&](const std::string& name) -> std::optional<ConfigProperty> {
    for (const auto& prop : props) {
      if (prop.name == name) {
        return prop;
      }
    }
    return std::nullopt;
  };

  // Check a boolean property.
  auto spillEnabled = findProp(QueryConfig::kSpillEnabled);
  ASSERT_TRUE(spillEnabled.has_value());
  EXPECT_EQ(spillEnabled->type, ConfigPropertyType::kBoolean);
  EXPECT_EQ(spillEnabled->defaultValue, "false");

  // Check a string property.
  auto sessionTz = findProp(QueryConfig::kSessionTimezone);
  ASSERT_TRUE(sessionTz.has_value());
  EXPECT_EQ(sessionTz->type, ConfigPropertyType::kString);
  EXPECT_EQ(sessionTz->defaultValue, "");

  // Check an integer property (macro-registered).
  auto startTime = findProp(QueryConfig::kSessionStartTime);
  ASSERT_TRUE(startTime.has_value());
  EXPECT_EQ(startTime->type, ConfigPropertyType::kInteger);
  EXPECT_EQ(startTime->defaultValue, "0");

  // Check a double property (macro-registered).
  auto cpuOverhead =
      findProp(QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct);
  ASSERT_TRUE(cpuOverhead.has_value());
  EXPECT_EQ(cpuOverhead->type, ConfigPropertyType::kDouble);
  EXPECT_EQ(cpuOverhead->defaultValue, "1");
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

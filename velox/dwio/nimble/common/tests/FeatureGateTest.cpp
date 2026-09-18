/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/FeatureGate.h"

#include <memory>
#include <string_view>

#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

// A gate that returns a fixed answer regardless of the requested default,
// modeling an internal gate that force-enables or gates-off a feature.
class FixedFeatureGate : public FeatureGate {
 public:
  explicit FixedFeatureGate(bool value) : value_{value} {}

  bool enabled(std::string_view /*feature*/, bool /*defaultValue*/)
      const override {
    return value_;
  }

 private:
  const bool value_;
};

class FeatureGateTest : public ::testing::Test {
 protected:
  // Restore the default no-op gate so tests don't leak process-wide state.
  void TearDown() override {
    registerFeatureGate(nullptr);
  }
};

TEST_F(FeatureGateTest, defaultGatePassesRequestedValueThrough) {
  const auto gate = featureGate();
  EXPECT_TRUE(gate->enabled("any_feature", true));
  EXPECT_FALSE(gate->enabled("any_feature", false));
}

TEST_F(FeatureGateTest, registeredGateOverridesDefault) {
  registerFeatureGate(std::make_shared<FixedFeatureGate>(false));
  EXPECT_FALSE(featureGate()->enabled("feature", true));

  registerFeatureGate(std::make_shared<FixedFeatureGate>(true));
  EXPECT_TRUE(featureGate()->enabled("feature", false));
}

TEST_F(FeatureGateTest, registerNullptrRestoresDefault) {
  registerFeatureGate(std::make_shared<FixedFeatureGate>(false));
  ASSERT_FALSE(featureGate()->enabled("feature", true));

  registerFeatureGate(nullptr);
  EXPECT_TRUE(featureGate()->enabled("feature", true));
  EXPECT_FALSE(featureGate()->enabled("feature", false));
}

TEST_F(FeatureGateTest, returnedGateOutlivesReRegistration) {
  // featureGate() hands out an owning pointer, so an already-fetched gate stays
  // valid (and unchanged) even after a different gate is registered.
  const auto original = featureGate();
  registerFeatureGate(std::make_shared<FixedFeatureGate>(false));
  EXPECT_TRUE(original->enabled("feature", true));
  EXPECT_FALSE(featureGate()->enabled("feature", true));
}

} // namespace
} // namespace facebook::nimble

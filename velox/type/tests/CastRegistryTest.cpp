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

#include "velox/type/CastRegistry.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {

class CastRulesRegistryTest : public testing::Test {
 protected:
  void SetUp() override {
    CastRulesRegistry::instance().clear();
  }

  void TearDown() override {
    CastRulesRegistry::instance().clear();
  }

  // Register rules directly on the registry, bypassing the free function's
  // CastOperator validation. Used for testing registry mechanics with
  // synthetic type names.
  void registerRules(const std::vector<CastRule>& rules) {
    CastRulesRegistry::instance().registerCastRules(rules);
  }
};

TEST_F(CastRulesRegistryTest, canCastAndCanCoerce) {
  registerRules({
      {.fromType = "BIGINT",
       .toType = "DOUBLE",
       .implicitAllowed = true,
       .validator = {}},
      {.fromType = "DOUBLE",
       .toType = "VARCHAR",
       .implicitAllowed = false,
       .validator = {}},
  });

  // Same type is always allowed.
  EXPECT_TRUE(CastRulesRegistry::instance().canCast(BIGINT(), BIGINT()));
  EXPECT_TRUE(CastRulesRegistry::instance().canCoerce(BIGINT(), BIGINT()));

  // Implicit rule: BIGINT -> DOUBLE allows both cast and coerce.
  EXPECT_TRUE(CastRulesRegistry::instance().canCast(BIGINT(), DOUBLE()));
  EXPECT_TRUE(CastRulesRegistry::instance().canCoerce(BIGINT(), DOUBLE()));

  // Explicit-only rule: DOUBLE -> VARCHAR allows cast but not coerce.
  EXPECT_TRUE(CastRulesRegistry::instance().canCast(DOUBLE(), VARCHAR()));
  EXPECT_FALSE(CastRulesRegistry::instance().canCoerce(DOUBLE(), VARCHAR()));

  // Unregistered: VARCHAR -> BIGINT.
  EXPECT_FALSE(CastRulesRegistry::instance().canCast(VARCHAR(), BIGINT()));
  EXPECT_FALSE(CastRulesRegistry::instance().canCoerce(VARCHAR(), BIGINT()));
}

TEST_F(CastRulesRegistryTest, unregisterRules) {
  registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = true,
       .validator = {}},
      {.fromType = "CUSTOM_TYPE",
       .toType = "VARCHAR",
       .implicitAllowed = false,
       .validator = {}},
  });

  unregisterCastRules("CUSTOM_TYPE");

  // After unregistering, re-registering with different flags should work
  // (no conflict because old rules were removed).
  EXPECT_NO_THROW(registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = false,
       .validator = {}},
  }));
}

TEST_F(CastRulesRegistryTest, duplicateRulesAllowedIfIdentical) {
  registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = true,
       .validator = {}},
  });

  // Identical rule should not throw.
  EXPECT_NO_THROW(registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = true,
       .validator = {}},
  }));
}

TEST_F(CastRulesRegistryTest, conflictingRulesThrow) {
  registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = true,
       .validator = {}},
  });

  // Different implicitAllowed flag should throw.
  EXPECT_THROW(
      registerRules({
          {.fromType = "BIGINT",
           .toType = "CUSTOM_TYPE",
           .implicitAllowed = false,
           .validator = {}},
      }),
      VeloxException);
}

TEST_F(CastRulesRegistryTest, parametricTypesReturnFalse) {
  // Parametric types (ARRAY, MAP, ROW, custom types with children like
  // IPPREFIX) are handled by callers recursing into children. The registry
  // only stores leaf-to-leaf rules, so parametric types return false.
  EXPECT_FALSE(
      CastRulesRegistry::instance().canCast(ARRAY(BIGINT()), ARRAY(DOUBLE())));
  EXPECT_FALSE(
      CastRulesRegistry::instance().canCoerce(
          ROW({BIGINT()}), ROW({DOUBLE()})));
}

TEST_F(CastRulesRegistryTest, clearRemovesAllRules) {
  registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = true,
       .validator = {}},
  });

  CastRulesRegistry::instance().clear();

  // After clear, the previously registered rule should no longer be found.
  // Re-registering with different flags should not conflict.
  EXPECT_NO_THROW(registerRules({
      {.fromType = "BIGINT",
       .toType = "CUSTOM_TYPE",
       .implicitAllowed = false,
       .validator = {}},
  }));
}

TEST_F(CastRulesRegistryTest, explicitOnlyCastNotCoercible) {
  // Register an explicit-only rule (implicitAllowed=false).
  registerRules({
      {.fromType = "VARCHAR",
       .toType = "BIGINT",
       .implicitAllowed = false,
       .validator = {}},
  });

  // Explicit cast is allowed.
  EXPECT_TRUE(CastRulesRegistry::instance().canCast(VARCHAR(), BIGINT()));

  // Implicit coercion is not allowed.
  EXPECT_FALSE(CastRulesRegistry::instance().canCoerce(VARCHAR(), BIGINT()));
}

TEST_F(CastRulesRegistryTest, validatorRejectsCast) {
  auto rejectAll = [](const TypePtr&, const TypePtr&) { return false; };

  registerRules({
      {.fromType = "BIGINT",
       .toType = "DOUBLE",
       .implicitAllowed = true,
       .validator = rejectAll},
  });

  // Rule exists but validator rejects — both canCast and canCoerce return
  // false.
  EXPECT_FALSE(CastRulesRegistry::instance().canCast(BIGINT(), DOUBLE()));
  EXPECT_FALSE(CastRulesRegistry::instance().canCoerce(BIGINT(), DOUBLE()));
}

TEST_F(CastRulesRegistryTest, decimalPrecisionValidator) {
  auto decimalWideningValidator = [](const TypePtr& from, const TypePtr& to) {
    if (!from->isDecimal() || !to->isDecimal()) {
      return false;
    }
    const auto [fromPrecision, fromScale] = getDecimalPrecisionScale(*from);
    const auto [toPrecision, toScale] = getDecimalPrecisionScale(*to);
    return toPrecision >= fromPrecision && toScale >= fromScale;
  };

  registerRules({
      {.fromType = "DECIMAL",
       .toType = "DECIMAL",
       .implicitAllowed = true,
       .validator = decimalWideningValidator},
  });

  // Widening: DECIMAL(10,2) -> DECIMAL(15,4).
  EXPECT_TRUE(
      CastRulesRegistry::instance().canCast(DECIMAL(10, 2), DECIMAL(15, 4)));
  EXPECT_TRUE(
      CastRulesRegistry::instance().canCoerce(DECIMAL(10, 2), DECIMAL(15, 4)));

  // Narrowing: DECIMAL(15,4) -> DECIMAL(10,2) — rejected by validator.
  EXPECT_FALSE(
      CastRulesRegistry::instance().canCast(DECIMAL(15, 4), DECIMAL(10, 2)));

  // Same type: always allowed (short-circuits before validator).
  EXPECT_TRUE(
      CastRulesRegistry::instance().canCast(DECIMAL(10, 2), DECIMAL(10, 2)));
}

TEST_F(CastRulesRegistryTest, standaloneRegistration) {
  registerRules({
      {.fromType = "TIMESTAMP",
       .toType = "CUSTOM_TZ",
       .implicitAllowed = true,
       .validator = {}},
      {.fromType = "CUSTOM_TZ",
       .toType = "VARCHAR",
       .implicitAllowed = false,
       .validator = {}},
  });

  // Verify rules were registered (test via unregister + re-register).
  unregisterCastRules("CUSTOM_TZ");

  // After unregister, can re-register with different flags.
  EXPECT_NO_THROW(registerRules({
      {.fromType = "TIMESTAMP",
       .toType = "CUSTOM_TZ",
       .implicitAllowed = false,
       .validator = {}},
  }));
}

TEST_F(CastRulesRegistryTest, rejectsRulesWithoutCastOperator) {
  // The free function registerCastRules() validates that at least one side
  // has a registered custom type with a CastOperator.
  EXPECT_THROW(
      registerCastRules({
          {.fromType = "BIGINT",
           .toType = "NONEXISTENT_TYPE",
           .implicitAllowed = true,
           .validator = {}},
      }),
      VeloxException);
}

} // namespace
} // namespace facebook::velox

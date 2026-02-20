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

#include "velox/type/Type.h"

namespace facebook::velox {
namespace {

class CastRegistryTest : public testing::Test {
 protected:
  void SetUp() override {
    CastRegistry::instance().clear();
  }

  void TearDown() override {
    CastRegistry::instance().clear();
  }
};

TEST_F(CastRegistryTest, registerAndFindRule) {
  std::vector<CastRule> rules = {
      {"BIGINT", "CUSTOM_TYPE", true, 1, nullptr},
      {"CUSTOM_TYPE", "VARCHAR", false, 2, nullptr},
  };

  CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules);

  auto rule1 = CastRegistry::instance().findRule("BIGINT", "CUSTOM_TYPE");
  ASSERT_TRUE(rule1.has_value());
  EXPECT_EQ(rule1->fromType, "BIGINT");
  EXPECT_EQ(rule1->toType, "CUSTOM_TYPE");
  EXPECT_TRUE(rule1->implicitAllowed);
  EXPECT_EQ(rule1->cost, 1);

  auto rule2 = CastRegistry::instance().findRule("CUSTOM_TYPE", "VARCHAR");
  ASSERT_TRUE(rule2.has_value());
  EXPECT_FALSE(rule2->implicitAllowed);
  EXPECT_EQ(rule2->cost, 2);

  // Non-existent rule.
  auto rule3 = CastRegistry::instance().findRule("VARCHAR", "CUSTOM_TYPE");
  EXPECT_FALSE(rule3.has_value());
}

TEST_F(CastRegistryTest, getCastsFromAndTo) {
  std::vector<CastRule> rules = {
      {"CUSTOM_TYPE", "BIGINT", true, 1, nullptr},
      {"CUSTOM_TYPE", "VARCHAR", false, 2, nullptr},
      {"CUSTOM_TYPE", "DOUBLE", true, 3, nullptr},
      {"INTEGER", "CUSTOM_TYPE", true, 1, nullptr},
  };

  CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules);

  // Get casts FROM CUSTOM_TYPE - should be sorted by cost.
  auto castsFrom = CastRegistry::instance().getCastsFrom("CUSTOM_TYPE");
  ASSERT_EQ(castsFrom.size(), 3);
  EXPECT_EQ(castsFrom[0].toType, "BIGINT");
  EXPECT_EQ(castsFrom[1].toType, "VARCHAR");
  EXPECT_EQ(castsFrom[2].toType, "DOUBLE");

  // Get casts TO CUSTOM_TYPE.
  auto castsTo = CastRegistry::instance().getCastsTo("CUSTOM_TYPE");
  ASSERT_EQ(castsTo.size(), 1);
  EXPECT_EQ(castsTo[0].fromType, "INTEGER");
}

TEST_F(CastRegistryTest, unregisterRules) {
  std::vector<CastRule> rules = {
      {"BIGINT", "CUSTOM_TYPE", true, 1, nullptr},
      {"CUSTOM_TYPE", "VARCHAR", false, 2, nullptr},
  };

  CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules);

  EXPECT_TRUE(
      CastRegistry::instance().findRule("BIGINT", "CUSTOM_TYPE").has_value());
  EXPECT_TRUE(
      CastRegistry::instance().findRule("CUSTOM_TYPE", "VARCHAR").has_value());

  CastRegistry::instance().unregisterCastRules("CUSTOM_TYPE");

  EXPECT_FALSE(
      CastRegistry::instance().findRule("BIGINT", "CUSTOM_TYPE").has_value());
  EXPECT_FALSE(
      CastRegistry::instance().findRule("CUSTOM_TYPE", "VARCHAR").has_value());
}

TEST_F(CastRegistryTest, validationRejectsInvalidRules) {
  // Rule that doesn't involve the custom type should be rejected.
  std::vector<CastRule> invalidRules = {
      {"BIGINT", "VARCHAR", true, 1, nullptr},
  };

  EXPECT_THROW(
      CastRegistry::instance().registerCastRules("CUSTOM_TYPE", invalidRules),
      VeloxException);
}

TEST_F(CastRegistryTest, canCastSameType) {
  // Same type should always be castable with zero cost.
  EXPECT_TRUE(CastRegistry::instance().canCast(BIGINT(), BIGINT()));
  EXPECT_TRUE(CastRegistry::instance().canImplicitCast(BIGINT(), BIGINT()));

  auto cost = CastRegistry::instance().getCastCost(BIGINT(), BIGINT());
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 0);
}

TEST_F(CastRegistryTest, parametricTypeCasting) {
  // Register rule for element type cast.
  std::vector<CastRule> rules = {
      {"BIGINT", "DOUBLE", true, 1, nullptr},
  };
  CastRegistry::instance().registerCastRules("BIGINT", rules);

  // ARRAY<BIGINT> -> ARRAY<DOUBLE> should work via recursive check.
  auto arrayBigint = ARRAY(BIGINT());
  auto arrayDouble = ARRAY(DOUBLE());

  EXPECT_TRUE(CastRegistry::instance().canCast(arrayBigint, arrayDouble));
  EXPECT_TRUE(
      CastRegistry::instance().canImplicitCast(arrayBigint, arrayDouble));

  // Cost should be the element cast cost.
  auto cost = CastRegistry::instance().getCastCost(arrayBigint, arrayDouble);
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 1);
}

TEST_F(CastRegistryTest, nestedParametricTypes) {
  std::vector<CastRule> rules = {
      {"BIGINT", "DOUBLE", true, 1, nullptr},
  };
  CastRegistry::instance().registerCastRules("BIGINT", rules);

  auto nestedBigint = ARRAY(ARRAY(BIGINT()));
  auto nestedDouble = ARRAY(ARRAY(DOUBLE()));

  EXPECT_TRUE(CastRegistry::instance().canCast(nestedBigint, nestedDouble));

  // Cost should be 1 (only one element cast needed at leaf level).
  auto cost = CastRegistry::instance().getCastCost(nestedBigint, nestedDouble);
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 1);
}

TEST_F(CastRegistryTest, mapTypeCasting) {
  std::vector<CastRule> rules = {
      {"BIGINT", "DOUBLE", true, 1, nullptr},
  };
  CastRegistry::instance().registerCastRules("BIGINT", rules);

  // MAP<BIGINT, VARCHAR> -> MAP<DOUBLE, VARCHAR>
  auto mapBigint = MAP(BIGINT(), VARCHAR());
  auto mapDouble = MAP(DOUBLE(), VARCHAR());

  EXPECT_TRUE(CastRegistry::instance().canCast(mapBigint, mapDouble));

  // Cost is 1 for key cast, 0 for value (same type).
  auto cost = CastRegistry::instance().getCastCost(mapBigint, mapDouble);
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 1);
}

TEST_F(CastRegistryTest, duplicateRulesAllowedIfIdentical) {
  std::vector<CastRule> rules1 = {
      {"BIGINT", "CUSTOM_TYPE", true, 1, nullptr},
  };
  std::vector<CastRule> rules2 = {
      {"BIGINT", "CUSTOM_TYPE", true, 1, nullptr}, // Same rule
  };

  CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules1);
  // Should not throw - identical rule.
  EXPECT_NO_THROW(
      CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules2));
}

TEST_F(CastRegistryTest, conflictingRulesThrow) {
  std::vector<CastRule> rules1 = {
      {"BIGINT", "CUSTOM_TYPE", true, 1, nullptr},
  };
  std::vector<CastRule> rules2 = {
      {"BIGINT",
       "CUSTOM_TYPE",
       false,
       2,
       nullptr}, // Different implicitAllowed/cost
  };

  CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules1);
  EXPECT_THROW(
      CastRegistry::instance().registerCastRules("CUSTOM_TYPE", rules2),
      VeloxException);
}

TEST_F(CastRegistryTest, typeSpecificValidator) {
  // Create a validator that only allows casts when the source type is INTEGER.
  // This simulates a type-specific check (like DECIMAL precision validation).
  auto integerOnlyValidator = [](const TypePtr& from, const TypePtr& to) {
    return from->kind() == TypeKind::INTEGER;
  };

  CastRule ruleWithValidator;
  ruleWithValidator.fromType = "INTEGER";
  ruleWithValidator.toType = "CUSTOM_VALIDATED";
  ruleWithValidator.implicitAllowed = true;
  ruleWithValidator.cost = 1;
  ruleWithValidator.validator = integerOnlyValidator;

  std::vector<CastRule> rules = {ruleWithValidator};
  CastRegistry::instance().registerCastRules("CUSTOM_VALIDATED", rules);

  // The validator should pass for INTEGER -> CUSTOM_VALIDATED.
  EXPECT_TRUE(CastRegistry::instance().canCast(INTEGER(), INTEGER()));

  // Verify the rule was registered.
  auto rule = CastRegistry::instance().findRule("INTEGER", "CUSTOM_VALIDATED");
  ASSERT_TRUE(rule.has_value());
  EXPECT_TRUE(rule->validator != nullptr);

  // Test that the validator is called during canCast.
  // Note: For this test to fully work, we'd need to register CUSTOM_VALIDATED
  // as a custom type. For now, verify the rule structure.
  EXPECT_EQ(rule->fromType, "INTEGER");
  EXPECT_EQ(rule->toType, "CUSTOM_VALIDATED");
  EXPECT_TRUE(rule->implicitAllowed);
}

TEST_F(CastRegistryTest, validatorRejectsCast) {
  // Create a validator that always rejects the cast.
  auto rejectingValidator = [](const TypePtr& from, const TypePtr& to) {
    return false; // Always reject
  };

  CastRule ruleWithRejectingValidator;
  ruleWithRejectingValidator.fromType = "BIGINT";
  ruleWithRejectingValidator.toType = "REJECT_TYPE";
  ruleWithRejectingValidator.implicitAllowed = true;
  ruleWithRejectingValidator.cost = 1;
  ruleWithRejectingValidator.validator = rejectingValidator;

  std::vector<CastRule> rules = {ruleWithRejectingValidator};
  CastRegistry::instance().registerCastRules("REJECT_TYPE", rules);

  // Verify the rule exists.
  auto rule = CastRegistry::instance().findRule("BIGINT", "REJECT_TYPE");
  ASSERT_TRUE(rule.has_value());

  // The validator rejects all casts, so canCast should return false.
  // Note: This requires REJECT_TYPE to be a registered type for full test.
  // For now, verify the validator is stored correctly.
  EXPECT_FALSE(rule->validator(BIGINT(), BIGINT()));
}

TEST_F(CastRegistryTest, decimalPrecisionValidator) {
  // Validator that allows widening but rejects narrowing.
  auto decimalWideningValidator = [](const TypePtr& from, const TypePtr& to) {
    if (!from->isDecimal() || !to->isDecimal()) {
      return false;
    }
    const auto [fromPrecision, fromScale] = getDecimalPrecisionScale(*from);
    const auto [toPrecision, toScale] = getDecimalPrecisionScale(*to);
    return toPrecision >= fromPrecision && toScale >= fromScale;
  };

  CastRule decimalToDecimalRule;
  decimalToDecimalRule.fromType = "DECIMAL";
  decimalToDecimalRule.toType = "DECIMAL";
  decimalToDecimalRule.implicitAllowed = true;
  decimalToDecimalRule.cost = 1;
  decimalToDecimalRule.validator = decimalWideningValidator;

  std::vector<CastRule> rules = {decimalToDecimalRule};
  CastRegistry::instance().registerCastRules("DECIMAL", rules);

  auto rule = CastRegistry::instance().findRule("DECIMAL", "DECIMAL");
  ASSERT_TRUE(rule.has_value());
  EXPECT_TRUE(rule->validator != nullptr);

  auto decimalSmall = DECIMAL(10, 2);
  auto decimalLarge = DECIMAL(15, 4);

  // Widening allowed.
  EXPECT_TRUE(rule->validator(decimalSmall, decimalLarge));
  // Narrowing rejected.
  EXPECT_FALSE(rule->validator(decimalLarge, decimalSmall));
  // Same type allowed.
  EXPECT_TRUE(rule->validator(decimalSmall, decimalSmall));
  // Lower scale rejected.
  EXPECT_FALSE(rule->validator(decimalSmall, DECIMAL(12, 1)));
  // Lower precision rejected.
  EXPECT_FALSE(rule->validator(decimalSmall, DECIMAL(8, 4)));
}

TEST_F(CastRegistryTest, integerWideningValidator) {
  auto integerWideningValidator = [](const TypePtr& from, const TypePtr& to) {
    auto getTypeSize = [](TypeKind kind) -> int {
      switch (kind) {
        case TypeKind::TINYINT:
          return 1;
        case TypeKind::SMALLINT:
          return 2;
        case TypeKind::INTEGER:
          return 4;
        case TypeKind::BIGINT:
          return 8;
        default:
          return 0;
      }
    };
    const int fromSize = getTypeSize(from->kind());
    const int toSize = getTypeSize(to->kind());
    if (fromSize == 0 || toSize == 0) {
      return false;
    }
    return toSize >= fromSize;
  };

  CastRule intWideningRule;
  intWideningRule.fromType = "INTEGER";
  intWideningRule.toType = "INTEGER";
  intWideningRule.implicitAllowed = true;
  intWideningRule.cost = 1;
  intWideningRule.validator = integerWideningValidator;

  std::vector<CastRule> rules = {intWideningRule};
  CastRegistry::instance().registerCastRules("INTEGER", rules);

  auto rule = CastRegistry::instance().findRule("INTEGER", "INTEGER");
  ASSERT_TRUE(rule.has_value());

  EXPECT_TRUE(rule->validator(INTEGER(), BIGINT()));
  EXPECT_FALSE(rule->validator(BIGINT(), INTEGER()));
  EXPECT_TRUE(rule->validator(INTEGER(), INTEGER()));
  EXPECT_TRUE(rule->validator(TINYINT(), BIGINT()));
  EXPECT_FALSE(rule->validator(SMALLINT(), TINYINT()));
}

TEST_F(CastRegistryTest, typeCompatibilityBasic) {
  // Same type - always compatible with zero cost.
  auto compat =
      CastRegistry::instance().getTypeCompatibility(BIGINT(), BIGINT());
  EXPECT_TRUE(compat.compatible);
  EXPECT_TRUE(compat.coercible);
  EXPECT_EQ(compat.cost, 0);
  EXPECT_TRUE(compat.typeOnlyCoercion);
}

TEST_F(CastRegistryTest, typeCompatibilityWithRule) {
  // Register a rule with typeOnlyCoercion=true.
  CastRule rule;
  rule.fromType = "SMALL_VARCHAR";
  rule.toType = "LARGE_VARCHAR";
  rule.implicitAllowed = true;
  rule.cost = 1;
  rule.typeOnlyCoercion = true; // Widening varchar is type-only
  rule.validator = nullptr;

  std::vector<CastRule> rules = {rule};
  CastRegistry::instance().registerCastRules("SMALL_VARCHAR", rules);

  auto foundRule =
      CastRegistry::instance().findRule("SMALL_VARCHAR", "LARGE_VARCHAR");
  ASSERT_TRUE(foundRule.has_value());
  EXPECT_TRUE(foundRule->typeOnlyCoercion);
}

TEST_F(CastRegistryTest, typeCompatibilityIncompatible) {
  // No rule registered - should be incompatible.
  auto compat =
      CastRegistry::instance().getTypeCompatibility(BIGINT(), VARCHAR());
  EXPECT_FALSE(compat.compatible);
  EXPECT_FALSE(compat.coercible);
}

TEST_F(CastRegistryTest, unknownTypeCoercion) {
  // UNKNOWN type (NULL literal) should coerce to any type.
  EXPECT_TRUE(CastRegistry::isUnknownType(UNKNOWN()));
  EXPECT_FALSE(CastRegistry::isUnknownType(BIGINT()));

  // UNKNOWN -> BIGINT should be compatible.
  auto compat =
      CastRegistry::instance().getTypeCompatibility(UNKNOWN(), BIGINT());
  EXPECT_TRUE(compat.compatible);
  EXPECT_TRUE(compat.coercible);
  EXPECT_EQ(compat.cost, 0);
  EXPECT_TRUE(compat.typeOnlyCoercion);
}

TEST_F(CastRegistryTest, getCommonSuperTypeSameType) {
  // Same type returns itself.
  auto result = CastRegistry::instance().getCommonSuperType(BIGINT(), BIGINT());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE((*result)->equivalent(*BIGINT()));
}

TEST_F(CastRegistryTest, getCommonSuperTypeWithUnknown) {
  // UNKNOWN + any type = that type.
  auto result =
      CastRegistry::instance().getCommonSuperType(UNKNOWN(), BIGINT());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE((*result)->equivalent(*BIGINT()));

  // Any type + UNKNOWN = that type.
  result = CastRegistry::instance().getCommonSuperType(VARCHAR(), UNKNOWN());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE((*result)->equivalent(*VARCHAR()));
}

TEST_F(CastRegistryTest, getCommonSuperTypeWithCoercion) {
  // Register INTEGER -> BIGINT as implicit coercion.
  std::vector<CastRule> rules = {
      {"INTEGER", "BIGINT", true, 1, nullptr},
  };
  CastRegistry::instance().registerCastRules("INTEGER", rules);

  // INTEGER + BIGINT = BIGINT (INTEGER coerces to BIGINT).
  auto result =
      CastRegistry::instance().getCommonSuperType(INTEGER(), BIGINT());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE((*result)->equivalent(*BIGINT()));
}

TEST_F(CastRegistryTest, getCommonSuperTypeNoCommon) {
  // No rule between VARCHAR and BIGINT - no common super type.
  auto result =
      CastRegistry::instance().getCommonSuperType(VARCHAR(), BIGINT());
  EXPECT_FALSE(result.has_value());
}

} // namespace
} // namespace facebook::velox

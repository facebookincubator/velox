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

#include "velox/expression/ExprToSubfieldFilter.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/Expr.h"
#include "velox/functions/lib/IsNull.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

using facebook::velox::common::Subfield;

namespace facebook::velox::exec {
namespace {

void validateSubfield(
    const Subfield& subfield,
    const std::vector<std::string>& expectedPath) {
  ASSERT_EQ(subfield.path().size(), expectedPath.size());
  for (int i = 0; i < expectedPath.size(); ++i) {
    ASSERT_TRUE(subfield.path()[i]);
    ASSERT_EQ(*subfield.path()[i], Subfield::NestedField(expectedPath[i]));
  }
}

class ExprToSubfieldFilterTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  core::TypedExprPtr parseExpr(
      const std::string& expr,
      const RowTypePtr& type) {
    return core::Expressions::inferTypes(
        parse::parseExpr(expr, {}), type, pool_.get());
  }

  core::CallTypedExprPtr parseCallExpr(
      const std::string& expr,
      const RowTypePtr& type) {
    auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(
        parseExpr(expr, type));
    VELOX_CHECK_NOT_NULL(call);
    return call;
  }

  core::ExpressionEvaluator* evaluator() {
    return &evaluator_;
  }

  std::pair<common::Subfield, std::unique_ptr<common::Filter>>
  leafCallToSubfieldFilter(const core::CallTypedExprPtr& call) {
    if (auto result =
            ExprToSubfieldFilterParser::getInstance()->leafCallToSubfieldFilter(
                *call, evaluator())) {
      return std::move(result.value());
    }

    return std::make_pair(common::Subfield(), nullptr);
  }

  std::pair<common::Subfield, std::unique_ptr<common::Filter>> toSubfieldFilter(
      const core::TypedExprPtr& expr) {
    return ExprToSubfieldFilterParser::getInstance()->toSubfieldFilter(
        expr, evaluator());
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  SimpleExpressionEvaluator evaluator_{queryCtx_.get(), pool_.get()};
};

TEST_F(ExprToSubfieldFilterTest, eq) {
  auto call = parseCallExpr("a = 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  auto bigintRange = dynamic_cast<common::BigintRange*>(filter.get());
  ASSERT_TRUE(bigintRange);
  ASSERT_EQ(bigintRange->lower(), 42);
  ASSERT_EQ(bigintRange->upper(), 42);
  ASSERT_FALSE(bigintRange->testNull());
}

TEST_F(ExprToSubfieldFilterTest, eqExpr) {
  auto call = parseCallExpr("a = 21 * 2", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  auto bigintRange = dynamic_cast<common::BigintRange*>(filter.get());
  ASSERT_TRUE(bigintRange);
  ASSERT_EQ(bigintRange->lower(), 42);
  ASSERT_EQ(bigintRange->upper(), 42);
  ASSERT_FALSE(bigintRange->testNull());
}

TEST_F(ExprToSubfieldFilterTest, eqSubfield) {
  auto call = parseCallExpr("a.b = 42", ROW({{"a", ROW({{"b", BIGINT()}})}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a", "b"});
  auto bigintRange = dynamic_cast<common::BigintRange*>(filter.get());
  ASSERT_TRUE(bigintRange);
  ASSERT_EQ(bigintRange->lower(), 42);
  ASSERT_EQ(bigintRange->upper(), 42);
  ASSERT_FALSE(bigintRange->testNull());
}

TEST_F(ExprToSubfieldFilterTest, neq) {
  auto call = parseCallExpr("a <> 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_TRUE(filter->testInt64(41));
  ASSERT_FALSE(filter->testInt64(42));
  ASSERT_TRUE(filter->testInt64(43));
}

TEST_F(ExprToSubfieldFilterTest, lte) {
  auto call = parseCallExpr("a <= 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_TRUE(filter->testInt64(41));
  ASSERT_TRUE(filter->testInt64(42));
  ASSERT_FALSE(filter->testInt64(43));
}

TEST_F(ExprToSubfieldFilterTest, lt) {
  auto call = parseCallExpr("a < 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_TRUE(filter->testInt64(41));
  ASSERT_FALSE(filter->testInt64(42));
  ASSERT_FALSE(filter->testInt64(43));
}

TEST_F(ExprToSubfieldFilterTest, gte) {
  auto call = parseCallExpr("a >= 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_FALSE(filter->testInt64(41));
  ASSERT_TRUE(filter->testInt64(42));
  ASSERT_TRUE(filter->testInt64(43));
}

TEST_F(ExprToSubfieldFilterTest, gt) {
  auto call = parseCallExpr("a > 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_FALSE(filter->testInt64(41));
  ASSERT_FALSE(filter->testInt64(42));
  ASSERT_TRUE(filter->testInt64(43));
}

TEST_F(ExprToSubfieldFilterTest, between) {
  auto call = parseCallExpr("a between 40 and 42", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  for (int i = 39; i <= 43; ++i) {
    ASSERT_EQ(filter->testInt64(i), 40 <= i && i <= 42);
  }
}

TEST_F(ExprToSubfieldFilterTest, in) {
  auto call = parseCallExpr("a in (40, 42)", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  for (int i = 39; i <= 43; ++i) {
    ASSERT_EQ(filter->testInt64(i), i == 40 || i == 42);
  }
}

TEST_F(ExprToSubfieldFilterTest, isNull) {
  auto call = parseCallExpr("a is null", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_FALSE(filter->testInt64(0));
  ASSERT_FALSE(filter->testInt64(42));
  ASSERT_TRUE(filter->testNull());
}

TEST_F(ExprToSubfieldFilterTest, isNotNull) {
  auto call = parseCallExpr("a is not null", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = toSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_TRUE(filter->testInt64(0));
  ASSERT_TRUE(filter->testInt64(42));
  ASSERT_FALSE(filter->testNull());
}

TEST_F(ExprToSubfieldFilterTest, or) {
  auto call = parseExpr("a = 42 OR a = 43", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = toSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_TRUE(filter->testInt64(42));
  ASSERT_TRUE(filter->testInt64(43));
  ASSERT_FALSE(filter->testInt64(41));
  ASSERT_FALSE(filter->testNull());
}

TEST_F(ExprToSubfieldFilterTest, unsupported) {
  const auto type = ROW({"a", "b", "c"}, BIGINT());

  VELOX_ASSERT_THROW(
      toSubfieldFilter(parseExpr("123", type)),
      "Unsupported expression for range filter");

  VELOX_ASSERT_THROW(
      toSubfieldFilter(parseCallExpr("a + b", type)),
      "Unsupported expression for range filter");

  VELOX_ASSERT_THROW(
      toSubfieldFilter(parseCallExpr("(a + b > 0) OR (c = 1)", type)),
      "Unsupported expression for range filter");

  // TODO Improve error message for this specific use case. Then use
  // VELOX_ASSERT_THROW.
  EXPECT_THROW(
      toSubfieldFilter(parseCallExpr("a = 1 OR c = 2", type)),
      VeloxRuntimeError);

  VELOX_ASSERT_THROW(
      toSubfieldFilter(parseCallExpr("a = 1 AND b = 2", type)),
      "Unsupported expression for range filter");
}

TEST_F(ExprToSubfieldFilterTest, like) {
  auto call = parseCallExpr("a like 'foo%'", ROW({{"a", VARCHAR()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, nonConstant) {
  auto call =
      parseCallExpr("a = b + 1", ROW({{"a", BIGINT()}, {"b", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, userError) {
  auto call = parseCallExpr("a = 1 / 0", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, dereferenceWithEmptyField) {
  auto call = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "is_null",
      std::make_shared<core::DereferenceTypedExpr>(
          REAL(),
          std::make_shared<core::FieldAccessTypedExpr>(
              ROW({{"", DOUBLE()}, {"", REAL()}, {"", BIGINT()}}),
              std::make_shared<core::InputTypedExpr>(ROW(
                  {{"c0",
                    ROW({{"", DOUBLE()}, {"", REAL()}, {"", BIGINT()}})}})),
              "c0"),
          1));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

class CustomExprToSubfieldFilterParser : public ExprToSubfieldFilterParser {
 public:
  std::pair<common::Subfield, std::unique_ptr<common::Filter>> toSubfieldFilter(
      const core::TypedExprPtr& expr,
      core::ExpressionEvaluator* evaluator) override {
    if (expr->isCallKind();
        auto* call = expr->asUnchecked<core::CallTypedExpr>()) {
      if (call->name() == "or") {
        auto left = toSubfieldFilter(call->inputs()[0], evaluator);
        auto right = toSubfieldFilter(call->inputs()[1], evaluator);
        VELOX_CHECK(left.first == right.first);
        return {
            std::move(left.first),
            makeOrFilter(std::move(left.second), std::move(right.second))};
      }
      if (call->name() == "not") {
        if (auto* inner =
                call->inputs()[0]->asUnchecked<core::CallTypedExpr>()) {
          if (auto result = leafCallToSubfieldFilter(*inner, evaluator, true)) {
            return std::move(result.value());
          }
        }
      } else {
        if (auto result = leafCallToSubfieldFilter(*call, evaluator, false)) {
          return std::move(result.value());
        }
      }
    }
    VELOX_UNSUPPORTED(
        "Unsupported expression for range filter: {}", expr->toString());
  }

  std::optional<std::pair<common::Subfield, std::unique_ptr<common::Filter>>>
  leafCallToSubfieldFilter(
      const core::CallTypedExpr& call,
      core::ExpressionEvaluator* evaluator,
      bool negated) override {
    if (call.inputs().empty()) {
      return std::nullopt;
    }

    const auto* leftSide = call.inputs()[0].get();

    common::Subfield subfield;
    if (call.name() == "custom_eq") {
      if (toSubfield(leftSide, subfield)) {
        auto filter = negated ? makeNotEqualFilter(call.inputs()[1], evaluator)
                              : makeEqualFilter(call.inputs()[1], evaluator);
        if (filter != nullptr) {
          return std::make_pair(std::move(subfield), std::move(filter));
        }
        return std::nullopt;
      }
    } else if (call.name() == "is_null") {
      if (toSubfield(call.inputs()[0].get(), subfield)) {
        if (negated) {
          return std::make_pair(std::move(subfield), isNotNull());
        }
        return std::make_pair(std::move(subfield), isNull());
      }
    }
    return std::nullopt;
  }
};

class CustomExprToSubfieldFilterTest : public ExprToSubfieldFilterTest {
 public:
  static void SetUpTestSuite() {
    functions::prestosql::registerAllScalarFunctions("custom_");
    functions::registerIsNullFunction("is_null");
    parse::registerTypeResolver();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    ExprToSubfieldFilterParser::registerParser(
        std::make_unique<CustomExprToSubfieldFilterParser>());
  }

  static void TearDownTestSuite() {
    ExprToSubfieldFilterParser::registerParser(
        std::make_unique<PrestoExprToSubfieldFilterParser>());
  }
};

TEST_F(CustomExprToSubfieldFilterTest, isNull) {
  auto call = parseCallExpr("a is null", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = toSubfieldFilter(call);
  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  ASSERT_FALSE(filter->testInt64(0));
  ASSERT_FALSE(filter->testInt64(42));
  ASSERT_TRUE(filter->testNull());
}

TEST_F(CustomExprToSubfieldFilterTest, eq) {
  auto call = parseCallExpr("custom_eq(a, 42)", ROW({{"a", BIGINT()}}));
  auto [subfield, filter] = toSubfieldFilter(call);
  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});
  auto bigintRange = dynamic_cast<common::BigintRange*>(filter.get());
  ASSERT_TRUE(bigintRange);
  ASSERT_EQ(bigintRange->lower(), 42);
  ASSERT_EQ(bigintRange->upper(), 42);
  ASSERT_FALSE(bigintRange->testNull());
}

TEST_F(CustomExprToSubfieldFilterTest, unsupported) {
  auto call = parseCallExpr("custom_neq(a, 42)", ROW({{"a", BIGINT()}}));
  VELOX_ASSERT_USER_THROW(
      toSubfieldFilter(call), "Unsupported expression for range filter");
}

} // namespace
} // namespace facebook::velox::exec

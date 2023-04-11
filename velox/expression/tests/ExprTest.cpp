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

#include <exception>
#include <fstream>
#include <stdexcept>
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "velox/expression/Expr.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/ConjunctExpr.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/tests/TestingAlwaysThrowsFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class ExprTest : public testing::Test, public VectorTestBase {
 protected:
  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
  }

  core::TypedExprPtr parseExpression(
      const std::string& text,
      const RowTypePtr& rowType) {
    auto untyped = parse::parseExpr(text, options_);
    return core::Expressions::inferTypes(untyped, rowType, execCtx_->pool());
  }

  template <typename T = exec::ExprSet>
  std::unique_ptr<T> compileExpression(
      const std::string& expr,
      const RowTypePtr& rowType) {
    std::vector<core::TypedExprPtr> expressions = {
        parseExpression(expr, rowType)};
    return std::make_unique<T>(std::move(expressions), execCtx_.get());
  }

  std::unique_ptr<exec::ExprSet> compileMultiple(
      const std::vector<std::string>& texts,
      const RowTypePtr& rowType) {
    std::vector<core::TypedExprPtr> expressions;
    expressions.reserve(texts.size());
    for (const auto& text : texts) {
      expressions.emplace_back(parseExpression(text, rowType));
    }
    return std::make_unique<exec::ExprSet>(
        std::move(expressions), execCtx_.get());
  }

  std::vector<VectorPtr> evaluateMultiple(
      const std::vector<std::string>& texts,
      const RowVectorPtr& input) {
    auto exprSet = compileMultiple(texts, asRowType(input->type()));

    exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

    SelectivityVector rows(input->size());
    std::vector<VectorPtr> result(texts.size());
    exprSet->eval(rows, context, result);
    return result;
  }

  std::pair<
      std::vector<VectorPtr>,
      std::unordered_map<std::string, exec::ExprStats>>
  evaluateMultipleWithStats(
      const std::vector<std::string>& texts,
      const RowVectorPtr& input,
      std::vector<VectorPtr> resultToReuse = {}) {
    auto exprSet = compileMultiple(texts, asRowType(input->type()));

    exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

    SelectivityVector rows(input->size());
    if (resultToReuse.empty()) {
      resultToReuse.resize(texts.size());
    }
    exprSet->eval(rows, context, resultToReuse);
    return {resultToReuse, exprSet->stats()};
  }

  VectorPtr evaluate(const std::string& text, const RowVectorPtr& input) {
    return evaluateMultiple({text}, input)[0];
  }

  std::pair<VectorPtr, std::unordered_map<std::string, exec::ExprStats>>
  evaluateWithStats(const std::string& expression, const RowVectorPtr& input) {
    auto exprSet = compileExpression(expression, asRowType(input->type()));

    SelectivityVector rows(input->size());
    std::vector<VectorPtr> results(1);

    exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());
    exprSet->eval(rows, context, results);

    return {results[0], exprSet->stats()};
  }

  template <
      typename T = exec::ExprSet,
      typename = std::enable_if_t<
          std::is_same_v<T, exec::ExprSet> ||
              std::is_same_v<T, exec::ExprSetSimplified>,
          bool>>
  VectorPtr evaluate(T* exprSet, const RowVectorPtr& input) {
    exec::EvalCtx context(execCtx_.get(), exprSet, input.get());

    SelectivityVector rows(input->size());
    std::vector<VectorPtr> result(1);
    exprSet->eval(rows, context, result);
    return result[0];
  }

  template <typename T = ComplexType>
  std::shared_ptr<core::ConstantTypedExpr> makeConstantExpr(
      const VectorPtr& base,
      vector_size_t index) {
    return std::make_shared<core::ConstantTypedExpr>(
        BaseVector::wrapInConstant(1, index, base));
  }

  /// Create constant expression from a variant of primitive type.
  std::shared_ptr<core::ConstantTypedExpr> makeConstantExpr(
      variant value,
      const TypePtr& type = nullptr) {
    auto valueType = type != nullptr ? type : value.inferType();
    return std::make_shared<core::ConstantTypedExpr>(
        valueType, std::move(value));
  }

  // Create LazyVector that produces a flat vector and asserts that is is being
  // loaded for a specific set of rows.
  template <typename T>
  std::shared_ptr<LazyVector> makeLazyFlatVector(
      vector_size_t size,
      std::function<T(vector_size_t /*row*/)> valueAt,
      std::function<bool(vector_size_t /*row*/)> isNullAt,
      vector_size_t expectedSize,
      const std::function<vector_size_t(vector_size_t /*index*/)>&
          expectedRowAt) {
    return std::make_shared<LazyVector>(
        execCtx_->pool(),
        CppToType<T>::create(),
        size,
        std::make_unique<SimpleVectorLoader>([=](RowSet rows) {
          VELOX_CHECK_EQ(rows.size(), expectedSize);
          for (auto i = 0; i < rows.size(); i++) {
            VELOX_CHECK_EQ(rows[i], expectedRowAt(i));
          }
          return makeFlatVector<T>(size, valueAt, isNullAt);
        }));
  }

  VectorPtr wrapInLazyDictionary(VectorPtr vector) {
    return std::make_shared<LazyVector>(
        execCtx_->pool(),
        vector->type(),
        vector->size(),
        std::make_unique<SimpleVectorLoader>([=](RowSet /*rows*/) {
          auto indices =
              makeIndices(vector->size(), [](auto row) { return row; });
          return wrapInDictionary(indices, vector->size(), vector);
        }));
  }

  /// Remove ". Input data: .*" from the 'context'.
  std::string trimInputPath(const std::string& context) {
    auto pos = context.find(". Input data: ");
    if (pos == std::string::npos) {
      return context;
    }

    return context.substr(0, pos);
  }

  /// Extract input path from the 'context':
  ///     "<expression>. Input data: <input path>."
  std::string extractInputPath(const std::string& context) {
    auto startPos = context.find(". Input data: ");
    VELOX_CHECK(startPos != std::string::npos);
    startPos += strlen(". Input data: ");
    auto endPos = context.find(".", startPos);
    VELOX_CHECK(endPos != std::string::npos);
    return context.substr(startPos, endPos - startPos);
  }

  VectorPtr restoreVector(const std::string& path) {
    std::ifstream inputFile(path, std::ifstream::binary);
    VELOX_CHECK(!inputFile.fail(), "Cannot open file: {}", path);
    auto copy = facebook::velox::restoreVector(inputFile, pool());
    inputFile.close();
    return copy;
  }

  std::string extractSqlPath(const std::string& context) {
    auto startPos = context.find(". SQL expression: ");
    VELOX_CHECK(startPos != std::string::npos);
    startPos += strlen(". SQL expression: ");
    auto endPos = context.find(".", startPos);
    VELOX_CHECK(endPos != std::string::npos);
    return context.substr(startPos, endPos - startPos);
  }

  void verifyDataAndSqlPaths(const VeloxException& e, const VectorPtr& data) {
    auto inputPath = extractInputPath(e.topLevelContext());
    auto copy = restoreVector(inputPath);
    assertEqualVectors(data, copy);

    auto sqlPath = extractSqlPath(e.topLevelContext());
    auto sql = readSqlFromFile(sqlPath);
    ASSERT_NO_THROW(compileExpression(sql, asRowType(data->type())));
  }

  std::string readSqlFromFile(const std::string& path) {
    std::ifstream inputFile(path, std::ifstream::binary);

    // Find out file size.
    auto begin = inputFile.tellg();
    inputFile.seekg(0, std::ios::end);
    auto end = inputFile.tellg();

    auto fileSize = end - begin;
    if (fileSize == 0) {
      return "";
    }

    // Read the file.
    std::string sql;
    sql.resize(fileSize);

    inputFile.seekg(begin);
    inputFile.read(sql.data(), fileSize);
    inputFile.close();
    return sql;
  }

  void assertError(
      const std::string& expression,
      const VectorPtr& input,
      const std::string& context,
      const std::string& topLevelContext,
      const std::string& message) {
    try {
      evaluate(expression, makeRowVector({input}));
      ASSERT_TRUE(false) << "Expected an error";
    } catch (VeloxException& e) {
      ASSERT_EQ(message, e.message());
      ASSERT_EQ(context, trimInputPath(e.context()));
      ASSERT_EQ(topLevelContext, trimInputPath(e.topLevelContext()));
    }
  }

  void assertErrorSimplified(
      const std::string& expression,
      const VectorPtr& input,
      const std::string& message) {
    try {
      auto inputVector = makeRowVector({input});
      auto exprSetSimplified = compileExpression<exec::ExprSetSimplified>(
          expression, asRowType(inputVector->type()));
      evaluate<exec::ExprSetSimplified>(exprSetSimplified.get(), inputVector);
      ASSERT_TRUE(false) << "Expected an error";
    } catch (VeloxException& e) {
      ASSERT_EQ(message, e.message());
    }
  }

  std::exception_ptr assertWrappedException(
      const std::string& expression,
      const VectorPtr& input,
      const std::string& context,
      const std::string& topLevelContext,
      const std::string& message) {
    try {
      evaluate(expression, makeRowVector({input}));
      EXPECT_TRUE(false) << "Expected an error";
    } catch (VeloxException& e) {
      EXPECT_EQ(context, trimInputPath(e.context()));
      EXPECT_EQ(topLevelContext, trimInputPath(e.topLevelContext()));
      EXPECT_EQ(message, e.message());
      return e.wrappedException();
    }

    return nullptr;
  }

  void testToSql(const std::string& expression, const RowTypePtr& rowType) {
    auto exprSet = compileExpression(expression, rowType);
    auto sql = exprSet->expr(0)->toSql();
    auto copy = compileExpression(sql, rowType);
    ASSERT_EQ(
        exprSet->toString(false /*compact*/), copy->toString(false /*compact*/))
        << sql;
  }

  bool propagatesNulls(const core::TypedExprPtr& typedExpr) {
    exec::ExprSet exprSet({typedExpr}, execCtx_.get(), true);
    return exprSet.exprs().front()->propagatesNulls();
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
  parse::ParseOptions options_;
};

TEST_F(ExprTest, moreEncodings) {
  const vector_size_t size = 1'000;
  std::vector<std::string> fruits = {"apple", "pear", "grapes", "pineapple"};
  VectorPtr a = makeFlatVector<int64_t>(size, [](auto row) { return row; });
  VectorPtr b = makeFlatVector(fruits);

  // Wrap b in a dictionary.
  auto indices =
      makeIndices(size, [&fruits](auto row) { return row % fruits.size(); });
  b = wrapInDictionary(indices, size, b);

  // Wrap both a and b in another dictionary.
  auto evenIndices = makeIndices(size / 2, [](auto row) { return row * 2; });

  a = wrapInDictionary(evenIndices, size / 2, a);
  b = wrapInDictionary(evenIndices, size / 2, b);

  auto result =
      evaluate("if(c1 = 'grapes', c0 + 10, c0)", makeRowVector({a, b}));
  ASSERT_EQ(VectorEncoding::Simple::DICTIONARY, result->encoding());
  ASSERT_EQ(size / 2, result->size());

  auto expected = makeFlatVector<int64_t>(size / 2, [&fruits](auto row) {
    return (fruits[row * 2 % 4] == "grapes") ? row * 2 + 10 : row * 2;
  });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, reorder) {
  constexpr int32_t kTestSize = 20'000;

  auto data = makeRowVector(
      {makeFlatVector<int64_t>(kTestSize, [](auto row) { return row; })});
  auto exprSet = compileExpression(
      "if (c0 % 409 < 300 and c0 % 103 < 30, 1, 2)", asRowType(data->type()));
  auto result = evaluate(exprSet.get(), data);

  auto expectedResult = makeFlatVector<int64_t>(kTestSize, [](auto row) {
    return (row % 409) < 300 && (row % 103) < 30 ? 1 : 2;
  });

  auto condition = std::dynamic_pointer_cast<exec::ConjunctExpr>(
      exprSet->expr(0)->inputs()[0]);
  EXPECT_TRUE(condition != nullptr);

  // Verify that more efficient filter is first.
  for (auto i = 1; i < condition->inputs().size(); ++i) {
    std::cout << condition->selectivityAt(i - 1).timeToDropValue() << std::endl;
    EXPECT_LE(
        condition->selectivityAt(i - 1).timeToDropValue(),
        condition->selectivityAt(i).timeToDropValue());
  }
}

TEST_F(ExprTest, constant) {
  auto exprSet = compileExpression("1 + 2 + 3 + 4", ROW({}));
  auto constExpr = dynamic_cast<exec::ConstantExpr*>(exprSet->expr(0).get());
  ASSERT_NE(constExpr, nullptr);
  auto constant = constExpr->value()->as<ConstantVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(10, constant);

  exprSet = compileExpression("a * (1 + 2 + 3)", ROW({"a"}, {BIGINT()}));
  ASSERT_EQ(2, exprSet->expr(0)->inputs().size());
  constExpr =
      dynamic_cast<exec::ConstantExpr*>(exprSet->expr(0)->inputs()[1].get());
  ASSERT_NE(constExpr, nullptr);
  constant = constExpr->value()->as<ConstantVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(6, constant);
}

// Tests that the eval does the right thing when it receives a NULL
// ConstantVector.
TEST_F(ExprTest, constantNull) {
  // Need to manually build the expression since our eval doesn't support type
  // promotion, to upgrade the UNKOWN type generated by the NULL constant.
  auto inputExpr =
      std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0");
  auto nullConstant = std::make_shared<core::ConstantTypedExpr>(
      INTEGER(), variant::null(TypeKind::INTEGER));

  // Builds the following expression: "plus(c0, plus(c0, null))"
  auto expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          inputExpr,
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{inputExpr, nullConstant},
              "plus"),
      },
      "plus");

  // Execute it and check it returns all null results.
  auto vector = makeNullableFlatVector<int32_t>({1, std::nullopt, 3});
  auto rowVector = makeRowVector({vector});
  SelectivityVector rows(rowVector->size());
  std::vector<VectorPtr> result(1);

  exec::ExprSet exprSet({expression}, execCtx_.get());
  exec::EvalCtx context(execCtx_.get(), &exprSet, rowVector.get());
  exprSet.eval(rows, context, result);

  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  assertEqualVectors(expected, result.front());
}

// Tests that exprCompiler throws if there's a return type mismatch between what
// the user specific in ConstantTypedExpr, and the available signatures.
TEST_F(ExprTest, validateReturnType) {
  auto inputExpr =
      std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0");

  // Builds a "eq(c0, c0)" expression.
  auto expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(), std::vector<core::TypedExprPtr>{inputExpr, inputExpr}, "eq");

  // Execute it and check it returns all null results.
  auto vector = makeNullableFlatVector<int32_t>({1, 2, 3});
  auto rowVector = makeRowVector({vector});
  SelectivityVector rows(rowVector->size());
  std::vector<VectorPtr> result(1);

  EXPECT_THROW(
      {
        exec::ExprSet exprSet({expression}, execCtx_.get());
        exec::EvalCtx context(execCtx_.get(), &exprSet, rowVector.get());
        exprSet.eval(rows, context, result);
      },
      VeloxUserError);
}

TEST_F(ExprTest, constantFolding) {
  auto typedExpr = parseExpression("1 + 2", ROW({}));

  auto extractConstant = [](exec::Expr* expr) {
    auto constExpr = dynamic_cast<exec::ConstantExpr*>(expr);
    return constExpr->value()->as<ConstantVector<int64_t>>()->valueAt(0);
  };

  // Check that the constants have been folded.
  {
    exec::ExprSet exprSetFolded({typedExpr}, execCtx_.get(), true);

    auto expr = exprSetFolded.exprs().front();
    auto constExpr = dynamic_cast<exec::ConstantExpr*>(expr.get());

    ASSERT_TRUE(constExpr != nullptr);
    EXPECT_TRUE(constExpr->inputs().empty());
    EXPECT_EQ(3, extractConstant(expr.get()));
  }

  // Check that the constants have NOT been folded.
  {
    exec::ExprSet exprSetUnfolded({typedExpr}, execCtx_.get(), false);
    auto expr = exprSetUnfolded.exprs().front();

    ASSERT_EQ(2, expr->inputs().size());
    EXPECT_EQ(1, extractConstant(expr->inputs()[0].get()));
    EXPECT_EQ(2, extractConstant(expr->inputs()[1].get()));
  }

  {
    // codepoint() takes a single character, so this expression
    // deterministically throws; however, we should never throw at constant
    // folding time. Ensure compiling this expression does not throw..
    auto typedExpr = parseExpression("codepoint('abcdef')", ROW({}));
    EXPECT_NO_THROW(exec::ExprSet exprSet({typedExpr}, execCtx_.get(), true));
  }
}

TEST_F(ExprTest, constantArray) {
  auto a = makeArrayVector<int32_t>(
      10, [](auto /*row*/) { return 5; }, [](auto row) { return row * 3; });
  auto b = makeArrayVector<int64_t>(
      10, [](auto /*row*/) { return 7; }, [](auto row) { return row; });

  std::vector<core::TypedExprPtr> expressions = {
      makeConstantExpr(a, 3), makeConstantExpr(b, 5)};

  auto exprSet =
      std::make_unique<exec::ExprSet>(std::move(expressions), execCtx_.get());

  const vector_size_t size = 1'000;
  auto input = makeRowVector(ROW({}), size);
  exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

  SelectivityVector rows(input->size());
  std::vector<VectorPtr> result(2);
  exprSet->eval(rows, context, result);

  ASSERT_TRUE(a->equalValueAt(result[0].get(), 3, 0));
  ASSERT_TRUE(b->equalValueAt(result[1].get(), 5, 0));
}

TEST_F(ExprTest, constantComplexNull) {
  std::vector<core::TypedExprPtr> expressions = {
      std::make_shared<const core::ConstantTypedExpr>(
          ARRAY(BIGINT()), variant::null(TypeKind::ARRAY)),
      std::make_shared<const core::ConstantTypedExpr>(
          MAP(VARCHAR(), BIGINT()), variant::null(TypeKind::MAP)),
      std::make_shared<const core::ConstantTypedExpr>(
          ROW({SMALLINT(), BIGINT()}), variant::null(TypeKind::ROW))};
  auto exprSet =
      std::make_unique<exec::ExprSet>(std::move(expressions), execCtx_.get());

  const vector_size_t size = 10;
  auto input = makeRowVector(ROW({}), size);
  exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

  SelectivityVector rows(size);
  std::vector<VectorPtr> result(3);
  exprSet->eval(rows, context, result);

  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, result[0]->encoding());
  ASSERT_EQ(TypeKind::ARRAY, result[0]->typeKind());
  ASSERT_TRUE(result[0]->as<ConstantVector<ComplexType>>()->isNullAt(0));

  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, result[1]->encoding());
  ASSERT_EQ(TypeKind::MAP, result[1]->typeKind());
  ASSERT_TRUE(result[1]->as<ConstantVector<ComplexType>>()->isNullAt(0));

  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, result[2]->encoding());
  ASSERT_EQ(TypeKind::ROW, result[2]->typeKind());
  ASSERT_TRUE(result[2]->as<ConstantVector<ComplexType>>()->isNullAt(0));
}

TEST_F(ExprTest, constantScalarEquals) {
  auto a = makeFlatVector<int32_t>(10, [](auto row) { return row; });
  auto b = makeFlatVector<int32_t>(10, [](auto row) { return row; });
  auto c = makeFlatVector<int64_t>(10, [](auto row) { return row; });

  ASSERT_EQ(*makeConstantExpr<int32_t>(a, 3), *makeConstantExpr<int32_t>(b, 3));
  // The types differ, so not equal
  ASSERT_FALSE(
      *makeConstantExpr<int32_t>(a, 3) == *makeConstantExpr<int64_t>(c, 3));
  // The values differ, so not equal
  ASSERT_FALSE(
      *makeConstantExpr<int32_t>(a, 3) == *makeConstantExpr<int32_t>(b, 4));
}

TEST_F(ExprTest, constantComplexEquals) {
  auto testConstantEquals =
      // a and b should be equal but distinct vectors.
      // a and c should be vectors with equal values but different types (e.g.
      // int32_t and int64_t).
      [&](const VectorPtr& a, const VectorPtr& b, const VectorPtr& c) {
        ASSERT_EQ(*makeConstantExpr(a, 3), *makeConstantExpr(b, 3));
        // The types differ, so not equal
        ASSERT_FALSE(*makeConstantExpr(a, 3) == *makeConstantExpr(c, 3));
        // The values differ, so not equal
        ASSERT_FALSE(*makeConstantExpr(a, 3) == *makeConstantExpr(b, 4));
      };

  testConstantEquals(
      makeArrayVector<int32_t>(
          10, [](auto /*row*/) { return 5; }, [](auto row) { return row * 3; }),
      makeArrayVector<int32_t>(
          10, [](auto /*row*/) { return 5; }, [](auto row) { return row * 3; }),
      makeArrayVector<int64_t>(
          10,
          [](auto /*row*/) { return 5; },
          [](auto row) { return row * 3; }));

  testConstantEquals(
      makeMapVector<int32_t, int32_t>(
          10,
          [](auto /*row*/) { return 5; },
          [](auto row) { return row; },
          [](auto row) { return row * 3; }),
      makeMapVector<int32_t, int32_t>(
          10,
          [](auto /*row*/) { return 5; },
          [](auto row) { return row; },
          [](auto row) { return row * 3; }),
      makeMapVector<int32_t, int64_t>(
          10,
          [](auto /*row*/) { return 5; },
          [](auto row) { return row; },
          [](auto row) { return row * 3; }));

  auto a = makeFlatVector<int32_t>(10, [](auto row) { return row; });
  auto b = makeFlatVector<int64_t>(10, [](auto row) { return row; });

  testConstantEquals(
      makeRowVector({a}), makeRowVector({a}), makeRowVector({b}));
}

namespace {
class PlusConstantFunction : public exec::VectorFunction {
 public:
  explicit PlusConstantFunction(int32_t addition) : addition_(addition) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);

    auto& arg = args[0];

    // The argument may be flat or constant.
    VELOX_CHECK(arg->isFlatEncoding() || arg->isConstantEncoding());

    BaseVector::ensureWritable(rows, INTEGER(), context.pool(), result);

    auto* flatResult = result->asFlatVector<int32_t>();
    auto* rawResult = flatResult->mutableRawValues();

    flatResult->clearNulls(rows);

    if (arg->isConstantEncoding()) {
      auto value = arg->as<ConstantVector<int32_t>>()->valueAt(0);
      rows.applyToSelected(
          [&](auto row) { rawResult[row] = value + addition_; });
    } else {
      auto* rawInput = arg->as<FlatVector<int32_t>>()->rawValues();

      rows.applyToSelected(
          [&](auto row) { rawResult[row] = rawInput[row] + addition_; });
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // integer -> integer
    return {exec::FunctionSignatureBuilder()
                .returnType("integer")
                .argumentType("integer")
                .build()};
  }

 private:
  const int32_t addition_;
};

} // namespace

TEST_F(ExprTest, dictionaryAndConstantOverLazy) {
  exec::registerVectorFunction(
      "plus5",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(5));

  const vector_size_t size = 1'000;

  // Make LazyVector with nulls.
  auto valueAt = [](vector_size_t row) { return row; };
  auto isNullAt = [](vector_size_t row) { return row % 5 == 0; };

  const auto lazyVector =
      vectorMaker_.lazyFlatVector<int32_t>(size, valueAt, isNullAt);
  auto row = makeRowVector({lazyVector});
  auto result = evaluate("plus5(c0)", row);

  auto expected = makeFlatVector<int32_t>(
      size, [](auto row) { return row + 5; }, isNullAt);
  assertEqualVectors(expected, result);

  // Wrap LazyVector in a dictionary (select only even rows).
  auto evenIndices = makeIndices(size / 2, [](auto row) { return row * 2; });

  auto vector = wrapInDictionary(evenIndices, size / 2, lazyVector);
  row = makeRowVector({vector});
  result = evaluate("plus5(c0)", row);

  expected = makeFlatVector<int32_t>(
      size / 2, [](auto row) { return row * 2 + 5; }, isNullAt);
  assertEqualVectors(expected, result);

  // non-null constant
  vector = BaseVector::wrapInConstant(size, 3, lazyVector);
  row = makeRowVector({vector});
  result = evaluate("plus5(c0)", row);

  expected = makeFlatVector<int32_t>(size, [](auto /*row*/) { return 3 + 5; });
  assertEqualVectors(expected, result);

  // null constant
  vector = BaseVector::wrapInConstant(size, 5, lazyVector);
  row = makeRowVector({vector});
  result = evaluate("plus5(c0)", row);

  expected = makeAllNullFlatVector<int32_t>(size);
  assertEqualVectors(expected, result);
}

// Test evaluating single-argument vector function on a non-zero row of
// constant vector.
TEST_F(ExprTest, vectorFunctionOnConstantInput) {
  exec::registerVectorFunction(
      "plus5",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(5));
  const vector_size_t size = 1'000;

  auto row = makeRowVector(
      {makeFlatVector<int64_t>(size, [](auto row) { return row; }),
       makeConstant(3, size)});

  VectorPtr expected = makeFlatVector<int32_t>(
      size, [](auto row) { return row > 5 ? 3 + 5 : 0; });
  auto result = evaluate("if (c0 > 5, plus5(c1), cast(0 as integer))", row);
  assertEqualVectors(expected, result);

  result = evaluate("is_null(c1)", row);
  expected = makeConstant(false, size);
  assertEqualVectors(expected, result);
}

namespace {
// f(n) = n + rand() - non-deterministict function with a single argument
class PlusRandomIntegerFunction : public exec::VectorFunction {
 public:
  bool isDeterministic() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    VELOX_CHECK_EQ(args[0]->typeKind(), facebook::velox::TypeKind::INTEGER);

    BaseVector::ensureWritable(rows, INTEGER(), context.pool(), result);
    auto flatResult = result->asFlatVector<int32_t>();

    DecodedVector decoded(*args[0], rows);
    std::srand(1);
    rows.applyToSelected([&](auto row) {
      if (decoded.isNullAt(row)) {
        flatResult->setNull(row, true);
      } else {
        flatResult->set(row, decoded.valueAt<int32_t>(row) + std::rand());
      }
      return true;
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // integer -> integer
    return {exec::FunctionSignatureBuilder()
                .returnType("integer")
                .argumentType("integer")
                .build()};
  }
};
} // namespace

// Test evaluating single-argument non-deterministic vector function on
// constant vector. The function must be called on each row, not just one.
TEST_F(ExprTest, nonDeterministicVectorFunctionOnConstantInput) {
  exec::registerVectorFunction(
      "plus_random",
      PlusRandomIntegerFunction::signatures(),
      std::make_unique<PlusRandomIntegerFunction>());

  const vector_size_t size = 1'000;
  auto row = makeRowVector({makeConstant(10, size)});

  auto result = evaluate("plus_random(c0)", row);

  std::srand(1);
  auto expected = makeFlatVector<int32_t>(
      size, [](auto /*row*/) { return 10 + std::rand(); });
  assertEqualVectors(expected, result);
}

// Verify constant folding doesn't apply to non-deterministic functions.
TEST_F(ExprTest, nonDeterministicConstantFolding) {
  exec::registerVectorFunction(
      "plus_random",
      PlusRandomIntegerFunction::signatures(),
      std::make_unique<PlusRandomIntegerFunction>());

  const vector_size_t size = 1'000;
  auto emptyRow = makeRowVector(ROW({}), size);

  auto result = evaluate("plus_random(cast(23 as integer))", emptyRow);

  std::srand(1);
  auto expected = makeFlatVector<int32_t>(
      size, [](auto /*row*/) { return 23 + std::rand(); });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, shortCircuit) {
  vector_size_t size = 4;

  auto a = makeConstant(10, size);
  auto b = makeFlatVector<int32_t>({-1, -2, -3, -4});

  auto result = evaluate("c0 > 0 OR c1 > 0", makeRowVector({a, b}));
  auto expectedResult = makeConstant(true, size);

  assertEqualVectors(expectedResult, result);

  result = evaluate("c0 < 0 AND c1 < 0", makeRowVector({a, b}));
  expectedResult = makeConstant(false, size);

  assertEqualVectors(expectedResult, result);
}

// Test common sub-expression (CSE) optimization with encodings.
// CSE evaluation may happen in different contexts, e.g. original input rows
// on first evaluation and base vectors uncovered through peeling of encodings
// on second. In this case, the row numbers from first evaluation and row
// numbers in the second evaluation are non-comparable.
//
// Consider two projections:
//  if (a > 0 AND c = 'apple')
//  if (b > 0 AND c = 'apple')
//
// c = 'apple' is CSE. Let a be flat vector, b and c be dictionaries with
// shared indices. On first evaluation, 'a' and 'c' don't share any encodings,
// no peeling happens and context contains the original vectors and rows. On
// second evaluation, 'b' and 'c' share dictionary encoding and it gets
// peeled. Context now contains base vectors and inner rows.
//
// Currently, this case doesn't work properly, hence, peeling disables CSE
// optimizations.
TEST_F(ExprTest, cseEncodings) {
  auto a = makeFlatVector<int32_t>({1, 2, 3, 4, 5});

  auto indices = makeIndices({0, 0, 0, 1, 1});
  auto b = wrapInDictionary(indices, 5, makeFlatVector<int32_t>({11, 15}));
  auto c = wrapInDictionary(
      indices, 5, makeFlatVector<std::string>({"apple", "banana"}));

  auto results = evaluateMultiple(
      {"if (c0 > 0 AND c2 = 'apple', 10, 3)",
       "if (c1 > 0 AND c2 = 'apple', 20, 5)"},
      makeRowVector({a, b, c}));

  auto expected = makeFlatVector<int64_t>({10, 10, 10, 3, 3});
  assertEqualVectors(expected, results[0]);

  expected = makeFlatVector<int64_t>({20, 20, 20, 5, 5});
  assertEqualVectors(expected, results[1]);
}

namespace {
class AddSuffixFunction : public exec::VectorFunction {
 public:
  explicit AddSuffixFunction(const std::string& suffix) : suffix_{suffix} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto input = args[0]->asFlatVector<StringView>();
    auto localResult = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), rows.end(), context.pool()));
    rows.applyToSelected([&](auto row) {
      auto value = fmt::format("{}{}", input->valueAt(row).str(), suffix_);
      localResult->set(row, StringView(value));
      return true;
    });

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // varchar -> varchar
    return {exec::FunctionSignatureBuilder()
                .returnType("varchar")
                .argumentType("varchar")
                .build()};
  }

 private:
  const std::string suffix_;
};
} // namespace

// Test CSE evaluation where first evaluation applies to fewer rows then
// second. Make sure values calculated on first evaluation are preserved when
// calculating additional rows on second evaluation. This could happen if CSE
// is a function that uses EvalCtx::moveOrCopyResult which relies on
// isFinalSelection flag.
TEST_F(ExprTest, csePartialEvaluation) {
  exec::registerVectorFunction(
      "add_suffix",
      AddSuffixFunction::signatures(),
      std::make_unique<AddSuffixFunction>("_xx"));

  auto a = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto b = makeFlatVector<std::string>({"a", "b", "c", "d", "e"});

  auto [results, stats] = evaluateMultipleWithStats(
      {
          "if (c0 >= 3, add_suffix(c1), 'n/a')",
          "add_suffix(c1)",
      },
      makeRowVector({a, b}));

  auto expected =
      makeFlatVector<std::string>({"n/a", "n/a", "c_xx", "d_xx", "e_xx"});
  assertEqualVectors(expected, results[0]);

  expected =
      makeFlatVector<std::string>({"a_xx", "b_xx", "c_xx", "d_xx", "e_xx"});
  assertEqualVectors(expected, results[1]);
  EXPECT_EQ(5, stats.at("add_suffix").numProcessedRows);

  std::tie(results, stats) = evaluateMultipleWithStats(
      {
          "if (c0 >= 3, add_suffix(c1), 'n/a')",
          "if (c0 < 2, 'n/a', add_suffix(c1))",
      },
      makeRowVector({a, b}));

  expected =
      makeFlatVector<std::string>({"n/a", "n/a", "c_xx", "d_xx", "e_xx"});
  assertEqualVectors(expected, results[0]);

  expected =
      makeFlatVector<std::string>({"n/a", "b_xx", "c_xx", "d_xx", "e_xx"});
  assertEqualVectors(expected, results[1]);
  EXPECT_EQ(4, stats.at("add_suffix").numProcessedRows);
}

TEST_F(ExprTest, csePartialEvaluationWithEncodings) {
  auto data = makeRowVector(
      {wrapInDictionary(
           makeIndicesInReverse(5),
           wrapInDictionary(
               makeIndicesInReverse(5),
               makeFlatVector<int64_t>({0, 10, 20, 30, 40}))),
       makeFlatVector<int64_t>({3, 33, 333, 3333, 33333})});

  // Compile the expressions once, then execute two times. First time, evaluate
  // on 2 rows (0, 1). Seconds time, one 4 rows (0, 1, 2, 3).
  auto exprSet = compileMultiple(
      {
          "concat(concat(cast(c0 as varchar), ',', cast(c1 as varchar)), 'xxx')",
          "concat(concat(cast(c0 as varchar), ',', cast(c1 as varchar)), 'yyy')",
      },
      asRowType(data->type()));

  std::vector<VectorPtr> results(2);
  {
    SelectivityVector rows(2);
    exec::EvalCtx context(execCtx_.get(), exprSet.get(), data.get());
    exprSet->eval(rows, context, results);

    std::vector<VectorPtr> expectedResults = {
        makeFlatVector<StringView>({"0,3xxx", "10,33xxx"}),
        makeFlatVector<StringView>({"0,3yyy", "10,33yyy"}),
    };

    assertEqualVectors(expectedResults[0], results[0]);
    assertEqualVectors(expectedResults[1], results[1]);
  }

  {
    SelectivityVector rows(4);
    exec::EvalCtx context(execCtx_.get(), exprSet.get(), data.get());
    exprSet->eval(rows, context, results);

    std::vector<VectorPtr> expectedResults = {
        makeFlatVector<StringView>(
            {"0,3xxx", "10,33xxx", "20,333xxx", "30,3333xxx"}),
        makeFlatVector<StringView>(
            {"0,3yyy", "10,33yyy", "20,333yyy", "30,3333yyy"}),
    };

    assertEqualVectors(expectedResults[0], results[0]);
    assertEqualVectors(expectedResults[1], results[1]);
  }
}

// Checks that vector function registry overwrites if multiple registry
// attempts are made for the same functions.
TEST_F(ExprTest, overwriteInRegistry) {
  exec::VectorFunctionMetadata metadata;
  auto inserted = exec::registerVectorFunction(
      "plus5",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(500),
      metadata,
      true);
  ASSERT_TRUE(inserted);

  auto vectorFunction = exec::getVectorFunction("plus5", {INTEGER()}, {});
  ASSERT_TRUE(vectorFunction != nullptr);

  inserted = exec::registerVectorFunction(
      "plus5",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(5),
      metadata,
      true);
  ASSERT_TRUE(inserted);

  auto vectorFunction2 = exec::getVectorFunction("plus5", {INTEGER()}, {});

  ASSERT_TRUE(vectorFunction2 != nullptr);
  ASSERT_TRUE(vectorFunction != vectorFunction2);

  ASSERT_TRUE(inserted);
}

// Check non overwriting path in the function registry

TEST_F(ExprTest, keepInRegistry) {
  // Adding a new function, overwrite = false;

  exec::VectorFunctionMetadata metadata;
  bool inserted = exec::registerVectorFunction(
      "NonExistingFunction",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(500),
      metadata,
      false);

  ASSERT_TRUE(inserted);

  auto vectorFunction = exec::getVectorFunction("NonExistingFunction", {}, {});

  inserted = exec::registerVectorFunction(
      "NonExistingFunction",
      PlusConstantFunction::signatures(),
      std::make_unique<PlusConstantFunction>(400),
      metadata,
      false);
  ASSERT_FALSE(inserted);
  ASSERT_EQ(
      vectorFunction, exec::getVectorFunction("NonExistingFunction", {}, {}));
}

TEST_F(ExprTest, lazyVectors) {
  vector_size_t size = 1'000;

  // Make LazyVector with no nulls
  auto valueAt = [](auto row) { return row; };
  auto vector = vectorMaker_.lazyFlatVector<int64_t>(size, valueAt);
  auto row = makeRowVector({vector});

  auto result = evaluate("c0 + coalesce(c0, 1)", row);

  auto expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row * 2; }, nullptr);
  assertEqualVectors(expected, result);

  // Make LazyVector with nulls
  auto isNullAt = [](auto row) { return row % 5 == 0; };
  vector = vectorMaker_.lazyFlatVector<int64_t>(size, valueAt, isNullAt);
  row = makeRowVector({vector});

  result = evaluate("c0 + coalesce(c0, 1)", row);

  expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row * 2; }, isNullAt);
  assertEqualVectors(expected, result);
}

// Tests that lazy vectors are not loaded unnecessarily.
TEST_F(ExprTest, lazyLoading) {
  const vector_size_t size = 1'000;
  VectorPtr vector =
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5; });
  VectorPtr lazyVector = std::make_shared<LazyVector>(
      execCtx_->pool(),
      BIGINT(),
      size,
      std::make_unique<test::SimpleVectorLoader>([&](RowSet /*rows*/) {
        VELOX_FAIL("This lazy vector is not expected to be loaded");
        return nullptr;
      }));

  auto result = evaluate(
      "if(c0 = 10, c1 + 5, c0 - 5)", makeRowVector({vector, lazyVector}));
  auto expected =
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5 - 5; });
  assertEqualVectors(expected, result);

  vector = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 5; }, nullEvery(7));

  result = evaluate(
      "if(c0 = 10, c1 + 5, c0 - 5)", makeRowVector({vector, lazyVector}));
  expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 5 - 5; }, nullEvery(7));
  assertEqualVectors(expected, result);

  // Wrap non-lazy vector in a dictionary (repeat each row twice).
  auto evenIndices = makeIndices(size, [](auto row) { return row / 2; });
  vector = wrapInDictionary(evenIndices, size, vector);

  result = evaluate(
      "if(c0 = 10, c1 + 5, c0 - 5)", makeRowVector({vector, lazyVector}));
  expected = makeFlatVector<int64_t>(
      size,
      [](auto row) { return (row / 2) % 5 - 5; },
      [](auto row) { return (row / 2) % 7 == 0; });
  assertEqualVectors(expected, result);

  // Wrap both vectors in the same dictionary.
  lazyVector = wrapInDictionary(evenIndices, size, lazyVector);

  result = evaluate(
      "if(c0 = 10, c1 + 5, c0 - 5)", makeRowVector({vector, lazyVector}));
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, selectiveLazyLoadingAnd) {
  const vector_size_t size = 1'000;

  // Evaluate AND expression on 3 lazy vectors and verify that each
  // subsequent vector is loaded for fewer rows than the one before.
  // Create 3 identical vectors with values set to row numbers. Use conditions
  // that pass on half of the rows for the first vector, a third for the
  // second, and a fifth for the third: a % 2 = 0 AND b % 3 = 0 AND c % 5 = 0.
  // Verify that all rows are loaded for the first vector, half for the second
  // and only 1/6 for the third.
  auto valueAt = [](auto row) { return row; };
  auto a = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });
  auto b = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, ceil(size / 2.0), [](auto row) {
        return row * 2;
      });
  auto c = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, ceil(size / 2.0 / 3.0), [](auto row) {
        return row * 2 * 3;
      });

  auto result = evaluate(
      "c0 % 2 = 0 AND c1 % 3 = 0 AND c2 % 5 = 0", makeRowVector({a, b, c}));
  auto expected = makeFlatVector<bool>(
      size, [](auto row) { return row % (2 * 3 * 5) == 0; });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, selectiveLazyLoadingOr) {
  const vector_size_t size = 1'000;

  // Evaluate OR expression. Columns under OR must be loaded for "all" rows
  // because the engine currently doesn't know whether a column is used
  // elsewhere or not.
  auto valueAt = [](auto row) { return row; };
  auto a = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });
  auto b = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });
  auto c = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });

  auto result = evaluate(
      "c0 % 2 <> 0 OR c1 % 4 <> 0 OR c2 % 8 <> 0", makeRowVector({a, b, c}));
  auto expected = makeFlatVector<bool>(size, [](auto row) {
    return row % 2 != 0 || row % 4 != 0 || row % 8 != 0;
  });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, lazyVectorAccessTwiceWithDifferentRows) {
  const vector_size_t size = 4;

  auto c0 = makeNullableFlatVector<int64_t>({1, 1, 1, std::nullopt});
  // [0, 1, 2, 3] if fully loaded
  std::vector<vector_size_t> loadedRows;
  auto valueAt = [](auto row) { return row; };
  VectorPtr c1 = std::make_shared<LazyVector>(
      pool_.get(),
      BIGINT(),
      size,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        for (auto row : rows) {
          loadedRows.push_back(row);
        }
        return makeFlatVector<int64_t>(rows.back() + 1, valueAt);
      }));

  auto result = evaluate(
      "row_constructor(c0 + c1, if (c1 >= 0, c1, 0))", makeRowVector({c0, c1}));

  auto expected = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt}),
       makeNullableFlatVector<int64_t>({0, 1, 2, 3})});

  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, lazyVectorAccessTwiceInDifferentExpressions) {
  const vector_size_t size = 1'000;

  // Fields referenced by multiple expressions will load lazy vector
  // immediately in ExprSet::eval().
  auto isNullAtColA = [](auto row) { return row % 4 == 0; };
  auto isNullAtColC = [](auto row) { return row % 2 == 0; };

  auto a = makeLazyFlatVector<int64_t>(
      size,
      [](auto row) { return row; },
      isNullAtColA,
      size,
      [](auto row) { return row; });
  auto b = makeLazyFlatVector<int64_t>(
      size,
      [](auto row) { return row * 2; },
      nullptr,
      size,
      [](auto row) { return row; });
  auto c = makeLazyFlatVector<int64_t>(
      size,
      [](auto row) { return row; },
      isNullAtColC,
      size,
      [](auto row) { return row; });

  auto result = evaluateMultiple(
      {"if(c0 is not null, c0, c1)", "if (c2 is not null, c2, c1)"},
      makeRowVector({a, b, c}));

  auto expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 4 == 0 ? row * 2 : row; });
  assertEqualVectors(expected, result[0]);

  expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 2 == 0 ? row * 2 : row; });
  assertEqualVectors(expected, result[1]);
}

TEST_F(ExprTest, selectiveLazyLoadingIf) {
  const vector_size_t size = 1'000;

  // Evaluate IF expression. Columns under IF must be loaded for "all" rows
  // because the engine currently doesn't know whether a column is used in a
  // single branch (then or else) or in both.
  auto valueAt = [](auto row) { return row; };

  auto a = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });
  auto b = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });
  auto c = makeLazyFlatVector<int64_t>(
      size, valueAt, nullptr, size, [](auto row) { return row; });

  auto result =
      evaluate("if (c0 % 2 = 0, c1 + c2, c2 / 3)", makeRowVector({a, b, c}));
  auto expected = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 2 == 0 ? row + row : row / 3; });
  assertEqualVectors(expected, result);
}

namespace {
class StatefulVectorFunction : public exec::VectorFunction {
 public:
  explicit StatefulVectorFunction(
      const std::string& /*name*/,
      const std::vector<exec::VectorFunctionArg>& inputs)
      : numInputs_(inputs.size()) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), numInputs_);
    auto numInputs = BaseVector::createConstant(
        INTEGER(), numInputs_, rows.size(), context.pool());
    if (!result) {
      result = numInputs;
    } else {
      BaseVector::ensureWritable(rows, INTEGER(), context.pool(), result);
      result->copy(numInputs.get(), rows, nullptr);
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T... -> integer
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("integer")
                .argumentType("T")
                .variableArity()
                .build()};
  }

 private:
  const int32_t numInputs_;
};
} // namespace

TEST_F(ExprTest, statefulVectorFunctions) {
  exec::registerStatefulVectorFunction(
      "test_function",
      StatefulVectorFunction::signatures(),
      exec::makeVectorFunctionFactory<StatefulVectorFunction>());

  vector_size_t size = 1'000;

  auto a = makeFlatVector<int64_t>(size, [](auto row) { return row; });
  auto b = makeFlatVector<int64_t>(size, [](auto row) { return row * 2; });
  auto row = makeRowVector({a, b});

  {
    auto result = evaluate("test_function(c0)", row);

    auto expected =
        makeFlatVector<int32_t>(size, [](auto /*row*/) { return 1; });
    assertEqualVectors(expected, result);
  }

  {
    auto result = evaluate("test_function(c0, c1)", row);

    auto expected =
        makeFlatVector<int32_t>(size, [](auto /*row*/) { return 2; });
    assertEqualVectors(expected, result);
  }
}

struct OpaqueState {
  static int constructed;
  static int destructed;

  static void clearStats() {
    constructed = 0;
    destructed = 0;
  }

  explicit OpaqueState(int x) : x(x) {
    ++constructed;
  }

  ~OpaqueState() {
    ++destructed;
  }

  int x;
};

int OpaqueState::constructed = 0;
int OpaqueState::destructed = 0;

template <typename T>
struct TestOpaqueCreateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<std::shared_ptr<OpaqueState>>& out,
      const arg_type<int64_t>& x) {
    out = std::make_shared<OpaqueState>(x);
    return true;
  }
};

template <typename T>
struct TestOpaqueAddFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<std::shared_ptr<OpaqueState>>& state,
      const arg_type<int64_t>& y) {
    out = state->x + y;
    return true;
  }
};

bool registerTestUDFs() {
  static bool once = [] {
    registerFunction<
        TestOpaqueCreateFunction,
        std::shared_ptr<OpaqueState>,
        int64_t>({"test_opaque_create"});
    registerFunction<
        TestOpaqueAddFunction,
        int64_t,
        std::shared_ptr<OpaqueState>,
        int64_t>({"test_opaque_add"});
    return true;
  }();
  return once;
}

TEST_F(ExprTest, opaque) {
  registerTestUDFs();

  static constexpr vector_size_t kRows = 100;

  OpaqueState::clearStats();

  auto data = makeRowVector({
      makeFlatVector<int64_t>(
          kRows, [](auto row) { return row; }, nullEvery(7)),
      makeFlatVector<int64_t>(
          kRows, [](auto row) { return row * 2; }, nullEvery(11)),
      BaseVector::wrapInConstant(
          kRows,
          0,
          makeFlatVector<std::shared_ptr<void>>(
              1,
              [](auto row) {
                return std::static_pointer_cast<void>(
                    std::make_shared<OpaqueState>(123));
              })),
  });

  EXPECT_EQ(1, OpaqueState::constructed);

  int nonNulls = 0;
  for (auto i = 0; i < kRows; ++i) {
    if (i % 7 != 0 && i % 11 != 0) {
      ++nonNulls;
    }
  }

  // Opaque value created each time.
  OpaqueState::clearStats();
  auto result = evaluate("test_opaque_add(test_opaque_create(c0), c1)", data);
  auto expectedResult = makeFlatVector<int64_t>(
      kRows,
      [](auto row) { return row + row * 2; },
      [](auto row) { return row % 7 == 0 || row % 11 == 0; });
  assertEqualVectors(expectedResult, result);

  EXPECT_EQ(OpaqueState::constructed, nonNulls);
  EXPECT_EQ(OpaqueState::destructed, nonNulls);

  // Opaque value passed in as a constant explicitly.
  OpaqueState::clearStats();
  result = evaluate("test_opaque_add(c2, c1)", data);
  expectedResult = makeFlatVector<int64_t>(
      kRows, [](auto row) { return 123 + row * 2; }, nullEvery(11));
  assertEqualVectors(expectedResult, result);

  // Nothing got created!
  EXPECT_EQ(OpaqueState::constructed, 0);
  EXPECT_EQ(OpaqueState::destructed, 0);

  // Opaque value created by a function taking a literal. Should be
  // constant-folded.
  OpaqueState::clearStats();
  result = evaluate("test_opaque_add(test_opaque_create(123), c1)", data);
  expectedResult = makeFlatVector<int64_t>(
      kRows, [](auto row) { return 123 + row * 2; }, nullEvery(11));
  assertEqualVectors(expectedResult, result);

  EXPECT_EQ(OpaqueState::constructed, 1);
  EXPECT_EQ(OpaqueState::destructed, 1);
}

TEST_F(ExprTest, switchExpr) {
  vector_size_t size = 1'000;
  auto vector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row; }),
       makeFlatVector<int32_t>(
           size, [](auto row) { return row; }, nullEvery(5)),
       makeConstant<int32_t>(0, size)});

  auto result =
      evaluate("case c0 when 7 then 1 when 11 then 2 else 0 end", vector);
  std::function<int32_t(vector_size_t)> expectedValueAt = [](auto row) {
    switch (row) {
      case 7:
        return 1;
      case 11:
        return 2;
      default:
        return 0;
    }
  };
  auto expected = makeFlatVector<int64_t>(size, expectedValueAt);
  assertEqualVectors(expected, result);

  // c1 has nulls
  result = evaluate("case c1 when 7 then 1 when 11 then 2 else 0 end", vector);
  assertEqualVectors(expected, result);

  // no "else" clause
  result = evaluate("case c0 when 7 then 1 when 11 then 2 end", vector);
  expected = makeFlatVector<int64_t>(
      size, expectedValueAt, [](auto row) { return row != 7 && row != 11; });
  assertEqualVectors(expected, result);

  result = evaluate("case c1 when 7 then 1 when 11 then 2 end", vector);
  assertEqualVectors(expected, result);

  // No "else" clause and no match.
  result = evaluate("case 0 when 100 then 1 when 200 then 2 end", vector);
  expected = makeAllNullFlatVector<int64_t>(size);
  assertEqualVectors(expected, result);

  result = evaluate("case c2 when 100 then 1 when 200 then 2 end", vector);
  assertEqualVectors(expected, result);

  // non-equality case expression
  result = evaluate(
      "case when c0 < 7 then 1 when c0 < 11 then 2 else 0 end", vector);
  expectedValueAt = [](auto row) {
    if (row < 7) {
      return 1;
    }
    if (row < 11) {
      return 2;
    }
    return 0;
  };
  expected = makeFlatVector<int64_t>(size, expectedValueAt);
  assertEqualVectors(expected, result);

  result = evaluate(
      "case when c1 < 7 then 1 when c1 < 11 then 2 else 0 end", vector);
  expected = makeFlatVector<int64_t>(size, [](auto row) {
    if (row % 5 == 0) {
      return 0;
    }
    if (row < 7) {
      return 1;
    }
    if (row < 11) {
      return 2;
    }
    return 0;
  });
  assertEqualVectors(expected, result);

  // non-equality case expression, no else clause
  result = evaluate("case when c0 < 7 then 1 when c0 < 11 then 2 end", vector);
  expected = makeFlatVector<int64_t>(
      size, expectedValueAt, [](auto row) { return row >= 11; });
  assertEqualVectors(expected, result);

  result = evaluate("case when c1 < 7 then 1 when c1 < 11 then 2 end", vector);
  expected = makeFlatVector<int64_t>(size, expectedValueAt, [](auto row) {
    return row >= 11 || row % 5 == 0;
  });
  assertEqualVectors(expected, result);

  // non-constant then expression
  result = evaluate(
      "case when c0 < 7 then c0 + 5 when c0 < 11 then c0 - 11 "
      "else c0::BIGINT end",
      vector);
  expectedValueAt = [](auto row) {
    if (row < 7) {
      return row + 5;
    }
    if (row < 11) {
      return row - 11;
    }
    return row;
  };
  expected = makeFlatVector<int64_t>(size, expectedValueAt);
  assertEqualVectors(expected, result);

  result = evaluate(
      "case when c1 < 7 then c1 + 5 when c1 < 11 then c1 - 11 "
      "else c1::BIGINT end",
      vector);
  expected = makeFlatVector<int64_t>(size, expectedValueAt, nullEvery(5));
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, swithExprSanityChecks) {
  auto vector = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  // Then clauses have different types.
  VELOX_ASSERT_THROW(
      evaluate(
          "case c0 when 7 then 1 when 11 then 'welcome' else 0 end", vector),
      "All then clauses of a SWITCH statement must have the same type. "
      "Expected BIGINT, but got VARCHAR.");

  // Else clause has different type.
  VELOX_ASSERT_THROW(
      evaluate("case c0 when 7 then 1 when 11 then 2 else 'hello' end", vector),
      "Else clause of a SWITCH statement must have the same type as 'then' clauses. "
      "Expected BIGINT, but got VARCHAR.");

  // Unknown is not implicitly casted.
  VELOX_ASSERT_THROW(
      evaluate("case c0 when 7 then 1 when 11 then null else 3 end", vector),
      "All then clauses of a SWITCH statement must have the same type. "
      "Expected BIGINT, but got UNKNOWN.");

  // Unknown is not implicitly casted.
  VELOX_ASSERT_THROW(
      evaluate(
          "case c0 when 7 then  row_constructor(null, 1) when 11 then  row_constructor(1, null) end",
          vector),
      "All then clauses of a SWITCH statement must have the same type. "
      "Expected ROW<c1:UNKNOWN,c2:BIGINT>, but got ROW<c1:BIGINT,c2:UNKNOWN>.");
}

TEST_F(ExprTest, switchExprWithNull) {
  vector_size_t size = 1'000;
  // Build an input with c0 column having nulls at odd row index.
  auto vector = makeRowVector(
      {makeFlatVector<int32_t>(
           size,
           [](auto /*unused*/) { return 7; },
           [](auto row) { return row % 2; }),
       makeFlatVector<int32_t>(
           size, [](auto row) { return row; }, nullEvery(5)),
       makeConstant<int32_t>(0, size)});

  auto result = evaluate("case c0 when 7 then 1 else 0 end", vector);
  // If 'c0' is null, then we shall get 0 from else branch.
  auto expected = makeFlatVector<int64_t>(size, [](auto row) {
    if (row % 2 == 0) {
      return 1;
    } else {
      return 0;
    }
  });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, ifWithConstant) {
  vector_size_t size = 4;

  auto a = makeFlatVector<int32_t>({-1, -2, -3, -4});
  auto b = makeNullConstant(TypeKind::INTEGER, size); // 4 nulls
  auto result = evaluate("is_null(if(c0 > 0, c0, c1))", makeRowVector({a, b}));
  EXPECT_EQ(VectorEncoding::Simple::CONSTANT, result->encoding());
  EXPECT_EQ(true, result->as<ConstantVector<bool>>()->valueAt(0));
}

namespace {
// Testing functions for generating intermediate results in different
// encodings. The test case passes vectors to these and these
// functions make constant/dictionary/sequence vectors from their arguments

// Returns the first value of the argument vector wrapped as a constant.
class TestingConstantFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& /*context*/,
      VectorPtr& result) const override {
    VELOX_CHECK(rows.isAllSelected());
    result = BaseVector::wrapInConstant(rows.size(), 0, args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .build()};
  }
};

// Returns a dictionary vector with values from the first argument
// vector and indices from the second.
class TestingDictionaryFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& /*context*/,
      VectorPtr& result) const override {
    VELOX_CHECK(rows.isAllSelected());
    auto& indices = args[1]->as<FlatVector<int32_t>>()->values();
    result = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices, rows.size(), args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T, integer -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .argumentType("integer")
                .build()};
  }
};

// Takes a vector  of values and a vector of integer run lengths.
// Wraps the values in a SequenceVector with the run lengths.

class TestingSequenceFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK(rows.isAllSelected());
    auto lengths = args[1]->as<FlatVector<int32_t>>()->values();
    result = BaseVector::wrapInSequence(lengths, rows.size(), args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T, integer -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .argumentType("integer")
                .build()};
  }
};

// Single-argument deterministic functions always receive their argument
// vector as flat or constant.
class TestingSingleArgDeterministicFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto& arg = args[0];
    VELOX_CHECK(arg->isFlatEncoding() || arg->isConstantEncoding());
    BaseVector::ensureWritable(rows, outputType, context.pool(), result);
    result->copy(arg.get(), rows, nullptr);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .build()};
  }
};

} // namespace
VELOX_DECLARE_VECTOR_FUNCTION(
    udf_testing_constant,
    TestingConstantFunction::signatures(),
    std::make_unique<TestingConstantFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_testing_dictionary,
    TestingDictionaryFunction::signatures(),
    std::make_unique<TestingDictionaryFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_testing_sequence,
    TestingSequenceFunction::signatures(),
    std::make_unique<TestingSequenceFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_testing_single_arg_deterministic,
    TestingSingleArgDeterministicFunction::signatures(),
    std::make_unique<TestingSingleArgDeterministicFunction>());

TEST_F(ExprTest, peelArgs) {
  constexpr int32_t kSize = 100;
  constexpr int32_t kDistinct = 10;
  VELOX_REGISTER_VECTOR_FUNCTION(udf_testing_constant, "testing_constant");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_testing_dictionary, "testing_dictionary");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_testing_sequence, "testing_sequence");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_testing_single_arg_deterministic, "testing_single_arg_deterministic");

  std::vector<int32_t> onesSource(kSize, 1);
  std::vector<int32_t> distinctSource(kDistinct);
  std::iota(distinctSource.begin(), distinctSource.end(), 11);
  std::vector<vector_size_t> indicesSource(kSize);
  for (auto i = 0; i < indicesSource.size(); ++i) {
    indicesSource[i] = i % kDistinct;
  }
  std::vector lengthSource(kDistinct, kSize / kDistinct);
  auto allOnes = makeFlatVector<int32_t>(onesSource);

  // constant
  auto result = evaluate("1 + testing_constant(c0)", makeRowVector({allOnes}));
  auto expected64 =
      makeFlatVector<int64_t>(kSize, [](int32_t /*i*/) { return 2; });
  assertEqualVectors(expected64, result);
  result = evaluate(
      "testing_constant(c0) + testing_constant(c1)",
      makeRowVector({allOnes, allOnes}));
  auto expected32 =
      makeFlatVector<int32_t>(kSize, [](int32_t /*i*/) { return 2; });
  assertEqualVectors(expected32, result);

  // Constant and dictionary
  auto distincts = makeFlatVector<int32_t>(distinctSource);
  auto indices = makeFlatVector<int32_t>(indicesSource);
  result = evaluate(
      "testing_constant(c0) + testing_dictionary(c1, c2)",
      makeRowVector({allOnes, distincts, indices}));
  expected32 = makeFlatVector<int32_t>(kSize, [&](int32_t row) {
    return 1 + distinctSource[indicesSource[row]];
  });
  assertEqualVectors(expected32, result);

  auto lengths = makeFlatVector<int32_t>(lengthSource);
  result = evaluate(
      "testing_constant(c0) + testing_sequence(c1, c2)",
      makeRowVector({allOnes, distincts, lengths}));
  expected32 = makeFlatVector<int32_t>(kSize, [&](int32_t row) {
    return 1 + distinctSource[row / (kSize / kDistinct)];
  });

  assertEqualVectors(expected32, result);

  // dictionary and single-argument deterministic
  indices = makeFlatVector<int32_t>(kSize, [](auto) {
    // having all indices to be the same makes DictionaryVector::isConstant()
    // returns true
    return 0;
  });
  result = evaluate(
      "testing_single_arg_deterministic(testing_dictionary(c1, c0))",
      makeRowVector({indices, distincts}));
  expected32 = makeFlatVector<int32_t>(kSize, [](int32_t /*i*/) { return 11; });
  assertEqualVectors(expected32, result);
}

class NullArrayFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // This function returns a vector of all nulls
    BaseVector::ensureWritable(rows, ARRAY(VARCHAR()), context.pool(), result);
    result->addNulls(nullptr, rows);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T... -> array(varchar)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(varchar)")
                .argumentType("T")
                .variableArity()
                .build()};
  }
};

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_null_array,
    NullArrayFunction::signatures(),
    std::make_unique<NullArrayFunction>());

TEST_F(ExprTest, complexNullOutput) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_null_array, "null_array");

  auto row = makeRowVector({makeAllNullFlatVector<int64_t>(1)});

  auto expectedResults =
      BaseVector::createNullConstant(ARRAY(VARCHAR()), 1, execCtx_->pool());
  auto resultForNulls = evaluate("null_array(NULL, NULL)", row);

  // Making sure the output of the function is the same when returning all
  // null or called on NULL constants
  assertEqualVectors(expectedResults, resultForNulls);
}

TEST_F(ExprTest, rewriteInputs) {
  // rewrite one field
  {
    auto expr = parseExpression(
        "(a + b) * 2.1", ROW({"a", "b"}, {INTEGER(), DOUBLE()}));
    expr = expr->rewriteInputNames({{"a", "alpha"}});

    auto expectedExpr = parseExpression(
        "(alpha + b) * 2.1", ROW({"alpha", "b"}, {INTEGER(), DOUBLE()}));
    ASSERT_EQ(*expectedExpr, *expr);
  }

  // rewrite 2 fields
  {
    auto expr = parseExpression(
        "a + b * c", ROW({"a", "b", "c"}, {INTEGER(), DOUBLE(), DOUBLE()}));
    expr = expr->rewriteInputNames({{"a", "alpha"}, {"b", "beta"}});

    auto expectedExpr = parseExpression(
        "alpha + beta * c",
        ROW({"alpha", "beta", "c"}, {INTEGER(), DOUBLE(), DOUBLE()}));
    ASSERT_EQ(*expectedExpr, *expr);
  }
}

TEST_F(ExprTest, memo) {
  auto base = makeArrayVector<int64_t>(
      1'000,
      [](auto row) { return row % 5 + 1; },
      [](auto row, auto index) { return (row % 3) + index; });

  auto evenIndices = makeIndices(100, [](auto row) { return 8 + row * 2; });
  auto oddIndices = makeIndices(100, [](auto row) { return 9 + row * 2; });

  auto rowType = ROW({"c0"}, {base->type()});
  auto exprSet = compileExpression("c0[1] = 1", rowType);

  auto result = evaluate(
      exprSet.get(), makeRowVector({wrapInDictionary(evenIndices, 100, base)}));
  auto expectedResult = makeFlatVector<bool>(
      100, [](auto row) { return (8 + row * 2) % 3 == 1; });
  assertEqualVectors(expectedResult, result);

  result = evaluate(
      exprSet.get(), makeRowVector({wrapInDictionary(oddIndices, 100, base)}));
  expectedResult = makeFlatVector<bool>(
      100, [](auto row) { return (9 + row * 2) % 3 == 1; });
  assertEqualVectors(expectedResult, result);

  auto everyFifth = makeIndices(100, [](auto row) { return row * 5; });
  result = evaluate(
      exprSet.get(), makeRowVector({wrapInDictionary(everyFifth, 100, base)}));
  expectedResult =
      makeFlatVector<bool>(100, [](auto row) { return (row * 5) % 3 == 1; });
  assertEqualVectors(expectedResult, result);
}

// This test triggers the situation when peelEncodings() produces an empty
// selectivity vector, which if passed to evalWithMemo() causes the latter to
// produce null Expr::dictionaryCache_, which leads to a crash in evaluation
// of subsequent rows. We have fixed that issue with condition and this test
// is for that.
TEST_F(ExprTest, memoNulls) {
  // Generate 5 rows with null string and 5 with a string.
  auto base = makeFlatVector<StringView>(
      10, [](vector_size_t /*row*/) { return StringView("abcdefg"); });

  // Two batches by 5 rows each.
  auto first5Indices = makeIndices(5, [](auto row) { return row; });
  auto last5Indices = makeIndices(5, [](auto row) { return row + 5; });
  // Nulls for the 1st batch.
  BufferPtr nulls =
      AlignedBuffer::allocate<bool>(5, execCtx_->pool(), bits::kNull);

  auto rowType = ROW({"c0"}, {base->type()});
  auto exprSet = compileExpression("STRPOS(c0, 'abc') >= 0", rowType);

  auto result = evaluate(
      exprSet.get(),
      makeRowVector(
          {BaseVector::wrapInDictionary(nulls, first5Indices, 5, base)}));
  // Expecting 5 nulls.
  auto expectedResult =
      BaseVector::createNullConstant(BOOLEAN(), 5, execCtx_->pool());
  assertEqualVectors(expectedResult, result);

  result = evaluate(
      exprSet.get(), makeRowVector({wrapInDictionary(last5Indices, 5, base)}));
  // Expecting 5 trues.
  expectedResult = makeConstant(true, 5);
  assertEqualVectors(expectedResult, result);
}

// This test is carefully constructed to exercise calling
// applyFunctionWithPeeling in a situation where inputValues_ can be peeled
// and applyRows and rows are distinct SelectivityVectors.  This test ensures
// we're using applyRows and rows in the right places, if not we should see a
// SIGSEGV.
TEST_F(ExprTest, peelNulls) {
  // Generate 5 distinct values for the c0 column.
  auto c0 = makeFlatVector<StringView>(5, [](vector_size_t row) {
    std::string val = "abcdefg";
    val.append(2, 'a' + row);
    return StringView(val);
  });
  // Generate 5 values for the c1 column.
  auto c1 = makeFlatVector<StringView>(
      5, [](vector_size_t /*row*/) { return StringView("xyz"); });

  // One batch of 5 rows.
  auto c0Indices = makeIndices(5, [](auto row) { return row; });
  auto c1Indices = makeIndices(5, [](auto row) { return row; });

  auto rowType = ROW({"c0", "c1"}, {c0->type(), c1->type()});
  // This expression is very deliberately written this way.
  // REGEXP_EXTRACT will return null for all but row 2, it is important we
  // get nulls and non-nulls so applyRows and rows will be distinct.
  // The result of REVERSE will be collapsed into a constant vector, which
  // is necessary so that the inputValues_ can be peeled.
  // REGEXP_LIKE is the function for which applyFunctionWithPeeling will be
  // called.
  auto exprSet = compileExpression(
      "REGEXP_LIKE(REGEXP_EXTRACT(c0, 'cc'), REVERSE(c1))", rowType);

  // It is important that both columns be wrapped in DictionaryVectors so
  // that they are not peeled until REGEXP_LIKE's children have been
  // evaluated.
  auto result = evaluate(
      exprSet.get(),
      makeRowVector(
          {BaseVector::wrapInDictionary(nullptr, c0Indices, 5, c0),
           BaseVector::wrapInDictionary(nullptr, c1Indices, 5, c1)}));

  // Since c0 only has 'cc' as a substring in row 2, all other rows should be
  // null.
  auto expectedResult = makeFlatVector<bool>(
      5,
      [](vector_size_t /*row*/) { return false; },
      [](vector_size_t row) { return row != 2; });
  assertEqualVectors(expectedResult, result);
}

TEST_F(ExprTest, peelLazyDictionaryOverConstant) {
  auto c0 = makeFlatVector<int64_t>(5, [](vector_size_t row) { return row; });
  auto c0Indices = makeIndices(5, [](auto row) { return row; });
  auto c1 = makeFlatVector<int64_t>(5, [](auto row) { return row; });

  auto result = evaluate(
      "if (not(is_null(if (c0 >= 0, c1, cast (null as bigint)))), coalesce(c0, 22), cast (null as bigint))",
      makeRowVector(
          {BaseVector::wrapInDictionary(
               nullptr, c0Indices, 5, wrapInLazyDictionary(c0)),
           BaseVector::wrapInDictionary(
               nullptr, c0Indices, 5, wrapInLazyDictionary(c1))}));
  assertEqualVectors(c0, result);
}

TEST_F(ExprTest, accessNested) {
  // Construct row(row(row(integer))) vector.
  auto base = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto level1 = makeRowVector({base});
  auto level2 = makeRowVector({level1});
  auto level3 = makeRowVector({level2});

  // Access level3->level2->level1->base.
  // TODO: Expression "c0.c0.c0" currently not supported by DuckDB
  // So we wrap with parentheses to force parsing as struct extract
  // Track https://github.com/duckdb/duckdb/issues/2568
  auto result = evaluate("(c0).c0.c0.c0", makeRowVector({level3}));

  assertEqualVectors(base, result);
}

TEST_F(ExprTest, accessNestedNull) {
  // Construct row(row(row(integer))) vector.
  auto base = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto level1 = makeRowVector({base});

  // Construct level 2 row with nulls.
  auto level2 = makeRowVector({level1}, nullEvery(2));
  auto level3 = makeRowVector({level2});

  auto result = evaluate("(c0).c0.c0.c0", makeRowVector({level3}));
  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, 2, std::nullopt, 4, std::nullopt});
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, accessNestedDictionaryEncoding) {
  // Construct row(row(row(integer))) vector.
  auto base = makeFlatVector<int32_t>({1, 2, 3, 4, 5});

  // Reverse order in dictionary encoding.
  auto indices = makeIndicesInReverse(5);

  auto level1 = makeRowVector({base});
  auto level2 = makeRowVector({wrapInDictionary(indices, 5, level1)});
  auto level3 = makeRowVector({level2});

  auto result = evaluate("(c0).c0.c0.c0", makeRowVector({level3}));

  assertEqualVectors(makeFlatVector<int32_t>({5, 4, 3, 2, 1}), result);
}

TEST_F(ExprTest, accessNestedConstantEncoding) {
  // Construct row(row(row(integer))) vector.
  VectorPtr base = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  // Wrap base in constant.
  base = BaseVector::wrapInConstant(5, 2, base);

  auto level1 = makeRowVector({base});
  auto level2 = makeRowVector({level1});
  auto level3 = makeRowVector({level2});

  auto result = evaluate("(c0).c0.c0.c0", makeRowVector({level3}));

  assertEqualVectors(makeConstant(3, 5), result);
}

TEST_F(ExprTest, testEmptyVectors) {
  auto a = makeFlatVector<int32_t>({});
  auto result = evaluate("c0 + c0", makeRowVector({a, a}));
  assertEqualVectors(a, result);
}

TEST_F(ExprTest, subsetOfDictOverLazy) {
  // We have dictionaries over LazyVector. We load for some indices in
  // the top dictionary. The intermediate dictionaries refer to
  // non-loaded items in the base of the LazyVector, including indices
  // past its end. We check that we end up with one level of
  // dictionary and no dictionaries that are invalid by through
  // referring to uninitialized/nonexistent positions.
  auto base = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  auto lazy = std::make_shared<LazyVector>(
      execCtx_->pool(),
      INTEGER(),
      1000,
      std::make_unique<test::SimpleVectorLoader>(
          [base](auto /*size*/) { return base; }));
  auto row = makeRowVector({BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(100, [](auto row) { return row; }),
      100,

      BaseVector::wrapInDictionary(
          nullptr,
          makeIndices(1000, [](auto row) { return row; }),
          1000,
          lazy))});

  // We expect a single level of dictionary.
  auto result = evaluate("c0", row);
  EXPECT_EQ(result->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(result->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  assertEqualVectors(result, base);
}

TEST_F(ExprTest, peeledConstant) {
  constexpr int32_t kSubsetSize = 80;
  constexpr int32_t kBaseSize = 160;
  auto indices = makeIndices(kSubsetSize, [](auto row) { return row * 2; });
  auto numbers =
      makeFlatVector<int32_t>(kBaseSize, [](auto row) { return row; });
  auto row = makeRowVector({
      wrapInDictionary(indices, kSubsetSize, numbers),
      makeConstant("Hans Pfaal", kBaseSize),
  });
  auto result = std::dynamic_pointer_cast<SimpleVector<StringView>>(
      evaluate("if (c0 % 4 = 0, c1, cast (null as VARCHAR))", row));
  EXPECT_EQ(kSubsetSize, result->size());
  for (auto i = 0; i < kSubsetSize; ++i) {
    if (result->isNullAt(i)) {
      continue;
    }
    EXPECT_LE(1, result->valueAt(i).size());
    // Check that the data is readable.
    EXPECT_NO_THROW(result->toString(i));
  }
}

TEST_F(ExprTest, exceptionContext) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  registerFunction<TestingAlwaysThrowsFunction, int32_t, int32_t>(
      {"always_throws"});

  // Disable saving vector and expression SQL on error.
  FLAGS_velox_save_input_on_expression_any_failure_path = "";
  FLAGS_velox_save_input_on_expression_system_failure_path = "";

  try {
    evaluate("always_throws(c0) + c1", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("always_throws(c0)", e.context());
    ASSERT_EQ("plus(always_throws(c0), c1)", e.topLevelContext());
  }

  try {
    evaluate("c0 + (c0 + c1) % 0", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT))",
        e.topLevelContext());
  }

  try {
    evaluate("c0 + (c1 % 0)", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((c1) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((c1) as BIGINT), 0:BIGINT))",
        e.topLevelContext());
  }

  // Enable saving vector and expression SQL for system errors only.
  auto tempDirectory = exec::test::TempDirectoryPath::create();
  FLAGS_velox_save_input_on_expression_system_failure_path =
      tempDirectory->path;

  try {
    evaluate("always_throws(c0) + c1", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("always_throws(c0)", e.context());
    ASSERT_EQ(
        "plus(always_throws(c0), c1)", trimInputPath(e.topLevelContext()));
    verifyDataAndSqlPaths(e, data);
  }

  try {
    evaluate("c0 + (c0 + c1) % 0", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT))",
        e.topLevelContext())
        << e.errorSource();
  }

  try {
    evaluate("c0 + (c0 + c1) % 0", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT))",
        e.topLevelContext());
  }

  // Enable saving vector and expression SQL for all errors.
  FLAGS_velox_save_input_on_expression_any_failure_path = tempDirectory->path;
  FLAGS_velox_save_input_on_expression_system_failure_path = "";

  try {
    evaluate("always_throws(c0) + c1", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("always_throws(c0)", e.context());
    ASSERT_EQ(
        "plus(always_throws(c0), c1)", trimInputPath(e.topLevelContext()));
    verifyDataAndSqlPaths(e, data);
  }

  try {
    evaluate("c0 + (c0 + c1) % 0", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((plus(c0, c1)) as BIGINT), 0:BIGINT))",
        trimInputPath(e.topLevelContext()));
    verifyDataAndSqlPaths(e, data);
  }

  try {
    evaluate("c0 + (c1 % 0)", data);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    ASSERT_EQ("mod(cast((c1) as BIGINT), 0:BIGINT)", e.context());
    ASSERT_EQ(
        "plus(cast((c0) as BIGINT), mod(cast((c1) as BIGINT), 0:BIGINT))",
        trimInputPath(e.topLevelContext()));
    verifyDataAndSqlPaths(e, data);
  }
}

namespace {

template <typename T>
struct AlwaysThrowsStdExceptionFunction {
  template <typename TResult, typename TInput>
  FOLLY_ALWAYS_INLINE void call(TResult&, const TInput&) {
    throw std::invalid_argument("This is a test");
  }
};
} // namespace

/// Verify exception context for the case when function throws std::exception.
TEST_F(ExprTest, stdExceptionContext) {
  auto data = makeFlatVector<int64_t>({1, 2, 3});

  registerFunction<AlwaysThrowsStdExceptionFunction, int64_t, int64_t>(
      {"throw_invalid_argument"});

  auto wrappedEx = assertWrappedException(
      "throw_invalid_argument(c0) + 5",
      data,
      "throw_invalid_argument(c0)",
      "plus(throw_invalid_argument(c0), 5:BIGINT)",
      "This is a test");
  ASSERT_THROW(std::rethrow_exception(wrappedEx), std::invalid_argument);

  wrappedEx = assertWrappedException(
      "throw_invalid_argument(c0 + 5)",
      data,
      "throw_invalid_argument(plus(c0, 5:BIGINT))",
      "Same as context.",
      "This is a test");
  ASSERT_THROW(std::rethrow_exception(wrappedEx), std::invalid_argument);
}

/// Verify the output of ConstantExpr::toString().
TEST_F(ExprTest, constantToString) {
  auto arrayVector =
      makeNullableArrayVector<float>({{1.2, 3.4, std::nullopt, 5.6}});

  exec::ExprSet exprSet(
      {std::make_shared<core::ConstantTypedExpr>(INTEGER(), 23),
       std::make_shared<core::ConstantTypedExpr>(
           DOUBLE(), variant::null(TypeKind::DOUBLE)),
       makeConstantExpr(arrayVector, 0)},
      execCtx_.get());

  ASSERT_EQ("23:INTEGER", exprSet.exprs()[0]->toString());
  ASSERT_EQ("null:DOUBLE", exprSet.exprs()[1]->toString());
  ASSERT_EQ(
      "4 elements starting at 0 {1.2000000476837158, 3.4000000953674316, null, 5.599999904632568}:ARRAY<REAL>",
      exprSet.exprs()[2]->toString());
}

TEST_F(ExprTest, constantToSql) {
  auto toSql = [&](const variant& value, const TypePtr& type = nullptr) {
    exec::ExprSet exprSet({makeConstantExpr(value, type)}, execCtx_.get());
    auto sql = exprSet.expr(0)->toSql();

    auto input = makeRowVector(ROW({}), 1);
    auto a = evaluate(&exprSet, input);
    auto b = evaluate(sql, input);

    if (a->type()->containsUnknown()) {
      EXPECT_TRUE(a->isNullAt(0));
      EXPECT_TRUE(b->isNullAt(0));
    } else {
      assertEqualVectors(a, b);
    }

    return sql;
  };

  ASSERT_EQ(toSql(true), "TRUE");
  ASSERT_EQ(toSql(false), "FALSE");
  ASSERT_EQ(toSql(variant::null(TypeKind::BOOLEAN)), "NULL::BOOLEAN");

  ASSERT_EQ(toSql((int8_t)23), "'23'::TINYINT");
  ASSERT_EQ(toSql(variant::null(TypeKind::TINYINT)), "NULL::TINYINT");

  ASSERT_EQ(toSql((int16_t)23), "'23'::SMALLINT");
  ASSERT_EQ(toSql(variant::null(TypeKind::SMALLINT)), "NULL::SMALLINT");

  ASSERT_EQ(toSql(23), "'23'::INTEGER");
  ASSERT_EQ(toSql(variant::null(TypeKind::INTEGER)), "NULL::INTEGER");

  ASSERT_EQ(toSql(2134456LL), "'2134456'::BIGINT");
  ASSERT_EQ(toSql(variant::null(TypeKind::BIGINT)), "NULL::BIGINT");

  ASSERT_EQ(toSql(Date(18'506)), "'2020-09-01'::DATE");
  ASSERT_EQ(toSql(variant::null(TypeKind::DATE)), "NULL::DATE");

  ASSERT_EQ(
      toSql(Timestamp(123'456, 123'000)),
      "'1970-01-02T10:17:36.000123000'::TIMESTAMP");
  ASSERT_EQ(toSql(variant::null(TypeKind::TIMESTAMP)), "NULL::TIMESTAMP");

  ASSERT_EQ(
      toSql(123'456LL, INTERVAL_DAY_TIME()),
      "'123456'::INTERVAL DAY TO SECOND");
  ASSERT_EQ(
      toSql(variant::null(TypeKind::BIGINT), INTERVAL_DAY_TIME()),
      "NULL::INTERVAL DAY TO SECOND");

  ASSERT_EQ(toSql(1.5f), "'1.5'::REAL");
  ASSERT_EQ(toSql(variant::null(TypeKind::REAL)), "NULL::REAL");

  ASSERT_EQ(toSql(-78.456), "'-78.456'::DOUBLE");
  ASSERT_EQ(toSql(variant::null(TypeKind::DOUBLE)), "NULL::DOUBLE");

  ASSERT_EQ(toSql("This is a test."), "'This is a test.'");
  ASSERT_EQ(
      toSql("This is a \'test\' with single quotes."),
      "'This is a \'\'test\'\' with single quotes.'");
  ASSERT_EQ(toSql(variant::null(TypeKind::VARCHAR)), "NULL::VARCHAR");

  auto toSqlComplex = [&](const VectorPtr& vector, vector_size_t index = 0) {
    exec::ExprSet exprSet({makeConstantExpr(vector, index)}, execCtx_.get());
    auto sql = exprSet.expr(0)->toSql();

    auto input = makeRowVector(ROW({}), 1);
    auto a = evaluate(&exprSet, input);
    auto b = evaluate(sql, input);

    if (a->type()->containsUnknown()) {
      EXPECT_TRUE(a->isNullAt(0));
      EXPECT_TRUE(b->isNullAt(0));
    } else {
      assertEqualVectors(a, b);
    }

    return sql;
  };

  ASSERT_EQ(
      toSqlComplex(makeArrayVector<int32_t>({{1, 2, 3}})),
      "ARRAY['1'::INTEGER, '2'::INTEGER, '3'::INTEGER]");
  ASSERT_EQ(
      toSqlComplex(makeArrayVector<int32_t>({{1, 2, 3}, {4, 5, 6}}), 1),
      "ARRAY['4'::INTEGER, '5'::INTEGER, '6'::INTEGER]");
  ASSERT_EQ(toSql(variant::null(TypeKind::ARRAY)), "NULL");

  ASSERT_EQ(
      toSqlComplex(makeMapVector<int32_t, int32_t>({
          {{1, 10}, {2, 20}, {3, 30}},
      })),
      "map(ARRAY['1'::INTEGER, '2'::INTEGER, '3'::INTEGER], ARRAY['10'::INTEGER, '20'::INTEGER, '30'::INTEGER])");
  ASSERT_EQ(
      toSqlComplex(
          makeMapVector<int32_t, int32_t>({
              {{1, 11}, {2, 12}},
              {{1, 10}, {2, 20}, {3, 30}},
          }),
          1),
      "map(ARRAY['1'::INTEGER, '2'::INTEGER, '3'::INTEGER], ARRAY['10'::INTEGER, '20'::INTEGER, '30'::INTEGER])");
  ASSERT_EQ(
      toSqlComplex(BaseVector::createNullConstant(
          MAP(INTEGER(), VARCHAR()), 10, pool())),
      "NULL::MAP(INTEGER, VARCHAR)");

  ASSERT_EQ(
      toSqlComplex(makeRowVector({
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<bool>({true, false, true}),
      })),
      "row_constructor('1'::INTEGER, TRUE)");
  ASSERT_EQ(
      toSqlComplex(BaseVector::createNullConstant(
          ROW({"a", "b"}, {BOOLEAN(), DOUBLE()}), 10, pool())),
      "NULL::STRUCT(a BOOLEAN, b DOUBLE)");
}

TEST_F(ExprTest, toSql) {
  auto rowType =
      ROW({"a", "b", "c.d", "e", "f"},
          {INTEGER(),
           BIGINT(),
           VARCHAR(),
           ARRAY(BIGINT()),
           MAP(VARCHAR(), DOUBLE())});

  // CAST.
  testToSql("a + 3", rowType);
  testToSql("a * b", rowType);
  testToSql("a * 1.5", rowType);
  testToSql("cast(e as varchar[])", rowType);
  testToSql("cast(f as map(bigint, varchar))", rowType);
  testToSql(
      "cast(row_constructor(a, b) as struct(x bigint, y double))", rowType);

  // SWITCH.
  testToSql("if(a > 0, 1, 10)", rowType);
  testToSql("if(a = 10, true, false)", rowType);
  testToSql("case a when 7 then 1 when 11 then 2 else 0 end", rowType);
  testToSql("case a when 7 then 1 when 11 then 2 when 17 then 3 end", rowType);
  testToSql(
      "case a when b + 3 then 1 when b * 11 then 2 when b - 17 then a + b end",
      rowType);

  // AND / OR.
  testToSql("a > 0 AND b < 100", rowType);
  testToSql("a > 0 AND b / a < 100", rowType);
  testToSql("is_null(a) OR is_null(b)", rowType);
  testToSql("a > 10 AND (b > 100 OR a < 0) AND b < 3", rowType);

  // COALESCE.
  testToSql("coalesce(a::bigint, b, 123)", rowType);

  // TRY.
  testToSql("try(a / b)", rowType);

  // String literals.
  testToSql("length(\"c.d\")", rowType);
  testToSql("concat(a::varchar, ',', b::varchar, '\'\'')", rowType);

  // Array, map and row literals.
  testToSql("contains(array[1, 2, 3], a)", rowType);
  testToSql("map(array[a, b, 5], array[10, 20, 30])", rowType);
  testToSql(
      "element_at(map(array[1, 2, 3], array['a', 'b', 'c']), a)", rowType);
  testToSql("row_constructor(a, b, 'test')", rowType);
  testToSql("row_constructor(true, 1.5, 'abc', array[1, 2, 3])", rowType);

  // Lambda functions.
  testToSql("filter(e, x -> (x > 10))", rowType);
  testToSql("transform(e, x -> x + b)", rowType);
  testToSql("map_filter(f, (k, v) -> (v > 10::double))", rowType);
  testToSql("reduce(e, b, (s, x) -> s + x, s -> s * 10)", rowType);

  // Function without inputs.
  testToSql("pi()", rowType);
}

namespace {
// A naive function that wraps the input in a dictionary vector resized to
// rows.end() - 1.  It assumes all selected values are non-null.
class TestingShrinkingDictionary : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), context.pool());
    auto rawIndices = indices->asMutable<vector_size_t>();
    rows.applyToSelected([&](int row) { rawIndices[row] = row; });

    result =
        BaseVector::wrapInDictionary(nullptr, indices, rows.end() - 1, args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("bigint")
                .argumentType("bigint")
                .build()};
  }
};
} // namespace

TEST_F(ExprTest, specialFormPropagateNulls) {
  exec::registerVectorFunction(
      "test_shrinking_dictionary",
      TestingShrinkingDictionary::signatures(),
      std::make_unique<TestingShrinkingDictionary>());

  // This test verifies an edge case where applyFunctionWithPeeling may produce
  // a result vector which is dictionary encoded and has fewer values than
  // are rows.
  // This can happen when the last value in a column used in an expression is
  // null which causes removeSureNulls to move the end of the SelectivityVector
  // forward.  When we incorrectly use rows.end() as the size of the
  // dictionary when rewrapping the results.
  // Normally this is masked when this vector is used in a function call which
  // produces a new output vector.  However, in SpecialForm expressions, we may
  // return the output untouched, and when we try to add back in the nulls, we
  // get an exception trying to resize the DictionaryVector.
  // This is difficult to reproduce, so this test artificially triggers the
  // issue by using a UDF that returns a dictionary one smaller than rows.end().

  // Making the last row NULL, so we call addNulls in eval.
  auto c0 = makeFlatVector<int64_t>(
      5,
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row == 4; });

  auto rowVector = makeRowVector({c0});
  auto evalResult = evaluate("test_shrinking_dictionary(\"c0\")", rowVector);

  auto expectedResult = makeFlatVector<int64_t>(
      5,
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row == 4; });
  assertEqualVectors(expectedResult, evalResult);
}

TEST_F(ExprTest, tryWithConstantFailure) {
  // This test verifies the behavior of constant peeling on a function wrapped
  // in a TRY.  Specifically the case when the UDF executed on the peeled
  // vector throws an exception on the constant value.

  // When wrapping a peeled ConstantVector, the result is wrapped in a
  // ConstantVector.  ConstantVector has special handling logic to copy the
  // underlying string when the type is Varchar.  When an exception is thrown
  // and the StringView isn't initialized, without special handling logic in
  // EvalCtx this results in reading uninitialized memory triggering ASAN
  // errors.
  registerFunction<TestingAlwaysThrowsFunction, Varchar, Varchar>(
      {"always_throws"});
  auto c0 = makeConstant("test", 5);
  auto c1 = makeFlatVector<int64_t>(5, [](vector_size_t row) { return row; });
  auto rowVector = makeRowVector({c0, c1});

  // We use strpos and c1 to ensure that the constant is peeled before calling
  // always_throws, not before the try.
  auto evalResult =
      evaluate("try(strpos(always_throws(\"c0\"), 't', c1))", rowVector);

  auto expectedResult = makeFlatVector<int64_t>(
      5, [](vector_size_t) { return 0; }, [](vector_size_t) { return true; });
  assertEqualVectors(expectedResult, evalResult);
}

TEST_F(ExprTest, castExceptionContext) {
  assertError(
      "cast(c0 as bigint)",
      makeFlatVector<std::string>({"1a"}),
      "cast((c0) as BIGINT)",
      "Same as context.",
      "Failed to cast from VARCHAR to BIGINT: 1a. Non-whitespace character found after end of conversion: \"a\"");

  assertError(
      "cast(c0 as timestamp)",
      makeFlatVector(std::vector<int8_t>{1}),
      "cast((c0) as TIMESTAMP)",
      "Same as context.",
      "Failed to cast from TINYINT to TIMESTAMP: 1. Conversion to Timestamp is not supported");
}

TEST_F(ExprTest, switchExceptionContext) {
  assertError(
      "case c0 when 7 then c0 / 0 else 0 end",
      makeFlatVector(std::vector<int64_t>{7}),
      "divide(c0, 0:BIGINT)",
      "switch(eq(c0, 7:BIGINT), divide(c0, 0:BIGINT), 0:BIGINT)",
      "division by zero");
}

TEST_F(ExprTest, conjunctExceptionContext) {
  auto data = makeFlatVector<int64_t>(20, [](auto row) { return row; });

  assertError(
      "if (c0 % 409 < 300 and c0 / 0 < 30, 1, 2)",
      data,
      "divide(c0, 0:BIGINT)",
      "switch(and(lt(mod(c0, 409:BIGINT), 300:BIGINT), lt(divide(c0, 0:BIGINT), 30:BIGINT)), 1:BIGINT, 2:BIGINT)",
      "division by zero");
}

TEST_F(ExprTest, lambdaExceptionContext) {
  auto array = makeArrayVector<int64_t>(
      10, [](auto /*row*/) { return 5; }, [](auto row) { return row * 3; });

  assertError(
      "filter(c0, x -> (x / 0 > 1))",
      array,
      "divide(x, 0:BIGINT)",
      "filter(c0, (x) -> gt(divide(x, 0:BIGINT), 1:BIGINT))",
      "division by zero");
}

/// Verify that null inputs result in exceptions, not crashes.
TEST_F(ExprTest, invalidInputs) {
  auto rowType = ROW({"a"}, {BIGINT()});
  auto exprSet = compileExpression("a + 5", rowType);

  // Try null top-level vector.
  RowVectorPtr input;
  ASSERT_THROW(
      exec::EvalCtx(execCtx_.get(), exprSet.get(), input.get()),
      VeloxRuntimeError);

  // Try non-null vector with null children.
  input = std::make_shared<RowVector>(
      pool_.get(), rowType, nullptr, 1024, std::vector<VectorPtr>{nullptr});
  ASSERT_THROW(
      exec::EvalCtx(execCtx_.get(), exprSet.get(), input.get()),
      VeloxRuntimeError);
}

TEST_F(ExprTest, lambdaWithRowField) {
  auto array = makeArrayVector<int64_t>(
      10, [](auto /*row*/) { return 5; }, [](auto row) { return row * 3; });
  auto row = makeRowVector(
      {"val"},
      {makeFlatVector<int64_t>(10, [](vector_size_t row) { return row; })});

  auto rowVector = makeRowVector({"c0", "c1"}, {row, array});

  // We use strpos and c1 to ensure that the constant is peeled before calling
  // always_throws, not before the try.
  auto evalResult = evaluate("filter(c1, x -> (x + c0.val >= 0))", rowVector);

  assertEqualVectors(array, evalResult);
}

TEST_F(ExprTest, flatNoNullsFastPath) {
  auto data = makeRowVector(
      {"a", "b", "c", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int32_t>({10, 20, 30}),
          makeFlatVector<float>({0.1, 0.2, 0.3}),
          makeFlatVector<float>({-1.2, 0.0, 10.67}),
      });
  auto rowType = asRowType(data->type());

  // Basic math expressions.

  auto exprSet = compileExpression("a + b", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  auto expectedResult = makeFlatVector<int32_t>({11, 22, 33});
  auto result = evaluate(exprSet.get(), data);
  assertEqualVectors(expectedResult, result);

  exprSet = compileExpression("a + b * 5::integer", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  exprSet = compileExpression("floor(c * 1.34::real) / d", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Switch expressions.

  exprSet = compileExpression("if (a > 10::integer, 0::integer, b)", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // If statement with 'then' or 'else' branch that can return null does not
  // support fast path.
  exprSet = compileExpression(
      "if (a > 10::integer, 0, cast (null as bigint))", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_FALSE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  exprSet = compileExpression(
      "case when a > 10::integer then 1 when b > 10::integer then 2 else 3 end",
      rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Switch without an else clause doesn't support fast path.
  exprSet = compileExpression(
      "case when a > 10::integer then 1 when b > 10::integer then 2 end",
      rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_FALSE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // AND / OR expressions.

  exprSet = compileExpression("a > 10::integer AND b < 0::integer", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  exprSet = compileExpression(
      "a > 10::integer OR (b % 7::integer == 4::integer)", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Coalesce expression.

  exprSet = compileExpression("coalesce(a, b)", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Multiplying an integer by a double requires a cast, but cast doesn't
  // support fast path.
  exprSet = compileExpression("a * 0.1 + b", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_FALSE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Try expression doesn't support fast path.
  exprSet = compileExpression("try(a / b)", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_FALSE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath())
      << exprSet->toString();

  // Field dereference.
  exprSet = compileExpression("a", rowType);
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_TRUE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath());

  exprSet = compileExpression("a.c0", ROW({"a"}, {ROW({"c0"}, {INTEGER()})}));
  ASSERT_EQ(1, exprSet->exprs().size());
  ASSERT_FALSE(exprSet->exprs()[0]->supportsFlatNoNullsFastPath());
}

TEST_F(ExprTest, commonSubExpressionWithEncodedInput) {
  // This test case does a sanity check of the code path that re-uses
  // precomputed results for common sub-expressions.
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({10, 10, 20, 20}),
       wrapInDictionary(
           makeIndices({0, 1, 2, 3}),
           4,
           wrapInDictionary(
               makeIndices({0, 1, 1, 0}),
               4,
               makeFlatVector<int64_t>({1, 2, 3, 4}))),
       makeConstant<int64_t>(1, 4)});

  // Case 1: When the input to the common sub-expression is a dictionary.
  // c2 > 1 is a common sub-expression. It is used in 3 top-level expressions.
  // In the first expression, c2 > 1 is evaluated for rows 2, 3.
  // In the second expression, c2 > 1 is evaluated for rows 0, 1.
  // In the third expression. c2 > 1 returns pre-computed results for rows 2, 3
  auto results = makeRowVector(evaluateMultiple(
      {"c0 = 2 AND c2 > 1", "c0 = 1 AND c2 > 1", "c1 = 20 AND c2 > 1"}, data));
  auto expectedResults = makeRowVector(
      {makeFlatVector<bool>({false, false, true, false}),
       makeFlatVector<bool>({false, true, false, false}),
       makeFlatVector<bool>({false, false, true, false})});
  assertEqualVectors(expectedResults, results);

  // Case 2: When the input to the common sub-expression is a constant.
  results = makeRowVector(evaluateMultiple(
      {"c0 = 2 AND c3 > 3", "c0 = 1 AND c3 > 3", "c1 = 20 AND c3 > 3"}, data));
  expectedResults = makeRowVector(
      {makeFlatVector<bool>({false, false, false, false}),
       makeFlatVector<bool>({false, false, false, false}),
       makeFlatVector<bool>({false, false, false, false})});
  assertEqualVectors(expectedResults, results);

  // Case 3: When cached rows in sub-expression are not present in final
  // selection.
  // In the first expression, c2 > 1 is evaluated for rows 2, 3.
  // In the second expression, c0 = 1 filters out row 2, 3 and the OR
  // expression sets the final selection to rows 0, 1. Finally, c2 > 1 is
  // evaluated for rows 0, 1. If finalSelection was not updated to the union of
  // cached rows and the existing finalSelection then the vector containing
  // cached values would have been override.
  // In the third expression. c2 > 1 returns pre-computed results for rows 3, 4
  // verifying that the vector containing cached values was not overridden.
  results = makeRowVector(evaluateMultiple(
      {"c0 = 2 AND c2 > 1",
       "c0 = 1 AND ( c1 = 20 OR c2 > 1 )",
       "c1 = 20 AND c2 > 1"},
      data));
  expectedResults = makeRowVector(
      {makeFlatVector<bool>({false, false, true, false}),
       makeFlatVector<bool>({false, true, false, false}),
       makeFlatVector<bool>({false, false, true, false})});
  assertEqualVectors(expectedResults, results);
}

TEST_F(ExprTest, preservePartialResultsWithEncodedInput) {
  // This test verifies that partially populated results are preserved when the
  // input contains an encoded vector. We do this by using an if statement where
  // partial results are passed between its children expressions based on the
  // condition.
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
      wrapInDictionary(
          makeIndices({0, 1, 2, 0, 1, 2}),
          6,
          makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6})),
  });

  // Create an expression which divides the input to be processed equally
  // between two different expressions.
  auto result = evaluate("if(c0 > 3, 7, c1 + 100)", data);
  assertEqualVectors(makeFlatVector<int64_t>({101, 102, 103, 7, 7, 7}), result);
}

// Verify code paths in Expr::applyFunctionWithPeeling for the case when one
// input is a constant of size N, while another input is a dictionary of size N
// over base vector with size > N. After peeling encodings, first input has size
// N, while second input has size > N (the size of the base vector). The
// translated set of rows now contains row numbers > N, hence, constant input
// needs to be resized, otherwise, accessing rows numbers > N will cause an
// error.
TEST_F(ExprTest, peelIntermediateResults) {
  auto data = makeRowVector({makeArrayVector<int32_t>({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {0, 1, 2, 33, 4, 5, 6, 7, 8},
  })});

  // element_at(c0, 200) returns a constant null of size 2.
  // element_at(c0, 4) returns a dictionary vector with base vector of size 17
  // (the size of the array's elements vector) and indices [0, 8].
  auto result = evaluate(
      "array_constructor(element_at(c0, 200), element_at(c0, 4))", data);
  auto expected = makeNullableArrayVector<int32_t>({
      {std::nullopt, 3},
      {std::nullopt, 33},
  });
  assertEqualVectors(expected, result);

  // Change the order of arguments.
  result = evaluate(
      "array_constructor(element_at(c0, 4), element_at(c0, 200))", data);

  expected = makeNullableArrayVector<int32_t>({
      {3, std::nullopt},
      {33, std::nullopt},
  });
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, peelWithDefaultNull) {
  // dict vector is [null, "b", null, "a", null, null].
  auto base =
      makeNullableFlatVector<StringView>({"a"_sv, "b"_sv, std::nullopt});
  auto indices = makeIndices({0, 1, 2, 0, 1, 2});
  auto nulls = makeNulls(6, nullEvery(2));
  auto dict = BaseVector::wrapInDictionary(nulls, indices, 6, base);
  auto data = makeRowVector({dict});

  // After peeling, to_utf8 is only evaluated on rows 0 and 1 in base vector.
  // The result is then wrapped with the dictionary encoding. Unevaluated rows
  // should be filled with nulls.
  auto result =
      evaluate("distinct_from(to_utf8('xB60ChtE03'),to_utf8(c0))", data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, true, true, true, true});
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, addNulls) {
  const vector_size_t kSize = 6;
  SelectivityVector rows{kSize + 1};
  rows.setValid(kSize, false);
  rows.updateBounds();

  auto nulls = allocateNulls(kSize, pool());
  auto* rawNulls = nulls->asMutable<uint64_t>();
  bits::setNull(rawNulls, kSize - 1);

  exec::EvalCtx context(execCtx_.get());

  auto checkConstantResult = [&](const VectorPtr& vector) {
    ASSERT_TRUE(vector->isConstantEncoding());
    ASSERT_EQ(vector->size(), kSize);
    ASSERT_TRUE(vector->isNullAt(0));
  };

  // Test vector that is nullptr.
  {
    VectorPtr vector;
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);
    ASSERT_NE(vector, nullptr);
    checkConstantResult(vector);
  }

  // Test vector that is already a constant null vector and is uniquely
  // referenced.
  {
    auto vector = makeNullConstant(TypeKind::BIGINT, kSize - 1);
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);
    checkConstantResult(vector);
  }

  // Test vector that is already a constant null vector and is not uniquely
  // referenced.
  {
    auto vector = makeNullConstant(TypeKind::BIGINT, kSize - 1);
    auto another = vector;
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);
    ASSERT_EQ(another->size(), kSize - 1);
    checkConstantResult(vector);
  }

  // Test vector that is a non-null constant vector.
  {
    auto vector = makeConstant<int64_t>(100, kSize - 1);
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);
    ASSERT_TRUE(vector->isFlatEncoding());
    ASSERT_EQ(vector->size(), kSize);
    for (auto i = 0; i < kSize - 1; ++i) {
      ASSERT_FALSE(vector->isNullAt(i));
      ASSERT_EQ(vector->asFlatVector<int64_t>()->valueAt(i), 100);
    }
    ASSERT_TRUE(vector->isNullAt(kSize - 1));
  }

  auto checkResult = [&](const VectorPtr& vector) {
    ASSERT_EQ(vector->size(), kSize);
    for (auto i = 0; i < kSize - 1; ++i) {
      ASSERT_FALSE(vector->isNullAt(i));
      ASSERT_EQ(vector->asFlatVector<int64_t>()->valueAt(i), i);
    }
    ASSERT_TRUE(vector->isNullAt(kSize - 1));
  };

  // Test vector that is not uniquely referenced.
  {
    VectorPtr vector =
        makeFlatVector<int64_t>(kSize - 1, [](auto row) { return row; });
    auto another = vector;
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);

    ASSERT_EQ(another->size(), kSize - 1);
    checkResult(vector);
  }

  // Test vector that is uniquely referenced.
  {
    VectorPtr vector =
        makeFlatVector<int64_t>(kSize - 1, [](auto row) { return row; });
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), vector);

    checkResult(vector);
  }

  // Test flat vector which has a shared values buffer. This is done by first
  // slicing the vector which creates buffer views of its nulls and values
  // buffer which are immutable.
  {
    VectorPtr vector =
        makeFlatVector<int64_t>(kSize, [](auto row) { return row; });
    auto slicedVector = vector->slice(0, kSize - 1);
    ASSERT_FALSE(slicedVector->values()->isMutable());
    exec::Expr::addNulls(rows, rawNulls, context, BIGINT(), slicedVector);

    checkResult(slicedVector);
  }

  // Lazy reading sometimes generates row vector that has child with length
  // shorter than the parent.  The extra rows in parent are all marked as nulls
  // so it is valid.  We need to handle this situation when propagating nulls
  // from parent to child.
  {
    auto a = makeFlatVector<int64_t>(kSize - 1, folly::identity);
    auto b = makeArrayVector<int64_t>(
        kSize - 1, [](auto) { return 1; }, [](auto i) { return i; });
    auto row = std::make_shared<RowVector>(
        pool_.get(),
        ROW({{"a", a->type()}, {"b", b->type()}}),
        nullptr,
        kSize,
        std::vector<VectorPtr>({a, b}));
    VectorPtr result = row;
    exec::Expr::addNulls(rows, rawNulls, context, row->type(), result);
    ASSERT_NE(result.get(), row.get());
    ASSERT_EQ(result->size(), kSize);
    for (int i = 0; i < kSize - 1; ++i) {
      ASSERT_FALSE(result->isNullAt(i));
      ASSERT_TRUE(result->equalValueAt(row.get(), i, i));
    }
    ASSERT_TRUE(result->isNullAt(kSize - 1));
  }
}

namespace {
class NoOpVectorFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& /* rows */,
      std::vector<VectorPtr>& /* args */,
      const TypePtr& /* outputType */,
      exec::EvalCtx& /* context */,
      VectorPtr& /* result */) const override {}

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("boolean")
                .argumentType("integer")
                .build()};
  }
};
} // namespace

TEST_F(ExprTest, applyFunctionNoResult) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  exec::registerVectorFunction(
      "always_throws_vector_function",
      TestingAlwaysThrowsVectorFunction::signatures(),
      std::make_unique<TestingAlwaysThrowsVectorFunction>(true));

  // At various places in the code, we don't check if result has been set or
  // not.  Conjuncts have the nice property that they set throwOnError to
  // false and don't check if the result VectorPtr is nullptr.
  assertError(
      "always_throws_vector_function(c0) AND true",
      makeFlatVector<int32_t>({1, 2, 3}),
      "always_throws_vector_function(c0)",
      "and(always_throws_vector_function(c0), true:BOOLEAN)",
      TestingAlwaysThrowsVectorFunction::kVeloxErrorMessage);

  exec::registerVectorFunction(
      "no_op",
      NoOpVectorFunction::signatures(),
      std::make_unique<NoOpVectorFunction>());

  assertError(
      "no_op(c0) AND true",
      makeFlatVector<int32_t>({1, 2, 3}),
      "no_op(c0)",
      "and(no_op(c0), true:BOOLEAN)",
      "Function neither returned results nor threw exception.");
}

TEST_F(ExprTest, mapKeysAndValues) {
  // Verify that the right size of maps and keys arrays are created. This is
  // done by executing eval with a selectivity vector larger than the size of
  // the input map but with the extra trailing rows marked invalid. Finally, if
  // map_keys/_values tried to create a larger result array (equivalent to
  // rows.size()) this will throw.
  vector_size_t vectorSize = 100;
  VectorPtr mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(BIGINT(), BIGINT()),
      makeNulls(vectorSize, nullEvery(3)),
      vectorSize,
      makeIndices(vectorSize, [](auto /* row */) { return 0; }),
      makeIndices(vectorSize, [](auto /* row */) { return 1; }),
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}));
  auto input = makeRowVector({mapVector});
  auto exprSet = compileMultiple(
      {"map_keys(c0)", "map_values(c0)"}, asRowType(input->type()));
  exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

  SelectivityVector rows(vectorSize + 1);
  rows.setValid(vectorSize, false);
  rows.updateBounds();
  std::vector<VectorPtr> result(2);
  ASSERT_NO_THROW(exprSet->eval(rows, context, result));
}

/// Test recursive constant peeling: in general expression evaluation first,
/// then in cast.
TEST_F(ExprTest, constantWrap) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt, 1, 25, 3}),
      makeConstant("5", 4),
  });

  auto result = evaluate("c0 < (cast(c1 as bigint) + 10)", {data});
  assertEqualVectors(
      makeNullableFlatVector<bool>({std::nullopt, true, false, true}), result);
}

TEST_F(ExprTest, stdExceptionInVectorFunction) {
  exec::registerVectorFunction(
      "always_throws_vector_function",
      TestingAlwaysThrowsVectorFunction::signatures(),
      std::make_unique<TestingAlwaysThrowsVectorFunction>(false));

  assertError(
      "always_throws_vector_function(c0)",
      makeFlatVector<int32_t>({1, 2, 3}),
      "always_throws_vector_function(c0)",
      "Same as context.",
      TestingAlwaysThrowsVectorFunction::kStdErrorMessage);

  assertErrorSimplified(
      "always_throws_vector_function(c0)",
      makeFlatVector<int32_t>({1, 2, 3}),
      TestingAlwaysThrowsVectorFunction::kStdErrorMessage);
}

TEST_F(ExprTest, cseUnderTry) {
  auto input = makeRowVector({
      makeNullableFlatVector<int8_t>({31, 3, 31, 31, 2, std::nullopt}),
  });

  // All rows trigger overflow.
  VELOX_ASSERT_THROW(
      evaluate(
          "72::tinyint * 31::tinyint <> 4 or 72::tinyint * 31::tinyint <> 5",
          input),
      "integer overflow: 72 * 31");

  auto result = evaluate(
      "try(72::tinyint * 31::tinyint <> 4 or 72::tinyint * 31::tinyint <> 5)",
      input);
  assertEqualVectors(makeNullConstant(TypeKind::BOOLEAN, 6), result);

  // Only some rows trigger overflow.
  VELOX_ASSERT_THROW(
      evaluate(
          "36::tinyint * c0 <> 4 or 36::tinyint * c0 <> 5 or 36::tinyint * c0 <> 6",
          input),
      "integer overflow: 36 * 31");

  result = evaluate(
      "try(36::tinyint * c0 <> 4 or 36::tinyint * c0 <> 5 or 36::tinyint * c0 <> 6)",
      input);

  assertEqualVectors(
      makeNullableFlatVector<bool>({
          std::nullopt,
          true,
          std::nullopt,
          std::nullopt,
          true,
          std::nullopt,
      }),
      result);
}

TEST_F(ExprTest, conjunctUnderTry) {
  auto input = makeRowVector({
      makeFlatVector<StringView>({"a"_sv, "b"_sv}),
      makeFlatVector<bool>({true, true}),
      makeFlatVector<bool>({true, true}),
  });

  VELOX_ASSERT_THROW(
      evaluate(
          "array_constructor(like(c0, 'test', 'escape'), c1 OR c2)", input),
      "Escape string must be a single character");

  auto result = evaluate(
      "try(array_constructor(like(c0, 'test', 'escape'), c1 OR c2))", input);
  auto expected =
      BaseVector::createNullConstant(ARRAY(BOOLEAN()), input->size(), pool());
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, flatNoNullsFastPathWithCse) {
  // Test CSE with flat-no-nulls fast path.
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({8, 9, 10, 11, 12}),
  });

  // Make sure CSE "c0 + c1" is evaluated only once for each row.
  auto [result, stats] = evaluateWithStats(
      "if((c0 + c1) > 100::bigint, 100::bigint, c0 + c1)", input);

  auto expected = makeFlatVector<int64_t>({9, 11, 13, 15, 17});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);

  std::tie(result, stats) = evaluateWithStats(
      "if((c0 + c1) >= 15::bigint, 100::bigint, c0 + c1)", input);

  expected = makeFlatVector<int64_t>({9, 11, 13, 100, 100});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);
}

TEST_F(ExprTest, cseOverLazyDictionary) {
  auto input = makeRowVector({
      makeConstant<int64_t>(10, 5),
      std::make_shared<LazyVector>(
          pool(),
          BIGINT(),
          5,
          std::make_unique<SimpleVectorLoader>([=](RowSet /*rows*/) {
            return wrapInDictionary(
                makeIndicesInReverse(5),
                makeFlatVector<int64_t>({8, 9, 10, 11, 12}));
          })),
      makeFlatVector<int64_t>({1, 2, 10, 11, 12}),
  });

  // if (c1 > 10, c0 + c1, c0 - c1) is a null-propagating conditional CSE.
  auto result = evaluate(
      "if (c2 > 10::bigint, "
      "   if (c1 > 10, c0 + c1, c0 - c1) + c2, "
      "   if (c1 > 10, c0 + c1, c0 - c1) - c2)",
      input);

  auto expected = makeFlatVector<int64_t>({21, 19, -10, 12, 14});
  assertEqualVectors(expected, result);
}

TEST_F(ExprTest, cseOverConstant) {
  auto input = makeRowVector({
      makeConstant<int64_t>(123, 5),
      makeConstant<int64_t>(-11, 5),
  });

  // Make sure CSE "c0 + c1" is evaluated only once for each row.
  auto [result, stats] =
      evaluateWithStats("if((c0 + c1) < 0::bigint, 0::bigint, c0 + c1)", input);

  auto expected = makeConstant<int64_t>(112, 5);
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);
}

TEST_F(ExprTest, cseOverDictionary) {
  auto indices = makeIndicesInReverse(5);
  auto input = makeRowVector({
      wrapInDictionary(indices, makeFlatVector<int64_t>({1, 2, 3, 4, 5})),
      wrapInDictionary(indices, makeFlatVector<int64_t>({8, 9, 10, 11, 12})),
  });

  // Make sure CSE "c0 + c1" is evaluated only once for each row.
  auto [result, stats] = evaluateWithStats(
      "if((c0 + c1) > 100::bigint, 100::bigint, c0 + c1)", input);

  auto expected = makeFlatVector<int64_t>({17, 15, 13, 11, 9});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);

  std::tie(result, stats) = evaluateWithStats(
      "if((c0 + c1) >= 15::bigint, 100::bigint, c0 + c1)", input);

  expected = makeFlatVector<int64_t>({100, 100, 13, 11, 9});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);
}

TEST_F(ExprTest, cseOverDictionaryOverConstant) {
  auto indices = makeIndicesInReverse(5);
  auto input = makeRowVector({
      wrapInDictionary(indices, makeFlatVector<int64_t>({1, 2, 3, 4, 5})),
      wrapInDictionary(indices, makeConstant<int64_t>(100, 5)),
  });

  // Make sure CSE "c0 + c1" is evaluated only once for each row.
  auto [result, stats] =
      evaluateWithStats("if((c0 + c1) < 0::bigint, 0::bigint, c0 + c1)", input);

  auto expected = makeFlatVector<int64_t>({105, 104, 103, 102, 101});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);

  std::tie(result, stats) = evaluateWithStats(
      "if((c0 + c1) < 103::bigint, 0::bigint, c0 + c1)", input);

  expected = makeFlatVector<int64_t>({105, 104, 103, 0, 0});
  assertEqualVectors(expected, result);
  EXPECT_EQ(5, stats.at("plus").numProcessedRows);
}

TEST_F(ExprTest, cseOverDictionaryAcrossMultipleExpressions) {
  // This test verifies that CSE across multiple expressions are evaluated
  // correctly, that is, make sure peeling is done before attempting to re-use
  // computed results from CSE.
  auto input = makeRowVector({
      wrapInDictionary(
          makeIndices({1, 3}),
          makeFlatVector<StringView>({"aa1"_sv, "bb2"_sv, "cc3"_sv, "dd4"_sv})),
  });
  // Case 1: Peeled and unpeeled set of rows have overlap. This will ensure the
  // right pre-computed values are used.
  // upper(c0) is the CSE here having c0 as a distinct field. Initially its
  // distinct fields is empty as concat (its parent) will have the same
  // fields. If during compilation distinct field is not set when it is
  // identified as a CSE then it will be empty and peeling
  // will not occur the second time CSE is employed. Here the peeled rows are
  // {0,1,2,3} and unpeeled are {0,1}. If peeling is performed in the first
  // encounter, rows to compute will be {_ , 1, _, 3} and in the second
  // instance if peeling is not performed then rows to computed would be {0,
  // 1} where row 0 will be computed and 1 will be re-used so row 1 would have
  // wrong result.
  {
    // Use an allocated result vector to force copying of values to the result
    // vector. Otherwise, we might end up with a result vector pointing directly
    // to the shared values vector from CSE.
    std::vector<VectorPtr> resultToReuse = {
        makeFlatVector<StringView>({"x"_sv, "y"_sv}),
        makeFlatVector<StringView>({"x"_sv, "y"_sv})};
    auto [result, stats] = evaluateMultipleWithStats(
        {"concat('foo_',upper(c0))", "upper(c0)"}, input, resultToReuse);
    std::vector<VectorPtr> expected = {
        makeFlatVector<StringView>({"foo_BB2"_sv, "foo_DD4"_sv}),
        makeFlatVector<StringView>({"BB2"_sv, "DD4"_sv})};
    assertEqualVectors(expected[0], result[0]);
    assertEqualVectors(expected[1], result[1]);
    EXPECT_EQ(2, stats.at("upper").numProcessedRows);
  }

  // Case 2: Here a CSE_1 "substr(upper(c0),2)" shared twice has a child
  // expression which itself is a CSE_2 "upper(c0)" shared thrice. If expression
  // compilation were not fixed, CSE_1 will have distinct fields set but
  // the CSE_2 has a parent in one of the other expression trees and therefore
  // will have its distinct fields set properly. This would result in CSE_1 not
  // peeling but CSE_2 will. In the first expression tree peeling happens
  // before CSE so both CSE_1 and CSE_2 are tracking peeled rows. In the second
  // expression CSE_2 is used again will peeled rows, however in third
  // expression CSE_1 is not peeled but its child CSE_2 attempts peeling and
  // runs into an error while creating the peel.
  {
    // Use an allocated result vector to force copying of values to the result.
    std::vector<VectorPtr> resultToReuse = {
        makeFlatVector<StringView>({"x"_sv, "y"_sv}),
        makeFlatVector<StringView>({"x"_sv, "y"_sv}),
        makeFlatVector<StringView>({"x"_sv, "y"_sv})};
    auto [result, stats] = evaluateMultipleWithStats(
        {"concat('foo_',substr(upper(c0),2))",
         "substr(upper(c0),3)",
         "substr(upper(c0),2)"},
        input,
        resultToReuse);
    std::vector<VectorPtr> expected = {
        makeFlatVector<StringView>({"foo_B2"_sv, "foo_D4"_sv}),
        makeFlatVector<StringView>({"2"_sv, "4"_sv}),
        makeFlatVector<StringView>({"B2"_sv, "D4"_sv})};
    assertEqualVectors(expected[0], result[0]);
    assertEqualVectors(expected[1], result[1]);
    assertEqualVectors(expected[2], result[2]);
    EXPECT_EQ(2, stats.at("upper").numProcessedRows);
  }
}

TEST_F(ExprTest, smallerWrappedBaseVector) {
  // This test verifies that in the case that wrapping the
  // result of a peeledResult (i.e result which is computed after
  // peeling input) results in a smaller result than baseVector,
  // then we don't fault if the rows to be copied are more than the
  // size of the wrapped result.
  // Typically, this happens when the results have a lot of trailing nulls.

  auto baseMap = createMapOfArraysVector<int64_t, int64_t>(
      {{{1, std::nullopt}},
       {{2, {{4, 5, std::nullopt}}}},
       {{2, {{7, 8, 9}}}},
       {{2, std::nullopt}},
       {{2, std::nullopt}},
       {{2, std::nullopt}}});
  auto indices = makeIndices(10, [](auto row) { return row % 6; });
  auto nulls = makeNulls(10, [](auto row) { return row > 5; });
  auto wrappedMap = BaseVector::wrapInDictionary(nulls, indices, 10, baseMap);
  auto input = makeRowVector({wrappedMap});

  auto exprSet = compileMultiple(
      {"element_at(element_at(c0, 2::bigint), 1::bigint)"},
      asRowType(input->type()));

  exec::EvalCtx context(execCtx_.get(), exprSet.get(), input.get());

  // We set finalSelection to false so that
  // we force copy of results into the flatvector.
  *context.mutableIsFinalSelection() = false;
  auto finalRows = SelectivityVector(10);
  *context.mutableFinalSelection() = &finalRows;

  // We need a different SelectivityVector for rows
  // otherwise the copy will not kick in.
  SelectivityVector rows(input->size());
  std::vector<VectorPtr> result(1);
  auto flatResult = makeFlatVector<int64_t>(input->size(), BIGINT());
  result[0] = flatResult;
  exprSet->eval(rows, context, result);

  assertEqualVectors(
      makeNullableFlatVector<int64_t>(
          {std::nullopt,
           4,
           7,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt}),
      result[0]);
}

TEST_F(ExprTest, nullPropagation) {
  auto singleString = parseExpression(
      "substr(c0, 1, if (length(c0) > 2, length(c0) - 1, 0))",
      ROW({"c0"}, {VARCHAR()}));
  auto twoStrings = parseExpression(
      "substr(c0, 1, if (length(c1) > 2, length(c0) - 1, 0))",
      ROW({"c0", "c1"}, {VARCHAR(), VARCHAR()}));
  EXPECT_TRUE(propagatesNulls(singleString));
  EXPECT_FALSE(propagatesNulls(twoStrings));
}

TEST_F(ExprTest, peelingWithSmallerConstantInput) {
  // This test ensures that when a dictionary-encoded vector is peeled together
  // with a constant vector whose size is smaller than the corresponding
  // selected rows of the dictionary base vector, the subsequent evaluation on
  // the constant vector doesn't access values beyond its size.
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2})});
  auto c0 = makeRowVector(
      {makeFlatVector<int64_t>({1, 3, 5, 7, 9})}, nullEvery(1, 2));
  auto indices = makeIndices({2, 3, 4});
  auto d0 = wrapInDictionary(indices, c0);
  auto c1 = BaseVector::wrapInConstant(3, 1, data);

  // After evaluating d0, Coalesce copies values from c1 to an existing result
  // vector. c1 should be large enough so that this copy step does not access
  // values out of bound.
  auto result = evaluate("coalesce(c0, c1)", makeRowVector({d0, c1}));
  assertEqualVectors(c1, result);
}

TEST_F(ExprTest, ifWithLazyNulls) {
  // Makes a null-propagating switch. Evaluates it so that null propagation
  // masks out errors.
  constexpr int32_t kSize = 100;
  const char* kExpr =
      "CASE WHEN 10 % (c0 - 2) < 0 then c0 + c1 when 10 % (c0 - 4) = 3 then c0 + c1 * 2 else c0 + c1 end";
  auto c0 = makeFlatVector<int64_t>(kSize, [](auto row) { return row % 10; });
  auto c1 = makeFlatVector<int64_t>(
      kSize,
      [](auto row) { return row; },
      [](auto row) { return row % 10 == 2 || row % 10 == 4; });

  auto result = evaluate(kExpr, makeRowVector({c0, c1}));
  auto resultFromLazy =
      evaluate(kExpr, makeRowVector({c0, wrapInLazyDictionary(c1)}));
  assertEqualVectors(result, resultFromLazy);
}

int totalDefaultNullFunc = 0;
template <typename T>
struct DefaultNullFunc {
  void call(int64_t& output, int64_t input1, int64_t input2) {
    output = input1 + input2;
    totalDefaultNullFunc++;
  }
};

int totalNotDefaultNullFunc = 0;
template <typename T>
struct NotDefaultNullFunc {
  bool callNullable(int64_t& output, const int64_t* input) {
    output = totalNotDefaultNullFunc++;
    return input;
  }
};

TEST_F(ExprTest, commonSubExpressionWithPeeling) {
  registerFunction<DefaultNullFunc, int64_t, int64_t, int64_t>(
      {"default_null"});
  registerFunction<NotDefaultNullFunc, int64_t, int64_t>({"not_default_null"});

  // func1(func2(c0), c0) propagates nulls of c0. since func1 is default null.
  std::string expr1 = "default_null(not_default_null(c0), c0)";
  EXPECT_TRUE(propagatesNulls(parseExpression(expr1, ROW({"c0"}, {BIGINT()}))));

  // func2(c0) does not propagate nulls.
  std::string expr2 = "not_default_null(c0)";
  EXPECT_FALSE(
      propagatesNulls(parseExpression(expr2, ROW({"c0"}, {BIGINT()}))));

  auto clearResults = [&]() {
    totalDefaultNullFunc = 0;
    totalNotDefaultNullFunc = 0;
  };

  // When the input does not have additional nulls, peeling happens for both
  // expr1, and expr2 identically, hence each of them will be evaluated only
  // once per peeled row.
  {
    auto data = makeRowVector({wrapInDictionary(
        makeIndices({0, 0, 0, 1}), 4, makeFlatVector<int64_t>({1, 2, 3, 4}))});
    auto check = [&](const std::vector<std::string>& expressions) {
      auto result = makeRowVector(evaluateMultiple(expressions, data));
      ASSERT_EQ(totalDefaultNullFunc, 2);
      ASSERT_EQ(totalNotDefaultNullFunc, 2);
      clearResults();
    };
    check({expr1, expr2});
    check({expr1});
    check({expr1, expr1, expr2});
    check({expr2, expr1, expr2});
  }

  // When the dictionary input have additional nulls, peeling won't happen for
  // expressions that do not propagate nulls. Hence when expr2 is reached it
  // shall be evaluated again for all rows.
  {
    auto data = makeRowVector({BaseVector::wrapInDictionary(
        makeNulls(4, nullEvery(2)),
        makeIndices({0, 0, 0, 1}),
        4,
        makeFlatVector<int64_t>({1, 2, 3, 4}))});
    {
      auto results = makeRowVector(evaluateMultiple({expr1, expr2}, data));

      ASSERT_EQ(totalDefaultNullFunc, 2);
      // It is evaluated twice during expr1 and 4 times during expr2.
      ASSERT_EQ(totalNotDefaultNullFunc, 6);
      clearResults();
    }
    {
      // if expr2 appears again it shall be not be re-evaluated.
      auto results =
          makeRowVector(evaluateMultiple({expr1, expr2, expr1, expr2}, data));
      ASSERT_EQ(totalDefaultNullFunc, 2);
      // It is evaluated twice during expr1 and 4 times during expr2.
      ASSERT_EQ(totalNotDefaultNullFunc, 6);
    }
  }
}

TEST_F(ExprTest, dictionaryOverLoadedLazy) {
  // This test verifies a corner case where peeling does not go past a loaded
  // lazy layer which caused wrong set of inputs being passed to shared
  // sub-expressions evaluation.
  // Inputs are of the form c0: Dict1(Lazy(Dict2(Flat1))) and c1: Dict1(Flat
  constexpr int32_t kSize = 100;

  // Generate inputs of the form c0: Dict1(Lazy(Dict2(Flat1))) and c1:
  // Dict1(Flat2). Note c0 and c1 have the same top encoding layer.

  // Generate indices that randomly point to different rows of the base flat
  // layer. This makes sure that wrong values are copied over if there is a bug
  // in shared sub-expressions evaluation.
  std::vector<int> indicesUnderLazy = {2, 5, 4, 1, 2, 4, 5, 6, 4, 9};
  auto smallFlat =
      makeFlatVector<int64_t>(kSize / 10, [](auto row) { return row * 2; });
  auto indices = makeIndices(kSize, [&indicesUnderLazy](vector_size_t row) {
    return indicesUnderLazy[row % 10];
  });
  auto lazyDict = std::make_shared<LazyVector>(
      execCtx_->pool(),
      smallFlat->type(),
      kSize,
      std::make_unique<SimpleVectorLoader>([=](RowSet /*rows*/) {
        return wrapInDictionary(indices, kSize, smallFlat);
      }));
  // Make sure it is loaded, otherwise during evaluation ensureLoaded() would
  // transform the input vector from Dict1(Lazy(Dict2(Flat1))) to
  // Dict1((Dict2(Flat1))) which recreates the buffers for the top layers and
  // disables any peeling that can happen between c0 and c1.
  lazyDict->loadedVector();

  auto sharedIndices = makeIndices(kSize / 2, [](auto row) { return row * 2; });
  auto c0 = wrapInDictionary(sharedIndices, kSize / 2, lazyDict);
  auto c1 = wrapInDictionary(
      sharedIndices,
      makeFlatVector<int64_t>(kSize, [](auto row) { return row; }));

  // "(c0 < 5 and c1 < 90)" would peel Dict1 layer in the top level conjunct
  // expression then when peeled c0 is passed to the inner "c0 < 5" expression,
  // a call to EvalCtx::getField() removes the lazy layer which ensures the last
  // dictionary layer is peeled. This means that shared sub-expression
  // evaluation is done on the lowest flat layer. In the second expression "c0 <
  // 5" the input is Dict1(Lazy(Dict2(Flat1))) and if peeling only removed till
  // the lazy layer, the shared sub-expression evaluation gets called on
  // Lazy(Dict2(Flat1)) which then results in wrong results.
  auto result = evaluateMultiple(
      {"(c0 < 5 and c1 < 90)", "c0 < 5"}, makeRowVector({c0, c1}));
  auto resultFromLazy = evaluate("c0 < 5", makeRowVector({c0, c1}));
  assertEqualVectors(result[1], resultFromLazy);
}

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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <fmt/format.h>

#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace facebook::velox;

namespace {

class CudfVectorDistanceFunctionsTest
    : public cudf_velox::CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    // registerCudf() installs the shared builtins only. Presto scalar
    // functions, including the distance kernels, are registered separately.
    cudf_velox::registerPrestoFunctions("");
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterCudf();
  }

  void SetUp() override {
    cudf_velox::CudfFunctionBaseTest::SetUp();
    // The CPU registers the array(real) overloads only when built with FAISS.
    if (!exec::simpleFunctions()
             .resolveFunction(
                 "cosine_similarity", {ARRAY(REAL()), ARRAY(REAL())})
             .has_value()) {
      GTEST_SKIP() << "Requires the FAISS-backed CPU distance functions";
    }
  }

  // Random arrays of 'dimensions' elements, with the special rows the CPU
  // treats differently: null arrays, null elements, empty arrays and zero
  // vectors.
  // 'includeEmpty' is false when the other argument has a fixed length, such
  // as a literal. An empty array is a size mismatch, and the CPU function
  // throws instead of returning null.
  template <typename T = float>
  ArrayVectorPtr makeVectors(
      vector_size_t numRows,
      vector_size_t dimensions,
      int seed,
      bool includeEmpty = true) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> distribution(T{-1}, T{1});
    std::vector<std::optional<std::vector<std::optional<T>>>> rows;
    for (vector_size_t row = 0; row < numRows; ++row) {
      if (row % 17 == 3) {
        rows.push_back(std::nullopt);
        continue;
      }
      if (includeEmpty && row % 23 == 5) {
        rows.push_back(std::vector<std::optional<T>>{});
        continue;
      }
      std::vector<std::optional<T>> values(dimensions);
      for (auto& value : values) {
        value = row % 29 == 7 ? T{0} : distribution(generator);
      }
      if (row % 31 == 11) {
        values[dimensions / 2] = std::nullopt;
      }
      rows.push_back(std::move(values));
    }
    return makeNullableArrayVector<T>(rows);
  }

  // Evaluates on the CPU and the GPU and compares the results with a
  // tolerance, since the two sum the products in different orders. The CPU
  // double overloads cast elements to float, so a true double GPU result can
  // differ by more than a float-versus-float comparison.
  template <typename T = float>
  void assertMatchesCpu(const std::string& sql, const RowVectorPtr& input) {
    SCOPED_TRACE(sql);
    const auto rowType = asRowType(input->type());
    auto exprSet = compileExpression(sql, rowType);
    const auto expected =
        functions::test::FunctionBaseTest::evaluate(*exprSet, input);
    const auto actual = evaluate(makeTypedExpr(sql, rowType), input);
    ASSERT_EQ(actual->size(), expected->size());
    constexpr auto kind = std::is_same_v<T, double> ? TypeKind::DOUBLE
                                                    : TypeKind::REAL;
    ASSERT_EQ(actual->type()->kind(), kind);
    const auto* expectedValues = expected->as<SimpleVector<T>>();
    const auto* actualValues = actual->as<SimpleVector<T>>();
    for (vector_size_t row = 0; row < expected->size(); ++row) {
      ASSERT_EQ(actual->isNullAt(row), expected->isNullAt(row))
          << "row " << row;
      if (expected->isNullAt(row)) {
        continue;
      }
      const auto expectedValue = expectedValues->valueAt(row);
      const auto actualValue = actualValues->valueAt(row);
      if (std::isnan(expectedValue)) {
        EXPECT_TRUE(std::isnan(actualValue)) << "row " << row;
        continue;
      }
      EXPECT_NEAR(
          actualValue,
          expectedValue,
          static_cast<T>(1e-4) * std::max(T{1}, std::abs(expectedValue)))
          << "row " << row;
    }
  }

  // Formats a query vector as an array literal of T.
  template <typename T>
  static std::string arrayLiteral(const std::vector<T>& values) {
    const char* sqlType = std::is_same_v<T, double> ? "DOUBLE" : "REAL";
    std::string sql = "ARRAY[";
    for (size_t i = 0; i < values.size(); ++i) {
      sql += fmt::format(
          "{}CAST({} AS {})", i == 0 ? "" : ", ", values[i], sqlType);
    }
    return sql + "]";
  }

  static constexpr const char* kFunctions[]{
      "cosine_similarity",
      "l2_squared",
      "dot_product",
  };
};

TEST_F(CudfVectorDistanceFunctionsTest, columnColumn) {
  constexpr vector_size_t kNumRows{1'000};
  constexpr vector_size_t kDimensions{96};
  const auto input = makeRowVector({
      makeVectors(kNumRows, kDimensions, /*seed=*/1),
      makeVectors(kNumRows, kDimensions, /*seed=*/2),
  });
  for (const auto* function : kFunctions) {
    assertMatchesCpu(fmt::format("{}(c0, c1)", function), input);
  }
}

TEST_F(CudfVectorDistanceFunctionsTest, columnConstant) {
  constexpr vector_size_t kNumRows{1'000};
  constexpr vector_size_t kDimensions{8};
  const auto input =
      makeRowVector({makeVectors(
          kNumRows, kDimensions, /*seed=*/3, /*includeEmpty=*/false)});
  const auto query = arrayLiteral(std::vector<float>{
      0.5f, -0.25f, 1.0f, 0.0f, 2.0f, -1.5f, 0.75f, 0.125f});
  for (const auto* function : kFunctions) {
    assertMatchesCpu(fmt::format("{}(c0, {})", function, query), input);
    assertMatchesCpu(fmt::format("{}({}, c0)", function, query), input);
  }
}

TEST_F(CudfVectorDistanceFunctionsTest, sizeMismatchThrows) {
  const auto input = makeRowVector({
      makeArrayVector<float>({{1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}}),
      makeArrayVector<float>({{1.0f, 2.0f}, {1.0f, 2.0f}}),
  });
  const auto rowType = asRowType(input->type());
  VELOX_ASSERT_USER_THROW(
      evaluate(makeTypedExpr("cosine_similarity(c0, c1)", rowType), input),
      "Both arrays need to have identical size");
  VELOX_ASSERT_USER_THROW(
      evaluate(makeTypedExpr("dot_product(c0, c1)", rowType), input),
      "Both array arguments must have identical sizes");

  const auto doubles = makeRowVector({
      makeArrayVector<double>({{1.0, 2.0}, {1.0, 2.0, 3.0}}),
      makeArrayVector<double>({{1.0, 2.0}, {1.0, 2.0}}),
  });
  const auto doubleType = asRowType(doubles->type());
  VELOX_ASSERT_USER_THROW(
      evaluate(
          makeTypedExpr("cosine_similarity(c0, c1)", doubleType), doubles),
      "Both arrays need to have identical size");
  VELOX_ASSERT_USER_THROW(
      evaluate(makeTypedExpr("l2_squared(c0, c1)", doubleType), doubles),
      "Both arrays need to have identical size");
  VELOX_ASSERT_USER_THROW(
      evaluate(makeTypedExpr("dot_product(c0, c1)", doubleType), doubles),
      "Both array arguments must have identical sizes");
}

TEST_F(CudfVectorDistanceFunctionsTest, doubleColumnColumn) {
  constexpr vector_size_t kNumRows{200};
  constexpr vector_size_t kDimensions{32};
  const auto input = makeRowVector({
      makeVectors<double>(kNumRows, kDimensions, /*seed=*/11),
      makeVectors<double>(kNumRows, kDimensions, /*seed=*/12),
  });
  for (const auto* function : kFunctions) {
    assertMatchesCpu<double>(fmt::format("{}(c0, c1)", function), input);
  }
}

TEST_F(CudfVectorDistanceFunctionsTest, doubleColumnConstant) {
  constexpr vector_size_t kNumRows{200};
  constexpr vector_size_t kDimensions{8};
  const auto input = makeRowVector({makeVectors<double>(
      kNumRows, kDimensions, /*seed=*/13, /*includeEmpty=*/false)});
  const auto query = arrayLiteral<double>(
      {0.5, -0.25, 1.0, 0.0, 2.0, -1.5, 0.75, 0.125});
  for (const auto* function : kFunctions) {
    assertMatchesCpu<double>(fmt::format("{}(c0, {})", function, query), input);
    assertMatchesCpu<double>(fmt::format("{}({}, c0)", function, query), input);
  }
}

TEST_F(CudfVectorDistanceFunctionsTest, nullElementSkipsSizeCheck) {
  const std::vector<std::vector<std::optional<float>>> left{
      {1.0f, std::nullopt, 3.0f}};
  const std::vector<std::vector<std::optional<float>>> right{{1.0f, 2.0f}};
  const auto input = makeRowVector({
      makeNullableArrayVector<float>(left),
      makeNullableArrayVector<float>(right),
  });
  for (const auto* function : kFunctions) {
    assertMatchesCpu(fmt::format("{}(c0, c1)", function), input);
  }

  const std::vector<std::vector<std::optional<double>>> doubleLeft{
      {1.0, std::nullopt, 3.0}};
  const std::vector<std::vector<std::optional<double>>> doubleRight{{1.0, 2.0}};
  const auto doubleInput = makeRowVector({
      makeNullableArrayVector<double>(doubleLeft),
      makeNullableArrayVector<double>(doubleRight),
  });
  for (const auto* function : kFunctions) {
    assertMatchesCpu<double>(fmt::format("{}(c0, c1)", function), doubleInput);
  }
}

} // namespace

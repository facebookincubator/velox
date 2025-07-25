#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::exec::test;

namespace {

class SubfieldFilterAstTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    facebook::velox::filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Generate a single test vector
  RowVectorPtr makeTestVector(const RowTypePtr& rowType, int32_t rows = 100) {
    return std::dynamic_pointer_cast<RowVector>(
        facebook::velox::test::BatchMaker::createBatch(rowType, rows, *pool_));
  }

  // Execute filter comparison between Velox and cuDF using a pre-created AST.
  // 'tree' and 'scalars' must out-live the computation because they own the
  // expression nodes and literal scalars referenced by 'expr'.
  void testFilterExecution(
      const RowTypePtr& rowType,
      const std::string& columnName,
      const common::Filter& filter,
      const RowVectorPtr& vector,
      const cudf::ast::expression& expr) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    {
      auto cudfTable =
          cudf_velox::with_arrow::toCudfTable(vector, pool_.get(), stream);
      ASSERT_NE(cudfTable, nullptr);

      auto cudfResult =
          cudf::compute_column(cudfTable->view(), expr, stream, mr);

      ASSERT_NE(cudfResult, nullptr);
      EXPECT_EQ(cudfResult->type().id(), cudf::type_id::BOOL8);
      EXPECT_EQ(cudfResult->size(), vector->size())
          << "Result size mismatch for column: " << columnName;

      // Convert cuDF bool result back to Velox to compare element-wise.
      auto resultTable = std::make_unique<cudf::table>(
          std::vector<std::unique_ptr<cudf::column>>{});
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.emplace_back(
          std::move(const_cast<std::unique_ptr<cudf::column>&>(cudfResult)));
      resultTable = std::make_unique<cudf::table>(std::move(cols));

      auto veloxBoolRow = cudf_velox::with_arrow::toVeloxColumn(
          resultTable->view(), pool_.get(), "cmp_", stream);
      auto boolVector = veloxBoolRow->childAt(0)->asFlatVector<bool>();
      boolVector->loadedVector();

      // Compare with Velox filter evaluation.
      auto inputFieldIdx = rowType->getChildIdx(columnName);
      auto fieldVec = vector->childAt(inputFieldIdx);

      for (int i = 0; i < vector->size(); ++i) {
        if (fieldVec->isNullAt(i)) {
          continue; // skip null comparison
        }

        bool veloxExpected = false;
        switch (fieldVec->typeKind()) {
          case TypeKind::BIGINT: {
            auto v = fieldVec->asFlatVector<int64_t>()->valueAt(i);
            veloxExpected = filter.testInt64(v);
            break;
          }
          case TypeKind::INTEGER: {
            auto v = fieldVec->asFlatVector<int32_t>()->valueAt(i);
            veloxExpected = filter.testInt64(static_cast<int64_t>(v));
            break;
          }
          case TypeKind::DOUBLE: {
            auto v = fieldVec->asFlatVector<double>()->valueAt(i);
            veloxExpected = filter.testDouble(v);
            break;
          }
          case TypeKind::REAL: {
            auto v = fieldVec->asFlatVector<float>()->valueAt(i);
            veloxExpected = filter.testFloat(v);
            break;
          }
          case TypeKind::VARCHAR: {
            auto sv = fieldVec->asFlatVector<StringView>()->valueAt(i);
            veloxExpected = filter.testBytes(sv.data(), sv.size());
            break;
          }
          default:
            veloxExpected = true;
        }
        bool cudfGot = boolVector->valueAt(i);
        EXPECT_EQ(veloxExpected, cudfGot)
            << "Mismatch at row " << i << " for " << columnName;
      }
    }
  }
};

// Basic AST generation tests
TEST_F(SubfieldFilterAstTest, Int32RangeInclusive) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, INTEGER()}});
  auto filter =
      std::make_unique<common::BigintRange>(10, 20, /*nullAllowed*/ false);

  // AST validation
  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
  ASSERT_GT(tree.size(), 0UL) << "No expressions created for test";
  EXPECT_LE(scalars.size(), 2UL) << "Too many scalars for range filter";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, Int64InList) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, BIGINT()}});
  std::vector<int64_t> inVals = {1, 2, 5, 7};
  int64_t min = *std::min_element(inVals.begin(), inVals.end());
  int64_t max = *std::max_element(inVals.begin(), inVals.end());
  auto filter = std::make_unique<common::BigintValuesUsingBitmask>(
      min, max, inVals, /*nullAllowed*/ false);

  // AST validation
  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
  ASSERT_GT(tree.size(), 0UL) << "No expressions created for test";
  EXPECT_EQ(scalars.size(), inVals.size())
      << "Scalar count mismatch for IN list";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, DoubleRange) {
  const std::string columnName = "c1";
  auto rowType = ROW({{columnName, DOUBLE()}});
  auto filter = std::make_unique<common::DoubleRange>(
      0.1, false, false, 10.5, false, false, /*nullAllowed*/ false);

  // AST validation
  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
  ASSERT_GT(tree.size(), 0UL) << "No expressions created for test";
  EXPECT_LE(scalars.size(), 2UL) << "Too many scalars for float range";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, StringInList) {
  const std::string columnName = "c2";
  auto rowType = ROW({{columnName, VARCHAR()}});
  std::vector<std::string> stringVals = {"apple", "cherry"};
  auto filter =
      std::make_unique<common::BytesValues>(stringVals, /*nullAllowed*/ false);

  // AST validation
  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
  ASSERT_GT(tree.size(), 0UL) << "No expressions created for test";
  EXPECT_EQ(scalars.size(), stringVals.size())
      << "Scalar count mismatch for string IN list";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, StringRangeNotSupported) {
  auto rowType = ROW({{"c2", VARCHAR()}});
  auto filter = std::make_unique<common::BytesRange>(
      "apple",
      /*lowerUnbounded*/ false,
      /*lowerExclusive*/ false,
      "orange",
      /*upperUnbounded*/ false,
      /*upperExclusive*/ false,
      /*nullAllowed*/ false);

  common::Subfield subfield("c2");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  // BytesRange is not yet implemented, should throw
  EXPECT_THROW(
      {
        createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
      },
      VeloxRuntimeError);
}

TEST_F(SubfieldFilterAstTest, BoolValueNotSupported) {
  auto rowType = ROW({{"flag", BOOLEAN()}});
  auto filter =
      std::make_unique<common::BoolValue>(true, /*nullAllowed*/ false);

  common::Subfield subfield("flag");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  // BoolValue is not yet implemented, should throw
  EXPECT_THROW(
      {
        createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
      },
      VeloxRuntimeError);
}

// Single value range tests
TEST_F(SubfieldFilterAstTest, BigintRangeSingleValue) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, BIGINT()}});
  auto filter =
      std::make_unique<common::BigintRange>(42, 42, /*nullAllowed*/ false);

  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL);
  // Single value range should create 1 scalar for equality comparison (c0 = 42)
  EXPECT_EQ(scalars.size(), 1UL)
      << "Single value range should create 1 scalar for equality";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, Int32SingleValue) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, INTEGER()}}); // 32-bit int
  auto filter =
      std::make_unique<common::BigintRange>(100, 100, /*nullAllowed*/ false);

  common::Subfield subfield("c0");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL);
  EXPECT_EQ(scalars.size(), 1UL)
      << "Single value on int32 should create 1 scalar for equality";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

// Single value that is outside the column's type range.
// For an INT32 column, pick a 64-bit value greater than INT32_MAX.
TEST_F(SubfieldFilterAstTest, Int32SingleValueOutOfRange) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, INTEGER()}}); // 32-bit int column

  // Value well above INT32_MAX.
  int64_t outOfRangeValue =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1000;

  auto filter = std::make_unique<common::BigintRange>(
      outOfRangeValue, outOfRangeValue, /*nullAllowed*/ false);

  // Build AST once.
  common::Subfield subfield(columnName);
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL)
      << "No expressions created for out-of-range single value test";
  EXPECT_EQ(scalars.size(), 0UL)
      << "Single value on int32 should create no scalars for out-of-range";

  // Execution validation – compare Velox filter vs cuDF AST results.
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

// Type boundary tests
TEST_F(SubfieldFilterAstTest, IntegerOverflowBounds) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, INTEGER()}}); // 32-bit int
  auto filter = std::make_unique<common::BigintRange>(
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(),
      /*nullAllowed*/ false);

  common::Subfield subfield("c0");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL);
  // Should have created no scalars since both bounds are beyond int32 range
  EXPECT_EQ(scalars.size(), 0UL)
      << "Should skip both bounds for full-range int32 filter";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, PartialBoundsOutsideTypeRange) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, INTEGER()}}); // 32-bit int
  auto filter = std::make_unique<common::BigintRange>(
      std::numeric_limits<int64_t>::min(), 1000, /*nullAllowed*/ false);

  common::Subfield subfield("c0");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL);
  // Should create 1 scalar (upper bound), lower bound should be skipped
  EXPECT_EQ(scalars.size(), 1UL)
      << "Should create only upper bound scalar when lower is out of range";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, SmallIntTypeBounds) {
  const std::string columnName = "c0";
  auto rowType = ROW({{columnName, SMALLINT()}}); // 16-bit int
  auto filter = std::make_unique<common::BigintRange>(
      -100000, 100000, /*nullAllowed*/ false);

  common::Subfield subfield("c0");
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  const auto& expr =
      createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);

  EXPECT_GT(tree.size(), 0UL);
  // Should skip both bounds since they exceed int16 range (-32768 to 32767)
  EXPECT_EQ(scalars.size(), 0UL)
      << "Should skip both bounds for range exceeding int16 limits";

  // Execution validation
  auto vec = makeTestVector(rowType, 100);
  testFilterExecution(rowType, columnName, *filter, vec, expr);
}

TEST_F(SubfieldFilterAstTest, EmptyInListHandling) {
  auto rowType = ROW({{"c0", BIGINT()}});
  std::vector<int64_t> emptyVals = {};

  // Empty IN list should throw or handle gracefully
  // Note: This test checks that we handle the edge case appropriately
  EXPECT_THROW(
      {
        auto filter = std::make_unique<common::BigintValuesUsingBitmask>(
            0, 1, emptyVals, /*nullAllowed*/ false);
        common::Subfield subfield("c0");
        cudf::ast::tree tree;
        std::vector<std::unique_ptr<cudf::scalar>> scalars;
        createAstFromSubfieldFilter(subfield, *filter, tree, scalars, rowType);
      },
      VeloxException);
}

} // namespace

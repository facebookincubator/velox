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

#include <gtest/gtest.h>
#include "FuzzerRunner.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/tests/ExpressionFuzzer.h"
#include "velox/expression/tests/ExpressionRunner.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

class ExpressionRunnerUnitTest : public testing::Test, public VectorTestBase {
 public:
  void SetUp() override {
    velox::functions::prestosql::registerAllScalarFunctions();
  }

 protected:
  std::shared_ptr<memory::MemoryPool> pool_{memory::addDefaultLeafMemoryPool()};
  core::QueryCtx queryCtx_{};
  core::ExecCtx execCtx_{pool_.get(), &queryCtx_};
};

TEST_F(ExpressionRunnerUnitTest, run) {
  auto inputFile = exec::test::TempFilePath::create();
  auto sqlFile = exec::test::TempFilePath::create();
  auto resultFile = exec::test::TempFilePath::create();
  const char* inputPath = inputFile->path.data();
  const char* resultPath = resultFile->path.data();
  const int vectorSize = 100;

  VectorMaker vectorMaker(pool_.get());
  auto inputVector = vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<StringView>(vectorSize, [](auto) {
        return "abc";
      })});
  auto resultVector = vectorMaker.flatVector<int64_t>(
      vectorSize, [](auto row) { return row * 100; });
  saveVectorToFile(inputVector.get(), inputPath);
  saveVectorToFile(resultVector.get(), resultPath);

  EXPECT_NO_THROW(ExpressionRunner::run(
      inputPath, "length(c0)", "", resultPath, "verify", 0, "", ""));
}

TEST_F(ExpressionRunnerUnitTest, persistAndReproComplexSql) {
  // Create a constant vector of ARRAY(Dictionary-Encoded INT)
  auto dictionaryVector = wrapInDictionary(
      makeIndices({{2, 4, 0, 1}}), makeFlatVector<int32_t>({{1, 2, 3, 4, 5}}));
  auto arrVector = makeArrayVector({0}, dictionaryVector, {});
  auto constantExpr = std::make_shared<core::ConstantTypedExpr>(
      BaseVector::wrapInConstant(1, 0, arrVector));

  ASSERT_EQ(
      constantExpr->toString(),
      "4 elements starting at 0 {[0->2] 3, [1->4] 5, [2->0] 1, [3->1] 2}");

  auto sqlExpr = exec::ExprSet({constantExpr}, &execCtx_, false).expr(0);

  // Self contained SQL that flattens complex constant.
  auto selfContainedSql = sqlExpr->toSql();
  ASSERT_EQ(
      selfContainedSql,
      "ARRAY['3'::INTEGER, '5'::INTEGER, '1'::INTEGER, '2'::INTEGER]");

  std::vector<VectorPtr> complexConstants;
  auto complexConstantsSql = sqlExpr->toSql(&complexConstants);
  ASSERT_EQ(complexConstantsSql, "__complex_constant(c0)");

  auto rowVector = makeRowVector(complexConstants);

  // Emulate a reproduce from complex constant SQL
  auto sqlFile = exec::test::TempFilePath::create();
  auto complexConstantsFile = exec::test::TempFilePath::create();
  auto sqlPath = sqlFile->path.c_str();
  auto complexConstantsPath = complexConstantsFile->path.c_str();

  // Write to file..
  saveStringToFile(complexConstantsSql, sqlPath);
  saveVectorToFile(rowVector.get(), complexConstantsPath);

  // Reproduce from file.
  auto reproSql = restoreStringFromFile(sqlPath);
  auto reproComplexConstants =
      restoreVectorFromFile(complexConstantsPath, pool_.get());

  auto reproExprs = ExpressionRunner::parseSql(
      reproSql, nullptr, pool_.get(), reproComplexConstants);
  ASSERT_EQ(reproExprs.size(), 1);
  ASSERT_EQ(
      reproExprs[0]->toString(),
      "4 elements starting at 0 {[0->2] 3, [1->4] 5, [2->0] 1, [3->1] 2}");
}
} // namespace facebook::velox::test

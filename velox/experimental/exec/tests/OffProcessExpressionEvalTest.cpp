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

#include "velox/experimental/exec/OffProcessExpressionEval.h"
#include <folly/Random.h>
#include <folly/init/Init.h>
#include "velox/connectors/fuzzer/tests/FuzzerConnectorTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/parse/Expressions.h"

namespace facebook::velox::exec::test {

using facebook::velox::test::assertEqualVectors;

// A simple implementation of the off-process expression eval client interface
// that just executes the expression in the same thread.
class MockOffProcessExpressionEvalClient
    : public OffProcessExpressionEvalClient {
 public:
  ~MockOffProcessExpressionEvalClient() override {}

  void setup(
      const std::vector<core::TypedExprPtr>& expressions,
      const RowTypePtr& inputRowType) override {
    std::vector<TypePtr> expressionTypes;
    expressionTypes.reserve(expressions.size());

    for (const auto& expr : expressions) {
      expressionTypes.push_back(expr->type());
    }

    inputRowType_ = inputRowType;
    outputRowType_ = ROW(std::move(expressionTypes));
    exprSet_ = std::make_unique<exec::ExprSet>(expressions, &execCtx_);
  }

  std::unique_ptr<folly::IOBuf> execute(
      std::unique_ptr<folly::IOBuf>&& ioBuf) override {
    // Deserialize the IOBuf.
    ByteStream byteStream_;
    std::vector<ByteRange> ranges;
    ranges.reserve(4);

    for (const auto& range : *ioBuf) {
      ranges.emplace_back(ByteRange{
          const_cast<uint8_t*>(range.data()), (int32_t)range.size(), 0});
    }
    byteStream_.resetInput(std::move(ranges));

    RowVectorPtr inputVector;
    VectorStreamGroup::read(
        &byteStream_, pool_.get(), inputRowType_, &inputVector);

    // Execute expression.
    const vector_size_t numRows = inputVector->size();

    SelectivityVector rows{numRows};
    exec::EvalCtx evalCtx(&execCtx_, exprSet_.get(), inputVector.get());

    std::vector<VectorPtr> result;
    exprSet_->eval(rows, evalCtx, result);

    // Create output vector.
    auto outputRowVector = std::make_shared<RowVector>(
        pool_.get(), outputRowType_, BufferPtr(), numRows, result);

    // Serialize results
    VectorStreamGroup streamGroup(pool_.get());
    streamGroup.createStreamTree(outputRowType_, outputRowVector->size());
    IndexRange range{0, outputRowVector->size()};
    streamGroup.append(outputRowVector, folly::Range<IndexRange*>(&range, 1));

    IOBufOutputStream stream(*pool_);
    streamGroup.flush(&stream);

    return stream.getIOBuf();
  }

  void close() override {
    exprSet_.reset();
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_{memory::getDefaultMemoryPool()};
  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};

  RowTypePtr inputRowType_;
  RowTypePtr outputRowType_;

  std::unique_ptr<exec::ExprSet> exprSet_;
};

class OffProcessExpressionEvalTest
    : public connector::fuzzer::test::FuzzerConnectorTestBase {
 protected:
  OffProcessExpressionEvalTest() {
    exec::Operator::registerOperator(
        std::make_unique<OffProcessExpressionEvalTranslator>());
    fuzzerOptions_.timestampPrecision =
        VectorFuzzer::Options::TimestampPrecision::MILLI;
  }

  std::vector<core::TypedExprPtr> parseExpressions(
      const std::vector<std::string>& expressions,
      const RowTypePtr& rowType) {
    std::vector<core::TypedExprPtr> typedExprs;
    for (const auto& expr : expressions) {
      typedExprs.push_back(
          OperatorTestBase::parseExpr(expr, rowType, parse::ParseOptions{}));
    }
    return typedExprs;
  }

  void testOffProcessExpression(
      const std::vector<RowVectorPtr>& rowVectors,
      const std::vector<std::string>& expressions) {
    // Create a "regular" plan execution expressions with a ProjectNode.
    auto regularPlan = exec::test::PlanBuilder()
                           .values(rowVectors)
                           .project(expressions)
                           .planNode();
    auto expectedResults =
        exec::test::AssertQueryBuilder(regularPlan).copyResults(pool());

    // Create a test plan using off-process expression eval operator.
    auto offProcessPlan =
        exec::test::PlanBuilder()
            .values(rowVectors)
            .addNode([&](std::string id, core::PlanNodePtr input) {
              return std::make_shared<OffProcessExpressionEvalNode>(
                  id,
                  parseExpressions(
                      expressions, asRowType(rowVectors.front()->type())),
                  std::make_shared<MockOffProcessExpressionEvalClient>(),
                  input);
            })
            .planNode();
    auto results =
        exec::test::AssertQueryBuilder(offProcessPlan).copyResults(pool());
    assertEqualVectors(expectedResults, results);
  }
};

TEST_F(OffProcessExpressionEvalTest, singleBatch) {
  auto rowVector = vectorMaker_.rowVector(
      {"col1", "col2", "col_str"},
      {
          vectorMaker_.flatVector({0, 1, 2, 3, 4}),
          vectorMaker_.flatVector({100, 200, 300, 400, 500}),
          vectorMaker_.flatVector({"a", "b", "c", "d", "e"}),
      });

  testOffProcessExpression({rowVector}, {"col1 + col2"});
  testOffProcessExpression(
      {rowVector}, {"10 * col1 + cast(sqrt(col2) as BIGINT)"});

  // Multiple expressions.
  testOffProcessExpression(
      {rowVector}, {"col1 * col2", "length(col_str) + col1", "10", "99.9"});
}

TEST_F(OffProcessExpressionEvalTest, multipleBatches) {
  std::vector<RowVectorPtr> inputVectors;

  for (int32_t i = 0; i < 100; i++) {
    inputVectors.push_back(vectorMaker_.rowVector(
        {"col", "col_str"},
        {
            vectorMaker_.flatVector({folly::Random::rand32()}),
            vectorMaker_.flatVector({std::to_string(i)}),
        }));
  }
  testOffProcessExpression(inputVectors, {"col + cast(col_str as BIGINT)"});
}

TEST_F(OffProcessExpressionEvalTest, fuzzer) {
  for (size_t i = 0; i < 1; i++) {
    auto randRowType = VectorFuzzer({}, pool()).randRowType();

    auto offProcessPlan =
        exec::test::PlanBuilder()
            .tableScan(randRowType, makeFuzzerTableHandle(), {})
            .addNode([&](std::string id, core::PlanNodePtr input) {
              return std::make_shared<OffProcessExpressionEvalNode>(
                  id,
                  parseExpressions(randRowType->names(), randRowType),
                  std::make_shared<MockOffProcessExpressionEvalClient>(),
                  input);
            })
            .planNode();
    auto results = exec::test::AssertQueryBuilder(offProcessPlan)
                       .splits(makeFuzzerSplits(50, 100))
                       .copyResults(pool());

    auto regularPlan = exec::test::PlanBuilder()
                           .tableScan(randRowType, makeFuzzerTableHandle(), {})
                           .project(randRowType->names())
                           .planNode();
    auto expectedResults = exec::test::AssertQueryBuilder(regularPlan)
                               .splits(makeFuzzerSplits(50, 100))
                               .copyResults(pool());

    assertEqualVectors(expectedResults, results);
  }
}

} // namespace facebook::velox::exec::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}

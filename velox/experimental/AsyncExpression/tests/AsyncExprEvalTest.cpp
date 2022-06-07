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

#include <folly/Executor.h>
#include <folly/experimental/coro/Collect.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <iostream>

#include <folly/experimental/coro/Task.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/FlatVector.h>
#include <velox/vector/SelectivityVector.h>
#include <velox/vector/TypeAliases.h>
#include "velox/experimental/AsyncExpression/AsyncExprEval.h"
#include "velox/experimental/AsyncExpression/AsyncVectorFunction.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/type/Type.h"

// #define PRINT_INFO
namespace facebook::velox::exec::test {

namespace {
// For every row, the execution will async sleep input[row]ms, and then result
// will be the number of rows that were completed before the execution of the
// row starts.
// This is used to make sure all rows from multiple functions start
// before any complete. Hence guarantee no blocking.
class RecordPreStartCompletedWork : public AsyncVectorFunction {
 public:
  folly::coro::Task<void> applyAsync(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    BaseVector::ensureWritable(rows, INTEGER(), context->pool(), result);

    auto flatArg = args[0]->asFlatVector<int32_t>();
    std::vector<folly::coro::Task<bool>> tasks;
    rows.applyToSelected([&](vector_size_t row) {
      tasks.push_back(perRowWork(
          row, flatArg->valueAt(row), (*result)->asFlatVector<int32_t>()));
    });

    // we can use collect windowed.
    co_await folly::coro::collectAllRange(std::move(tasks));
    co_return;
  }
  static int32_t finishedItems;

  bool isDeterministic() const override {
    return false;
  }

 private:
  // The results record the number of finished items at the time right
  // before the work per row started.
  folly::coro::Task<bool> perRowWork(
      vector_size_t row,
      int32_t input,
      FlatVector<int32_t>* result) const {
    int32_t rowResult = finishedItems;

#ifdef PRINT_INFO
    std::cout << "start function for row:" << row << "\n";
#endif
    co_await folly::futures::sleep(std::chrono::milliseconds{input});

#ifdef PRINT_INFO
    std::cout << "finish function for row:" << row << "\n";
#endif

    finishedItems++;
    result->set(row, rowResult);
    co_return true;
  }
};

int32_t RecordPreStartCompletedWork::finishedItems = 0;
} // namespace

class AsyncExprEvalTest : public functions::test::FunctionBaseTest {
 public:
  AsyncExprEvalTest() : FunctionBaseTest() {
    facebook::velox::exec::registerVectorFunction(
        "async_call",
        {exec::FunctionSignatureBuilder()
             .returnType("integer")
             .argumentType("integer")
             .build()},
        std::make_unique<RecordPreStartCompletedWork>());
  }

  exec::ExprSet compileExpression(
      const std::string& text,
      const TypePtr& rowType) {
    auto untyped = parse::parseExpr(text);
    auto typed =
        core::Expressions::inferTypes(untyped, rowType, execCtx_.pool());
    return exec::ExprSet({typed}, &execCtx_);
  }
};

TEST_F(AsyncExprEvalTest, test) {
  auto input =
      vectorMaker_.rowVector({makeFlatVector<int32_t>({10, 100, 200})});
  auto exprSet =
      compileExpression("async_call(c0) + async_call(c0)", input->type());

  SelectivityVector rows(3);
  exec::EvalCtx evalCtx(&execCtx_, &exprSet, input.get());
  std::vector<VectorPtr> results(1);

  // The output shall be 0+0, 0+0, 0+0 when we ran evalAsync.
  // Since  All will start before any finishes.
  // start function for row:0
  // start function for row:1
  // start function for row:2
  // start function for row:0
  // start function for row:1
  // start function for row:2
  // finish function for row:0
  // finish function for row:0
  // finish function for row:1
  // finish function for row:1
  // finish function for row:2
  // finish function for row:2
  folly::coro::blockingWait(
      AsyncExprEval::evalExprSetAsync(exprSet, rows, evalCtx, results));

  auto flatResultAsync = results[0]->asFlatVector<int32_t>();
  VELOX_CHECK_EQ(flatResultAsync->valueAt(0), 0);
  VELOX_CHECK_EQ(flatResultAsync->valueAt(1), 0);
  VELOX_CHECK_EQ(flatResultAsync->valueAt(2), 0);
  VELOX_CHECK_EQ(RecordPreStartCompletedWork::finishedItems, 6);
  // Run in non-async mode.

  // Rows will run async within an async vector function, then in the apply
  // function call will block, so two inputs of + are not running
  // concurrently.

  // Output will be 3, 3, 3. The first vector function will finish then the
  // second. For each row starting in the second function, it will see 3
  // finished rows.

  // start function for row:0
  // start function for row:1
  // start function for row:2
  // finish function for row:0
  // finish function for row:1
  // finish function for row:2
  // start function for row:0
  // start function for row:1
  // start function for row:2
  // finish function for row:0
  // finish function for row:1
  // finish function for row:2
  RecordPreStartCompletedWork::finishedItems = 0;
  exprSet.eval(rows, &evalCtx, &results);
  auto flatResultSync = results[0]->asFlatVector<int32_t>();
  VELOX_CHECK_EQ(flatResultSync->valueAt(0), 3);
  VELOX_CHECK_EQ(flatResultSync->valueAt(1), 3);
  VELOX_CHECK_EQ(flatResultSync->valueAt(2), 3);
  VELOX_CHECK_EQ(RecordPreStartCompletedWork::finishedItems, 6);
}
} // namespace facebook::velox::exec::test

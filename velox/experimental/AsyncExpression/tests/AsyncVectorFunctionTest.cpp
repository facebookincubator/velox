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

#include <velox/vector/BaseVector.h>
#include <velox/vector/FlatVector.h>
#include <velox/vector/SelectivityVector.h>
#include <velox/vector/TypeAliases.h>
#include "velox/experimental/AsyncExpression/AsyncVectorFunction.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec {
namespace {
// For every row, the execution will async sleep input[row]ms, and then result
// will be the number of rows that were completed before the execution of the
// row starts.
// This is used to make sure all rows from multiple functions start
// before any complete. Hence guarantee no blocking.
class RecordWakeUpOrder : public AsyncVectorFunction {
 public:
  // Such expansion of the work shall be done through the
  // AsyncFunctionAdapter eventually.
  folly::coro::Task<void> applyAsync(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    // TODO: if we want to allow threadSafe()-> true, we need to make sure this
    // is thread-safe.
    BaseVector::ensureWritable(rows, INTEGER(), context->pool(), result);

    auto flatArg = args[0]->asFlatVector<int32_t>();
    std::vector<folly::coro::Task<bool>> tasks;
    int32_t counter = 0;
    rows.applyToSelected([&](vector_size_t row) {
      tasks.push_back(perRowWork(
          row,
          flatArg->valueAt(row),
          (*result)->asFlatVector<int32_t>(),
          counter));
    });

    // we can use collect windowed.
    co_await folly::coro::collectAllRange(std::move(tasks));
    co_return;
  }

 private:
  // The results record the order at which rows execution finished.
  // Not thread-safe but ok since collectAllRange run all tasks on the same
  // thread.
  folly::coro::Task<bool> perRowWork(
      vector_size_t row,
      int32_t input,
      FlatVector<int32_t>* result,
      int32_t& counter) const {
    co_await folly::futures::sleep(std::chrono::milliseconds{input});
    // record the order in which rows finished.
    result->set(row, counter++);
    co_return true;
  }
};
} // namespace

class AsyncVectorFunctionTest : public functions::test::FunctionBaseTest {};

TEST_F(AsyncVectorFunctionTest, testRowsRunAsync) {
  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::createForTest()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};
  ExprSet exprSet({}, &execCtx_);
  auto inputs = makeRowVector({});
  exec::EvalCtx evalCtx(&execCtx_, &exprSet, inputs.get());

  RecordWakeUpOrder function;
  VectorPtr result;
  SelectivityVector rows(3);
  {
    auto input = makeFlatVector<int32_t>({10, 200, 1000});
    std::vector<VectorPtr> inputsVector = {input};
    function.apply(rows, inputsVector, INTEGER(), &evalCtx, &result);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(0), 0);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(1), 1);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(2), 2);
  }

  {
    auto input = makeFlatVector<int32_t>({1000, 10, 500});
    std::vector<VectorPtr> inputsVector = {input};
    function.apply(rows, inputsVector, INTEGER(), &evalCtx, &result);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(0), 2);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(1), 0);
    ASSERT_EQ(result->asFlatVector<int32_t>()->valueAt(2), 1);
  }
}

// For every row, it will sleep input[i] ms, and the result will be the rows
// completed before the execution of the row starts. We use this to make sure
// all rows from multiple functions start before any complete. Hence
// guarantee no blocking.
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

 private:
  // The results record the number of finished items at the time right
  // before the work per row started.
  folly::coro::Task<bool> perRowWork(
      vector_size_t row,
      int32_t input,
      FlatVector<int32_t>* result) const {
    static int32_t finishedItems = 0;
    int32_t rowResult = finishedItems;
    co_await folly::futures::sleep(std::chrono::milliseconds{input});
    finishedItems++;
    result->set(row, rowResult);
    co_return true;
  }
};

TEST_F(AsyncVectorFunctionTest, testVectorsRunAsync) {
  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::createForTest()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};
  ExprSet exprSet({}, &execCtx_);
  auto inputs = makeRowVector({});
  exec::EvalCtx evalCtx(&execCtx_, &exprSet, inputs.get());

  RecordPreStartCompletedWork function;
  auto input = makeFlatVector<int32_t>({100, 200, 1000});
  std::vector<VectorPtr> inputsVector = {input};

  VectorPtr result1, result2;
  SelectivityVector rows(3);

  auto task1 =
      function.applyAsync(rows, inputsVector, INTEGER(), &evalCtx, &result1);
  auto task2 =
      function.applyAsync(rows, inputsVector, INTEGER(), &evalCtx, &result2);
  folly::coro::blockingWait(
      folly::coro::collectAll(std::move(task1), std::move(task2)));

  auto assertAllZeros = [&](auto& vector) {
    for (auto i = 0; i < vector->size(); i++) {
      ASSERT_EQ(vector->template asFlatVector<int32_t>()->valueAt(i), 0);
    }
  };
  assertAllZeros(result1);
  assertAllZeros(result2);
}
} // namespace facebook::velox::exec

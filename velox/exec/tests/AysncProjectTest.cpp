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
#include <folly/executors/EDFThreadPoolExecutor.h>
#include <folly/executors/GlobalExecutor.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <velox/exec/tests/utils/Cursor.h>
#include <velox/exec/tests/utils/QueryAssertions.h>
#include <atomic>
#include <cstdint>
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/AsyncExpression/AsyncExprEval.h"
#include "velox/experimental/AsyncExpression/AsyncVectorFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
namespace {
using facebook::velox::test::BatchMaker;
class PlusAsync : public AsyncVectorFunction {
 public:
  folly::coro::Task<void> applyAsync(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    static std::mutex mtx; // mutex for critical section
    mtx.lock();
    BaseVector::ensureWritable(rows, INTEGER(), context->pool(), result);
    mtx.unlock();

    auto flatArg1 = args[0]->asFlatVector<int32_t>();
    auto flatArg2 = args[1]->asFlatVector<int32_t>();

    std::vector<folly::coro::Task<bool>> tasks;
    rows.applyToSelected([&](vector_size_t row) {
      tasks.push_back(perRowWork(
          row,
          flatArg1->valueAt(row),
          flatArg2->valueAt(row),
          (*result)->asFlatVector<int32_t>()));
    });

    // we can use collect windowed.
    co_await folly::coro::collectAllRange(std::move(tasks));
    co_return;
  }

  static std::atomic<int64_t> finishedItems;

  bool isDeterministic() const override {
    return false;
  }

 private:
  // The results record the number of finished items at the time right
  // before the work per row started.
  folly::coro::Task<bool> perRowWork(
      vector_size_t row,
      int32_t input1,
      int32_t input2,

      FlatVector<int32_t>* result) const {
    finishedItems++;

    // std::cout << "Start time:"
    // << "row" << row << "id" << finishedItems << "\n";

    co_await folly::futures::sleep(std::chrono::milliseconds{200});

    // std::cout << "End time:"
    // << "row" << row << "id" << finishedItems << "\n";

    finishedItems;
    // look needed if multi threaded executor.
    result->set(row, input1 + input2);
    co_return true;
  }
};
std::atomic<int64_t> PlusAsync::finishedItems = 0;
} // namespace

class AsyncProjectTest : public OperatorTestBase {
 protected:
  void runQuery(
      const std::shared_ptr<const core::PlanNode>& planNode,
      int threadCount,
      int driverCount) {
    auto start = std::chrono::high_resolution_clock::now();
    PlusAsync::finishedItems = 0;
    AsyncExprEval::maxLatency = 0;
    CursorParameters params;
    params.planNode = planNode;

    params.queryCtx = std::make_shared<core::QueryCtx>(
        std::make_shared<folly::IOThreadPoolExecutor>(threadCount));

    params.maxDrivers = driverCount;

    auto [taskCursor, results] = readCursor(params, [](auto x) {});

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << "row throughput is  "
              << (PlusAsync::finishedItems * 1000 / elapsed.count()) / 2
              << " request/s" << std::endl;

    std::cout << "time per rpc "
              << elapsed.count() * 2 / PlusAsync::finishedItems
              << " milli seconds" << std::endl;

    std::cout << "total item count is " << PlusAsync::finishedItems
              << std::endl;
    std::cout << "MAx latency is  " << AsyncExprEval::maxLatency << "ms"
              << std::endl;
    // std::cout << "total time for all requests " << elapsed.count() <<
    // std::endl;
  }

  void runExprB(
      std::string name,
      size_t threadCount,
      size_t driverCount,
      bool async,
      bool yeild,
      int totalRows) {
    std::cout << "yeild:" << yeild << "async:" << async
              << "thread-count:" << threadCount << "driver-count" << driverCount
              << std::endl;
    std::shared_ptr<const RowType> rowType_{
        ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};
    if (totalRows % driverCount != 0) {
      assert(false);
    }

    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < totalRows / driverCount; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType_, 1, *pool_));
      vectors.push_back(vector);
    }

    auto planPre =
        PlanBuilder()
            .values(vectors, true)
            .project(
                {"async_plus(c0, c1) +async_plus(c0, c0+ cast (1 as int))  AS c1_c2"},
                /*async*/ async,
                !yeild)
            .planNode();
    runQuery(planPre, threadCount, driverCount);
  }

  void runExprC(
      std::string name,
      size_t threadCount,
      size_t driverCount,
      bool async,
      bool yeild,
      int totalRows) {
    std::cout << "yeild:" << yeild << "async:" << async
              << "thread-count:" << threadCount << "driver-count" << driverCount
              << std::endl;
    std::shared_ptr<const RowType> rowType_{
        ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};
    if (totalRows % driverCount != 0) {
      assert(false);
    }
    auto totalVectors = totalRows / (driverCount * 1000);

    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < totalVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType_, 1000, *pool_));
      vectors.push_back(vector);
    }

    auto planPre =
        PlanBuilder()
            .values(vectors, true)
            .project(
                {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int))  AS c1_c2"},
                /*async*/ async,
                !yeild)
            .planNode();
    runQuery(planPre, threadCount, driverCount);
  }
};

// one driver, 100 threads.
// TEST_F(AsyncProjectTest, expr1A) {
//   facebook::velox::exec::registerVectorFunction(
//       "async_plus",
//       {exec::FunctionSignatureBuilder()
//            .returnType("integer")
//            .argumentType("integer")
//            .argumentType("integer")
//            .build()},
//       std::make_unique<PlusAsync>());

//   std::shared_ptr<const RowType> rowType_{
//       ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};

//   {
//     std::cout << "one driver and one thread, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 16; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 1, 1);
//   }

//   {
//     std::cout << "two drivers and two threads, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 8; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 2, 2);
//   }

//   {
//     std::cout << "4 drivers and 4 threads, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 4; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 4, 4);
//   }

//   {
//     std::cout << "8 drivers and 8 threads, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 2; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 8, 8);
//   }

//   {
//     std::cout << "16 drivers and 16 threads, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 1; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 16, 16);
//   }

//   std::cout << "############################################\n";

//   // need number 1. limited number of threads available.
//   // We are limited with the numnber if
//   {
//     std::cout << "16 drivers and 1 threads, no async"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 1; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ false)
//             .planNode();
//     runQuery(planPre, 1, 16);
//   }

//   // need number 1. limited number of threads available.
//   // We are limited with the numnber if
//   {
//     std::cout << "16 drivers and 1 threads, async no yeild"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 1; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true,
//                 true)
//             .planNode();
//     runQuery(planPre, 1, 16);
//   }

//   // Need number 1. limited number of threads available.
//   {
//     std::cout << "16 drivers and 1 threads, async and yeild"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 1; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true,
//                 false)
//             .planNode();
//     runQuery(planPre, 1, 16);
//   }

//   std::cout
//       << "can we acheive same with out yeild? yes async expr eval + blocking
//       + multi threading"
//       << "\n";
//   {
//     std::cout << "16 drivers and 16 threads,  async but no yeild"
//               << "\n";
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 1; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true,
//                 true)
//             .planNode();
//     runQuery(planPre, 16, 16);
//   }
//   // this last guy utilies multi threading instead of multi tasking,
//   // whuch one scales better, the expr bellow will tell us.
//   return;
// }

// // Multi-tasking Vs multi-threading assuming we can run as threads as we
// want. TEST_F(AsyncProjectTest, exp2) {
//   facebook::velox::exec::registerVectorFunction(
//       "async_plus",
//       {exec::FunctionSignatureBuilder()
//            .returnType("integer")
//            .argumentType("integer")
//            .argumentType("integer")
//            .build()},
//       std::make_unique<PlusAsync>());

//   std::shared_ptr<const RowType> rowType_{
//       ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};
//   {
//     std::cout << "multi-threading, async expr , no driver yeild "
//               << "\n";

//     // no benefits for drivers>threads in non-yeild form.
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 10; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true,
//                 true)
//             .planNode();
//     runQuery(planPre, 30000, 30000);
//   }
// }

// // Multi-tasking Vs multi-threading assuming we can run as threads as we
// want. TEST_F(AsyncProjectTest, exp3) {
//   facebook::velox::exec::registerVectorFunction(
//       "async_plus",
//       {exec::FunctionSignatureBuilder()
//            .returnType("integer")
//            .argumentType("integer")
//            .argumentType("integer")
//            .build()},
//       std::make_unique<PlusAsync>());

//   std::shared_ptr<const RowType> rowType_{
//       ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};

//   {
//     std::cout << "multi-tasking, async expr , no driver yeild "
//               << "\n";

//     // no benefits for drivers>threads in non-yeild form.
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 10; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true)
//             .planNode();
//     runQuery(planPre, 1, 30000);
//   }
// }

// // Multi-tasking Vs multi-threading assuming we can run as threads as we
// want. TEST_F(AsyncProjectTest, exp4) {
//   facebook::velox::exec::registerVectorFunction(
//       "async_plus",
//       {exec::FunctionSignatureBuilder()
//            .returnType("integer")
//            .argumentType("integer")
//            .argumentType("integer")
//            .build()},
//       std::make_unique<PlusAsync>());

//   std::shared_ptr<const RowType> rowType_{
//       ROW({"c0", "c1"}, {INTEGER(), INTEGER()})};
//   {
//     std::cout << "multi-tasking and multi threading , async expr"
//               << "\n";
//     // 0.0213955
//     // no benefits for drivers>threads in non-yeild form.
//     std::vector<RowVectorPtr> vectors;
//     for (int32_t i = 0; i < 10; ++i) {
//       auto vector = std::dynamic_pointer_cast<RowVector>(
//           BatchMaker::createBatch(rowType_, 1, *pool_));
//       vectors.push_back(vector);
//     }

//     auto planPre =
//         PlanBuilder()
//             .values(vectors, true)
//             .project(
//                 {"async_plus(c0, c1)+async_plus(c0, c0+ cast (1 as int)) AS
//                 c1_c2"},
//                 /*async*/ true)
//             .planNode();
//     runQuery(planPre, std::thread::hardware_concurrency(), 30000);
//   }
// }

// Multi-tasking Vs multi-threading assuming we can run as threads as we want.
TEST_F(AsyncProjectTest, expB) {
  facebook::velox::exec::registerVectorFunction(
      "async_plus",
      {exec::FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .argumentType("integer")
           .build()},
      std::make_unique<PlusAsync>());
  auto totalRows = 5000000;

  // this give best perf.
  auto runAsyncWithYeild = [&]() {
    //// ASYNC limited threads.

    // row throughput is 30372.4 request
    // runExprB(
    //     "",
    //     totalRows / 100,
    //     totalRows / 100,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  55198.9 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency(),
    //     totalRows / 10,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  34787.6 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency(),
    //     totalRows / 100,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  63386.1 request/s
    // 18 drivers used.
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 2,
    //     totalRows / 10,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  32389.5 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 2,
    //     totalRows / 5,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  87749.1 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 2,
    //     totalRows / 20,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  48159.2 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 4,
    //     totalRows / 100,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  52005.2 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 4,
    //     totalRows / 10,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  29871.8 request/s
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 4,
    //     totalRows / 5,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // too slow lol.
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency() / 8,
    //     totalRows / 1000,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // too slow.
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency(),
    //     totalRows / 1,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // Very slow, due to low max concurrency.
    // runExprB(
    //     "",
    //     std::thread::hardware_concurrency(),
    //     totalRows / 1000,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);
  };

  auto runAsyncNoYeild = [&]() {
    // row throughput is  35107.3 request/s
    // (slightly better than the async lol)
    // runExprB(
    //     "",
    //     totalRows / 100,
    //     totalRows / 100,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);

    // row throughput is  29694.9 request/s
    // runExprB(
    //     "",
    //     totalRows / 50,
    //     totalRows / 50,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);

    // row throughput is  42894.7 request/s
    // runExprB(
    //     "",
    //     totalRows / 20,
    //     totalRows / 20,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);

    // row throughput is  35979.9 request/s
    // runExprB(
    //     "",
    //     totalRows / 20,
    //     totalRows / 20,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);

    // row throughput is  90538 request/s
    runExprB("", 20, totalRows / 20, true /*async*/, true /*yeild*/, totalRows);

    // row throughput is  39878.4 request/s
    // runExprB(
    //     "",
    //     totalRows / 15,
    //     totalRows / 15,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);

    // row throughput is  71183.7 request/s
    runExprB("", 15, totalRows / 15, true /*async*/, true /*yeild*/, totalRows);

    // row throughput is  32277.7 request/s
    // runExprB(
    //     "",
    //     totalRows / 15,
    //     totalRows / 15,
    //     true /*async*/,
    //     true /*yeild*/,
    //     totalRows);
  };

  // runAsyncWithYeild();

  // runAsyncNoYeild();

  auto runForOptimalLatency = [&]() {
    // row throughput is  35107.3 request/s
    // MAx latency is  572.161ms
    // runExprB(
    //     "",
    //     totalRows / 100,
    //     totalRows / 100,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);
    // row throughput is  36713.3 request/s
    // MAx latency is  285.297ms
    // runExprB(
    //     "",
    //     totalRows / 100,
    //     totalRows / 50,
    //     true /*async*/,
    //     false /*yeild*/,
    //     totalRows);
  };

  // runExprB("", 18, totalRows / 20, true /*async*/, true /*yeild*/,
  // totalRows);

  runForOptimalLatency();
  // Note : if we use globalIOExecutor when we do run with iyt yeikd we get bad
  // (max latency) 800ms vs 283ms
  // by not using globalIOExecutor we construct a thread on the fly and run it.

  // runExprB(
  //     "",
  //     totalRows / 20,
  //     totalRows / 20,
  //     true /*async*/,
  //     false /*yeild*/,
  //     totalRows);

  // yeild:1async:1thread-count:18driver-count20000
  // row throughput is  90467.4 request/s
  // total item count is 2000000

  // less sensetive to number of threads us
  //   row throughput is  96172.3 request/s
  // time per rpc 0.010398 milli seconds
  // total item count is 10000000
  // MAx latency is  213.156ms
  runExprB("", 18, 20000, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:5driver-count20000
  // row throughput is  96844.1 request/s
  // time per rpc 0.0103259 milli seconds
  // total item count is 10000000
  // MAx latency is  245.868ms
  runExprB("", 5, 20000, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:5driver-count20000
  // row throughput is  96883.6 request/s
  // time per rpc 0.0103217 milli seconds
  // total item count is 10000000
  // MAx latency is  238.135ms
  runExprB("", 5, 20000, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:18driver-count200000
  // row throughput is  97621.6 request/s
  // time per rpc 0.0102436 milli seconds
  // total item count is 10000000
  // MAx latency is  2299.83ms
  runExprB("", 18, 200000, true /*async*/, true /*yeild*/, totalRows);

  // does increasing the number of rows effect numbers (answer is no)
}

// This expr evalautes remote calls on full vectors (vectors of size 1000)
// We are looking at yeild vs no yeild.
// the difference is that there is more concurrency with in a blocked thread
// unlike exprB.
TEST_F(AsyncProjectTest, expC) {
  facebook::velox::exec::registerVectorFunction(
      "async_plus",
      {exec::FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .argumentType("integer")
           .build()},
      std::make_unique<PlusAsync>());
  auto totalRows = 6000000;

  // Async Projection
  // yeild:1async:1thread-count:18driver-count20
  // row throughput is  96585.7 request/s
  // time per rpc 0.0103535 milli seconds
  // total item count is 12000000
  // MAx latency is  346.764ms
  // runExprC("", 18, 20, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:18driver-count200
  // row throughput is  475045 request/s
  // time per rpc 0.00210507 milli seconds
  // total item count is 12000000
  // MAx latency is  1795.04ms
  // runExprC("", 18, 200, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:18driver-count300
  // row throughput is  485591 request/s
  // time per rpc 0.00205935 milli seconds
  // total item count is 12000000
  // MAx latency is  2560.82ms
  // runExprC("", 18, 300, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:18driver-count100
  // row throughput is  319569 request/s
  // time per rpc 0.00312921 milli seconds
  // total item count is 12000000
  // MAx latency is  923.561ms
  // runExprC("", 18, 100, true /*async*/, true /*yeild*/, totalRows);

  // yeild:1async:1thread-count:18driver-count50
  // row throughput is  220596 request/s
  // time per rpc 0.00453317 milli seconds
  // total item count is 20000000
  // MAx latency is  510.044ms
  // runExprC("", 18, 50, true /*async*/, true /*yeild*/, totalRows);

  // No yeild

  // yeild:0async:1thread-count:100driver-count100
  // row throughput is  251612 request/s
  // time per rpc 0.00397437 milli seconds
  // total item count is 10000000
  // MAx latency is  942.716ms
  runExprC("", 100, 100, true /*async*/, false /*yeild*/, totalRows);

  // yeild:0async:1thread-count:150driver-count150
  // row throughput is  290106 request/s
  // time per rpc 0.00344702 milli seconds
  // total item count is 9900000
  // MAx latency is 1147.65ms
  runExprC("", 150, 150, true /*async*/, false /*yeild*/, totalRows);

  // yeild:0async:1thread-count:200driver-count200
  // row throughput is  342505 request/s
  // time per rpc 0.00291966 milli seconds
  // total item count is 10000000
  // MAx latency is  1470.63ms
  runExprC("", 200, 200, true /*async*/, false /*yeild*/, totalRows);

  // yeild:0async:1thread-count:300driver-count300
  // row throughput is  330511 request/s
  // time per rpc 0.00302562 milli seconds
  // total item count is 12000000
  // MAx latency is  2253.58ms
  runExprC("", 300, 300, true /*async*/, false /*yeild*/, totalRows);
}

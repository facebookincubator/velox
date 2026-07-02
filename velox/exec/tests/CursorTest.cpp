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

#include "velox/exec/Cursor.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {

class CursorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    input_ = makeRowVector({
        makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
    });
  }

  void TearDown() override {
    // Release the input before OperatorTestBase::TearDown() resets pool_.
    input_.reset();
    OperatorTestBase::TearDown();
  }

  RowVectorPtr input_;

  CursorParameters makeParams(bool serialExecution) {
    CursorParameters params;
    params.planNode =
        PlanBuilder()
            .values(
                {input_, input_, input_}, /*parallelizable=*/!serialExecution)
            .planNode();
    params.serialExecution = serialExecution;
    return params;
  }

  // Drains a cursor through the non-blocking moveNext(future) protocol, waiting
  // on the returned future whenever the cursor reports it is blocked. Returns
  // the total number of output rows.
  static int64_t drainAsync(TaskCursor& cursor) {
    int64_t rows{0};
    for (;;) {
      ContinueFuture future = ContinueFuture::makeEmpty();
      if (cursor.moveNext(&future)) {
        rows += cursor.current()->size();
        continue;
      }
      if (!future.valid()) {
        break;
      }
      future.wait();
    }
    return rows;
  }

  static int64_t drainSync(TaskCursor& cursor) {
    int64_t rows{0};
    while (cursor.moveNext()) {
      rows += cursor.current()->size();
    }
    return rows;
  }
};

// The async drain protocol returns the same data as the blocking drain, for
// both the parallel (multi-threaded) and serial cursors.
TEST_F(CursorTest, asyncDrainParallel) {
  auto cursor = TaskCursor::create(makeParams(/*serialExecution=*/false));
  EXPECT_EQ(drainAsync(*cursor), 30);
}

TEST_F(CursorTest, asyncDrainSerial) {
  auto cursor = TaskCursor::create(makeParams(/*serialExecution=*/true));
  EXPECT_EQ(drainAsync(*cursor), 30);
}

TEST_F(CursorTest, asyncMatchesSync) {
  auto syncCursor = TaskCursor::create(makeParams(/*serialExecution=*/false));
  const int64_t syncRows = drainSync(*syncCursor);

  auto asyncCursor = TaskCursor::create(makeParams(/*serialExecution=*/false));
  EXPECT_EQ(drainAsync(*asyncCursor), syncRows);
}

// With multiple output drivers (producers), a blocked async consumer must still
// be woken and drain to completion. Exercises terminal-only consumer signaling:
// a non-terminal producer finishing must not be relied on to wake the consumer,
// and the terminal one must — a missed wakeup here would hang the drain.
TEST_F(CursorTest, asyncDrainParallelMultipleProducers) {
  constexpr int32_t kNumDrivers = 4;
  // input_ holds 10 rows; the plan emits 3 copies and, being parallelizable,
  // replicates that full 30-row set on each driver.
  constexpr int64_t kRowsPerDriver = 3 * 10;

  CursorParameters params;
  params.planNode =
      PlanBuilder()
          .values({input_, input_, input_}, /*parallelizable=*/true)
          .planNode();
  params.serialExecution = false;
  params.maxDrivers = kNumDrivers;

  auto cursor = TaskCursor::create(params);
  cursor->start();
  // The driver count is fixed by maxDrivers at task creation (not timing), so
  // pin it exactly: a regression that ran a single producer would both miss the
  // multi-producer wakeup path and fail here loudly.
  EXPECT_EQ(cursor->task()->numOutputDrivers(), kNumDrivers);
  EXPECT_EQ(drainAsync(*cursor), kNumDrivers * kRowsPerDriver);
}

// After the task is cancelled, the parallel cursor surfaces the error from
// moveNext() rather than returning data. requestCancel().wait() makes the
// terminal state deterministic before we pull.
TEST_F(CursorTest, cancelSurfacesErrorParallel) {
  auto cursor = TaskCursor::create(makeParams(/*serialExecution=*/false));
  cursor->start();
  cursor->task()->requestCancel().wait();

  ContinueFuture future = ContinueFuture::makeEmpty();
  VELOX_ASSERT_THROW(cursor->moveNext(&future), "Cancelled");
}

// The serial cursor surfaces the cancellation error too, rather than reporting
// a terminated task as end-of-stream (false with an invalid future).
TEST_F(CursorTest, cancelSurfacesErrorSerial) {
  auto cursor = TaskCursor::create(makeParams(/*serialExecution=*/true));
  cursor->start();
  cursor->task()->requestCancel().wait();

  ContinueFuture future = ContinueFuture::makeEmpty();
  VELOX_ASSERT_THROW(cursor->moveNext(&future), "Cancelled");
}

} // namespace facebook::velox::exec::test

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
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

struct TaskCompletedEvent {
  std::string taskUuid;
  exec::TaskState state;
  std::exception_ptr error;
  exec::TaskStats stats;

  std::string errorMessage() const {
    if (error) {
      try {
        std::rethrow_exception(error);
      } catch (const VeloxException& e) {
        return e.message();
      } catch (const std::exception& e) {
        return e.what();
      }
    } else {
      return "";
    }
  }
};

class TestTaskListener : public exec::TaskListener {
 public:
  void onTaskCompletion(
      const std::string& taskUuid,
      const std::string& taskId,
      exec::TaskState state,
      std::exception_ptr error,
      exec::TaskStats stats) override {
    events_.push_back({taskUuid, state, error, stats});
  }

  std::vector<TaskCompletedEvent>& events() {
    return events_;
  }

 private:
  std::vector<TaskCompletedEvent> events_;
};

class TaskListenerTest : public OperatorTestBase {};

TEST_F(TaskListenerTest, success) {
  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});

  auto plan = PlanBuilder().values({data}).planNode();

  // Register event listener to collect task completion events.
  auto listener = std::make_shared<TestTaskListener>();
  auto& events = listener->events();
  ASSERT_TRUE(exec::registerTaskListener(listener));

  assertQuery(plan, "VALUES (0), (1), (2), (3), (4)");
  ASSERT_EQ(1, events.size());
  ASSERT_EQ(nullptr, events.back().error);

  // Register the same listener again. This should have no effect as the
  // registration logic should detect a duplicate and not register it again.
  ASSERT_FALSE(exec::registerTaskListener(listener));

  // Clear the events, re-run the query and verify that a new event is received.
  events.clear();

  assertQuery(plan, "VALUES (0), (1), (2), (3), (4)");
  ASSERT_EQ(1, events.size());
  ASSERT_EQ(nullptr, events.back().error);

  // Clear the events, unregister the listener, re-run the query and verify that
  // no event is received.
  events.clear();
  ASSERT_TRUE(exec::unregisterTaskListener(listener));

  assertQuery(plan, "VALUES (0), (1), (2), (3), (4)");
  ASSERT_TRUE(events.empty());

  // Try to unregister the listener again.
  ASSERT_FALSE(exec::unregisterTaskListener(listener));
}

TEST_F(TaskListenerTest, error) {
  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});

  auto plan = PlanBuilder().values({data}).project({"10 / c0"}).planNode();

  CursorParameters params;
  params.planNode = plan;

  // Register event listener to collect task completion events.
  auto listener = std::make_shared<TestTaskListener>();
  auto& events = listener->events();
  ASSERT_TRUE(exec::registerTaskListener(listener));

  EXPECT_THROW(readCursor(params, [](auto) {}), VeloxException);

  ASSERT_EQ(1, events.size());
  ASSERT_EQ("division by zero", events.back().errorMessage());

  ASSERT_TRUE(exec::unregisterTaskListener(listener));
}

TEST_F(TaskListenerTest, multipleListeners) {
  auto listener1 = std::make_shared<TestTaskListener>();
  auto listener2 = std::make_shared<TestTaskListener>();
  ASSERT_TRUE(exec::registerTaskListener(listener1));
  ASSERT_TRUE(exec::registerTaskListener(listener2));

  ASSERT_TRUE(listener1->events().empty());
  ASSERT_TRUE(listener2->events().empty());

  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});
  auto plan = PlanBuilder().values({data}).planNode();
  assertQuery(plan, "VALUES (0), (1), (2), (3), (4)");

  ASSERT_EQ(1, listener1->events().size());
  ASSERT_EQ(1, listener2->events().size());

  {
    const auto& event1 = listener1->events().front();
    const auto& event2 = listener2->events().front();
    ASSERT_FALSE(event1.taskUuid.empty());
    ASSERT_EQ(event1.taskUuid, event2.taskUuid);
    ASSERT_EQ(event1.state, exec::TaskState::kFinished);
    ASSERT_EQ(event2.state, exec::TaskState::kFinished);
  }

  ASSERT_TRUE(exec::unregisterTaskListener(listener1));
  ASSERT_TRUE(exec::unregisterTaskListener(listener2));
}

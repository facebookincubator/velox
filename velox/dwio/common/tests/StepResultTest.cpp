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

#include "velox/dwio/common/StepResult.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <list>

using namespace ::testing;
using namespace ::facebook::velox::dwio::common;

namespace {

enum class ReaderState {
  READ,
  NEEDS_MORE_IO,
  END_OF_FILE,
};

enum class SplitState {
  READ,
  NEEDS_MORE_IO,
  END_OF_SPLIT,
};

std::ostream& operator<<(std::ostream& os, ReaderState state) {
  return os << static_cast<int>(state);
}

using ReaderResult = StepResult<ReaderState, ReaderState::READ, uint64_t>;
using SplitResult = StepResult<SplitState, SplitState::READ, uint64_t>;
using VoidResult = StepResult<ReaderState, ReaderState::READ, void>;

using actions = std::vector<int>;

auto getAction(actions& executedActions) {
  executedActions.push_back(0);
  return [&executedActions, i = executedActions.size() - 1]() {
    executedActions.at(i)++;
  };
}

std::vector<std::function<void()>> getActions(
    actions& executedActions,
    int numActions) {
  std::vector<std::function<void()>> actions;
  actions.reserve(numActions);
  for (int i = 0; i < numActions; i++) {
    actions.push_back(getAction(executedActions));
  }
  return actions;
}

template <typename Result>
class StepsJob {
 public:
  explicit StepsJob(std::vector<Result> results)
      : results_{std::move(results)} {}

  Result tryNext() {
    VELOX_CHECK_LT(i_, results_.size());
    return results_.at(i_++);
  }

 private:
  std::vector<Result> results_;
  size_t i_ = 0;
};

template <typename T>
class StepResultTypedTest : public testing::Test {};

} // namespace

TEST(StepResultTest, Read) {
  ReaderResult readerResult(10);
  EXPECT_EQ(readerResult.state(), ReaderState::READ);
  ASSERT_TRUE(readerResult.hasResult());
  EXPECT_EQ(readerResult.result(), 10);
  EXPECT_EQ(readerResult.actions().size(), 0);
}

TEST(StepResultTest, Void) {
  {
    VoidResult readerResult;
    EXPECT_EQ(readerResult.state(), ReaderState::READ);
    EXPECT_EQ(readerResult.actions().size(), 0);
  }
  {
    VoidResult readerResult(ReaderState::READ);
    EXPECT_EQ(readerResult.state(), ReaderState::READ);
    EXPECT_EQ(readerResult.actions().size(), 0);
  }
  actions executedActions;
  {
    VoidResult readerResult(ReaderState::READ, getAction(executedActions));
    EXPECT_EQ(readerResult.state(), ReaderState::READ);
    EXPECT_EQ(readerResult.actions().size(), 1);
  }
  {
    VoidResult readerResult(ReaderState::READ, getActions(executedActions, 2));
    EXPECT_EQ(readerResult.state(), ReaderState::READ);
    EXPECT_EQ(readerResult.actions().size(), 2);
  }
}

TEST(StepResultTest, MergeToResult) {
  SplitResult splitResult(1);
  actions executedActions;

  ReaderResult readerResult(
      ReaderState::NEEDS_MORE_IO, getAction(executedActions));

  EXPECT_THAT(
      [&]() { splitResult.mergeActionsFrom(std::move(readerResult)); },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::message,
          HasSubstr("Can't merge actions if destination class has a result"))));
}

TEST(StepResultTest, MergeFromResult) {
  SplitResult splitResult(SplitState::NEEDS_MORE_IO);
  ReaderResult readerResult(1);

  EXPECT_THAT(
      [&]() { splitResult.mergeActionsFrom(std::move(readerResult)); },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::message,
          HasSubstr("Can't merge actions of class that has a result"))));
}

TEST(StepResultTest, ResultStateMustHaveAResultForNonVoid) {
  actions executedActions;
  EXPECT_THAT(
      [&]() { ReaderResult readerResult(ReaderState::READ); },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::message,
          HasSubstr(
              "Result state must have a result for non-void ResultType"))));
  EXPECT_THAT(
      [&]() {
        ReaderResult readerResult(
            ReaderState::READ, getAction(executedActions));
      },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::message,
          HasSubstr(
              "Result state must have a result for non-void ResultType"))));
  EXPECT_THAT(
      [&]() {
        ReaderResult readerResult(
            ReaderState::READ, getActions(executedActions, 2));
      },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::message,
          HasSubstr(
              "Result state must have a result for non-void ResultType"))));
}

using ReaderResultTypes = ::testing::Types<ReaderResult, VoidResult>;
TYPED_TEST_CASE(StepResultTypedTest, ReaderResultTypes);

TYPED_TEST(StepResultTypedTest, EndOfFile) {
  TypeParam readerResult(ReaderState::END_OF_FILE);
  EXPECT_EQ(readerResult.state(), ReaderState::END_OF_FILE);
  if constexpr (!std::is_same_v<TypeParam, VoidResult>) {
    ASSERT_FALSE(readerResult.hasResult());
    EXPECT_THAT(
        [&]() { readerResult.result(); },
        Throws<facebook::velox::VeloxRuntimeError>(Property(
            &facebook::velox::VeloxRuntimeError::message,
            HasSubstr("Result is not set"))));
  }
  EXPECT_EQ(readerResult.actions().size(), 0);
  EXPECT_EQ(readerResult.runAllActions(), 0);
}

TYPED_TEST(StepResultTypedTest, ActionNeeded) {
  actions executedActions;
  TypeParam readerResult(
      ReaderState::NEEDS_MORE_IO, getAction(executedActions));
  EXPECT_EQ(readerResult.state(), ReaderState::NEEDS_MORE_IO);
  if constexpr (!std::is_same_v<TypeParam, VoidResult>) {
    ASSERT_FALSE(readerResult.hasResult());
    EXPECT_THAT(
        [&]() { readerResult.result(); },
        Throws<facebook::velox::VeloxRuntimeError>(Property(
            &facebook::velox::VeloxRuntimeError::message,
            HasSubstr("Result is not set"))));
  }
  EXPECT_EQ(readerResult.actions().size(), 1);
  EXPECT_EQ(executedActions, actions({0}));
  readerResult.actions()[0]();
  EXPECT_EQ(executedActions, actions({1}));
  EXPECT_EQ(readerResult.runAllActions(), 1);
  EXPECT_EQ(executedActions, actions({2}));
}

TYPED_TEST(StepResultTypedTest, ActionsNeeded) {
  actions executedActions;
  TypeParam readerResult(
      ReaderState::NEEDS_MORE_IO, getActions(executedActions, 2));
  EXPECT_EQ(readerResult.state(), ReaderState::NEEDS_MORE_IO);
  if constexpr (!std::is_same_v<TypeParam, VoidResult>) {
    ASSERT_FALSE(readerResult.hasResult());
    EXPECT_THAT(
        [&]() { readerResult.result(); },
        Throws<facebook::velox::VeloxRuntimeError>(Property(
            &facebook::velox::VeloxRuntimeError::message,
            HasSubstr("Result is not set"))));
  }
  EXPECT_EQ(readerResult.actions().size(), 2);
  EXPECT_EQ(executedActions, actions({0, 0}));
  readerResult.actions()[0]();
  EXPECT_EQ(executedActions, actions({1, 0}));
  readerResult.actions()[1]();
  EXPECT_EQ(executedActions, actions({1, 1}));
  EXPECT_EQ(readerResult.runAllActions(), 2);
  EXPECT_EQ(executedActions, actions({2, 2}));
}

TYPED_TEST(StepResultTypedTest, MergeActionsIncremental) {
  SplitResult splitResult(SplitState::NEEDS_MORE_IO);
  actions executedActions;
  {
    TypeParam readerResult(
        ReaderState::NEEDS_MORE_IO, getAction(executedActions));
    splitResult.mergeActionsFrom(std::move(readerResult));
  }
  ASSERT_EQ(splitResult.actions().size(), 1);
  EXPECT_EQ(executedActions, actions({0}));
  splitResult.actions()[0]();
  EXPECT_EQ(executedActions, actions({1}));
  {
    TypeParam readerResult(
        ReaderState::NEEDS_MORE_IO, getAction(executedActions));
    splitResult.mergeActionsFrom(std::move(readerResult));
  }
  ASSERT_EQ(splitResult.actions().size(), 2);
  EXPECT_EQ(executedActions, actions({1, 0}));
  splitResult.actions()[0]();
  EXPECT_EQ(executedActions, actions({2, 0}));
  splitResult.actions()[1]();
  EXPECT_EQ(executedActions, actions({2, 1}));
  {
    TypeParam readerResult(
        ReaderState::NEEDS_MORE_IO, getActions(executedActions, 2));
    splitResult.mergeActionsFrom(std::move(readerResult));
  }
  ASSERT_EQ(splitResult.actions().size(), 4);
  EXPECT_EQ(executedActions, actions({2, 1, 0, 0}));
  splitResult.actions()[2]();
  EXPECT_EQ(executedActions, actions({2, 1, 1, 0}));
  splitResult.actions()[3]();
  EXPECT_EQ(executedActions, actions({2, 1, 1, 1}));
  EXPECT_EQ(splitResult.runAllActions(), 4);
  EXPECT_EQ(executedActions, actions({3, 2, 2, 2}));
}

TYPED_TEST(StepResultTypedTest, MergeActions) {
  SplitResult splitResult(SplitState::NEEDS_MORE_IO);
  actions executedActions;

  {
    TypeParam readerResult(
        ReaderState::NEEDS_MORE_IO, getActions(executedActions, 2));
    splitResult.mergeActionsFrom(std::move(readerResult));
  }
  EXPECT_EQ(splitResult.actions().size(), 2);
  EXPECT_EQ(executedActions, actions({0, 0}));
  splitResult.actions()[0]();
  EXPECT_EQ(executedActions, actions({1, 0}));
  splitResult.actions()[1]();
  EXPECT_EQ(executedActions, actions({1, 1}));
  EXPECT_EQ(splitResult.runAllActions(), 2);
  EXPECT_EQ(executedActions, actions({2, 2}));
}

TYPED_TEST(StepResultTypedTest, TryUntilState) {
  actions executedActions;
  StepsJob<TypeParam> stepsJob(
      {TypeParam(ReaderState::NEEDS_MORE_IO, getAction(executedActions)),
       TypeParam(ReaderState::NEEDS_MORE_IO, getActions(executedActions, 2)),
       TypeParam(ReaderState::END_OF_FILE, getAction(executedActions))});
  auto lastState = tryUntilState(
      ReaderState::END_OF_FILE, [&stepsJob]() { return stepsJob.tryNext(); });
  EXPECT_EQ(executedActions, actions({1, 1, 1, 0}));
  EXPECT_EQ(lastState.state(), ReaderState::END_OF_FILE);
}

TYPED_TEST(StepResultTypedTest, TryUntilNotState) {
  actions executedActions;
  StepsJob<TypeParam> stepsJob(
      {TypeParam(ReaderState::NEEDS_MORE_IO, getAction(executedActions)),
       TypeParam(ReaderState::NEEDS_MORE_IO, getActions(executedActions, 2)),
       TypeParam(ReaderState::END_OF_FILE, getAction(executedActions))});
  auto lastState = tryUntilNotState(
      ReaderState::NEEDS_MORE_IO, [&stepsJob]() { return stepsJob.tryNext(); });
  EXPECT_EQ(executedActions, actions({1, 1, 1, 0}));
  EXPECT_EQ(lastState.state(), ReaderState::END_OF_FILE);
}

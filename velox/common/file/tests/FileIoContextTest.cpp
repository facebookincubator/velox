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

#include "velox/common/file/FileIoContext.h"

#include <string>
#include <string_view>
#include <thread>

#include <folly/coro/Baton.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"

using namespace facebook::velox;

namespace {

FileIoContext makeContext(std::string_view key) {
  FileIoContext context;
  context.fileOpts = {{"key", std::string(key)}};
  return context;
}

// Returns the 'key' file option of the active scope's context, or "none".
std::string scopedKey() {
  static const FileIoContext kNoScope;
  const auto& context = ScopedFileIoContext::resolve(kNoScope);
  return &context == &kNoScope ? "none" : context.fileOpts.at("key");
}

void expectNestedScopeThrows() {
  VELOX_ASSERT_USER_THROW(
      ScopedFileIoContext{makeContext("nested")},
      "File IO context scopes must not nest");
}

// Checks that no scope can start here, and returns the active scope's key.
folly::coro::Task<std::string> observeScope() {
  expectNestedScopeThrows();
  co_return scopedKey();
}

// Waits for 'scopeEnded', then observes the scope on 'executor'.
folly::coro::Task<std::string> observeScopeAfterItEnds(
    folly::coro::Baton& scopeEnded,
    folly::Executor::KeepAlive<> executor) {
  co_await scopeEnded;
  co_return co_await folly::coro::co_withExecutor(
      std::move(executor), observeScope());
}

} // namespace

TEST(FileIoContextTest, scopeReplacesContext) {
  IoStats inputStats;
  FileIoContext input;
  input.ioStats = &inputStats;
  input.fileOpts = {{"inputKey", "input"}};
  input.cacheable = true;
  EXPECT_EQ(&ScopedFileIoContext::resolve(input), &input);

  {
    const ScopedFileIoContext scope{makeContext("scope")};
    const auto& context = ScopedFileIoContext::resolve(input);
    const folly::F14FastMap<std::string, std::string> expectedFileOpts{
        {"key", "scope"}};
    EXPECT_EQ(context.fileOpts, expectedFileOpts);
    EXPECT_EQ(context.ioStats, nullptr);
    EXPECT_FALSE(context.cacheable);
  }
  EXPECT_EQ(&ScopedFileIoContext::resolve(input), &input);
}

TEST(FileIoContextTest, nestedScopeThrows) {
  {
    const ScopedFileIoContext scope{makeContext("outer")};
    expectNestedScopeThrows();
    EXPECT_EQ(scopedKey(), "outer");
  }
  EXPECT_EQ(scopedKey(), "none");

  const ScopedFileIoContext scope{makeContext("next")};
  EXPECT_EQ(scopedKey(), "next");
}

TEST(FileIoContextTest, scopeFollowsCoroutinesNotRawThreads) {
  folly::CPUThreadPoolExecutor startExecutor{1};
  folly::CPUThreadPoolExecutor hopExecutor{1};
  folly::coro::Baton scopeEnded;
  auto keyAfterScope = folly::SemiFuture<std::string>::makeEmpty();
  std::string threadKey;
  {
    const ScopedFileIoContext scope{makeContext("scope")};
    keyAfterScope =
        folly::coro::co_withExecutor(
            &startExecutor, observeScopeAfterItEnds(scopeEnded, &hopExecutor))
            .start();
    std::thread([&threadKey] { threadKey = scopedKey(); }).join();
  }
  scopeEnded.post();

  // The coroutine keeps the scope it started in, on another thread and after
  // the scope ends. A raw thread never sees the scope.
  EXPECT_EQ(std::move(keyAfterScope).get(), "scope");
  EXPECT_EQ(threadKey, "none");
}

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

#include "velox/dwio/common/Writer.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

using namespace ::testing;

namespace facebook::velox::dwio::common {
namespace {

class MockWriter : public Writer {
 public:
  MockWriter() = default;

  void setStateForTest(State state) {
    setState(state);
  }

  void callCheckRunning() const {
    checkRunning();
  }

  void write(const VectorPtr& /*data*/) override {}

  void flush() override {}

  bool finish() override {
    return true;
  }

  void abort() override {}

  void close() override {}
};

TEST(WriterTest, stateString) {
  ASSERT_EQ(Writer::stateString(Writer::State::kInit), "INIT");
  ASSERT_EQ(Writer::stateString(Writer::State::kRunning), "RUNNING");
  ASSERT_EQ(Writer::stateString(Writer::State::kClosed), "CLOSED");
  ASSERT_EQ(Writer::stateString(Writer::State::kFinishing), "FINISHING");
  ASSERT_EQ(Writer::stateString(Writer::State::kAborted), "ABORTED");
  VELOX_ASSERT_THROW(
      Writer::stateString(static_cast<Writer::State>(100)), "BAD STATE: 100");
}

TEST(WriterTest, checkRunning) {
  MockWriter writer;
  VELOX_ASSERT_THROW(writer.callCheckRunning(), "Writer is not running: INIT");
  writer.setStateForTest(Writer::State::kRunning);
  ASSERT_NO_THROW(writer.callCheckRunning());
  writer.setStateForTest(Writer::State::kClosed);
  VELOX_ASSERT_USER_THROW(
      writer.callCheckRunning(),
      "Writer is not running: CLOSED. Write operations are not allowed on a closed writer.");
}
} // namespace
} // namespace facebook::velox::dwio::common

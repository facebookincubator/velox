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

  bool callIsRunning() const {
    return isRunning();
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
  VELOX_ASSERT_THROW(
      writer.callCheckRunning(), "Writer is not running: CLOSED");
}

TEST(WriterTest, stateTransitions) {
  MockWriter writer;
  ASSERT_EQ(writer.state(), Writer::State::kInit);

  // Valid transition: kInit -> kRunning
  writer.setStateForTest(Writer::State::kRunning);
  ASSERT_EQ(writer.state(), Writer::State::kRunning);

  // Valid transition: kRunning -> kFinishing
  writer.setStateForTest(Writer::State::kFinishing);
  ASSERT_EQ(writer.state(), Writer::State::kFinishing);

  // Valid transition: kFinishing -> kFinishing (reentry)
  writer.setStateForTest(Writer::State::kFinishing);
  ASSERT_EQ(writer.state(), Writer::State::kFinishing);

  // Valid transition: kFinishing -> kClosed
  writer.setStateForTest(Writer::State::kClosed);
  ASSERT_EQ(writer.state(), Writer::State::kClosed);
}

TEST(WriterTest, invalidStateTransitions) {
  {
    MockWriter writer;
    // Invalid: kInit -> kClosed
    VELOX_ASSERT_THROW(
        writer.setStateForTest(Writer::State::kClosed),
        "Unexpected state transition from INIT to CLOSED");
  }
  {
    MockWriter writer;
    // Invalid: kInit -> kFinishing
    VELOX_ASSERT_THROW(
        writer.setStateForTest(Writer::State::kFinishing),
        "Unexpected state transition from INIT to FINISHING");
  }
  {
    MockWriter writer;
    writer.setStateForTest(Writer::State::kRunning);
    writer.setStateForTest(Writer::State::kClosed);
    // Invalid: kClosed -> kRunning
    VELOX_ASSERT_THROW(
        writer.setStateForTest(Writer::State::kRunning),
        "Unexpected state transition from CLOSED to RUNNING");
  }
}

TEST(WriterTest, stateGetter) {
  MockWriter writer;
  ASSERT_EQ(writer.state(), Writer::State::kInit);

  writer.setStateForTest(Writer::State::kRunning);
  ASSERT_EQ(writer.state(), Writer::State::kRunning);

  writer.setStateForTest(Writer::State::kFinishing);
  ASSERT_EQ(writer.state(), Writer::State::kFinishing);

  writer.setStateForTest(Writer::State::kClosed);
  ASSERT_EQ(writer.state(), Writer::State::kClosed);
}

TEST(WriterTest, isRunning) {
  MockWriter writer;
  ASSERT_FALSE(writer.callIsRunning());

  writer.setStateForTest(Writer::State::kRunning);
  ASSERT_TRUE(writer.callIsRunning());

  writer.setStateForTest(Writer::State::kFinishing);
  ASSERT_FALSE(writer.callIsRunning());

  writer.setStateForTest(Writer::State::kClosed);
  ASSERT_FALSE(writer.callIsRunning());
}

} // namespace
} // namespace facebook::velox::dwio::common

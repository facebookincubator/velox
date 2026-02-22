/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <gtest/gtest.h>

#include <sstream>

#include "velox/tool/trace/TraceReplayerConfigPrompt.h"

DECLARE_string(root_dir);
DECLARE_string(query_id);
DECLARE_string(task_id);
DECLARE_string(node_id);
DECLARE_string(driver_ids);
DECLARE_string(table_writer_output_dir);

namespace facebook::velox::tool::trace::test {
namespace {

class TraceReplayerConfigPromptTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original flag values
    originalRootDir_ = FLAGS_root_dir;
    originalQueryId_ = FLAGS_query_id;
    originalTaskId_ = FLAGS_task_id;
    originalNodeId_ = FLAGS_node_id;
    originalDriverIds_ = FLAGS_driver_ids;
    originalTableWriterOutputDir_ = FLAGS_table_writer_output_dir;
    originalFast_ = FLAGS_fast;

    // Reset flags to empty/default state
    FLAGS_root_dir = "";
    FLAGS_query_id = "";
    FLAGS_task_id = "";
    FLAGS_node_id = "";
    FLAGS_driver_ids = "";
    FLAGS_table_writer_output_dir = "";
    FLAGS_fast = false;
  }

  void TearDown() override {
    // Restore original flag values
    FLAGS_root_dir = originalRootDir_;
    FLAGS_query_id = originalQueryId_;
    FLAGS_task_id = originalTaskId_;
    FLAGS_node_id = originalNodeId_;
    FLAGS_driver_ids = originalDriverIds_;
    FLAGS_table_writer_output_dir = originalTableWriterOutputDir_;
    FLAGS_fast = originalFast_;

    // Restore cin buffer if it was redirected
    if (originalCinBuf_ != nullptr) {
      std::cin.rdbuf(originalCinBuf_);
      originalCinBuf_ = nullptr;
    }
  }

  // Helper to redirect stdin from a string
  void setStdinInput(const std::string& input) {
    stdinStream_ = std::make_unique<std::istringstream>(input);
    originalCinBuf_ = std::cin.rdbuf(stdinStream_->rdbuf());
  }

  std::string originalRootDir_;
  std::string originalQueryId_;
  std::string originalTaskId_;
  std::string originalNodeId_;
  std::string originalDriverIds_;
  std::string originalTableWriterOutputDir_;
  bool originalFast_;
  std::unique_ptr<std::istringstream> stdinStream_;
  std::streambuf* originalCinBuf_{nullptr};
};

TEST_F(TraceReplayerConfigPromptTest, fastModeSkipsPrompts) {
  // Set up flags with values
  FLAGS_root_dir = "/test/root";
  FLAGS_query_id = "query-123";
  FLAGS_task_id = "task-456";
  FLAGS_node_id = "node-789";
  FLAGS_fast = true;

  // Redirect stdout to capture output
  std::ostringstream capturedOutput;
  std::streambuf* originalCoutBuf = std::cout.rdbuf(capturedOutput.rdbuf());

  TraceReplayerConfigPrompt prompt;
  prompt.run();

  // Restore stdout
  std::cout.rdbuf(originalCoutBuf);

  // Verify flags remain unchanged
  EXPECT_EQ(FLAGS_root_dir, "/test/root");
  EXPECT_EQ(FLAGS_query_id, "query-123");
  EXPECT_EQ(FLAGS_task_id, "task-456");
  EXPECT_EQ(FLAGS_node_id, "node-789");

  // Verify output indicates fast mode
  std::string output = capturedOutput.str();
  EXPECT_TRUE(output.find("--fast mode") != std::string::npos);
  EXPECT_TRUE(output.find("Skipping interactive prompts") != std::string::npos);
}

TEST_F(TraceReplayerConfigPromptTest, interactivePromptsSetFlags) {
  // Simulate user entering values for all prompts
  setStdinInput(
      "/root/dir\n" // root_dir
      "query-id-1\n" // query_id
      "task-id-1\n" // task_id
      "node-id-1\n" // node_id
      "0,1,2\n" // driver_ids
      "/output/dir\n"); // table_writer_output_dir

  std::ostringstream capturedOutput;
  std::streambuf* originalCoutBuf = std::cout.rdbuf(capturedOutput.rdbuf());

  FLAGS_fast = false;
  TraceReplayerConfigPrompt prompt;
  prompt.run();

  std::cout.rdbuf(originalCoutBuf);

  EXPECT_EQ(FLAGS_root_dir, "/root/dir");
  EXPECT_EQ(FLAGS_query_id, "query-id-1");
  EXPECT_EQ(FLAGS_task_id, "task-id-1");
  EXPECT_EQ(FLAGS_node_id, "node-id-1");
  EXPECT_EQ(FLAGS_driver_ids, "0,1,2");
  EXPECT_EQ(FLAGS_table_writer_output_dir, "/output/dir");
}

TEST_F(TraceReplayerConfigPromptTest, interactivePromptsShowFastCommand) {
  // Simulate user entering values
  setStdinInput(
      "/root/dir\n"
      "query-123\n"
      "task-456\n"
      "node-789\n"
      "\n" // empty driver_ids
      "\n"); // empty table_writer_output_dir

  std::ostringstream capturedOutput;
  std::streambuf* originalCoutBuf = std::cout.rdbuf(capturedOutput.rdbuf());

  FLAGS_fast = false;
  TraceReplayerConfigPrompt prompt;
  prompt.run();

  std::cout.rdbuf(originalCoutBuf);

  std::string output = capturedOutput.str();
  // Should show how to run with --fast next time
  EXPECT_TRUE(output.find("--fast") != std::string::npos);
  EXPECT_TRUE(output.find("--root_dir") != std::string::npos);
  EXPECT_TRUE(output.find("--query_id") != std::string::npos);
}

TEST_F(TraceReplayerConfigPromptTest, interactivePromptsKeepExistingValues) {
  // Set up existing flag values
  FLAGS_root_dir = "/existing/root";
  FLAGS_query_id = "existing-query";
  FLAGS_task_id = "existing-task";
  FLAGS_node_id = "existing-node";
  FLAGS_driver_ids = "1,2,3";
  FLAGS_table_writer_output_dir = "/existing/output";

  // Simulate user pressing Enter for all prompts (keep existing values)
  setStdinInput("\n\n\n\n\n\n");

  std::ostringstream capturedOutput;
  std::streambuf* originalCoutBuf = std::cout.rdbuf(capturedOutput.rdbuf());

  FLAGS_fast = false;
  TraceReplayerConfigPrompt prompt;
  prompt.run();

  std::cout.rdbuf(originalCoutBuf);

  // All flags should remain unchanged
  EXPECT_EQ(FLAGS_root_dir, "/existing/root");
  EXPECT_EQ(FLAGS_query_id, "existing-query");
  EXPECT_EQ(FLAGS_task_id, "existing-task");
  EXPECT_EQ(FLAGS_node_id, "existing-node");
  EXPECT_EQ(FLAGS_driver_ids, "1,2,3");
  EXPECT_EQ(FLAGS_table_writer_output_dir, "/existing/output");
}

TEST_F(TraceReplayerConfigPromptTest, interactivePromptsPartialOverride) {
  // Set up existing flag values
  FLAGS_root_dir = "/existing/root";
  FLAGS_query_id = "existing-query";
  FLAGS_task_id = "";
  FLAGS_node_id = "existing-node";

  // Simulate: keep root_dir, keep query_id, enter new task_id, keep node_id,
  // etc.
  setStdinInput(
      "\n" // keep root_dir
      "\n" // keep query_id
      "new-task-id\n" // override task_id
      "\n" // keep node_id
      "\n" // keep driver_ids
      "\n"); // keep table_writer_output_dir

  std::ostringstream capturedOutput;
  std::streambuf* originalCoutBuf = std::cout.rdbuf(capturedOutput.rdbuf());

  FLAGS_fast = false;
  TraceReplayerConfigPrompt prompt;
  prompt.run();

  std::cout.rdbuf(originalCoutBuf);

  EXPECT_EQ(FLAGS_root_dir, "/existing/root");
  EXPECT_EQ(FLAGS_query_id, "existing-query");
  EXPECT_EQ(FLAGS_task_id, "new-task-id");
  EXPECT_EQ(FLAGS_node_id, "existing-node");
}

} // namespace
} // namespace facebook::velox::tool::trace::test

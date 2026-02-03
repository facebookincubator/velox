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

#include "velox/tool/trace/TraceReplayerConfigPrompt.h"

#include <gflags/gflags.h>
#include <iostream>
#include <sstream>
#include <string>

DEFINE_bool(
    fast,
    false,
    "Skip interactive prompts and use existing gflag values directly. "
    "Use this flag when you already know the config values and want to "
    "bypass the interactive configuration.");

DECLARE_string(root_dir);
DECLARE_string(query_id);
DECLARE_string(task_id);
DECLARE_string(node_id);
DECLARE_string(driver_ids);
DECLARE_string(table_writer_output_dir);

namespace facebook::velox::tool::trace {

namespace {
constexpr const char* kColorReset = "\033[0m";
constexpr const char* kColorBold = "\033[1m";
constexpr const char* kColorGreen = "\033[32m";
constexpr const char* kColorYellow = "\033[33m";
constexpr const char* kColorCyan = "\033[36m";

// Reads a line from stdin, trimming leading/trailing whitespace.
std::string readLine() {
  std::string input;
  std::getline(std::cin, input);

  // Trim leading whitespace
  size_t start = input.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }

  // Trim trailing whitespace
  size_t end = input.find_last_not_of(" \t\n\r");
  return input.substr(start, end - start + 1);
}

// Prints a separator line for visual clarity.
void printSeparator() {
  std::cout << kColorCyan
            << "============================================================"
            << kColorReset << std::endl;
}

// Prints a header for the config prompt session.
void printHeader() {
  std::cout << std::endl;
  printSeparator();
  std::cout << kColorBold << kColorGreen
            << "          Trace Replayer Configuration Prompt" << kColorReset
            << std::endl;
  printSeparator();
  std::cout << std::endl;
  std::cout << "Enter config values or press " << kColorYellow << "Enter"
            << kColorReset << " to keep current gflag value." << std::endl;
  std::cout << std::endl;
}

// Prints the equivalent command with --fast flag for future use.
void printFastCommand() {
  std::ostringstream cmd;
  cmd << "--fast";

  if (!FLAGS_root_dir.empty()) {
    cmd << " \\\n  --root_dir=\"" << FLAGS_root_dir << "\"";
  }
  if (!FLAGS_query_id.empty()) {
    cmd << " \\\n  --query_id=\"" << FLAGS_query_id << "\"";
  }
  if (!FLAGS_task_id.empty()) {
    cmd << " \\\n  --task_id=\"" << FLAGS_task_id << "\"";
  }
  if (!FLAGS_node_id.empty()) {
    cmd << " \\\n  --node_id=\"" << FLAGS_node_id << "\"";
  }
  if (!FLAGS_driver_ids.empty()) {
    cmd << " \\\n  --driver_ids=\"" << FLAGS_driver_ids << "\"";
  }
  if (!FLAGS_table_writer_output_dir.empty()) {
    cmd << " \\\n  --table_writer_output_dir=\""
        << FLAGS_table_writer_output_dir << "\"";
  }

  std::cout << std::endl;
  printSeparator();
  std::cout << kColorBold << kColorGreen << "Quick Command for Next Run"
            << kColorReset << std::endl;
  printSeparator();
  std::cout << std::endl;
  std::cout
      << "To skip prompts next time, copy and run with the following flags:\n"
      << std::endl;
  std::cout << kColorYellow << cmd.str() << kColorReset << std::endl;
  std::cout << std::endl;
  printSeparator();
  std::cout << std::endl;
}

// Prompts for a single config value.
// @param configName The name of the config (for display purposes).
// @param currentValue The current value from gflag.
// @param description A brief description of what this config is for.
// @param required Whether this config is required (cannot be empty).
// @return The final value (either user input or the original gflag value).
std::string promptForConfig(
    const std::string& configName,
    const std::string& currentValue,
    const std::string& description,
    bool required) {
  std::cout << kColorBold << configName << kColorReset << std::endl;
  std::cout << "  " << description << std::endl;
  std::cout << "  Current gflag value: " << kColorYellow
            << (currentValue.empty() ? "(empty)" : currentValue) << kColorReset
            << std::endl;

  if (required) {
    std::cout << "  " << kColorCyan << "[REQUIRED]" << kColorReset << std::endl;
  }

  std::cout << "  Enter new value: ";
  std::string input = readLine();

  std::string finalValue = input.empty() ? currentValue : input;

  if (required && finalValue.empty()) {
    std::cout << kColorYellow << "  Warning: This config is required but empty."
              << kColorReset << std::endl;
  }

  std::cout << "  Using: " << kColorGreen << finalValue << kColorReset
            << std::endl;
  std::cout << std::endl;

  return finalValue;
}
} // namespace

void TraceReplayerConfigPrompt::run() {
  if (FLAGS_fast) {
    std::cout << kColorCyan << "[--fast mode] "
              << "Skipping interactive prompts, using existing gflag values."
              << kColorReset << std::endl;
    std::cout << std::endl;

    // Print summary of current config values
    std::cout << kColorBold << "Current Configuration:" << kColorReset
              << std::endl;
    std::cout << "  root_dir: " << FLAGS_root_dir << std::endl;
    std::cout << "  query_id: " << FLAGS_query_id << std::endl;
    std::cout << "  task_id: "
              << (FLAGS_task_id.empty() ? "(empty - summary mode)"
                                        : FLAGS_task_id)
              << std::endl;
    std::cout << "  node_id: " << FLAGS_node_id << std::endl;
    std::cout << "  driver_ids: "
              << (FLAGS_driver_ids.empty() ? "(all drivers)" : FLAGS_driver_ids)
              << std::endl;
    std::cout << "  table_writer_output_dir: "
              << (FLAGS_table_writer_output_dir.empty()
                      ? "(not set)"
                      : FLAGS_table_writer_output_dir)
              << std::endl;
    std::cout << std::endl;
    return;
  }

  printHeader();

  // Prompt for critical required configs
  FLAGS_root_dir = promptForConfig(
      "--root_dir",
      FLAGS_root_dir,
      "Root directory where the replayer reads traced data",
      true);

  FLAGS_query_id = promptForConfig(
      "--query_id", FLAGS_query_id, "The target query id to replay", true);

  FLAGS_task_id = promptForConfig(
      "--task_id",
      FLAGS_task_id,
      "The target task id to replay (leave empty for summary mode)",
      false);

  FLAGS_node_id = promptForConfig(
      "--node_id", FLAGS_node_id, "The target node id to replay", true);

  // Prompt for optional configs
  std::cout << kColorBold << "Optional Configs:" << kColorReset << std::endl;
  std::cout << std::endl;

  FLAGS_driver_ids = promptForConfig(
      "--driver_ids",
      FLAGS_driver_ids,
      "A comma-separated list of target driver ids",
      false);

  FLAGS_table_writer_output_dir = promptForConfig(
      "--table_writer_output_dir",
      FLAGS_table_writer_output_dir,
      "Output directory for TableWriter (required for TableWrite replays)",
      false);

  printSeparator();
  std::cout << kColorGreen << "Configuration complete!" << kColorReset
            << std::endl;
  std::cout << std::endl;

  // Print summary
  std::cout << kColorBold << "Final Configuration Summary:" << kColorReset
            << std::endl;
  std::cout << "  root_dir: " << FLAGS_root_dir << std::endl;
  std::cout << "  query_id: " << FLAGS_query_id << std::endl;
  std::cout << "  task_id: "
            << (FLAGS_task_id.empty() ? "(empty - summary mode)"
                                      : FLAGS_task_id)
            << std::endl;
  std::cout << "  node_id: " << FLAGS_node_id << std::endl;
  std::cout << "  driver_ids: "
            << (FLAGS_driver_ids.empty() ? "(all drivers)" : FLAGS_driver_ids)
            << std::endl;
  std::cout << "  table_writer_output_dir: "
            << (FLAGS_table_writer_output_dir.empty()
                    ? "(not set)"
                    : FLAGS_table_writer_output_dir)
            << std::endl;

  // Print the fast command for next time
  printFastCommand();
}

} // namespace facebook::velox::tool::trace

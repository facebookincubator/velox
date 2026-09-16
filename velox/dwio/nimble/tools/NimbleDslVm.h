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
#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

#include "velox/dwio/nimble/tools/NimbleDslLib.h"

namespace facebook::nimble::tools {

enum class OpCode {
  Select,
  Describe,
  ShowSchema,
  ShowInfo,
  ShowStats,
  ShowStripes,
  ShowStreams,
  ShowEncoding,
  ShowIndex,
  ShowHistogram,
  ShowContent,
  ShowFileLayout,
  ShowStripesMetadata,
  ShowStripeGroups,
  ShowOptionalSections,
  Help,
  Quit,
};

/// Keywords recognized by the compiler/tokenizer.
enum class Keyword {
  Unknown,
  // Top-level commands
  Quit,
  Exit,
  Help,
  Describe,
  Select,
  Show,
  // SHOW subcommands
  Schema,
  Info,
  Stats,
  Stripes,
  Streams,
  Encoding,
  Index,
  Histogram,
  Content,
  File,
  Stripe,
  Optional,
  // Clauses / modifiers
  Limit,
  Offset,
  From,
  Layout,
  Groups,
  Sections,
  Top,
  Metadata,
  Star,
};

struct NoOperands {};

struct SelectOperands {
  std::vector<std::string> columns;
  uint64_t limit{20};
  uint64_t offset{0};
  std::optional<uint32_t> stripeId;
};

struct ShowStreamsOperands {
  std::optional<uint32_t> stripeId;
};

struct ShowEncodingOperands {
  std::optional<uint32_t> stripeId;
};

struct ShowHistogramOperands {
  bool topLevel{false};
  std::optional<uint32_t> stripeId;
};

struct ShowContentOperands {
  uint32_t streamId{};
  std::optional<uint32_t> stripeId;
};

using Operands = std::variant<
    NoOperands,
    SelectOperands,
    ShowStreamsOperands,
    ShowEncodingOperands,
    ShowHistogramOperands,
    ShowContentOperands>;

struct Instruction {
  OpCode opcode;
  Operands operands;
};

struct CompileResult {
  std::vector<Instruction> program;
  std::string error;
};

/// Convert a token string to a Keyword enum value.
Keyword toKeyword(const std::string& token);

/// String utility: convert to uppercase.
std::string toUpper(std::string s);

/// String utility: trim leading/trailing whitespace.
std::string trim(const std::string& s);

/// Tokenize input, stripping trailing semicolons and commas.
std::vector<std::string> tokenize(const std::string& input);

/// Compile tokens into a program (list of instructions).
CompileResult compile(const std::vector<std::string>& tokens);

/// Print the REPL help message.
void printHelp(std::ostream& out, bool color);

/// Virtual machine that executes compiled programs against a NimbleDslLib.
class NimbleDslVm {
 public:
  NimbleDslVm(NimbleDslLib& lib, std::ostream& out, bool enableColors)
      : lib_{lib}, out_{out}, enableColors_{enableColors} {}

  /// Execute a program. Returns false if a Quit instruction is encountered.
  bool execute(const std::vector<Instruction>& program);

 private:
  NimbleDslLib& lib_;
  std::ostream& out_;
  bool enableColors_;
};

} // namespace facebook::nimble::tools

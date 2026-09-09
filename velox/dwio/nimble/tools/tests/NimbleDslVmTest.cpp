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

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/tools/NimbleDslLib.h"
#include "velox/dwio/nimble/tools/NimbleDslVm.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::tools {
namespace {

// ===== Compile tests =====

TEST(CompileTest, emptyTokens) {
  auto result = compile({});
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.program.empty());
}

TEST(CompileTest, quit) {
  auto result = compile({"quit"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Quit);
}

TEST(CompileTest, quitCaseInsensitive) {
  auto result = compile({"QUIT"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Quit);
}

TEST(CompileTest, exit) {
  auto result = compile({"EXIT"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Quit);
}

TEST(CompileTest, help) {
  auto result = compile({"help"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Help);
}

TEST(CompileTest, describe) {
  auto result = compile({"DESCRIBE"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Describe);
}

TEST(CompileTest, describeCaseInsensitive) {
  auto result = compile({"describe"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Describe);
}

TEST(CompileTest, selectStar) {
  auto result = compile({"SELECT", "*"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::Select);
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  EXPECT_TRUE(ops.columns.empty());
  EXPECT_EQ(ops.limit, 20);
  EXPECT_EQ(ops.offset, 0);
  EXPECT_FALSE(ops.stripeId.has_value());
}

TEST(CompileTest, selectColumns) {
  auto result = compile({"SELECT", "col1", "col2"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  const std::vector<std::string> expected{"col1", "col2"};
  EXPECT_EQ(ops.columns, expected);
}

TEST(CompileTest, selectWithLimit) {
  auto result = compile({"SELECT", "*", "LIMIT", "5"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  EXPECT_EQ(ops.limit, 5);
}

TEST(CompileTest, selectWithOffset) {
  auto result = compile({"SELECT", "*", "OFFSET", "10"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  EXPECT_EQ(ops.offset, 10);
}

TEST(CompileTest, selectWithStripe) {
  auto result = compile({"SELECT", "*", "STRIPE", "2"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 2);
}

TEST(CompileTest, selectWithAllClauses) {
  auto result = compile(
      {"select", "name", "age", "LIMIT", "50", "OFFSET", "10", "STRIPE", "3"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  const std::vector<std::string> expectedCols{"name", "age"};
  EXPECT_EQ(ops.columns, expectedCols);
  EXPECT_EQ(ops.limit, 50);
  EXPECT_EQ(ops.offset, 10);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 3);
}

TEST(CompileTest, selectWithFrom) {
  auto result = compile({"SELECT", "*", "FROM", "table1"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<SelectOperands>(result.program[0].operands);
  EXPECT_TRUE(ops.columns.empty());
}

TEST(CompileTest, showSchema) {
  auto result = compile({"SHOW", "SCHEMA"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowSchema);
}

TEST(CompileTest, showInfo) {
  auto result = compile({"show", "info"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowInfo);
}

TEST(CompileTest, showStats) {
  auto result = compile({"SHOW", "STATS"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStats);
}

TEST(CompileTest, showStripes) {
  auto result = compile({"SHOW", "STRIPES"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStripes);
}

TEST(CompileTest, showStreams) {
  auto result = compile({"SHOW", "STREAMS"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStreams);
  const auto& ops = std::get<ShowStreamsOperands>(result.program[0].operands);
  EXPECT_FALSE(ops.stripeId.has_value());
}

TEST(CompileTest, showStreamsWithStripe) {
  auto result = compile({"SHOW", "STREAMS", "STRIPE", "1"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<ShowStreamsOperands>(result.program[0].operands);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 1);
}

TEST(CompileTest, showEncoding) {
  auto result = compile({"SHOW", "ENCODING"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowEncoding);
  const auto& ops = std::get<ShowEncodingOperands>(result.program[0].operands);
  EXPECT_FALSE(ops.stripeId.has_value());
}

TEST(CompileTest, showEncodingWithStripe) {
  auto result = compile({"SHOW", "ENCODING", "STRIPE", "1"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowEncoding);
  const auto& ops = std::get<ShowEncodingOperands>(result.program[0].operands);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 1);
}

TEST(CompileTest, showIndex) {
  auto result = compile({"SHOW", "INDEX"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowIndex);
}

TEST(CompileTest, showHistogram) {
  auto result = compile({"SHOW", "HISTOGRAM"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowHistogram);
  const auto& ops = std::get<ShowHistogramOperands>(result.program[0].operands);
  EXPECT_FALSE(ops.topLevel);
  EXPECT_FALSE(ops.stripeId.has_value());
}

TEST(CompileTest, showHistogramTop) {
  auto result = compile({"SHOW", "HISTOGRAM", "TOP"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowHistogram);
  const auto& ops = std::get<ShowHistogramOperands>(result.program[0].operands);
  EXPECT_TRUE(ops.topLevel);
}

TEST(CompileTest, showHistogramWithStripe) {
  auto result = compile({"SHOW", "HISTOGRAM", "STRIPE", "2"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<ShowHistogramOperands>(result.program[0].operands);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 2);
}

TEST(CompileTest, showHistogramTopWithStripe) {
  auto result = compile({"SHOW", "HISTOGRAM", "TOP", "STRIPE", "1"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<ShowHistogramOperands>(result.program[0].operands);
  EXPECT_TRUE(ops.topLevel);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 1);
}

TEST(CompileTest, showContent) {
  auto result = compile({"SHOW", "CONTENT", "5"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowContent);
  const auto& ops = std::get<ShowContentOperands>(result.program[0].operands);
  EXPECT_EQ(ops.streamId, 5);
  EXPECT_FALSE(ops.stripeId.has_value());
}

TEST(CompileTest, showContentWithStripe) {
  auto result = compile({"SHOW", "CONTENT", "3", "STRIPE", "1"});
  EXPECT_TRUE(result.error.empty());
  const auto& ops = std::get<ShowContentOperands>(result.program[0].operands);
  EXPECT_EQ(ops.streamId, 3);
  ASSERT_TRUE(ops.stripeId.has_value());
  EXPECT_EQ(*ops.stripeId, 1);
}

TEST(CompileTest, showContentMissingStreamId) {
  auto result = compile({"SHOW", "CONTENT"});
  EXPECT_FALSE(result.error.empty());
}

TEST(CompileTest, showFileLayout) {
  auto result = compile({"SHOW", "FILE", "LAYOUT"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowFileLayout);
}

TEST(CompileTest, showStripesMetadata) {
  auto result = compile({"SHOW", "STRIPES", "METADATA"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStripesMetadata);
}

TEST(CompileTest, showStripesStillWorks) {
  auto result = compile({"SHOW", "STRIPES"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStripes);
}

TEST(CompileTest, showStripeGroups) {
  auto result = compile({"SHOW", "STRIPE", "GROUPS"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowStripeGroups);
}

TEST(CompileTest, showOptionalSections) {
  auto result = compile({"SHOW", "OPTIONAL", "SECTIONS"});
  EXPECT_TRUE(result.error.empty());
  ASSERT_EQ(result.program.size(), 1);
  EXPECT_EQ(result.program[0].opcode, OpCode::ShowOptionalSections);
}

TEST(CompileTest, unknownCommand) {
  auto result = compile({"FOOBAR"});
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.program.empty());
  EXPECT_NE(result.error.find("FOOBAR"), std::string::npos);
}

TEST(CompileTest, unknownShowSubcommand) {
  auto result = compile({"SHOW", "BANANAS"});
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.program.empty());
  EXPECT_NE(result.error.find("BANANAS"), std::string::npos);
}

TEST(CompileTest, showAlone) {
  // "SHOW" with no subcommand is an unknown command.
  auto result = compile({"SHOW"});
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.program.empty());
}

// ===== Tokenize tests =====

TEST(TokenizeTest, simpleTokens) {
  auto tokens = tokenize("SELECT * FROM table1");
  const std::vector<std::string> expected{"SELECT", "*", "FROM", "table1"};
  EXPECT_EQ(tokens, expected);
}

TEST(TokenizeTest, stripsSemicolons) {
  auto tokens = tokenize("SELECT *;");
  const std::vector<std::string> expected{"SELECT", "*"};
  EXPECT_EQ(tokens, expected);
}

TEST(TokenizeTest, stripsCommas) {
  auto tokens = tokenize("SELECT col1, col2, col3");
  const std::vector<std::string> expected{"SELECT", "col1", "col2", "col3"};
  EXPECT_EQ(tokens, expected);
}

TEST(TokenizeTest, emptyInput) {
  auto tokens = tokenize("");
  EXPECT_TRUE(tokens.empty());
}

TEST(TokenizeTest, whitespaceOnly) {
  auto tokens = tokenize("   \t  ");
  EXPECT_TRUE(tokens.empty());
}

// ===== Trim tests =====

TEST(TrimTest, basic) {
  EXPECT_EQ(trim("  hello  "), "hello");
}

TEST(TrimTest, noTrimNeeded) {
  EXPECT_EQ(trim("hello"), "hello");
}

TEST(TrimTest, empty) {
  EXPECT_EQ(trim(""), "");
}

TEST(TrimTest, whitespaceOnly) {
  EXPECT_EQ(trim("   "), "");
}

// ===== ToUpper tests =====

TEST(ToUpperTest, basic) {
  EXPECT_EQ(toUpper("hello"), "HELLO");
}

TEST(ToUpperTest, alreadyUpper) {
  EXPECT_EQ(toUpper("HELLO"), "HELLO");
}

TEST(ToUpperTest, mixed) {
  EXPECT_EQ(toUpper("HeLLo"), "HELLO");
}

// ===== VM execute tests =====

class NimbleDslVmTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("nimble_dsl_vm_test");
    leafPool_ = rootPool_->addLeafChild("leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(NimbleDslVmTest, quitReturnsFalse) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::Quit, NoOperands{}});
  EXPECT_FALSE(vm.execute(program));
}

TEST_F(NimbleDslVmTest, helpPrintsOutput) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::Help, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Commands"), std::string::npos);
}

TEST_F(NimbleDslVmTest, describeDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"id", "name"},
      {maker.flatVector<int64_t>({1, 2}),
       maker.flatVector<velox::StringView>({"a", "b"})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::Describe, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("id"), std::string::npos);
  EXPECT_NE(out.str().find("name"), std::string::npos);
}

TEST_F(NimbleDslVmTest, selectDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"x", "y"},
      {maker.flatVector<int32_t>({10, 20}),
       maker.flatVector<int32_t>({30, 40})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  SelectOperands ops;
  ops.limit = 10;
  std::vector<Instruction> program;
  program.push_back({OpCode::Select, std::move(ops)});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("(2 rows)"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showSchemaDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowSchema, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("col"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showInfoDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowInfo, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Nimble File"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showStripesDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowStripes, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Stripe Id"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showStreamsDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  ShowStreamsOperands ops;
  std::vector<Instruction> program;
  program.push_back({OpCode::ShowStreams, std::move(ops)});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Stream Id"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showEncodingDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  ShowEncodingOperands ops;
  std::vector<Instruction> program;
  program.push_back({OpCode::ShowEncoding, std::move(ops)});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Encoding"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showIndexDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowIndex, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Not configured"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showHistogramDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  ShowHistogramOperands ops;
  std::vector<Instruction> program;
  program.push_back({OpCode::ShowHistogram, std::move(ops)});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("Encoding Type"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showContentDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  ShowContentOperands ops;
  ops.streamId = 1;
  std::vector<Instruction> program;
  program.push_back({OpCode::ShowContent, std::move(ops)});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_FALSE(out.str().empty());
}

TEST_F(NimbleDslVmTest, showFileLayoutDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowFileLayout, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_NE(out.str().find("File Footer"), std::string::npos);
}

TEST_F(NimbleDslVmTest, showStripesMetadataDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowStripesMetadata, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_FALSE(out.str().empty());
}

TEST_F(NimbleDslVmTest, showStripeGroupsDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowStripeGroups, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_FALSE(out.str().empty());
}

TEST_F(NimbleDslVmTest, showOptionalSectionsDispatches) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  std::vector<Instruction> program;
  program.push_back({OpCode::ShowOptionalSections, NoOperands{}});
  EXPECT_TRUE(vm.execute(program));
  EXPECT_FALSE(out.str().empty());
}

TEST_F(NimbleDslVmTest, endToEndShowHistogram) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  auto tokens = tokenize("SHOW HISTOGRAM;");
  auto compiled = compile(tokens);
  EXPECT_TRUE(compiled.error.empty());
  EXPECT_TRUE(vm.execute(compiled.program));
  EXPECT_NE(out.str().find("Encoding Type"), std::string::npos);
}

TEST_F(NimbleDslVmTest, endToEndShowFileLayout) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  auto tokens = tokenize("SHOW FILE LAYOUT;");
  auto compiled = compile(tokens);
  EXPECT_TRUE(compiled.error.empty());
  EXPECT_TRUE(vm.execute(compiled.program));
  EXPECT_NE(out.str().find("File Footer"), std::string::npos);
}

TEST_F(NimbleDslVmTest, emptyProgramReturnsTrue) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  EXPECT_TRUE(vm.execute({}));
}

TEST_F(NimbleDslVmTest, endToEndCompileAndExecute) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"name"}, {maker.flatVector<velox::StringView>({"Alice", "Bob"})});
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
  std::ostringstream out;
  NimbleDslLib lib{out, false, std::move(readFile)};
  NimbleDslVm vm{lib, out, false};

  auto tokens = tokenize("SELECT * LIMIT 10;");
  auto compiled = compile(tokens);
  EXPECT_TRUE(compiled.error.empty());
  EXPECT_TRUE(vm.execute(compiled.program));
  EXPECT_NE(out.str().find("Alice"), std::string::npos);
  EXPECT_NE(out.str().find("Bob"), std::string::npos);
}

} // namespace
} // namespace facebook::nimble::tools

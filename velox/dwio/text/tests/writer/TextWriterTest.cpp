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

#include "velox/dwio/text/writer/TextWriter.h"
#include <gtest/gtest.h>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::text {
std::vector<std::vector<std::string>> readFile(const std::string& name) {
  std::ifstream file(name);
  std::string line;
  std::vector<std::vector<std::string>> table;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> row;

    while (std::getline(ss, cell, TextWriter::SOH)) {
      row.push_back(cell);
    }

    table.push_back(row);
  }
  return table;
}

class TextWriterTest : public testing::Test,
                       public velox::test::VectorTestBase {
 public:
  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
    dwio::common::LocalFileSink::registerFactory();
    rootPool_ = memory::memoryManager()->addRootPool("TextWriterTests");
    leafPool_ = rootPool_->addLeafChild("TextWriterTests");
    tempPath_ = exec::test::TempDirectoryPath::create();
  }

 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::shared_ptr<exec::test::TempDirectoryPath> tempPath_;
};

TEST_F(TextWriterTest, write) {
  auto schema = ROW({"c0", "c1"}, {BIGINT(), BOOLEAN()});
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeConstant(true, 3),
      });

  WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  auto filePath =
      fs::path(fmt::format("{}/test_abort.txt", tempPath_->getPath()));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto writer = std::make_unique<TextWriter>(
      schema,
      std::move(sink),
      std::make_shared<text::WriterOptions>(writerOptions));
  writer->write(data);
  writer->close();

  std::vector<std::vector<std::string>> result = readFile(filePath);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0].size(), 2);
  EXPECT_EQ(result[0][0], "1");
  EXPECT_EQ(result[0][1], "1");
  EXPECT_EQ(result[1][0], "2");
  EXPECT_EQ(result[1][1], "1");
  EXPECT_EQ(result[2][0], "3");
  EXPECT_EQ(result[2][1], "1");
}
} // namespace facebook::velox::text

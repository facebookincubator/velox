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

#include <gtest/gtest.h>

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/json/RegisterJsonReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::json {
namespace {

class JsonReaderTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    registerJsonReaderFactory();
  }

  void TearDown() override {
    unregisterJsonReaderFactory();
  }
};

TEST_F(JsonReaderTest, factoryRegistration) {
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);
  ASSERT_NE(factory, nullptr);
  EXPECT_EQ(factory->fileFormat(), dwio::common::FileFormat::JSON);
}

TEST_F(JsonReaderTest, emptyFileReturnsZeroRows) {
  auto type = ROW({{"a", BIGINT()}});
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

  dwio::common::ReaderOptions readerOptions{pool()};
  readerOptions.setFileSchema(type);

  auto readFile = std::make_shared<InMemoryReadFile>(std::string{});
  auto input =
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
  auto reader = factory->createReader(std::move(input), readerOptions);
  auto rowReader = reader->createRowReader(dwio::common::RowReaderOptions{});

  VectorPtr result;
  EXPECT_EQ(rowReader->next(10, result), 0);
}

} // namespace
} // namespace facebook::velox::json

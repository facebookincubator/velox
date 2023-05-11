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

#include "velox/dwio/common/InputStream.h"

#include <string_view>

#include "gtest/gtest.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

TEST(ReadFileInputStream, SimpleUsage) {
  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append("aaaaa");
    writeFile.append("bbbbb");
    writeFile.append("ccccc");
  }
  auto readFile = std::make_shared<InMemoryReadFile>(fileData);
  ReadFileInputStream inputStream(readFile);
  ASSERT_EQ(inputStream.getLength(), 15);
  auto buf = std::make_unique<char[]>(15);

  inputStream.read(buf.get(), 7, 4, LogType::STREAM);
  std::string_view read_value(buf.get(), 7);
  ASSERT_EQ(read_value, "abbbbbc");

  inputStream.read(buf.get(), 15, 0, LogType::STREAM);
  read_value = {buf.get(), 15};
  ASSERT_EQ(read_value, "aaaaabbbbbccccc");
}

TEST(ReadFileInputStream, vread) {
  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append("aaaaa");
    writeFile.append("bbbbb");
    writeFile.append("ccccc");
  }
  auto readFile = std::make_shared<InMemoryReadFile>(fileData);
  ReadFileInputStream inputStream(readFile);
  ASSERT_EQ(inputStream.getLength(), 15);

  std::vector<std::string> buffers = {"1234567", "890"};

  inputStream.vread(
      {reinterpret_cast<void*>(buffers[0].data()),
       reinterpret_cast<void*>(buffers[1].data())},
      {{2, buffers[0].size()}, {10, buffers[1].size()}},
      LogType::STREAM);

  std::vector<std::string> result = {"aaabbbb", "ccc"};
  ASSERT_EQ(buffers, result);
}

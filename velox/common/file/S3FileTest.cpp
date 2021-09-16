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

#include "velox/common/file/S3File.h"
#include "velox/common/file/File.h"
#include "velox/core/Context.h"

#include "gtest/gtest.h"

using namespace facebook::velox;

constexpr int kOneMB = 1 << 20;

void writeData(WriteFile* writeFile) {
  writeFile->append("aaaaa");
  writeFile->append("bbbbb");
  writeFile->append(std::string(kOneMB, 'c'));
  writeFile->append("ddddd");
  ASSERT_EQ(writeFile->size(), 15 + kOneMB);
}

void readData(ReadFile* readFile) {
  Arena arena;
  // ASSERT_EQ(readFile->size(), 15 + kOneMB);
  ASSERT_EQ(readFile->pread(10 + kOneMB, 5, &arena), "ddddd");
  ASSERT_EQ(readFile->pread(0, 10, &arena), "aaaaabbbbb");
  ASSERT_EQ(readFile->pread(10, kOneMB, &arena), std::string(kOneMB, 'c'));
  // ASSERT_EQ(readFile->size(), 15 + kOneMB);
  const std::string_view arf = readFile->pread(5, 10, &arena);
  const std::string zarf = readFile->pread(kOneMB, 15);
  auto buf = std::make_unique<char[]>(8);
  const std::string_view warf = readFile->pread(4, 8, buf.get());
  const std::string_view warfFromBuf(buf.get(), 8);
  ASSERT_EQ(arf, "bbbbbccccc");
  ASSERT_EQ(zarf, "ccccccccccddddd");
  ASSERT_EQ(warf, "abbbbbcc");
  ASSERT_EQ(warfFromBuf, "abbbbbcc");
}

TEST(S3File, WriteAndRead) {
  const char* filename = "/Users/deepak/workspace/minio/tmp/test.txt";
  const char* s3File = "tmp/test.txt";
  remove(filename);
  {
    LocalWriteFile writeFile(filename);
    writeData(&writeFile);
  }
  std::unordered_map<std::string, std::string> hiveConnectorConfigs = {
      {"hive.s3.aws-access-key", "admin"},
      {"hive.s3.aws-secret-key", "password"},
      {"hive.s3.endpoint", "127.0.0.1:9000"}};
  std::shared_ptr<const Config> config =
      std::make_shared<const core::MemConfig>(std::move(hiveConnectorConfigs));
  InitializeS3();
  S3FileSystem s3fs(config);
  s3fs.init();
  auto readFile = s3fs.openReadFile(s3File);
  readData(readFile.get());
}
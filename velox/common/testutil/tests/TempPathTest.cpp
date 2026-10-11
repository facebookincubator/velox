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
#include <filesystem>
#include <fstream>
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TempFilePath.h"

namespace facebook::velox::common::testutil {

TEST(TempPathTest, directoryLifetime) {
  for (const bool faultInjection : {false, true}) {
    std::string path;
    {
      const auto directory = TempDirectoryPath::create(faultInjection);
      path = directory->getDelegatePath();
      EXPECT_TRUE(std::filesystem::is_directory(path));
      EXPECT_TRUE(
          std::filesystem::equivalent(
              std::filesystem::path(path).parent_path(),
              std::filesystem::temp_directory_path()));
      EXPECT_EQ(directory->getPath(), faultInjection ? "faulty:" + path : path);
      const auto other = TempDirectoryPath::create(faultInjection);
      EXPECT_NE(other->getDelegatePath(), path);
      std::ofstream(std::filesystem::path(path) / "contents") << "value";
    }
    EXPECT_FALSE(std::filesystem::exists(path));
  }
}

TEST(TempPathTest, fileLifetime) {
  for (const bool faultInjection : {false, true}) {
    std::string path;
    {
      const auto file = TempFilePath::create(faultInjection);
      path = file->tempFilePath();
      EXPECT_TRUE(std::filesystem::is_regular_file(path));
      EXPECT_TRUE(
          std::filesystem::equivalent(
              std::filesystem::path(path).parent_path(),
              std::filesystem::temp_directory_path()));
      EXPECT_EQ(file->getPath(), faultInjection ? "faulty:" + path : path);
      const auto other = TempFilePath::create(faultInjection);
      EXPECT_NE(other->tempFilePath(), path);
      file->append("value");
      EXPECT_EQ(file->fileSize(), 5);
      EXPECT_GT(file->fileModifiedTime(), 0);
    }
    EXPECT_FALSE(std::filesystem::exists(path));
  }
}

} // namespace facebook::velox::common::testutil

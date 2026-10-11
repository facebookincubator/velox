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

#include "velox/common/base/Fs.h"
#include <gtest/gtest.h>
#include "boost/filesystem.hpp"
#include "velox/common/testutil/TempDirectoryPath.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::common {

class FsTest : public testing::Test {};

TEST_F(FsTest, createDirectory) {
  auto dir = TempDirectoryPath::create();
  auto rootPath = dir->getPath();
  auto tmpDirectoryPath = rootPath + "/first/second/third";
  // First time should generate directory successfully.
  EXPECT_FALSE(fs::exists(tmpDirectoryPath.c_str()));
  EXPECT_TRUE(generateFileDirectory(tmpDirectoryPath.c_str()));
  EXPECT_TRUE(fs::exists(tmpDirectoryPath.c_str()));

  // Directory already exist, not creating but should return success.
  EXPECT_TRUE(generateFileDirectory(tmpDirectoryPath.c_str()));
  EXPECT_TRUE(fs::exists(tmpDirectoryPath.c_str()));
  dir.reset();
  EXPECT_FALSE(fs::exists(rootPath.c_str()));
}

TEST_F(FsTest, temporaryPaths) {
  const auto dir = TempDirectoryPath::create();
  const auto path = dir->getPath();
  const auto file = generateTempFilePath(path.c_str(), "file");
  ASSERT_TRUE(file.has_value());
  EXPECT_TRUE(fs::is_regular_file(*file));
  EXPECT_TRUE(fs::remove(*file));

  const auto folder = generateTempFolderPath(path.c_str(), "folder");
  ASSERT_TRUE(folder.has_value());
  EXPECT_TRUE(fs::is_directory(*folder));
}

} // namespace facebook::velox::common

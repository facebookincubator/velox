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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"

#include "gtest/gtest.h"

using namespace facebook::velox::filesystems;

TEST(AbfsPathTest, encodedPath) {
    auto abfssAccountWithSpecialCharacters = AbfsPath(
      "abfss://testc@test.dfs.core.windows.net/main@dir/brand#51/sub dir/test.txt");
  EXPECT_EQ(
      abfssAccountWithSpecialCharacters.accountNameWithSuffix(),
      "test.dfs.core.windows.net");
  EXPECT_EQ(abfssAccountWithSpecialCharacters.accountName(), "test");
  EXPECT_EQ(abfssAccountWithSpecialCharacters.fileSystem(), "testc");
  EXPECT_EQ(
      abfssAccountWithSpecialCharacters.filePath(),
      "main@dir/brand#51/sub dir/test.txt");
  EXPECT_EQ(
      abfssAccountWithSpecialCharacters.getUrl(true),
      "https://test.blob.core.windows.net/testc/main%40dir/brand%2351/sub%20dir/test.txt");
}
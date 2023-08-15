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

#include "velox/common/caching/SsdCache.h"
#include "velox/common/base/Fs.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/FileInfoMap.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/TempFilePath.h"

#include "gtest/gtest.h"

#include <fcntl.h>
#ifdef linux
#include <linux/fs.h>
#endif // linux

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cache {

class SsdCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
  }
};

TEST_F(SsdCacheTest, fileInfoMapCheckpoint) {
  auto tempPath = TempFilePath::create();
  SsdCache ssd(tempPath->path, 1 << 20, 2, nullptr, 128 << 10, true);
  FileInfoMap::create();

  for (auto i = 0; i < 1000; i++) {
    auto fileNum = fileIds().makeId(fmt::format("file{}", i));
    FileInfoMap::getInstance()->addOpenFileInfo(fileNum);
  }
  auto fileMapBefore = FileInfoMap::getInstance()->getMap();

  // Test writing the file info map checkpoint and successfully reading back the
  // file info map.
  {
    ssd.makeFileInfoMapCheckpoint();
    FileInfoMap::getInstance()->clear();
    ssd.readFileInfoMapCheckpoint();

    auto fileMapAfter = FileInfoMap::getInstance()->getMap();
    ASSERT_EQ(fileMapBefore.size(), fileMapAfter.size());
    ASSERT_TRUE(std::all_of(
        fileMapAfter.begin(),
        fileMapAfter.end(),
        [&fileMapBefore](auto& entry) {
          return fileMapBefore.at(entry.first) == entry.second;
        }));
  }
  {
    // Test clearing the file info map and deleting the checkpoint file if the
    // read is broken.
    auto checkpointPath =
        tempPath->path + SsdCache::kFileInfoMapCheckpointExtension;
    auto fd = open(checkpointPath.c_str(), O_WRONLY);
    ftruncate(fd, 100);
    FileInfoMap::getInstance()->clear();
    ssd.readFileInfoMapCheckpoint();
    close(fd);

    auto fileMapAfter = FileInfoMap::getInstance()->getMap();
    ASSERT_EQ(fileMapAfter.size(), 0);
    ASSERT_FALSE(fs::exists(checkpointPath));
  }
}

} // namespace facebook::velox::cache

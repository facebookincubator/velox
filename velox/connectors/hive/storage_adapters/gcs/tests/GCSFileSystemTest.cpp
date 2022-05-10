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

#include "connectors/hive/storage_adapters/gcs/GCSFileSystem.h"
#include "connectors/hive/storage_adapters/gcs/GCSUtil.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/exec/tests/utils/TempFilePath.h"

#include "gtest/gtest.h"

using namespace facebook::velox;

constexpr int kOneMB = 1 << 20;

class GCSFileSystemTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    filesystems::registerGCSFileSystem();
  }

  static void TearDownTestSuite() {
    // TODO
  }
};

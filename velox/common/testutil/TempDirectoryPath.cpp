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

#include "velox/common/testutil/TempDirectoryPath.h"

#include "boost/filesystem.hpp"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#endif

namespace facebook::velox::common::testutil {

std::shared_ptr<TempDirectoryPath> TempDirectoryPath::create(bool injectFault) {
  auto* tempDirPath = new TempDirectoryPath(injectFault);
  return std::shared_ptr<TempDirectoryPath>(tempDirPath);
}

TempDirectoryPath::~TempDirectoryPath() {
  LOG(INFO) << "TempDirectoryPath:: removing all files from " << tempPath_;
  try {
    boost::filesystem::remove_all(tempPath_.c_str());
  } catch (...) {
    LOG(WARNING)
        << "TempDirectoryPath:: destructor failed while calling boost::filesystem::remove_all";
  }
}

std::string TempDirectoryPath::createTempDirectory() {
#ifdef _WIN32
  char tempDir[MAX_PATH];
  if (::GetTempPathA(MAX_PATH, tempDir) == 0) {
    VELOX_FAIL("Cannot determine temp directory");
  }
  char path[MAX_PATH];
  if (::GetTempFileNameA(tempDir, "vlx", 0, path) == 0) {
    VELOX_FAIL("Cannot create temp directory name");
  }
  // GetTempFileNameA created a file with this name; replace it with a directory.
  ::_unlink(path);
  if (::_mkdir(path) != 0) {
    VELOX_FAIL("Cannot create temp directory");
  }
  return path;
#else
  char tempPath[] = "/tmp/velox_test_XXXXXX";
  const char* tempDirectoryPath = ::mkdtemp(tempPath);
  VELOX_CHECK_NOT_NULL(tempDirectoryPath, "Cannot open temp directory");
  return tempDirectoryPath;
#endif
}

} // namespace facebook::velox::common::testutil

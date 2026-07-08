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

#include "velox/common/testutil/TempFilePath.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace facebook::velox::common::testutil {

TempFilePath::~TempFilePath() {
#ifdef _WIN32
  // On Windows an open file generally cannot be unlinked, so close first.
  if (fd_ != -1) {
    ::_close(fd_);
  }
  ::_unlink(tempPath_.c_str());
#else
  ::unlink(tempPath_.c_str());
  if (fd_ != -1) {
    ::close(fd_);
  }
#endif
}

std::shared_ptr<TempFilePath> TempFilePath::create(bool enableFaultInjection) {
  auto* tempFilePath = new TempFilePath(enableFaultInjection);
  return std::shared_ptr<TempFilePath>(tempFilePath);
}

std::string TempFilePath::createTempFile(TempFilePath* tempFilePath) {
#ifdef _WIN32
  char tempDir[MAX_PATH];
  if (::GetTempPathA(MAX_PATH, tempDir) == 0) {
    VELOX_FAIL("Cannot determine temp directory");
  }
  char path[MAX_PATH];
  if (::GetTempFileNameA(tempDir, "vlx", 0, path) == 0) {
    VELOX_FAIL("Cannot create temp file");
  }
  // GetTempFileNameA created the file; open it for read/write in binary mode.
  const errno_t err = ::_sopen_s(
      &tempFilePath->fd_,
      path,
      _O_RDWR | _O_BINARY,
      _SH_DENYNO,
      _S_IREAD | _S_IWRITE);
  if (err != 0 || tempFilePath->fd_ == -1) {
    VELOX_FAIL("Cannot open temp file: {}", folly::errnoStr(errno));
  }
  return path;
#else
  char path[] = "/tmp/velox_test_XXXXXX";
  tempFilePath->fd_ = ::mkstemp(path);
  if (tempFilePath->fd_ == -1) {
    VELOX_FAIL("Cannot open temp file: {}", folly::errnoStr(errno));
  }
  return path;
#endif
}

std::vector<std::string> toFilePaths(
    const std::vector<std::shared_ptr<TempFilePath>>& tempFiles) {
  std::vector<std::string> filePaths;
  filePaths.reserve(tempFiles.size());
  for (const auto& tempFile : tempFiles) {
    filePaths.push_back(tempFile->getPath());
  }
  return filePaths;
}
} // namespace facebook::velox::common::testutil

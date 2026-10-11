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

#include <boost/filesystem.hpp>
#include <folly/portability/Stdlib.h>
#include <folly/portability/Unistd.h>

namespace facebook::velox::common::testutil {

TempFilePath::~TempFilePath() {
  if (fd_ != -1) {
    ::close(fd_);
  }
  ::unlink(tempPath_.c_str());
}

std::shared_ptr<TempFilePath> TempFilePath::create(bool enableFaultInjection) {
  auto* tempFilePath = new TempFilePath(enableFaultInjection);
  return std::shared_ptr<TempFilePath>(tempFilePath);
}

std::string TempFilePath::createTempFile(TempFilePath* tempFilePath) {
  auto path =
      (boost::filesystem::temp_directory_path() / "velox_test_XXXXXX").string();
  tempFilePath->fd_ = ::mkstemp(path.data());
  if (tempFilePath->fd_ == -1) {
    VELOX_FAIL("Cannot open temp file: {}", folly::errnoStr(errno));
  }
  return path;
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

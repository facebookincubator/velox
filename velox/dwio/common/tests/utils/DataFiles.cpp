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

#include "velox/dwio/common/tests/utils/DataFiles.h"
#include <string_view>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Fs.h"

namespace facebook::velox::test {

namespace {

constexpr std::string_view kDataFilesSourceSuffix =
    "velox/dwio/common/tests/utils/DataFiles.cpp";

// Computes the project root from this source file location so test data lookup
// does not depend on the process working directory.
const fs::path& projectRoot() {
  static const fs::path root = [] {
    const std::string sourcePath =
        fs::path{__FILE__}.lexically_normal().string();
    VELOX_CHECK(
        sourcePath.ends_with(kDataFilesSourceSuffix),
        "DataFiles.cpp path does not end with expected suffix: {}",
        sourcePath);
    return fs::path{sourcePath.substr(
        0, sourcePath.size() - kDataFilesSourceSuffix.size())};
  }();
  return root;
}

// Preserves the legacy cwd-based lookup for builds where __FILE__ is rewritten
// to be source-relative, e.g. by -ffile-prefix-map.
std::string getDataFilePathFromCurrentDirectory(
    const std::string& baseDir,
    const fs::path& requestedPath) {
  const std::string currentPath = fs::current_path().string();
  if (currentPath.ends_with("fbcode")) {
    return (fs::path{currentPath} / baseDir / requestedPath)
        .lexically_normal()
        .string();
  }
  return (fs::path{currentPath} / requestedPath).lexically_normal().string();
}

} // namespace

std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  const fs::path requestedPath{filePath};
  if (requestedPath.is_absolute()) {
    return requestedPath.string();
  }

  const auto& root = projectRoot();
  if (root.is_absolute()) {
    return (root / baseDir / requestedPath).lexically_normal().string();
  }

  return getDataFilePathFromCurrentDirectory(baseDir, requestedPath);
}

} // namespace facebook::velox::test

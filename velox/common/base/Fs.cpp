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
#include <fmt/format.h>
#include <glog/logging.h>
#include <io.h>
#include <direct.h>
namespace facebook::velox::common {

bool generateFileDirectory(const fs::path::value_type* dirPath) {
  std::error_code errorCode;
  const auto success = fs::create_directories(dirPath, errorCode);
  fs::permissions(dirPath, fs::perms::all, fs::perm_options::replace);
  if (!success && errorCode.value() != 0) {
    LOG(ERROR) << "Failed to create file directory '" << dirPath
               << "'. Error: " << errorCode.message() << " errno "
               << errorCode.value();
    return false;
  }
  return true;
}
bool generateFileDirectory(const char* dirPath) {
  std::error_code errorCode;
  const auto success = fs::create_directories(dirPath, errorCode);
  fs::permissions(dirPath, fs::perms::all, fs::perm_options::replace);
  if (!success && errorCode.value() != 0) {
    LOG(ERROR) << "Failed to create file directory '" << dirPath
               << "'. Error: " << errorCode.message() << " errno "
               << errorCode.value();
    return false;
  }
  return true;
}

std::optional<std::string> generateTempFilePath(
    const char* basePath,
    const char* prefix) {
  auto path = fmt::format("{}/velox_{}_XXXXXX", basePath, prefix);

  // Use _mktemp_s to create a unique temporary path
  if (_mktemp_s(&path[0], path.size() + 1) != 0) {
    return std::nullopt;
  }

  // Open the file for writing
  FILE* file;
  if (fopen_s(&file, path.c_str(), "w") != 0) {
    return std::nullopt;
  }

  // Close the file immediately to ensure it's created and empty

  fclose(file);
  return path;
}

std::optional<std::string> generateTempFolderPath(
    const char* basePath,
    const char* prefix) {
  auto path = fmt::format("{}/velox_{}_XXXXXX", basePath, prefix);

    // Use _mktemp_s to create a unique temporary path
  if (_mktemp_s(&path[0], path.size() + 1) != 0) {
    return std::nullopt;
  }

  // Create the temporary directory
  if (_mkdir(path.c_str()) != 0) {
    return std::nullopt;
  }


  return path;
}

} // namespace facebook::velox::common

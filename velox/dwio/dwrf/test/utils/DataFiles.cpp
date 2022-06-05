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
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include <iostream>
#include "velox/common/base/tests/Fs.h"

namespace facebook::velox::test {

namespace {
bool endsWith(const std::string& s, const std::string& suffix) {
  return s.length() >= suffix.length() &&
      s.compare(s.length() - suffix.length(), suffix.length(), suffix) == 0;
}
} // namespace

std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  std::string current_path = fs::current_path().c_str();

  std::cout << "baseDir " << baseDir << " filePath " << filePath << std::endl;

  std::cout << "current_path " << current_path << std::endl;
  for (const auto& entry : fs::directory_iterator(current_path))
    std::cout << entry.path() << std::endl;

  auto parentPath = current_path + "/../";
  std::cout << "parentPath " << parentPath << std::endl;
  for (const auto& entry : fs::directory_iterator(parentPath)) {
    std::cout << entry.path() << std::endl;
    if (std::strstr(entry.path().c_str(), "examples") !=
        NULL) { // Strstr says does b contain a
      auto examplesPath = parentPath + "examples";
      for (const auto& entry : fs::directory_iterator(examplesPath))
        std::cout << entry.path() << std::endl;
    }
  }

  if (endsWith(current_path, "fbcode")) {
    return current_path + "/" + baseDir + "/" + filePath;
  }
  return current_path + "/" + filePath;
}
} // namespace facebook::velox::test

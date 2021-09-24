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
#include "velox/common/file/FileSystems.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::filesystems {

using RegisteredFileSystems = std::vector<std::pair<
    std::function<bool(std::string_view)>,
    std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>>>;

RegisteredFileSystems& registeredFileSystems() {
  // Meyers singleton.
  static RegisteredFileSystems* fss = new RegisteredFileSystems();
  return *fss;
}

void registerFileSystemClass(
    std::function<bool(std::string_view)> schemeMatcher,
    std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
        fileSystemGenerator) {
  registeredFileSystems().emplace_back(schemeMatcher, fileSystemGenerator);
}

std::shared_ptr<FileSystem> getFileSystem(
    std::string_view filename,
    std::shared_ptr<const Config> properties) {
  const auto& filesystems = registeredFileSystems();
  for (const auto& p : filesystems) {
    if (p.first(filename)) {
      return p.second(properties);
    }
  }
  throw std::runtime_error(fmt::format(
      "No registered file system matched with filename '{}'", filename));
}

void registerFileSystems() {
  VELOX_REGISTER_FILE_SYSTEM(Linux);
}

} // namespace facebook::velox::filesystems

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
#pragma once

#include <functional>
#include <string_view>

#define _VELOX_REGISTER_FILE_SYSTEM(tag) registerFileSystem_##tag

// Registers a filesystem associated with a given tag.
#define VELOX_REGISTER_FILE_SYSTEM(tag)             \
  {                                                 \
    extern void _VELOX_REGISTER_FILE_SYSTEM(tag)(); \
    _VELOX_REGISTER_FILE_SYSTEM(tag)();             \
  }

namespace facebook::velox {
class Config;
class ReadFile;
class WriteFile;
} // namespace facebook::velox

namespace facebook::velox::filesystems {

// An abstract FileSystem
class FileSystem : public std::enable_shared_from_this<FileSystem> {
 public:
  FileSystem(std::shared_ptr<const Config> config) : config_(config) {}
  virtual ~FileSystem() {}
  virtual std::string name() const = 0;
  virtual std::unique_ptr<ReadFile> openReadFile(std::string_view path) = 0;
  virtual std::unique_ptr<WriteFile> openWriteFile(std::string_view path) = 0;

 protected:
  std::shared_ptr<const Config> config_;
};

std::shared_ptr<FileSystem> getFileSystem(
    std::string_view filename,
    std::shared_ptr<const Config> config);

// FileSystems must be registered explicitly
// The registration function take two parameters: a
// std::function<bool(std::string_view)> that says whether the registered
// FileSystem subclass should be used for that filename, and a lambda that
// generates the actual file system. Each registered file system is tried in the
// order it was registered, so keep this in mind if multiple file systems could
// match the same filename.

void registerFileSystemClass(
    std::function<bool(std::string_view)> schemeMatcher,
    std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
        fileSystemGenerator);

// Register all filesystems.
void registerFileSystems();

} // namespace facebook::velox::filesystems

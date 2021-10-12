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
#include <memory>
#include <string_view>

namespace facebook::velox {
class Config;
class ReadFile;
class WriteFile;
} // namespace facebook::velox

namespace facebook::velox::filesystems {

// An abstract FileSystem
class FileSystem {
 public:
  FileSystem(std::shared_ptr<const Config> config)
      : config_(std::move(config)) {}
  virtual ~FileSystem() {}
  // Returns the name of the File System
  virtual std::string name() const = 0;
  // Returns a ReadFile handle for a given file path
  virtual std::unique_ptr<ReadFile> openFileForRead(std::string_view path) = 0;
  // Returns a WriteFile handle for a given file path
  virtual std::unique_ptr<WriteFile> openFileForWrite(
      std::string_view path) = 0;

 protected:
  std::shared_ptr<const Config> config_;
};

std::shared_ptr<FileSystem> getFileSystem(
    std::string_view filename,
    std::shared_ptr<const Config> config);

// FileSystems must be registered explicitly.
// The registration function takes two parameters:
// a std::function<bool(std::string_view)> that says whether the registered
// FileSystem subclass should be used for that filename,
// and a lambda that generates the actual file system.
void registerFileSystem(
    std::function<bool(std::string_view)> schemeMatcher,
    std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
        fileSystemGenerator);

// Register the local filesystem.
void registerLocalFileSystem();

} // namespace facebook::velox::filesystems

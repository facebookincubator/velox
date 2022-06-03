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
#include "HdfsReadFile.h"
#include "libhdfs3/src/client/hdfs.h"
#include "service_discover/consul/discovery.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::filesystems {
class HdfsFileSystem : public FileSystem {
 private:
  hdfsFS hdfsClient_;
  static std::string_view scheme;
  static std::vector<cpputil::consul::ServiceEndpoint> getServiceEndpoint(
      const std::shared_ptr<const Config>& config);

 public:
  explicit HdfsFileSystem(const std::shared_ptr<const Config>& config);

  ~HdfsFileSystem() override;

  std::string name() const override;

  std::unique_ptr<ReadFile> openFileForRead(std::string_view path) override;

  std::unique_ptr<WriteFile> openFileForWrite(std::string_view path) override;

  static std::function<bool(std::string_view)> schemeMatcher();

  void remove(std::string_view path) override {
    VELOX_UNSUPPORTED("remove for HDFS not implemented");
  }

  static std::function<
      std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
  fileSystemGenerator();
};

// Register the HDFS.
void registerHdfsFileSystem();
} // namespace facebook::velox::filesystems

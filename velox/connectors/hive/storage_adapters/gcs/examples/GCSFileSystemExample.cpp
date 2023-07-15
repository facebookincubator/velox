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
#include "connectors/hive/storage_adapters/gcs/GCSFileSystem.h"
#include "velox/common/file/File.h"
#include "velox/core/Config.h"

#include <folly/init/Init.h>

#include <gflags/gflags.h>

#include <iostream>

DEFINE_string(gcs_path, "", "Path of GCS bucket");

auto newConfiguration() {
  using namespace facebook::velox;
  std::unordered_map<std::string, std::string> configOverride = {};
  return std::make_shared<const core::MemConfig>(std::move(configOverride));
}

int main(int argc, char** argv) {
  using namespace facebook::velox;
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_gcs_path.empty()) {
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }
  filesystems::registerGCSFileSystem();
  filesystems::GCSFileSystem gcfs(newConfiguration());
  gcfs.initializeClient();
  std::cout << "Opening file " << FLAGS_gcs_path << std::endl;
  std::unique_ptr<ReadFile> file_read = gcfs.openFileForRead(FLAGS_gcs_path);
  std::size_t file_size = file_read->size();
  std::cout << "File size = " << file_size << std::endl;
  std::string buffer(file_size + 1, '\0');
  file_read->pread(0 /*offset*/, file_size /*lenght*/, buffer.data());
  std::cout << "File Content = " << buffer << std::endl;
}

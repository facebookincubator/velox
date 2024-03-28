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
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/storage_adapters/gcs/GCSFileSystem.h"
#include "velox/core/Config.h"

#include <folly/init/Init.h>

#include <gflags/gflags.h>

#include <iostream>

DEFINE_string(gcs_path, "", "Path of GCS bucket");
DEFINE_string(gcs_max_retry_count, "", "Max retry count");
DEFINE_string(gcs_max_retry_time, "", "Max retry time (seconds)");

auto newConfiguration() {
  using namespace facebook::velox;
  std::unordered_map<std::string, std::string> configOverride = {};
  if (!FLAGS_gcs_max_retry_count.empty()) {
    configOverride.emplace(
        "hive.gcs.max-retry-count", FLAGS_gcs_max_retry_count);
  }
  if (!FLAGS_gcs_max_retry_time.empty()) {
    configOverride.emplace("hive.gcs.max-retry-time", FLAGS_gcs_max_retry_time);
  }
  return std::make_shared<const core::MemConfig>(std::move(configOverride));
}

void write_buffer(std::string_view buffer) {
  std::cout << buffer;
}

int main(int argc, char** argv) {
  char buffer[2UL << 20];
  constexpr std::size_t buffer_size = sizeof(buffer);

  using namespace facebook::velox;
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_gcs_path.empty()) {
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  filesystems::GCSFileSystem gcfs(newConfiguration());
  gcfs.initializeClient();

  std::cerr << "Opening file for write " << FLAGS_gcs_path << std::endl;
  auto file_write = gcfs.openFileForWrite(FLAGS_gcs_path);

  while (true) {
    std::cin.read(buffer, buffer_size);
    std::size_t count = std::cin.gcount();
    file_write->append(std::string_view(buffer, count));
    if (std::cin.eof()) {
      break;
    } else if (std::cin.fail() || std::cin.bad()) {
      std::cerr << "There was an error reading from stdin" << std::endl;
      break;
    }
  }

  file_write->flush();
  file_write->close();
}
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

#include "velox/connectors/hive/storage_adapters/s3fs/benchmark/S3ReadBenchmark.h"
#include "velox/core/Context.h"

#include <fstream>

DEFINE_string(s3_config, "", "Path of S3 config file");

namespace facebook::velox {

// From presto-cpp
std::shared_ptr<Config> readConfig(const std::string& filePath) {
  std::ifstream configFile(filePath);
  if (!configFile.is_open()) {
    throw std::runtime_error(
        fmt::format("Couldn't open config file {} for reading.", filePath));
  }

  std::unordered_map<std::string, std::string> properties;
  std::string line;
  while (getline(configFile, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line[0] == '#' || line.empty()) {
      continue;
    }
    auto delimiterPos = line.find('=');
    auto name = line.substr(0, delimiterPos);
    auto value = line.substr(delimiterPos + 1);
    properties.emplace(name, value);
  }

  return std::make_shared<facebook::velox::core::MemConfig>(properties);
}

void S3ReadBenchmark::run() {
  if (FLAGS_bytes) {
    modes(FLAGS_bytes, FLAGS_gap, FLAGS_num_in_run);
    return;
  }
  modes(1100, 0, 10);
  modes(1100, 1200, 10);
  modes(16 * 1024, 0, 10);
  modes(16 * 1024, 10000, 10);
  modes(1000000, 0, 8);
  modes(1000000, 100000, 8);
}
} // namespace facebook::velox

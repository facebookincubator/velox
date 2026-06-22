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

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

  void shutdown() override;
};

namespace facebook::velox::cudf_velox {

/// Parse a properties file into a key-value map.
/// Each line should be key=value. Lines starting with '#' and blank lines are
/// skipped. Lines without '=' are logged as warnings and ignored.
inline std::unordered_map<std::string, std::string> loadPropertiesFile(
    const std::string& path) {
  auto fsPath = std::filesystem::path(path);
  VELOX_USER_CHECK(
      std::filesystem::exists(fsPath), "Properties file not found: {}", path);
  std::unordered_map<std::string, std::string> properties;
  std::string line;
  std::ifstream configFile(fsPath);
  while (std::getline(configFile, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line.empty() || line[0] == '#') {
      continue;
    }
    LOG(INFO) << "Setting property " << line;
    const auto delimiterPos = line.find('=');
    if (delimiterPos == std::string::npos) {
      LOG(WARNING) << "Skipping malformed config line (no '='): " << line;
      continue;
    }
    properties.emplace(
        line.substr(0, delimiterPos), line.substr(delimiterPos + 1));
  }
  return properties;
}

} // namespace facebook::velox::cudf_velox

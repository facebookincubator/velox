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

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace facebook {
namespace dwio {
namespace catalog {
namespace fbhive {

class FileUtils {
 public:
  static bool shouldEscape(char c);

  static std::string toLower(const std::string& data) {
    std::string ret;
    ret.reserve(data.size());
    std::transform(
        data.begin(), data.end(), std::back_inserter(ret), [](auto& c) {
          return std::tolower(c);
        });
    return ret;
  }

  static std::string escapePathName(const std::string& data);

  static std::string unescapePathName(const std::string& data);

  static std::string makePartName(
      const std::vector<std::pair<std::string, std::string>>& entries);

  static std::vector<std::pair<std::string, std::string>> parsePartKeyValues(
      const std::string& partName);

  static std::vector<std::pair<std::string, std::string>>
  extractPartitionKeyValues(
      const std::string& filePathSubstr,
      std::function<void(
          const std::string& partitionPart,
          std::vector<std::pair<std::string, std::string>>& parsedParts)>
          parserFunc);

  // Strong assumption that all expressions in the form of a=b means a partition
  // key value pair in '/' separated tokens. We could have stricter validation
  // on what a file path should look like, but it doesn't belong to this layer.
  static std::vector<std::pair<std::string, std::string>>
  extractPartitionKeyValues(const std::string& filePath);

  // Not the most efficient impl, but utilizes the above methods that are used
  // by koski dwio connector.
  static std::string extractPartitionName(const std::string& filePath) {
    const auto& partitionParts = extractPartitionKeyValues(filePath);
    return partitionParts.empty() ? "" : makePartName(partitionParts);
  }

 private:
  static size_t countEscape(const std::string& val);
};

} // namespace fbhive
} // namespace catalog
} // namespace dwio
} // namespace facebook

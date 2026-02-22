/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
namespace velox {
namespace dwio {
namespace catalog {
namespace fbhive {

class FileUtils {
 public:
  /// Function type for encoding partition key/value strings.
  /// Takes a string to encode and returns the encoded string.
  using EncodeFunction = std::function<std::string(const std::string&)>;

  /// Converts the path name to be hive metastore compliant, will do
  /// url-encoding when needed.
  static std::string escapePathName(const std::string& data);

  /// Converts the hive-metastore-compliant path name back to its original path
  /// name.
  static std::string unescapePathName(const std::string& data);

  /// Creates the partition directory path from the list of partition key/value
  /// pairs, will do url-encoding when needed.
  /// @param entries Vector of (key, value) pairs for partition columns. Cannot
  ///        be empty.
  /// @param partitionPathAsLowerCase Whether to convert keys to lowercase
  /// @param useDefaultPartitionValue If true, empty values are replaced with
  ///        kDefaultPartitionValue. If false, empty values are encoded as-is.
  ///        Defaults to true for Hive compatibility.
  /// @param encodeFunc Function to use for encoding keys and values.
  ///                   Defaults to escapePathName.
  static std::string makePartName(
      const std::vector<std::pair<std::string, std::string>>& entries,
      bool partitionPathAsLowerCase,
      bool useDefaultPartitionValue = true,
      const EncodeFunction& encodeFunc = escapePathName);

  /// Converts the hive-metastore-compliant path name back to the corresponding
  /// partition key/value pairs.
  static std::vector<std::pair<std::string, std::string>> parsePartKeyValues(
      const std::string& partName);

  /// Converts a path name to a hive-metastore-compliant path name.
  static std::string extractPartitionName(const std::string& filePath);

  inline static const std::string kDefaultPartitionValue =
      "__HIVE_DEFAULT_PARTITION__";
};

} // namespace fbhive
} // namespace catalog
} // namespace dwio
} // namespace velox
} // namespace facebook

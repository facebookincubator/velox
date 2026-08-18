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

#include <string>
#include <string_view>
#include <vector>

namespace facebook::nimble {

/// File-level feature state that readers need before loading feature-specific
/// optional metadata.
class FileFeatures {
 public:
  FileFeatures(
      bool compactRowCountEncoding,
      bool clusterIndexKeyColumnStorageOmitted,
      std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage);

  /// Returns whether encoded stream row counts use compact varint encoding.
  bool compactRowCountEncoding() const {
    return compactRowCountEncoding_;
  }

  /// Returns whether cluster index key columns were omitted from data storage.
  bool clusterIndexKeyColumnStorageOmitted() const {
    return clusterIndexKeyColumnStorageOmitted_;
  }

  /// Returns cluster index key columns whose normal data streams are absent.
  const std::vector<std::string>& clusterIndexKeyColumnsWithOmittedStorage()
      const {
    return clusterIndexKeyColumnsWithOmittedStorage_;
  }

  /// Serializes file features into the `columnar.features` optional section.
  std::string serialize() const;

  /// Deserializes the `columnar.features` optional section.
  static FileFeatures deserialize(std::string_view data);

 private:
  bool compactRowCountEncoding_{false};
  bool clusterIndexKeyColumnStorageOmitted_{false};
  std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage_;
};

} // namespace facebook::nimble

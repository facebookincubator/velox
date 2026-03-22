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

#include <optional>
#include <unordered_map>

#include "velox/connectors/Connector.h"
#include "velox/connectors/common/FileProperties.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector {

/// Base split for file-based connectors. Represents a byte range within a
/// single data file, along with partition keys and synthesized column values.
struct FileSplit : public ConnectorSplit {
  const std::string filePath;
  dwio::common::FileFormat fileFormat;
  const uint64_t start;
  const uint64_t length;
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  std::unordered_map<std::string, std::string> infoColumns;
  std::optional<FileProperties> properties;

  FileSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      dwio::common::FileFormat _fileFormat,
      uint64_t _start = 0,
      uint64_t _length = std::numeric_limits<uint64_t>::max(),
      const std::unordered_map<std::string, std::optional<std::string>>&
          _partitionKeys = {},
      int64_t splitWeight = 0,
      bool cacheable = true,
      const std::unordered_map<std::string, std::string>& _infoColumns = {},
      std::optional<FileProperties> _properties = std::nullopt)
      : ConnectorSplit(connectorId, splitWeight, cacheable),
        filePath(_filePath),
        fileFormat(_fileFormat),
        start(_start),
        length(_length),
        partitionKeys(_partitionKeys),
        infoColumns(_infoColumns),
        properties(std::move(_properties)) {}

  ~FileSplit() override = default;

  uint64_t size() const override {
    return length;
  }

  std::string toString() const override {
    return fmt::format(
        "FileSplit: [file: {}, format: {}, range: [{}, {})]",
        filePath,
        dwio::common::toString(fileFormat),
        start,
        length);
  }

  std::string getFileName() const {
    const auto i = filePath.rfind('/');
    return i == std::string::npos ? filePath : filePath.substr(i + 1);
  }
};

} // namespace facebook::velox::connector

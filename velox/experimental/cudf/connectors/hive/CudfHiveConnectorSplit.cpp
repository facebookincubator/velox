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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"

#include <cudf/io/types.hpp>

#include <charconv>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {
std::string stripFilePrefix(const std::string& targetPath) {
  const std::string prefix = "file:";
  if (targetPath.rfind(prefix, 0) == 0) {
    return targetPath.substr(prefix.length());
  }
  return targetPath;
}
} // namespace

std::optional<size_t> knownKvikioFileSize(const CudfHiveConnectorSplit& split) {
  // Presto forwards HiveFileSplit.fileSize as this synthesized column. It is
  // object metadata, not the byte range assigned to this particular split.
  const auto it = split.infoColumns.find("$file_size");
  if (it == split.infoColumns.end() || it->second.empty()) {
    return std::nullopt;
  }
  const auto& text = it->second;
  int64_t parsedSize;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsedSize);
  if (error != std::errc{} || end != text.data() + text.size() ||
      parsedSize < 0) {
    return std::nullopt;
  }
  const auto fileSize = static_cast<uint64_t>(parsedSize);
  if (fileSize > std::numeric_limits<size_t>::max() || split.start > fileSize ||
      (split.length != std::numeric_limits<uint64_t>::max() &&
       split.length > fileSize - split.start)) {
    return std::nullopt;
  }
  return static_cast<size_t>(fileSize);
}

std::string CudfHiveConnectorSplit::toString() const {
  return fmt::format("CudfHive: {}", filePath);
}

std::string CudfHiveConnectorSplit::getFileName() const {
  const auto i = filePath.rfind('/');
  return i == std::string::npos ? filePath : filePath.substr(i + 1);
}

uint64_t CudfHiveConnectorSplit::size() const {
  return length;
}

CudfHiveConnectorSplit::CudfHiveConnectorSplit(
    const std::string& connectorId,
    const std::string& _filePath,
    uint64_t _start,
    uint64_t _length,
    int64_t _splitWeight,
    const std::unordered_map<std::string, std::string>& _infoColumns,
    bool _cacheable)
    : facebook::velox::connector::ConnectorSplit(
          connectorId,
          _splitWeight,
          _cacheable),
      filePath(stripFilePrefix(_filePath)),
      start(_start),
      length(_length),
      cudfSourceInfo(std::make_unique<cudf::io::source_info>(filePath)),
      infoColumns(_infoColumns) {}

// static
std::shared_ptr<CudfHiveConnectorSplit> CudfHiveConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto filePath = obj["filePath"].asString();
  const auto start = static_cast<uint64_t>(obj["start"].asInt());
  const auto length = static_cast<uint64_t>(obj["length"].asInt());
  const auto splitWeight = obj["splitWeight"].asInt();
  const auto cacheable = obj.getDefault("cacheable", true).asBool();

  std::unordered_map<std::string, std::string> infoColumns;
  for (const auto& [key, value] : obj["infoColumns"].items()) {
    infoColumns[key.asString()] = value.asString();
  }

  return std::make_shared<CudfHiveConnectorSplit>(
      connectorId,
      filePath,
      start,
      length,
      splitWeight,
      infoColumns,
      cacheable);
}

folly::dynamic CudfHiveConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["connectorId"] = connectorId;
  obj["filePath"] = filePath;
  obj["start"] = start;
  obj["length"] = length;
  obj["splitWeight"] = splitWeight;
  obj["cacheable"] = cacheable;

  folly::dynamic infoColumnsObj = folly::dynamic::object;
  for (const auto& [key, value] : infoColumns) {
    infoColumnsObj[key] = value;
  }
  obj["infoColumns"] = infoColumnsObj;

  return obj;
}

} // namespace facebook::velox::cudf_velox::connector::hive

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

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {
std::string stripFilePrefix(const std::string& targetPath) {
  const std::string prefix = "file:";
  if (targetPath.rfind(prefix, 0) == 0) {
    return targetPath.substr(prefix.length());
  }
  return targetPath;
}

std::string normalizeBatchedFilePath(const std::string& targetPath) {
  auto path = stripFilePrefix(targetPath);
  constexpr std::string_view kS3APrefix{"s3a:"};
  if (path.starts_with(kS3APrefix)) {
    path.erase(kS3APrefix.size() - 2, 1);
  }
  return path;
}

std::vector<std::string> stripFilePrefixes(
    const std::vector<std::string>& paths) {
  VELOX_USER_CHECK(!paths.empty(), "A batched split must contain a file");
  std::vector<std::string> result;
  result.reserve(paths.size());
  for (const auto& p : paths) {
    result.push_back(normalizeBatchedFilePath(p));
  }
  return result;
}

} // namespace

std::string CudfHiveConnectorSplit::toString() const {
  if (filePaths.size() <= 1) {
    return fmt::format("CudfHive: {}", filePath);
  }
  return fmt::format(
      "CudfHive: {} files [{}..{}]",
      filePaths.size(),
      filePaths.front(),
      filePaths.back());
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
    const std::unordered_map<std::string, std::string>& _infoColumns)
    : facebook::velox::connector::ConnectorSplit(connectorId, _splitWeight),
      filePaths({stripFilePrefix(_filePath)}),
      filePath(filePaths.front()),
      start(_start),
      length(_length),
      cudfSourceInfo(std::make_unique<cudf::io::source_info>(filePath)),
      infoColumns(_infoColumns) {}

CudfHiveConnectorSplit::CudfHiveConnectorSplit(
    BatchTag,
    const std::string& connectorId,
    const std::vector<std::string>& paths,
    int64_t splitWeight)
    : facebook::velox::connector::ConnectorSplit(connectorId, splitWeight),
      filePaths(stripFilePrefixes(paths)),
      filePath(filePaths.front()),
      start(0),
      length(std::numeric_limits<uint64_t>::max()),
      cudfSourceInfo(std::make_unique<cudf::io::source_info>(filePaths)) {}

// static
std::shared_ptr<CudfHiveConnectorSplit> CudfHiveConnectorSplit::makeBatch(
    const std::string& connectorId,
    const std::vector<std::string>& filePaths,
    int64_t splitWeight) {
  return std::shared_ptr<CudfHiveConnectorSplit>(new CudfHiveConnectorSplit(
      BatchTag{}, connectorId, filePaths, splitWeight));
}

// static
CudfHiveConnectorSplitBuilder CudfHiveConnectorSplitBuilder::forFilePaths(
    std::vector<std::string> filePaths) {
  return CudfHiveConnectorSplitBuilder(BatchTag{}, std::move(filePaths));
}

std::vector<std::shared_ptr<CudfHiveConnectorSplit>>
makeCudfHiveConnectorSplitBatches(
    const std::string& connectorId,
    const std::vector<std::string>& filePaths,
    size_t maxFilesPerBatch,
    int64_t splitWeight) {
  if (filePaths.empty()) {
    return {};
  }
  const auto batchSize =
      maxFilesPerBatch == 0 ? filePaths.size() : maxFilesPerBatch;
  std::vector<std::shared_ptr<CudfHiveConnectorSplit>> result;
  result.reserve((filePaths.size() + batchSize - 1) / batchSize);
  for (size_t start = 0; start < filePaths.size(); start += batchSize) {
    const auto end = std::min(start + batchSize, filePaths.size());
    result.push_back(
        CudfHiveConnectorSplit::makeBatch(
            connectorId,
            std::vector<std::string>(
                filePaths.begin() + start, filePaths.begin() + end),
            splitWeight));
  }
  return result;
}

// static
std::shared_ptr<CudfHiveConnectorSplit> CudfHiveConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto splitWeight = obj["splitWeight"].asInt();

  if (obj.count("filePaths")) {
    std::vector<std::string> filePaths;
    filePaths.reserve(obj["filePaths"].size());
    for (const auto& path : obj["filePaths"]) {
      filePaths.push_back(path.asString());
    }
    return CudfHiveConnectorSplit::makeBatch(
        connectorId, filePaths, splitWeight);
  }

  const auto filePath = obj["filePath"].asString();
  const auto start = static_cast<uint64_t>(obj["start"].asInt());
  const auto length = static_cast<uint64_t>(obj["length"].asInt());

  std::unordered_map<std::string, std::string> infoColumns;
  for (const auto& [key, value] : obj["infoColumns"].items()) {
    infoColumns[key.asString()] = value.asString();
  }

  return std::make_shared<CudfHiveConnectorSplit>(
      connectorId, filePath, start, length, splitWeight, infoColumns);
}

folly::dynamic CudfHiveConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["connectorId"] = connectorId;
  obj["filePath"] = filePath;
  obj["start"] = start;
  obj["length"] = length;
  obj["splitWeight"] = splitWeight;

  if (filePaths.size() > 1) {
    folly::dynamic paths = folly::dynamic::array;
    for (const auto& path : filePaths) {
      paths.push_back(path);
    }
    obj["filePaths"] = std::move(paths);
  }

  folly::dynamic infoColumnsObj = folly::dynamic::object;
  for (const auto& [key, value] : infoColumns) {
    infoColumnsObj[key] = value;
  }
  obj["infoColumns"] = infoColumnsObj;

  return obj;
}

} // namespace facebook::velox::cudf_velox::connector::hive

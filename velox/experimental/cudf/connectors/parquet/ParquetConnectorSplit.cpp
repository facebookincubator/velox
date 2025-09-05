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

#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"

#include <string>

namespace facebook::velox::cudf_velox::connector::parquet {

namespace {
std::string stripFilePrefix(const std::string& targetPath) {
  const std::string prefix = "file:";
  if (targetPath.rfind(prefix, 0) == 0) {
    return targetPath.substr(prefix.length());
  }
  return targetPath;
}
} // namespace

std::string ParquetConnectorSplit::toString() const {
  return fmt::format("Parquet: {}", filePath);
}

std::string ParquetConnectorSplit::getFileName() const {
  const auto i = filePath.rfind('/');
  return i == std::string::npos ? filePath : filePath.substr(i + 1);
}

// static
std::shared_ptr<ParquetConnectorSplit> ParquetConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto splitWeight = obj["splitWeight"].asInt();
  const auto filePath = obj["filePath"].asString();

  return std::make_shared<ParquetConnectorSplit>(
      connectorId, filePath, splitWeight);
}

ParquetConnectorSplit::ParquetConnectorSplit(
    const std::string& connectorId,
    const std::string& _filePath,
    int64_t _splitWeight)
    : facebook::velox::connector::ConnectorSplit(connectorId, _splitWeight),
      filePath(stripFilePrefix(_filePath)),
      cudfSourceInfo({filePath}) {
  VELOX_CHECK(
      start <=
          static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
      "ParquetConnectorSplit `start` must be less than or equal to 2^31");
  VELOX_CHECK(
      length <=
          static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
      "ParquetConnectorSplit `length` must be less than or equal to 2^31");
}

} // namespace facebook::velox::cudf_velox::connector::parquet

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

#include "velox/connectors/hive/hudi/HudiLogBlock.h"

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::hudi {

const std::string& HudiLogBlock::instantTime() const {
  const auto it = header.find(HudiLogBlockMetadataKey::kInstantTime);
  VELOX_USER_CHECK(
      it != header.end(), "Hudi log block header is missing an instant time");
  return it->second;
}

std::optional<std::string> HudiLogBlock::targetInstantTime() const {
  const auto it = header.find(HudiLogBlockMetadataKey::kTargetInstantTime);
  if (it == header.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> HudiLogBlock::schemaJson() const {
  const auto it = header.find(HudiLogBlockMetadataKey::kSchema);
  if (it == header.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<HudiCommandBlockType> HudiLogBlock::commandBlockType() const {
  if (blockType != HudiLogBlockType::kCommand) {
    return std::nullopt;
  }
  const auto it = header.find(HudiLogBlockMetadataKey::kCommandBlockType);
  if (it == header.end()) {
    return std::nullopt;
  }
  // The command sub-type is stored as its decimal string representation.
  return static_cast<HudiCommandBlockType>(folly::to<uint32_t>(it->second));
}

bool HudiLogBlock::isDataBlock() const {
  return blockType == HudiLogBlockType::kAvroData ||
      blockType == HudiLogBlockType::kParquetData ||
      blockType == HudiLogBlockType::kCdcData;
}

bool HudiLogBlock::isDeleteBlock() const {
  return blockType == HudiLogBlockType::kDelete;
}

bool HudiLogBlock::isRollbackBlock() const {
  const auto commandType = commandBlockType();
  return commandType.has_value() &&
      commandType.value() == HudiCommandBlockType::kRollback;
}

} // namespace facebook::velox::connector::hive::hudi

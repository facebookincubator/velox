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

#include "velox/common/caching/FileProperties.h"

namespace facebook::velox {

folly::dynamic FileProperties::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["fileSize"] =
      fileSize.has_value() ? folly::dynamic(fileSize.value()) : nullptr;
  obj["modificationTime"] = modificationTime.has_value()
      ? folly::dynamic(modificationTime.value())
      : nullptr;
  obj["readRangeHint"] = readRangeHint.has_value()
      ? folly::dynamic(readRangeHint.value())
      : nullptr;
  obj["extraFileInfo"] =
      extraFileInfo == nullptr ? nullptr : folly::dynamic(*extraFileInfo);

  folly::dynamic fileReadOpsObj = folly::dynamic::object;
  for (const auto& [key, value] : fileReadOps) {
    fileReadOpsObj[key] = value;
  }
  obj["fileReadOps"] = fileReadOpsObj;

  return obj;
}

namespace {
std::optional<int64_t> optionalInt(const folly::dynamic& obj, const char* key) {
  const auto& value = obj.getDefault(key, nullptr);
  return value.isNull() ? std::nullopt : std::optional<int64_t>(value.asInt());
}
} // namespace

// static
FileProperties FileProperties::create(const folly::dynamic& obj) {
  FileProperties properties;
  properties.fileSize = optionalInt(obj, "fileSize");
  properties.modificationTime = optionalInt(obj, "modificationTime");
  properties.readRangeHint = optionalInt(obj, "readRangeHint");

  const auto& extraFileInfoObj = obj.getDefault("extraFileInfo", nullptr);
  if (!extraFileInfoObj.isNull()) {
    properties.extraFileInfo =
        std::make_shared<std::string>(extraFileInfoObj.asString());
  }

  const auto& fileReadOpsObj = obj.getDefault("fileReadOps", nullptr);
  if (!fileReadOpsObj.isNull()) {
    for (const auto& [key, value] : fileReadOpsObj.items()) {
      properties.fileReadOps[key.asString()] = value.asString();
    }
  }

  return properties;
}

} // namespace facebook::velox

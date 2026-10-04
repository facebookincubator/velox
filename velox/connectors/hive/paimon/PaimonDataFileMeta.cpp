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
#include "velox/connectors/hive/paimon/PaimonDataFileMeta.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/paimon/PaimonMetadata.h"

namespace facebook::velox::connector::hive::paimon {

// static
std::string PaimonDataFile::typeString(Type type) {
  switch (type) {
    case Type::kData:
      return "DATA";
    case Type::kChangelog:
      return "CHANGELOG";
    default:
      VELOX_FAIL("Unknown PaimonDataFile::Type: {}", static_cast<int>(type));
  }
}

// static
PaimonDataFile::Type PaimonDataFile::typeFromString(const std::string& str) {
  if (str == "DATA") {
    return Type::kData;
  }
  if (str == "CHANGELOG") {
    return Type::kChangelog;
  }
  VELOX_FAIL("Unknown PaimonDataFile::Type: {}", str);
}

// static
std::string PaimonDataFile::sourceString(Source source) {
  switch (source) {
    case Source::kAppend:
      return "APPEND";
    case Source::kCompact:
      return "COMPACT";
    default:
      VELOX_FAIL(
          "Unknown PaimonDataFile::Source: {}", static_cast<int>(source));
  }
}

// static
PaimonDataFile::Source PaimonDataFile::sourceFromString(
    const std::string& str) {
  if (str == "APPEND") {
    return Source::kAppend;
  }
  if (str == "COMPACT") {
    return Source::kCompact;
  }
  VELOX_FAIL("Unknown PaimonDataFile::Source: {}", str);
}

std::string PaimonDataFile::toString() const {
  return fmt::format(
      "{{path={}, size={}, rows={}, level={}, type={}, source={}, "
      "deletionFile={}}}",
      path,
      size,
      rowCount,
      level,
      typeString(type),
      sourceString(source),
      deletionFile.has_value() ? deletionFile->toString() : "none");
}

folly::dynamic PaimonDataFile::serialize() const {
  validate();
  folly::dynamic obj = folly::dynamic::object;
  obj["filePath"] = path;
  obj["fileSize"] = size;
  obj["rowCount"] = rowCount;
  obj["level"] = level;
  obj["minSequenceNumber"] = minSequenceNumber;
  obj["maxSequenceNumber"] = maxSequenceNumber;
  obj["deleteRowCount"] = deleteRowCount ? folly::dynamic(*deleteRowCount)
                                         : folly::dynamic(nullptr);
  obj["schemaId"] =
      schemaId ? folly::dynamic(*schemaId) : folly::dynamic(nullptr);
  if (fileFormat) {
    obj["fileFormat"] = dwio::common::FileFormatName::toName(*fileFormat);
  }
  obj["physicalFilePath"] = physicalFilePath;
  if (properties) {
    obj["properties"] = properties->serialize();
  }
  obj["creationTimeMs"] = creationTimeMs;
  obj["fileType"] = typeString(type);
  obj["sourceType"] = sourceString(source);
  if (deletionFile.has_value()) {
    obj["deletionFile"] = deletionFile->serialize();
  }
  return obj;
}

// static
PaimonDataFile PaimonDataFile::create(const folly::dynamic& obj) {
  PaimonDataFile file;
  file.path = obj["filePath"].asString();
  file.size = paimonInt(obj, "fileSize");
  file.rowCount = paimonInt(obj, "rowCount");
  file.level = static_cast<int32_t>(
      paimonInt(obj, "level", 0, std::numeric_limits<int32_t>::max()));
  file.minSequenceNumber =
      paimonInt(obj, "minSequenceNumber", std::numeric_limits<int64_t>::min());
  file.maxSequenceNumber =
      paimonInt(obj, "maxSequenceNumber", std::numeric_limits<int64_t>::min());
  if (obj.count("deleteRowCount") && !obj["deleteRowCount"].isNull()) {
    file.deleteRowCount = paimonInt(obj, "deleteRowCount");
  }
  if (obj.count("schemaId") && !obj["schemaId"].isNull()) {
    file.schemaId = paimonInt(obj, "schemaId");
  }
  if (obj.count("fileFormat")) {
    file.fileFormat = dwio::common::toFileFormat(obj["fileFormat"].asString());
  }
  if (obj.count("physicalFilePath")) {
    file.physicalFilePath = obj["physicalFilePath"].asString();
  }
  if (obj.count("properties")) {
    const auto& properties = obj["properties"];
    for (const auto* field :
         {"fileSize", "readRangeHint", "modificationTime"}) {
      if (properties.count(field) && !properties[field].isNull()) {
        paimonInt(properties, field, std::numeric_limits<int64_t>::min());
      }
    }
    file.properties = FileProperties::create(properties);
  }
  file.creationTimeMs =
      paimonInt(obj, "creationTimeMs", std::numeric_limits<int64_t>::min());
  file.type = typeFromString(obj["fileType"].asString());
  file.source = sourceFromString(obj["sourceType"].asString());
  if (obj.count("deletionFile") > 0) {
    file.deletionFile = PaimonDeletionFile::create(obj["deletionFile"]);
  }
  file.validate();
  return file;
}

void PaimonDataFile::validate() const {
  VELOX_USER_CHECK(!path.empty(), "Paimon filePath must not be empty");
  constexpr auto max = std::numeric_limits<int64_t>::max();
  VELOX_USER_CHECK_LE(size, max, "Paimon fileSize out of range");
  VELOX_USER_CHECK_LE(rowCount, max, "Paimon rowCount out of range");
  VELOX_USER_CHECK_GE(level, 0, "Paimon level out of range");
  VELOX_USER_CHECK_LE(
      minSequenceNumber,
      maxSequenceNumber,
      "Paimon sequence bounds are reversed");
  if (schemaId) {
    VELOX_USER_CHECK_GE(*schemaId, 0, "Paimon schemaId out of range");
  }
  if (deleteRowCount) {
    VELOX_USER_CHECK_GE(
        *deleteRowCount, 0, "Paimon deleteRowCount out of range");
    VELOX_USER_CHECK_LE(
        *deleteRowCount, rowCount, "Paimon deleteRowCount exceeds rowCount");
  }
  if (fileFormat) {
    VELOX_USER_CHECK_NE(
        *fileFormat,
        dwio::common::FileFormat::UNKNOWN,
        "Paimon fileFormat is unknown");
  }
  if (properties) {
    if (properties->fileSize) {
      VELOX_USER_CHECK_GE(
          *properties->fileSize, 0, "Paimon properties.fileSize out of range");
      VELOX_USER_CHECK_EQ(
          *properties->fileSize,
          size,
          "Paimon properties.fileSize differs from fileSize");
    }
    if (properties->readRangeHint) {
      VELOX_USER_CHECK_GE(
          *properties->readRangeHint, 0, "Paimon readRangeHint out of range");
    }
  }
}

} // namespace facebook::velox::connector::hive::paimon

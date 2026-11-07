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
#include "velox/connectors/paimon/PaimonDataFileMeta.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::paimon {

PaimonDataFile::PaimonDataFile(
    int64_t schemaId,
    std::string path,
    uint64_t size,
    uint64_t rowCount,
    int32_t level,
    int64_t minSequenceNumber,
    int64_t maxSequenceNumber,
    int64_t deleteRowCount,
    int64_t creationTimeMs,
    Type type,
    Source source,
    std::optional<PaimonDeletionFile> deletionFile)
    : path(std::move(path)),
      schemaId(schemaId),
      size(size),
      rowCount(rowCount),
      level(level),
      minSequenceNumber(minSequenceNumber),
      maxSequenceNumber(maxSequenceNumber),
      deleteRowCount(deleteRowCount),
      creationTimeMs(creationTimeMs),
      type(type),
      source(source),
      deletionFile(std::move(deletionFile)) {
  VELOX_USER_CHECK_GE(schemaId, 0, "Paimon file schema ID must be nonnegative");
}

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
  folly::dynamic obj = folly::dynamic::object;
  obj["filePath"] = path;
  obj["fileSize"] = size;
  obj["rowCount"] = rowCount;
  obj["level"] = level;
  obj["minSequenceNumber"] = minSequenceNumber;
  obj["maxSequenceNumber"] = maxSequenceNumber;
  obj["deleteRowCount"] = deleteRowCount;
  obj["creationTimeMs"] = creationTimeMs;
  obj["fileType"] = typeString(type);
  obj["sourceType"] = sourceString(source);
  obj["schemaId"] = schemaId;
  if (deletionFile.has_value()) {
    obj["deletionFile"] = deletionFile->serialize();
  }
  return obj;
}

// static
PaimonDataFile PaimonDataFile::create(const folly::dynamic& obj) {
  const auto schemaId = obj.find("schemaId");
  VELOX_USER_CHECK(
      schemaId != obj.items().end() && schemaId->second.isInt(),
      "Expected Paimon file schema ID");
  return PaimonDataFile(
      schemaId->second.asInt(),
      obj["filePath"].asString(),
      static_cast<uint64_t>(obj["fileSize"].asInt()),
      static_cast<uint64_t>(obj["rowCount"].asInt()),
      static_cast<int32_t>(obj["level"].asInt()),
      obj["minSequenceNumber"].asInt(),
      obj["maxSequenceNumber"].asInt(),
      obj["deleteRowCount"].asInt(),
      obj["creationTimeMs"].asInt(),
      typeFromString(obj["fileType"].asString()),
      sourceFromString(obj["sourceType"].asString()),
      obj.count("deletionFile") > 0
          ? std::optional<PaimonDeletionFile>(
                PaimonDeletionFile::create(obj["deletionFile"]))
          : std::nullopt);
}

} // namespace facebook::velox::connector::paimon

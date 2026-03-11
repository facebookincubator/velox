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
  if (deletionFile.has_value()) {
    obj["deletionFile"] = deletionFile->serialize();
  }
  return obj;
}

// static
PaimonDataFile PaimonDataFile::create(const folly::dynamic& obj) {
  PaimonDataFile file;
  file.path = obj["filePath"].asString();
  file.size = static_cast<uint64_t>(obj["fileSize"].asInt());
  file.rowCount = static_cast<uint64_t>(obj["rowCount"].asInt());
  file.level = static_cast<int32_t>(obj["level"].asInt());
  file.minSequenceNumber = obj["minSequenceNumber"].asInt();
  file.maxSequenceNumber = obj["maxSequenceNumber"].asInt();
  file.deleteRowCount = obj["deleteRowCount"].asInt();
  file.creationTimeMs = obj["creationTimeMs"].asInt();
  file.type = typeFromString(obj["fileType"].asString());
  file.source = sourceFromString(obj["sourceType"].asString());
  if (obj.count("deletionFile") > 0) {
    file.deletionFile = PaimonDeletionFile::create(obj["deletionFile"]);
  }
  return file;
}

} // namespace facebook::velox::connector::hive::paimon

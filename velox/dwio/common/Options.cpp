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

#include "velox/dwio/common/Options.h"
#include "velox/common/compression/Compression.h"

namespace facebook::velox::dwio::common {

FileFormat toFileFormat(std::string_view s) {
  if (s == "dwrf") {
    return FileFormat::DWRF;
  } else if (s == "rc") {
    return FileFormat::RC;
  } else if (s == "rc:text") {
    return FileFormat::RC_TEXT;
  } else if (s == "rc:binary") {
    return FileFormat::RC_BINARY;
  } else if (s == "text") {
    return FileFormat::TEXT;
  } else if (s == "json") {
    return FileFormat::JSON;
  } else if (s == "parquet") {
    return FileFormat::PARQUET;
  } else if (s == "nimble" || s == "alpha") {
    return FileFormat::NIMBLE;
  } else if (s == "orc") {
    return FileFormat::ORC;
  } else if (s == "sst") {
    return FileFormat::SST;
  }
  return FileFormat::UNKNOWN;
}

std::string_view toString(FileFormat fmt) {
  switch (fmt) {
    case FileFormat::DWRF:
      return "dwrf";
    case FileFormat::RC:
      return "rc";
    case FileFormat::RC_TEXT:
      return "rc:text";
    case FileFormat::RC_BINARY:
      return "rc:binary";
    case FileFormat::TEXT:
      return "text";
    case FileFormat::JSON:
      return "json";
    case FileFormat::PARQUET:
      return "parquet";
    case FileFormat::NIMBLE:
      return "nimble";
    case FileFormat::ORC:
      return "orc";
    case FileFormat::SST:
      return "sst";
    default:
      return "unknown";
  }
}

ColumnReaderOptions makeColumnReaderOptions(const ReaderOptions& options) {
  ColumnReaderOptions columnReaderOptions;
  columnReaderOptions.useColumnNamesForColumnMapping_ =
      options.useColumnNamesForColumnMapping();
  return columnReaderOptions;
}

folly::dynamic WriterOptions::serialize() const {
  folly::dynamic obj = folly::dynamic::object;

  // 1) Schema
  if (schema) {
    obj["schema"] = schema->serialize();
  }

  // 2) compressionKind
  if (compressionKind) {
    obj["compressionKind"] = static_cast<int>(*compressionKind);
  }

  // 3) serdeParameters
  if (!serdeParameters.empty()) {
    folly::dynamic mapObj = folly::dynamic::object;
    for (auto& [k, v] : serdeParameters) {
      mapObj[k] = v;
    }
    obj["serdeParameters"] = std::move(mapObj);
  }

  // 4) sessionTimezoneName
  if (!sessionTimezoneName.empty()) {
    obj["sessionTimezoneName"] = sessionTimezoneName;
  }

  // 5) adjustTimestampToTimezone
  obj["adjustTimestampToTimezone"] = adjustTimestampToTimezone;

  // (We do *not* serialize pool, spillConfig, nonReclaimableSection,
  //  or the factory functions—they must be re‐injected by the host.)

  return obj;
}

std::shared_ptr<WriterOptions> WriterOptions::deserialize(
    const folly::dynamic& obj) {
  auto opts = std::make_shared<WriterOptions>();

  // 1) schema
  if (auto p = obj.get_ptr("schema")) {
    opts->schema = ISerializable::deserialize<velox::Type>(*p, nullptr);
  }

  // 2) compressionKind
  if (auto p = obj.get_ptr("compressionKind")) {
    opts->compressionKind =
        static_cast<velox::common::CompressionKind>(p->asInt());
  }

  // 3) serdeParameters
  if (auto p = obj.get_ptr("serdeParameters")) {
    opts->serdeParameters.clear();
    for (auto& kv : p->items()) {
      opts->serdeParameters.emplace(kv.first.asString(), kv.second.asString());
    }
  }

  // 4) sessionTimezoneName
  if (auto p = obj.get_ptr("sessionTimezoneName")) {
    opts->sessionTimezoneName = p->asString();
  }

  // 5) adjustTimestampToTimezone
  if (auto p = obj.get_ptr("adjustTimestampToTimezone")) {
    opts->adjustTimestampToTimezone = p->asBool();
  }

  // pool, spillConfig, nonReclaimableSection, factories remain at default
  return opts;
}

void WriterOptions::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("WriterOptions", WriterOptions::deserialize);
}

// force registration at load‐time
static bool _writerOptionsSerdeRegistered = []() {
  WriterOptions::registerSerDe();
  return true;
}();

} // namespace facebook::velox::dwio::common

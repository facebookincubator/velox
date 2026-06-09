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

#include "velox/common/EnumDefine.h"

namespace facebook::velox::dwio::common {

namespace {

const auto& columnMappingModeNames() {
  static const folly::F14FastMap<ColumnMappingMode, std::string_view> kNames = {
      {ColumnMappingMode::kPosition, "POSITION"},
      {ColumnMappingMode::kName, "NAME"},
      {ColumnMappingMode::kParquetFieldId, "PARQUET_FIELD_ID"},
  };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(ColumnMappingMode, columnMappingModeNames);

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
  } else if (s == "flux") {
    return FileFormat::FLUX;
  } else if (s == "avro") {
    return FileFormat::AVRO;
  } else if (s == "puffin") {
    return FileFormat::PUFFIN;
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
    case FileFormat::FLUX:
      return "flux";
    case FileFormat::AVRO:
      return "avro";
    case FileFormat::PUFFIN:
      return "puffin";
    default:
      return "unknown";
  }
}

ColumnReaderOptions makeColumnReaderOptions(const ReaderOptions& options) {
  ColumnReaderOptions columnReaderOptions;
  columnReaderOptions.columnMappingMode_ = options.columnMappingMode();
  return columnReaderOptions;
}

} // namespace facebook::velox::dwio::common

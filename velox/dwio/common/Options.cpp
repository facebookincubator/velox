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

const auto& fileFormatNames() {
  static const folly::F14FastMap<FileFormat, std::string_view> kNames = {
      {FileFormat::UNKNOWN, "unknown"},
      {FileFormat::DWRF, "dwrf"},
      {FileFormat::RC, "rc"},
      {FileFormat::RC_TEXT, "rc:text"},
      {FileFormat::RC_BINARY, "rc:binary"},
      {FileFormat::TEXT, "text"},
      {FileFormat::JSON, "json"},
      {FileFormat::PARQUET, "parquet"},
      {FileFormat::NIMBLE, "nimble"},
      {FileFormat::ORC, "orc"},
      {FileFormat::SST, "sst"},
      {FileFormat::FLUX, "flux"},
      {FileFormat::AVRO, "avro"},
      {FileFormat::PUFFIN, "puffin"},
  };
  return kNames;
}

const auto& columnMappingModeNames() {
  static const folly::F14FastMap<ColumnMappingMode, std::string_view> kNames = {
      {ColumnMappingMode::kPosition, "POSITION"},
      {ColumnMappingMode::kName, "NAME"},
      {ColumnMappingMode::kParquetFieldId, "PARQUET_FIELD_ID"},
      {ColumnMappingMode::kFieldId, "FIELD_ID"},
  };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(FileFormat, fileFormatNames);
VELOX_DEFINE_ENUM_NAME(ColumnMappingMode, columnMappingModeNames);

FileFormat toFileFormat(std::string_view s) {
  if (s == "alpha") {
    return FileFormat::NIMBLE;
  }
  return FileFormatName::tryToFileFormat(s).value_or(FileFormat::UNKNOWN);
}

std::string formatConfigPrefix(FileFormat fmt, std::string_view separator) {
  if (fmt == FileFormat::UNKNOWN) {
    return "";
  }
  if (fmt == FileFormat::DWRF) {
    fmt = FileFormat::ORC;
  }
  return std::string(FileFormatName::toName(fmt)) + std::string(separator);
}

ColumnReaderOptions makeColumnReaderOptions(const ReaderOptions& options) {
  ColumnReaderOptions columnReaderOptions;
  columnReaderOptions.columnMappingMode_ = options.columnMappingMode();
  return columnReaderOptions;
}

} // namespace facebook::velox::dwio::common

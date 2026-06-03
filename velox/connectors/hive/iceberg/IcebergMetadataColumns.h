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

#pragma once

#include <string>

#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

struct IcebergMetadataColumn {
  int id;
  std::string name;
  std::shared_ptr<const Type> type;
  std::string doc;

  // Reserved Field IDs for Iceberg tables; see
  // https://iceberg.apache.org/spec/#reserved-field-ids
  static constexpr int32_t kPosId = 2'147'483'545;
  static constexpr int32_t kFilePathId = 2'147'483'546;
  static constexpr int32_t kRowId = 2'147'483'540;
  static constexpr int32_t kLastUpdatedSequenceNumber = 2'147'483'539;

  static constexpr const char* kRowIdColumnName = "_row_id";
  static constexpr const char* kLastUpdatedSequenceNumberColumnName =
      "_last_updated_sequence_number";
  // Info column keys provided in the split's infoColumns map.
  static constexpr const char* kFirstRowIdInfoColumn = "$first_row_id";
  static constexpr const char* kDataSequenceNumberInfoColumn =
      "$data_sequence_number";

  IcebergMetadataColumn(
      int _id,
      const std::string& _name,
      std::shared_ptr<const Type> _type,
      const std::string& _doc)
      : id(_id), name(_name), type(_type), doc(_doc) {}

  static std::shared_ptr<IcebergMetadataColumn> icebergDeleteFilePathColumn() {
    return std::make_shared<IcebergMetadataColumn>(
        kFilePathId,
        "file_path",
        VARCHAR(),
        "Path of a file in which a deleted row is stored");
  }

  static std::shared_ptr<IcebergMetadataColumn> icebergDeletePosColumn() {
    return std::make_shared<IcebergMetadataColumn>(
        kPosId,
        "pos",
        BIGINT(),
        "Ordinal position of a deleted row in the data file");
  }
};

} // namespace facebook::velox::connector::hive::iceberg

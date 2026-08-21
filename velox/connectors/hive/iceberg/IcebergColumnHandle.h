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

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergFieldMetadata.h"
#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergColumnHandle : public HiveColumnHandle {
 public:
  IcebergColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      parquet::ParquetFieldId icebergField,
      std::vector<common::Subfield> requiredSubfields = {},
      std::optional<std::string> initialDefaultValue = std::nullopt,
      IcebergFieldMetadata icebergMetadata = {},
      std::function<void(VectorPtr&)> postProcessor = {});

  const parquet::ParquetFieldId& field() const;

  /// Iceberg V3 type-disambiguation attributes for this column, parallel to
  /// field(). Empty for callers that do not supply V3 metadata.
  const IcebergFieldMetadata& icebergMetadata() const {
    return icebergMetadata_;
  }

  /// Initial default value for an Iceberg V3 added column.
  ///
  /// The coordinator supplies this using its internal epoch-duration encoding,
  /// not the Iceberg table metadata JSON literal form: DATE defaults are passed
  /// as days since epoch and TIMESTAMP/TIMESTAMP_UTC defaults are passed as
  /// microseconds since epoch, both serialized as decimal strings.
  const std::optional<std::string>& initialDefaultValue() const {
    return initialDefaultValue_;
  }

 private:
  const parquet::ParquetFieldId field_;
  const std::optional<std::string> initialDefaultValue_;
  const IcebergFieldMetadata icebergMetadata_;
};

using IcebergColumnHandlePtr = std::shared_ptr<const IcebergColumnHandle>;

} // namespace facebook::velox::connector::hive::iceberg

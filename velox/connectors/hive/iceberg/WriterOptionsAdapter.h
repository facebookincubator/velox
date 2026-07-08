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

#include <memory>
#include <string>

#include "velox/connectors/hive/iceberg/IcebergFieldId.h"
#include "velox/connectors/hive/iceberg/IcebergFieldMetadata.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive::iceberg {

/// Format-specific adapter for an Iceberg-bound WriterOptions instance.
/// Encapsulates the per-file-format behavior (manifest format string and
/// writer option overrides) behind a virtual interface. Dispatched at runtime
/// via createWriterOptionsAdapter() based on the table's storage format.
class WriterOptionsAdapter {
 public:
  virtual ~WriterOptionsAdapter() = default;

  /// Identifier reported in Iceberg manifest commit messages. Iceberg's
  /// file-format vocabulary has no DWRF enum; per the Iceberg SDK
  /// convention, DWRF files are reported as "ORC" so downstream consumers
  /// (Presto coordinator, catalog) can interpret the commit message.
  virtual std::string manifestFormatString() const = 0;

  /// Hook applied before format-specific writers consume the common writer
  /// options. Used for settings that flow through common WriterOptions fields.
  virtual void applyPreConfigs(dwio::common::WriterOptions& /*options*/) const {
  }

  /// Hook applied after HiveDataSink creates and configures WriterOptions.
  /// Used for direct field assignments that must not be overwritten by
  /// config-driven processing.
  virtual void applyPostConfigs(
      dwio::common::WriterOptions& /*options*/) const {}
};

/// Returns the adapter for the given file format, or nullptr for
/// unsupported formats. Single source of truth for which file formats the
/// Iceberg DataSink supports on the write path.
///
/// `icebergFieldIds` carries the per-input-column Iceberg field-id tree.
/// Honored only by the NIMBLE adapter, which uses it to stamp
/// `iceberg.id` (and other Iceberg V3 keys) onto each NIMBLE schema node
/// via `VeloxWriterOptions::attributesByColumn`. Pass an empty
/// `IcebergFieldId{}` for formats / call sites that have no field-id tree
/// available (the NIMBLE adapter then produces files without
/// `iceberg.id` attributes, the same wire shape as a pre-attributes
/// writer).
///
/// `icebergMetadata` carries the parallel Iceberg V3 type-attribute tree
/// (`iceberg.required`, `iceberg.long-type`, etc.). Also honored only by
/// the NIMBLE adapter; empty by default so the stamped output is
/// byte-identical to an `iceberg.id`-only writer.
std::unique_ptr<WriterOptionsAdapter> createWriterOptionsAdapter(
    dwio::common::FileFormat format,
    IcebergFieldId icebergFieldIds = {},
    IcebergFieldMetadata icebergMetadata = {});

/// True if the Iceberg DataSink can write the given file format.
/// Supported formats: PARQUET, ORC, DWRF, NIMBLE. ORC and DWRF share
/// the same on-disk family and route through the same adapter.
bool isSupportedFileFormat(dwio::common::FileFormat format);

/// Maps a Velox file format to the string used in Iceberg manifest commit
/// messages. Throws if the format is not supported.
std::string toManifestFormatString(dwio::common::FileFormat format);

} // namespace facebook::velox::connector::hive::iceberg

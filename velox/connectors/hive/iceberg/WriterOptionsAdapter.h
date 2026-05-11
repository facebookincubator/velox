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

#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive::iceberg {

/// Format-specific adapter for an Iceberg-bound WriterOptions instance.
/// Encapsulates the per-file-format behavior (manifest format string,
/// pre/post-processConfigs hooks on the writer options) behind a virtual
/// interface. Dispatched at runtime via createWriterOptionsAdapter() based
/// on the table's storage format.
class WriterOptionsAdapter {
 public:
  virtual ~WriterOptionsAdapter() = default;

  /// Identifier reported in Iceberg manifest commit messages. Iceberg's
  /// file-format vocabulary has no DWRF enum; per the Iceberg SDK
  /// convention, DWRF files are reported as "ORC" so downstream consumers
  /// (Presto coordinator, catalog) can interpret the commit message.
  virtual std::string manifestFormatString() const = 0;

  /// Hook applied to WriterOptions BEFORE processConfigs() runs. Used for
  /// settings that flow through serdeParameters.
  virtual void applyPreConfigs(dwio::common::WriterOptions& /*options*/) const {
  }

  /// Hook applied to WriterOptions AFTER processConfigs() runs. Used for
  /// direct field assignments that must not be overwritten by
  /// config-driven processing.
  virtual void applyPostConfigs(
      dwio::common::WriterOptions& /*options*/) const {}
};

/// Returns the adapter for the given file format, or nullptr for
/// unsupported formats. Single source of truth for which file formats the
/// Iceberg DataSink supports on the write path.
std::unique_ptr<WriterOptionsAdapter> createWriterOptionsAdapter(
    dwio::common::FileFormat format);

/// True if the Iceberg DataSink can write the given file format.
bool isSupportedFileFormat(dwio::common::FileFormat format);

/// Maps a Velox file format to the string used in Iceberg manifest commit
/// messages. Throws if the format is not supported.
std::string toManifestFormatString(dwio::common::FileFormat format);

} // namespace facebook::velox::connector::hive::iceberg

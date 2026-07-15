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

#include "velox/connectors/hive/iceberg/WriterOptionsAdapter.h"

#include "velox/common/base/Exceptions.h"
#ifdef VELOX_ENABLE_NIMBLE
#include "velox/connectors/hive/iceberg/fb/NimbleWriterOptionsAdapter.h"
#endif
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/dwio/parquet/common/ParquetConfig.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Manifest format string emitted in Iceberg commit messages for files that
// share the ORC on-disk family. Iceberg's manifest vocabulary has no DWRF or
// NIMBLE enum, so DWRF and NIMBLE files are reported as "ORC" per the
// cross-engine convention shared with the Java planner (see
// FileFormat.{DWRF,NIMBLE}.toIceberg() in presto-facebook-iceberg).
constexpr std::string_view kOrcManifestFormat{"ORC"};

class ParquetWriterOptionsAdapter : public WriterOptionsAdapter {
 public:
  std::string manifestFormatString() const override {
    return "PARQUET";
  }

  void applyPreConfigs(dwio::common::WriterOptions& options) const override {
    // Per Iceberg spec (https://iceberg.apache.org/spec/#parquet):
    // - Timestamps must be stored with microsecond precision.
    // - Timestamps must NOT be adjusted to UTC; written as-is without
    //   timezone conversion (empty string disables conversion).
    //
    // Settings are routed through serdeParameters so the common writer options
    // can carry them until the Parquet writer constructor reads them. The value
    // "6" represents microseconds (TimestampPrecision::kMicroseconds).
    options.serdeParameters[std::string(
        parquet::ParquetConfig::kWriterSerdeTimestampUnit)] = "6";
    options.serdeParameters[std::string(
        parquet::ParquetConfig::kWriterSerdeTimestampTimezone)] = "";
  }
};

class DwrfWriterOptionsAdapter : public WriterOptionsAdapter {
 public:
  std::string manifestFormatString() const override {
    return std::string{kOrcManifestFormat};
  }

  void applyPostConfigs(dwio::common::WriterOptions& options) const override {
    // DWRF stores microsecond-precision timestamps natively, so no
    // precision conversion is required; only timezone adjustment must be
    // disabled per the Iceberg spec. Unlike Parquet, DWRF exposes
    // timestamp configuration as direct fields on dwrf::WriterOptions
    // rather than serdeParameters.
    auto* dwrfOptions = dynamic_cast<dwrf::WriterOptions*>(&options);
    if (dwrfOptions == nullptr) {
      return;
    }
    dwrfOptions->adjustTimestampToTimezone = false;
    dwrfOptions->sessionTimezone = nullptr;
  }
};

} // namespace

std::unique_ptr<WriterOptionsAdapter> createWriterOptionsAdapter(
    dwio::common::FileFormat format,
    IcebergFieldId icebergFieldIds,
    IcebergFieldMetadata icebergMetadata) {
  switch (format) {
    case dwio::common::FileFormat::PARQUET:
      return std::make_unique<ParquetWriterOptionsAdapter>();
    case dwio::common::FileFormat::ORC:
    case dwio::common::FileFormat::DWRF:
      // ORC and DWRF share the same on-disk family — Meta's DWRF is an
      // ORC implementation. Iceberg manifests have no DWRF enum, so both
      // are reported as "ORC" per the cross-engine convention shared
      // with the Java planner (see FileFormat.DWRF.toIceberg() in
      // presto-facebook-iceberg).
      return std::make_unique<DwrfWriterOptionsAdapter>();
#ifdef VELOX_ENABLE_NIMBLE
    case dwio::common::FileFormat::NIMBLE:
      return createNimbleWriterOptionsAdapter(
          std::move(icebergFieldIds), std::move(icebergMetadata));
#endif
    default:
      return nullptr;
  }
}

bool isSupportedFileFormat(dwio::common::FileFormat format) {
  return createWriterOptionsAdapter(format) != nullptr;
}

std::string toManifestFormatString(dwio::common::FileFormat format) {
  auto adapter = createWriterOptionsAdapter(format);
  VELOX_CHECK_NOT_NULL(
      adapter,
      "Unsupported file format for Iceberg manifest: {}",
      dwio::common::FileFormatName::toName(format));
  return adapter->manifestFormatString();
}

} // namespace facebook::velox::connector::hive::iceberg

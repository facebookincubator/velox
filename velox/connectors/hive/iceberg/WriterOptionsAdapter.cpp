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
#include "velox/dwio/dwrf/writer/Writer.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

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
    // Settings are routed through serdeParameters to avoid pulling in
    // parquet-specific headers. Keys must match
    // kParquetSerdeTimestampUnit and kParquetSerdeTimestampTimezone in
    // velox/dwio/parquet/writer/Writer.h. The value "6" represents
    // microseconds (TimestampPrecision::kMicroseconds).
    options.serdeParameters["parquet.writer.timestamp.unit"] = "6";
    options.serdeParameters["parquet.writer.timestamp.timezone"] = "";
  }
};

class DwrfWriterOptionsAdapter : public WriterOptionsAdapter {
 public:
  std::string manifestFormatString() const override {
    return "ORC";
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
    dwio::common::FileFormat format) {
  // ORC is intentionally excluded until a dedicated ORC end-to-end test
  // exists.
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
    case dwio::common::FileFormat::PARQUET:
      return std::make_unique<ParquetWriterOptionsAdapter>();
    case dwio::common::FileFormat::DWRF:
      return std::make_unique<DwrfWriterOptionsAdapter>();
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
      dwio::common::toString(format));
  return adapter->manifestFormatString();
}

} // namespace facebook::velox::connector::hive::iceberg

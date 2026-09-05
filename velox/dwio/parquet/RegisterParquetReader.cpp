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

#ifdef VELOX_ENABLE_PARQUET
#include "velox/common/base/RuntimeMetrics.h" // @manual
#include "velox/dwio/parquet/common/ParquetRuntimeStats.h" // @manual
#include "velox/dwio/parquet/reader/ParquetReader.h" // @manual
#endif

namespace facebook::velox::parquet {

#ifdef VELOX_ENABLE_PARQUET
namespace {

std::string parquetRuntimeMetricName(std::string_view name) {
  return fmt::format(
      "{}.{}",
      dwio::common::FileFormatName::toName(dwio::common::FileFormat::PARQUET),
      name);
}

} // namespace
#endif

void registerParquetReaderFactory() {
#ifdef VELOX_ENABLE_PARQUET
  dwio::common::registerReaderFactory(std::make_shared<ParquetReaderFactory>());
  registerRuntimeMetricForOperatorAggregation(
      parquetRuntimeMetricName(ParquetRuntimeStats::kPageLoadTimeNs));
#endif
}

void unregisterParquetReaderFactory() {
#ifdef VELOX_ENABLE_PARQUET
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::PARQUET);
  unregisterRuntimeMetricForOperatorAggregation(
      parquetRuntimeMetricName(ParquetRuntimeStats::kPageLoadTimeNs));
#endif
}

} // namespace facebook::velox::parquet

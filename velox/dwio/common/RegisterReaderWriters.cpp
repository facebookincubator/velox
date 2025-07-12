//
// Created by Ying Su on 7/4/25.
//
#include "velox/dwio/common/RegisterReaderWriters.h"

#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/dwio/orc/reader/OrcReader.h"
#include "velox/dwio/text/RegisterTextReader.h"
#include "velox/dwio/text/RegisterTextWriter.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#endif

namespace facebook::velox::dwio::common {

void registerReaderFactories() {
  dwrf::registerDwrfReaderFactory();
  orc::registerOrcReaderFactory();
  // TODO: either move registerTextReaderFactory() to text namespace, or move
  // all registrations to dwio/common
  registerTextReaderFactory();
#ifdef VELOX_ENABLE_PARQUET
  parquet::registerParquetReaderFactory();
#endif
}

void registerWriterFactories() {
  dwrf::registerDwrfWriterFactory();
  text::registerTextWriterFactory();
#ifdef VELOX_ENABLE_PARQUET
  parquet::registerParquetWriterFactory();
#endif
}

void unregisterReaderFactories() {
  dwrf::unregisterDwrfReaderFactory();
  orc::unregisterOrcReaderFactory();
  // TODO: either move registerTextReaderFactory() to text namespace, or move
  // all registrations to dwio/common
  unregisterTextReaderFactory();
#ifdef VELOX_ENABLE_PARQUET
  parquet::unregisterParquetReaderFactory();
#endif
}

void unregisterWriterFactories() {
  dwrf::unregisterDwrfWriterFactory();
  text::unregisterTextWriterFactory();
#ifdef VELOX_ENABLE_PARQUET
  parquet::unregisterParquetWriterFactory();
#endif
}

} // namespace facebook::velox::dwio::common

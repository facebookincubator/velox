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

//#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/reader/ParquetReader.h" // @manual
//#endif

namespace facebook::velox::parquet {

namespace {
// Self‑register the Parquet reader factory at load time
const bool kParquetReaderRegistered = []() {
  facebook::velox::dwio::common::registerReaderFactory(
      std::make_shared<facebook::velox::parquet::ParquetReaderFactory>());

  VLOG(0) << "Registered Parquet reader";
  return true;
}();
} // anonymous namespace

void registerParquetReaderFactory() {
  // In case manual registration is ever needed
  dwio::common::registerReaderFactory(std::make_shared<ParquetReaderFactory>());
}

void unregisterParquetReaderFactory() {
  // In case manual de-registration is ever needed
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::PARQUET);
}

} // namespace facebook::velox::parquet

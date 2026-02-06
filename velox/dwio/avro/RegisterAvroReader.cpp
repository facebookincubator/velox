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

#include "velox/dwio/avro/RegisterAvroReader.h"

#include "velox/common/base/Exceptions.h"

#ifdef VELOX_ENABLE_AVRO
#include "velox/dwio/avro/reader/AvroReader.h"
#endif

namespace facebook::velox::avro {

std::unique_ptr<dwio::common::Reader> AvroReaderFactory::createReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options) {
#ifdef VELOX_ENABLE_AVRO
  return std::make_unique<AvroReader>(std::move(input), options);
#else
  VELOX_UNSUPPORTED("Avro reader is not enabled in this build.");
#endif
}

void registerAvroReaderFactory() {
#ifdef VELOX_ENABLE_AVRO
  dwio::common::registerReaderFactory(std::make_shared<AvroReaderFactory>());
#endif
}

void unregisterAvroReaderFactory() {
#ifdef VELOX_ENABLE_AVRO
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::AVRO);
#endif
}

} // namespace facebook::velox::avro

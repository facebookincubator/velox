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

#include "velox/dwio/orc/writer/OrcWriter.h"

namespace facebook::velox::orc {
std::unique_ptr<dwio::common::Writer> OrcWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<dwio::common::WriterOptions>& options) {
  auto dwrfOptions = std::dynamic_pointer_cast<dwrf::WriterOptions>(options);
  VELOX_CHECK_NOT_NULL(
      dwrfOptions, "DWRF writer factory expected a DWRF WriterOptions object.");
  return std::make_unique<dwrf::Writer>(
      std::move(sink), *dwrfOptions, dwio::common::FileFormat::ORC);
}

std::unique_ptr<dwio::common::WriterOptions>
OrcWriterFactory::createWriterOptions() {
  return std::make_unique<dwrf::WriterOptions>();
}

void registerOrcWriterFactory() {
  dwio::common::registerWriterFactory(std::make_shared<OrcWriterFactory>());
}

void unregisterOrcWriterFactory() {
  dwio::common::unregisterWriterFactory(dwio::common::FileFormat::ORC);
}
} // namespace facebook::velox::orc

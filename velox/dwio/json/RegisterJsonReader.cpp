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

#include "velox/dwio/json/RegisterJsonReader.h"
#include "velox/dwio/json/reader/JsonReader.h"

namespace facebook::velox::json {

std::unique_ptr<dwio::common::Reader> JsonReaderFactory::createReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options) {
  return std::make_unique<JsonReader>(options, std::move(input));
}

void registerJsonReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<JsonReaderFactory>());
}

void unregisterJsonReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::JSON);
}

} // namespace facebook::velox::json

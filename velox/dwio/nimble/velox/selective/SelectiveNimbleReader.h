/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::nimble {

class SelectiveNimbleReaderFactory : public velox::dwio::common::ReaderFactory {
 public:
  SelectiveNimbleReaderFactory()
      : ReaderFactory(velox::dwio::common::FileFormat::NIMBLE) {}

  std::unique_ptr<velox::dwio::common::Reader> createReader(
      std::unique_ptr<velox::dwio::common::BufferedInput>,
      const velox::dwio::common::ReaderOptions&) override;
};

inline void registerSelectiveNimbleReaderFactory() {
  velox::dwio::common::registerReaderFactory(
      std::make_shared<SelectiveNimbleReaderFactory>());
}

inline void unregisterSelectiveNimbleReaderFactory() {
  velox::dwio::common::unregisterReaderFactory(
      velox::dwio::common::FileFormat::NIMBLE);
}

} // namespace facebook::nimble

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
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"

namespace facebook::velox::nimble {

namespace detail {

/// Creates the batch (non-selective) NIMBLE reader, or nullptr when that
/// reader is not part of the build. Defined once per build flavour: the batch
/// reader depends on internal-only code, so only the internal build can supply
/// it.
std::unique_ptr<dwio::common::Reader> createBatchReader(
    const dwio::common::ReaderOptions& options,
    const std::shared_ptr<ReadFile>& readFile);

} // namespace detail

/// Registered against FileFormat::NIMBLE. Chooses between the selective reader
/// and the batch reader per read, based on
/// `ReaderOptions::selectiveNimbleReaderEnabled()`.
class NimbleReaderFactory : public dwio::common::ReaderFactory {
 public:
  NimbleReaderFactory() : ReaderFactory(dwio::common::FileFormat::NIMBLE) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions&) override;

 private:
  facebook::nimble::SelectiveNimbleReaderFactory selectiveFactory_;
};

void registerNimbleReaderFactory();

void unregisterNimbleReaderFactory();

} // namespace facebook::velox::nimble

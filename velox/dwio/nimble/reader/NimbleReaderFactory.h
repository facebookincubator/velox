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

/// Creates the batch (non-selective) NIMBLE reader. Defined in
/// NimbleReader.cpp.
std::unique_ptr<dwio::common::Reader> createBatchReader(
    const dwio::common::ReaderOptions& options,
    const std::shared_ptr<ReadFile>& readFile,
    const FileIoContext& fileIoContext);

} // namespace detail

/// Registered against FileFormat::NIMBLE. Chooses between the selective reader
/// and the batch reader per read, based on
/// `ReaderOptions::selectiveNimbleReaderEnabled()`.
class NimbleReaderFactory : public dwio::common::ReaderFactory {
 public:
  /// Passes 'tabletReaderCache' to the selective reader factory, which shares
  /// tablets among the readers that load the cluster index. See
  /// facebook::nimble::SelectiveNimbleReaderFactory.
  explicit NimbleReaderFactory(
      facebook::nimble::TabletReaderCache* tabletReaderCache = nullptr)
      : ReaderFactory(dwio::common::FileFormat::NIMBLE),
        selectiveFactory_{tabletReaderCache} {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions&) override;

 private:
  facebook::nimble::SelectiveNimbleReaderFactory selectiveFactory_;
};

/// Registers a NimbleReaderFactory with 'tabletReaderCache'. Does nothing if a
/// Nimble reader factory is already registered.
void registerNimbleReaderFactory(
    facebook::nimble::TabletReaderCache* tabletReaderCache = nullptr);

void unregisterNimbleReaderFactory();

} // namespace facebook::velox::nimble

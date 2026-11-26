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

class ReaderBase;
class TabletReaderCache;

/// Creates selective Nimble readers.
class SelectiveNimbleReaderFactory : public velox::dwio::common::ReaderFactory {
 public:
  /// With 'tabletReaderCache' set, the readers that load the cluster index,
  /// such as index lookups, share one TabletReader per file from the cache.
  /// The shared tablet keeps the file metadata and the cluster index parsed in
  /// memory, so a lookup on a file the cache holds reads and parses no
  /// metadata. Readers that do not load the cluster index are unaffected.
  ///
  /// The cache keys tablets by file name and builds each one with the options
  /// of the first reader of its file. A file name must therefore always refer
  /// to the same contents, and the readers of one file must use compatible
  /// options. The cache must outlive every reader this factory creates.
  explicit SelectiveNimbleReaderFactory(
      TabletReaderCache* tabletReaderCache = nullptr)
      : ReaderFactory(velox::dwio::common::FileFormat::NIMBLE),
        tabletReaderCache_{tabletReaderCache} {}

  std::unique_ptr<velox::dwio::common::Reader> createReader(
      std::unique_ptr<velox::dwio::common::BufferedInput>,
      const velox::dwio::common::ReaderOptions&) override;

  /// Creates a reader sharing already loaded metadata and its input. Throws if
  /// 'readerBase' is null.
  std::unique_ptr<velox::dwio::common::Reader> createReader(
      std::shared_ptr<ReaderBase> readerBase,
      const velox::dwio::common::ReaderOptions& options);

 private:
  // Shares tablets among the readers that load the cluster index. Null shares
  // none.
  TabletReaderCache* const tabletReaderCache_;
};

/// Registers a SelectiveNimbleReaderFactory with 'tabletReaderCache'. Does
/// nothing if a Nimble reader factory is already registered.
inline void registerSelectiveNimbleReaderFactory(
    TabletReaderCache* tabletReaderCache = nullptr) {
  velox::dwio::common::registerReaderFactory(
      std::make_shared<SelectiveNimbleReaderFactory>(tabletReaderCache));
}

inline void unregisterSelectiveNimbleReaderFactory() {
  velox::dwio::common::unregisterReaderFactory(
      velox::dwio::common::FileFormat::NIMBLE);
}

} // namespace facebook::nimble

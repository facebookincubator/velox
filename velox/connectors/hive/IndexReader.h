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
#pragma once

#include <mutex>
#include <unordered_map>

#include <folly/CPortability.h>

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive {

struct HiveConnectorSplit;

/// Abstract interface for format-specific index readers.
///
/// Provides the customization point for storage formats to plug into
/// HiveIndexSource. Each format implements this interface
/// with its own I/O and decoding logic. HiveIndexSource owns all
/// format-agnostic orchestration (partition routing, remaining filter
/// evaluation, non-index condition filtering, output projection) and delegates
/// storage-specific reads to SplitIndexReader implementations.
///
/// File-based readers (e.g., Nimble) can internally use ScanSpec and filter
/// pushdown without leaking those concepts through this interface.
class SplitIndexReader {
 public:
  using Request = IndexSource::Request;
  using Result = IndexSource::Result;
  using Options = dwio::common::IndexReader::Options;

  /// The total number of output rows returned across all next() calls.
  static constexpr std::string_view kNumIndexReaderOutputRows{
      "numIndexReaderOutputRows"};

  virtual ~SplitIndexReader() = default;

  /// Initializes a lookup for the given probe request.
  ///
  /// The reader handles format-specific I/O (e.g., file reads for Nimble,
  /// network RPCs for remote stores). HiveIndexSource handles everything above
  /// this level (filters, projection, partition routing).
  ///
  /// After calling startLookup(), the caller iterates results via hasNext()
  /// and next().
  ///
  /// @param request The probe-side input rows containing lookup keys.
  /// @param options Lookup options (e.g., max rows per request batch).
  virtual void startLookup(
      const Request& request,
      const Options& options = {}) = 0;

  /// Returns true if there are more result batches to read.
  virtual bool hasNext() = 0;

  /// Returns the next batch of results, or nullptr if no more results.
  ///
  /// @param maxOutputRows Maximum number of rows to return in this batch.
  virtual std::unique_ptr<Result> next(vector_size_t maxOutputRows) = 0;

  /// Returns runtime statistics collected by this reader.
  virtual std::unordered_map<std::string, RuntimeMetric> runtimeStats() = 0;
};

/// Factory function type for creating IndexReader instances.
///
/// Creates one IndexReader per split during HiveIndexSource::addSplits().
/// Receives all the context needed to set up a reader for the given storage
/// format.
///
/// @param split The split to create the reader for.
/// @param tableHandle The table handle containing table metadata.
/// @param connectorQueryCtx Query context (memory pool, session config, etc.).
/// @return A unique_ptr to the created IndexReader.
using IndexReaderFactory = std::function<std::unique_ptr<SplitIndexReader>(
    const std::shared_ptr<const HiveConnectorSplit>& split,
    const ConnectorTableHandlePtr& tableHandle,
    ConnectorQueryCtx* connectorQueryCtx)>;

/// Thread-safe registry for IndexReaderFactory instances keyed by file format.
///
/// External storage formats register their reader
/// factories at application startup. HiveIndexSource consults this registry
/// during addSplits() to find the appropriate reader for each split's file
/// format.
///
/// Example registration:
///   IndexReaderFactoryRegistry::getInstance()->registerFactory(
///       FileFormat::DWRF, myDwrfReaderFactory);
class IndexReaderFactoryRegistry {
 public:
  FOLLY_EXPORT static IndexReaderFactoryRegistry* getInstance() {
    static IndexReaderFactoryRegistry instance;
    return &instance;
  }

  /// Registers a factory for the given file format. Throws if a factory is
  /// already registered for the format.
  void registerFactory(
      dwio::common::FileFormat format,
      IndexReaderFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = factories_.emplace(format, std::move(factory));
    VELOX_CHECK(
        inserted,
        "IndexReaderFactory already registered for format: {}",
        dwio::common::toString(format));
  }

  /// Unregisters the factory for the given file format. Returns true if a
  /// factory was removed, false if none was registered.
  bool unregisterFactory(dwio::common::FileFormat format) {
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.erase(format) > 0;
  }

  /// Returns the registered factory for the given format, or nullptr if none
  /// is registered.
  const IndexReaderFactory* getFactory(dwio::common::FileFormat format) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(format);
    return it != factories_.end() ? &it->second : nullptr;
  }

 private:
  IndexReaderFactoryRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<dwio::common::FileFormat, IndexReaderFactory> factories_;
};

} // namespace facebook::velox::connector::hive

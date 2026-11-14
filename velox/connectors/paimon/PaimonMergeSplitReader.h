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

#include "velox/connectors/paimon/PaimonSplitReader.h"

namespace facebook::velox::connector::paimon {

/// Merge-on-read reader for primary-key Paimon splits that are not
/// rawConvertible, whose files may hold several rows of a key.
///
/// The first next() merges the keys of all files. It reads the trimmed primary
/// keys, sequence fields, _SEQUENCE_NUMBER and _VALUE_KIND of every file,
/// orders the rows of each key by sequence fields and then by sequence number
/// as Paimon Java does, and picks the row the merge engine returns: the last
/// for deduplicate, the first for first-row. A key whose picked row is a
/// DELETE or UPDATE_BEFORE returns nothing, and rows deleted by a deletion
/// vector take no part. The files are then read one by one as for raw splits,
/// with every row that was not picked deleted, so filters, sampling and lazy
/// loading see merged rows only. Paimon Java pushes only key filters below its
/// merge, so filtering merged rows returns the same result.
///
/// Supports the deduplicate and first-row merge engines.
class PaimonMergeSplitReader : public PaimonSplitReader {
 public:
  PaimonMergeSplitReader(
      const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
      std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
      const std::shared_ptr<const PaimonTableHandle>& tableHandle,
      const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
          partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const connector::FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  uint64_t next(uint64_t size, VectorPtr& output) override;

 protected:
  const uint64_t* keptRows() const override;

 private:
  // Merges the keys of all files into keptRows_, reading 'batchSize' rows of
  // a file at a time.
  void mergeKeys(uint64_t batchSize);

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;

  // One bitmap per data file with a bit per row, set for the rows the split
  // returns. Empty until the keys are merged.
  std::vector<BufferPtr> keptRows_;
};

} // namespace facebook::velox::connector::paimon

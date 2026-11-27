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
/// The first next() merges the keys of all files. It reads the primary keys
/// that are not partition columns, the sequence fields, _SEQUENCE_NUMBER and
/// _VALUE_KIND of every file, orders the rows of each key by sequence fields
/// and then by sequence number as Paimon Java does, and picks the row the
/// merge engine returns: the last for deduplicate, the first for first-row.
/// When compaction looks up keys, deduplicate considers only the rows of level
/// 0 and the first row of the lowest level above 0, as Paimon Java's
/// LookupMergeFunction does. A key whose picked row is a DELETE or
/// UPDATE_BEFORE returns nothing. Rows deleted by a deletion vector take no
/// part, as Paimon Java applies the deletion vector of each file before it
/// merges. The picked rows of each file form its row selection. The files are
/// then read one by one as for raw splits, with the rows outside the selection
/// deleted, so filters, sampling and lazy loading see merged rows only. Paimon
/// Java pushes only key filters below its merge, so filtering merged rows
/// returns the same result.
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
  const uint64_t* rowSelection() const override;

 private:
  // Merges the keys of all files into rowSelections_, reading 'batchSize' rows
  // of a file at a time.
  void mergeKeys(uint64_t batchSize);

  // Split whose data files the merge reads.
  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;
  // Primary keys and merge options of the table.
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;
  // True to read _SEQUENCE_NUMBER and _VALUE_KIND under lowercase names. DWRF,
  // ORC and Parquet readers read file column names as lowercase when the
  // session asks for it, while Nimble readers keep the names as written.
  const bool lowerCaseSystemColumns_;

  // Row selection of each data file, as rowSelection() returns it. Empty until
  // the keys are merged.
  std::vector<BufferPtr> rowSelections_;
};

} // namespace facebook::velox::connector::paimon

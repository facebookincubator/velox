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

#include "velox/connectors/paimon/PaimonMergeFunction.h"
#include "velox/connectors/paimon/PaimonSplitReader.h"

namespace facebook::velox::connector::paimon {

/// Merge-on-read reader for primary-key splits of the aggregation merge engine
/// that are not rawConvertible. Merges the rows of each key into one row by
/// column with PaimonMergeFunction.
///
/// Reads the merge columns and every column the scan reads from all files at
/// once, in merge order as PaimonMergeSplitReader does, and buffers the rows of
/// a key. As in Paimon Java, a key with one row returns the row unmerged. A key
/// with more rows passes to the merge function the rows the lookup rule keeps
/// when compaction looks up keys, as Paimon Java's LookupMergeFunction does.
/// A key whose single or merged row is a DELETE or UPDATE_BEFORE is dropped,
/// as Paimon Java's DropDeleteReader does, and rows deleted by a deletion
/// vector take no part. The merged rows then take the filters, sampling and
/// partition values of the scan. Paimon Java pushes only key filters below its
/// merge, so filtering merged rows returns the same result.
class PaimonFieldMergeSplitReader : public PaimonSplitReader {
 public:
  PaimonFieldMergeSplitReader(
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

  /// Merges up to 'size' keys and returns their rows that pass the filters in
  /// 'output'. Returns the number of file rows merged, which is 0 only at the
  /// end of the split.
  uint64_t next(uint64_t size, VectorPtr& output) override;

 private:
  // A row of the current key. Holding its batch keeps the row readable after
  // its stream moves on.
  struct KeyRow {
    RowVectorPtr batch;
    vector_size_t row;
    int32_t level;
    PaimonRowKind kind;
  };

  // Sets the partition values of the scan and opens the file streams, which
  // read 'batchSize' rows at a time. Returns false if a partition value fails
  // its filter, so the split returns no row.
  bool open(uint64_t batchSize);

  // Writes the row of the current key, if any, to row numOutputRows_ of
  // 'output', and clears the rows of the key.
  void finishKey(RowVector& output);

  // Returns the row of the current key the lookup rule keeps from the levels
  // above 0: the first row of the lowest such level. Returns nullptr if the
  // key has no row above level 0.
  const KeyRow* pickHighLevelRow() const;

  // Copies row 'row' of 'batch' to row numOutputRows_ of 'output'. Copies of
  // consecutive rows of a batch run together in flushCopies().
  void copyRow(const RowVectorPtr& batch, vector_size_t row, RowVector& output);

  // Runs the pending copies of copyRow().
  void flushCopies(RowVector& output);

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;
  // Columns the scan reads that the merge reads after the merge columns: all
  // but the partition columns, which take their values from the split, and the
  // merge columns. The merge function builds each of them from several rows of
  // a key, so the merge reads them with the keys.
  const RowTypePtr valueColumns_;
  // Layout of the rows of the file streams and of the merged rows: the merge
  // columns and then 'valueColumns_'.
  const PaimonMergeLayout layout_;
  // Merges every column of 'layout_' but _SEQUENCE_NUMBER and _VALUE_KIND.
  const std::unique_ptr<PaimonMergeFunction> mergeFunction_;

  // Null until the first next().
  std::unique_ptr<TreeOfLosers<PaimonMergeStream>> merge_;
  bool atEnd_{false};

  std::vector<KeyRow> keyRows_;
  vector_size_t numOutputRows_{0};
  // Batch and ranges of the pending copies of copyRow().
  RowVectorPtr copySource_;
  std::vector<BaseVector::CopyRange> copyRanges_;
};

} // namespace facebook::velox::connector::paimon

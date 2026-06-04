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

#include <memory>
#include <string>
#include <vector>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"

namespace facebook::velox::connector::hive::iceberg {

/// V3 deletion-vector sink. Consumes (file_path, pos) rows produced by the
/// coordinator's DELETE plan and emits one Puffin file per referenced data
/// file. Each Puffin file contains a single deletion-vector-v1 blob holding
/// the deleted positions encoded as a 64-bit roaring bitmap.
///
/// Selected by IcebergConnector::createDataSink when the incoming
/// IcebergInsertTableHandle has writeKind() == kDeletionVector. Native
/// execution dispatches to this sink for V3 tables; V2 tables continue to
/// flow through the row-id-rewrite path on the coordinator and never reach
/// this sink.
///
/// Commit messages emitted by close() follow the existing Iceberg connector
/// CommitTaskData JSON contract with the V3 additions:
///   {
///     "path": "<puffin file path>",
///     "fileSizeInBytes": <total file size>,
///     "metrics": {"recordCount": <numPositions>},
///     "partitionSpecJson": 0,
///     "fileFormat": "PUFFIN",
///     "referencedDataFile": "<data file path>",
///     "content": "POSITION_DELETES",
///     "contentOffset": <blob offset within puffin>,
///     "contentSizeInBytes": <blob length>
///   }
class IcebergDeletionVectorSink : public DataSink {
 public:
  IcebergDeletionVectorSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy);

  void appendData(RowVectorPtr input) override;

  bool finish() override;

  std::vector<std::string> close() override;

  void abort() override;

  Stats stats() const override;

  // Per-referenced-data-file accumulator. Builds a roaring bitmap from
  // incoming positions; flushed to a Puffin file in close(). Public so the
  // helpers in IcebergDeletionVectorSink.cpp's anonymous namespace can
  // construct one.
  struct PerFileState {
    DeletionVectorWriter writer;
  };

 private:
  // Resolves the puffin output path for 'dataFile'. Today returns
  // "<base location>/<uuid>.puffin"; in production this should be folded
  // through the same LocationHandle / FileNameGenerator path that
  // IcebergDataSink uses for data files so the puffin lands next to the data.
  std::string puffinPathFor(const std::string& dataFile) const;

  const RowTypePtr inputType_;
  const IcebergInsertTableHandlePtr insertTableHandle_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const CommitStrategy commitStrategy_;

  // Indexed by referenced data file path; preserves insertion order so
  // commit messages come back deterministically.
  std::vector<std::pair<std::string, PerFileState>> perFile_;

  // Cached commit messages produced by close(); populated once and returned
  // on each close() call to match the DataSink contract.
  std::vector<std::string> commitMessages_;

  bool finished_{false};
  bool aborted_{false};

  // Cumulative stats for the Stats() accessor.
  uint64_t numWrittenBytes_{0};
  uint32_t numWrittenFiles_{0};
};

} // namespace facebook::velox::connector::hive::iceberg

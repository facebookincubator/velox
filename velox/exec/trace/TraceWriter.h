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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec {
struct Split;
}

namespace facebook::velox::core {
class PlanNode;
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec::trace {

/// Abstract interface for capturing traced input. Implementations are
/// responsible for serializing and/or storing row batches during query
/// execution tracing, along with associated metadata and summaries.
class TraceInputWriter {
 public:
  virtual ~TraceInputWriter() = default;

  /// Serializes rows and writes out each batch. Return whether the driver
  /// should block the pipeline. If it returns true, a future needs to be set
  /// (returned) to signal the driver when it can resume execution.
  virtual bool write(const RowVectorPtr& rows, ContinueFuture* future) = 0;

  /// Closes the data file and writes out the data summary.
  virtual void finish() = 0;
};

/// Abstract interface for capturing traced split information. Implementations
/// are responsible for processing and/or recording the splits found during
/// query execution tracing, enabling replay and analysis of query input
/// patterns.
class TraceSplitWriter {
 public:
  virtual ~TraceSplitWriter() = default;

  virtual void write(const exec::Split& split) const = 0;

  virtual void finish() = 0;
};

/// Abstract interface for capturing task metadata.
class TraceMetadataWriter {
 public:
  virtual ~TraceMetadataWriter() = default;

  virtual void write(
      const core::QueryCtx& queryCtx,
      const core::PlanNode& planNode) = 0;
};

} // namespace facebook::velox::exec::trace

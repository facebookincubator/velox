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

#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/trace/TraceCtx.h"

namespace facebook::velox::exec::trace {

class TraceInputWriter;
class TraceSplitWriter;

/// The callback used to update and aggregate the trace bytes of a query. If the
/// query trace limit is set, the callback return true if the aggregate traced
/// bytes exceed the set limit otherwise return false.
using UpdateAndCheckTraceLimitCB = std::function<void(uint64_t)>;

class OperatorTraceCtx : public TraceCtx {
 public:
  OperatorTraceCtx(
      std::string queryNodeId,
      std::string queryTraceDir,
      UpdateAndCheckTraceLimitCB updateAndCheckTraceLimitCB,
      std::string taskRegExp,
      bool dryRun);

  static std::unique_ptr<OperatorTraceCtx> maybeCreate(
      core::QueryCtx& queryCtx,
      const core::PlanFragment& planFragment,
      const std::string& taskId);

  bool shouldTrace(const Operator& op) const override;

  std::unique_ptr<TraceInputWriter> createInputTracer(
      Operator& op) const override;

  std::unique_ptr<TraceSplitWriter> createSplitTracer(
      Operator& op) const override;

  std::unique_ptr<TraceMetadataWriter> createMetadataTracer() const override;

 private:
  /// Target query trace node id.
  const std::string queryNodeId_;

  /// Base dir of query trace.
  const std::string queryTraceDir_;

  /// The trace task regexp.
  const std::string taskRegExp_;

  UpdateAndCheckTraceLimitCB updateAndCheckTraceLimitCB_;
};

} // namespace facebook::velox::exec::trace

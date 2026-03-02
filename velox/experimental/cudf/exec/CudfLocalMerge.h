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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/MergeSource.h"
#include "velox/exec/Operator.h"

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated LocalMerge operator. Replaces the CPU LocalMerge by
/// draining all MergeSource queues (which carry CudfVectors as RowVectorPtrs),
/// then performing a k-way sorted merge via cudf::merge() on the GPU.
///
/// Inherits from SourceOperator (not from Merge) because the base Merge class's
/// SourceMerger/TreeOfLosers/SourceStream all require CPU child vectors that
/// CudfVector does not have.
class CudfLocalMerge : public exec::SourceOperator, public NvtxHelper {
 public:
  CudfLocalMerge(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::LocalMergeNode>& localMergeNode);

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  void close() override;

 private:
  void addMergeSources();

  std::vector<std::shared_ptr<exec::MergeSource>> sources_;
  bool sourcesAdded_{false};
  bool sourcesStarted_{false};

  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::order> columnOrder_;
  std::vector<cudf::null_order> nullOrder_;

  std::unique_ptr<CudaEvent> cudaEvent_;
  /// Per-source accumulated batches. sourceData_[i] holds all CudfVectors
  /// received from sources_[i].
  std::vector<std::vector<CudfVectorPtr>> sourceData_;
  /// Per-source done flag. True when sources_[i] has been fully drained.
  std::vector<bool> sourceDone_;

  std::vector<ContinueFuture> blockingFutures_;
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox

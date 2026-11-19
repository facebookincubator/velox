/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/experimental/cudf/exec/CudfOperator.h"

namespace facebook::velox::exec {
class VectorSearchJoinProbe;
class VectorSearchJoinTable;
} // namespace facebook::velox::exec

namespace facebook::velox::cudf_velox {

class CudfVectorSearchJoinProbe final : public CudfOperatorBase {
 public:
  CudfVectorSearchJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      core::VectorSearchJoinNodePtr node);

  ~CudfVectorSearchJoinProbe() override;

  bool needsInput() const override;
  bool isFinished() override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void prepareSlice();
  void prepareMatchTile();
  bool fillResults();
  void collectUnmatched();
  RowVectorPtr makeOutput();
  void releaseInput();

  const core::VectorSearchJoinNodePtr node_;
  column_index_t embeddingChannel_;
  std::vector<int32_t> probeKeyChannels_;
  std::vector<int32_t> probeColumns_;
  std::vector<int32_t> buildColumns_;
  int64_t maxK_{1};
  std::shared_ptr<const exec::VectorSearchJoinTable> table_;
  std::unique_ptr<exec::VectorSearchJoinProbe> fallback_;
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace facebook::velox::cudf_velox

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

#include "velox/exec/Delete.h"
#include "velox/exec/CommitAwareOperator.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

Delete::Delete(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::DeleteNode>& deleteNode)
    : Operator(
          driverCtx,
          deleteNode->outputType(),
          operatorId,
          deleteNode->id(),
          "Delete"),
      rowIdChannel_{exprToChannel(
          deleteNode->rowId().get(),
          deleteNode->sources()[0]->outputType())},
      numDeletedRows_(0),
      finished_(false),
      closed_(false) {
  VELOX_CHECK_NE(
      rowIdChannel_,
      kConstantChannel,
      "Delete doesn't allow constant row id channel");
}

void Delete::addInput(RowVectorPtr input) {
  if (!updatableDataSourceSupplier_) {
    auto updatableDataSourceSupplier =
        operatorCtx_->driverCtx()->driver->updatableDataSourceSupplier();
    VELOX_CHECK_NOT_NULL(
        updatableDataSourceSupplier, "UpdatableDataSource is missing");
    updatableDataSourceSupplier_ = updatableDataSourceSupplier;
  }

  const auto& updatable = updatableDataSourceSupplier_();
  if (updatable.has_value()) {
    const auto& ids = input->childAt(rowIdChannel_);
    updatable.value()->deleteRows(ids);
    numDeletedRows_ += ids->size();
  }
}

RowVectorPtr Delete::getOutput() {
  // Making sure the output is read only once after the deletion is fully done
  if (!noMoreInput_ || finished_) {
    return nullptr;
  }
  finished_ = true;

  std::vector<VectorPtr> columns;
  CommitAwareOperator::buildDMLOutput(
      numDeletedRows_,
      operatorCtx_->driverCtx(),
      pool(),
      columns,
      outputType_->size() > 1);

  return std::make_shared<RowVector>(
      pool(), outputType_, BufferPtr(nullptr), 1, columns);
}

} // namespace facebook::velox::exec

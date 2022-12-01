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
#include "velox/exec/ArrowStream.h"
#include "velox/vector/arrow/Abi.h"

namespace facebook::velox::exec {

ArrowStream::ArrowStream(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::ArrowStreamNode> arrowStream)
    : SourceOperator(
          driverCtx,
          arrowStream->outputType(),
          operatorId,
          arrowStream->id(),
          "Arrow Stream") {
  arrowStream_ = arrowStream->arrowStream();
  pool_ = arrowStream->memoryPool();
}

ArrowStream::~ArrowStream() {
  if (!isFinished0()) {
    close0();
  }
}

RowVectorPtr ArrowStream::getOutput() {
  struct ArrowArray arrowArray;
  if (arrowStream_->get_next(&(*arrowStream_), &arrowArray)) {
    VELOX_FAIL(
        "Failed to call get_next on ArrowStream: " + std::string(GetError()))
  }
  if (arrowArray.release == NULL) {
    // End of Stream.
    closed_ = true;
    return nullptr;
  }
  struct ArrowSchema arrowSchema;
  if (arrowStream_->get_schema(&(*arrowStream_), &arrowSchema)) {
    VELOX_FAIL(
        "Failed to call get_schema on ArrowStream: " + std::string(GetError()))
  }
  // Convert Arrow data into RowVector.
  rowVector_ = std::dynamic_pointer_cast<RowVector>(
      facebook::velox::importFromArrowAsOwner(arrowSchema, arrowArray, pool_));
  return rowVector_;
}

const char* ArrowStream::GetError() {
  return arrowStream_->get_last_error(arrowStream_.get());
}

void ArrowStream::close() {
  close0();
  SourceOperator::close();
}

bool ArrowStream::isFinished() {
  return isFinished0();
}

void ArrowStream::close0() {
  arrowStream_->release(arrowStream_.get());
  closed_ = true;
}

bool ArrowStream::isFinished0() {
  return closed_;
}

} // namespace facebook::velox::exec

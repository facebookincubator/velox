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

#include <folly/String.h>

#include "velox/exec/OperatorType.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::exec {

ArrowStream::ArrowStream(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ArrowStreamNode>& arrowStreamNode)
    : SourceOperator(
          driverCtx,
          arrowStreamNode->outputType(),
          operatorId,
          arrowStreamNode->id(),
          OperatorType::kArrowStream) {
  arrowStream_ = arrowStreamNode->arrowStream();
}

ArrowStream::~ArrowStream() {
  close();
}

RowVectorPtr ArrowStream::getOutput() {
  // Get Arrow array. Zero-initialize it, and the schema below, because a
  // producer may return an error without writing to its output.
  struct ArrowArray arrowArray{};
  if (const int errorCode =
          arrowStream_->get_next(arrowStream_.get(), &arrowArray);
      errorCode != 0) {
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
    VELOX_FAIL(
        "Failed to call get_next on ArrowStream: {}", getError(errorCode));
  }
  if (arrowArray.release == nullptr) {
    // End of Stream.
    finished_ = true;
    return nullptr;
  }

  // Get Arrow schema.
  struct ArrowSchema arrowSchema{};
  if (const int errorCode =
          arrowStream_->get_schema(arrowStream_.get(), &arrowSchema);
      errorCode != 0) {
    if (arrowSchema.release) {
      arrowSchema.release(&arrowSchema);
    }
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
    VELOX_FAIL(
        "Failed to call get_schema on ArrowStream: {}", getError(errorCode));
  }

  // Convert Arrow Array into RowVector and return.
  return std::dynamic_pointer_cast<RowVector>(
      importFromArrowAsOwner(arrowSchema, arrowArray, pool()));
}

bool ArrowStream::isFinished() {
  return finished_;
}

std::string ArrowStream::getError(int errorCode) const {
  const char* lastError = arrowStream_->get_last_error(arrowStream_.get());
  if (lastError == nullptr) {
    return folly::errnoStr(errorCode);
  }
  return lastError;
}

void ArrowStream::close() {
  if (arrowStream_->release) {
    arrowStream_->release(arrowStream_.get());
  }
  SourceOperator::close();
}

} // namespace facebook::velox::exec

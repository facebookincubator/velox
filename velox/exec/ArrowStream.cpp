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

namespace facebook::velox::exec {

ArrowStream::ArrowStream(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ArrowStreamNode>& arrowStream)
    : SourceOperator(
          driverCtx,
          arrowStream->outputType(),
          operatorId,
          arrowStream->id(),
          "Arrow Stream") {
  arrowStream_ = arrowStream->arrowStream();

  // Get arrow schema.
  if (arrowStream_->get_schema(arrowStream_.get(), &arrowSchema_)) {
    VELOX_FAIL(
        "Failed to call get_schema on ArrowStream: " + std::string(getError()))
  }
}

ArrowStream::~ArrowStream() {
  if (!isFinished()) {
    close();
  }
}

RowVectorPtr ArrowStream::getOutput() {
  struct ArrowArray arrowArray;
  if (arrowStream_->get_next(arrowStream_.get(), &arrowArray)) {
    VELOX_FAIL(
        "Failed to call get_next on ArrowStream: " + std::string(getError()))
  }
  if (arrowArray.release == NULL) {
    // End of Stream.
    finished_ = true;
    return nullptr;
  }

  // Convert Arrow Array into RowVector and return.
  return std::dynamic_pointer_cast<RowVector>(
      facebook::velox::importFromArrowAsOwner(
          arrowSchema_, arrowArray, pool()));
}

void ArrowStream::close() {
  arrowStream_->release(arrowStream_.get());
  SourceOperator::close();
}

bool ArrowStream::isFinished() {
  return finished_;
}

} // namespace facebook::velox::exec

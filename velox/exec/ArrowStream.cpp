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
    std::shared_ptr<const core::ArrowStreamNode> arrowStream)
    : SourceOperator(
          driverCtx,
          arrowStream->outputType(),
          operatorId,
          arrowStream->id(),
          "Arrow Stream") {
  arrowStream_ = arrowStream->arrowStream();
}

RowVectorPtr ArrowStream::getOutput() {
  struct ArrowArray arrowArray;
  arrowStream_->get_next(&(*arrowStream_), &arrowArray);
  if (arrowArray.release == NULL) {
    // End of Stream.
    closed_ = true;
    return nullptr;
  }
  struct ArrowSchema arrowSchema;
  arrowStream_->get_schema(&(*arrowStream_), &arrowSchema);
  // Convert Arrow data into RowVector.
  rowVector_ = std::dynamic_pointer_cast<RowVector>(
      facebook::velox::importFromArrowAsViewer(arrowSchema, arrowArray));
  return rowVector_;
}

void ArrowStream::close() {
  closed_ = true;
}

bool ArrowStream::isFinished() {
  return closed_;
}

} // namespace facebook::velox::exec

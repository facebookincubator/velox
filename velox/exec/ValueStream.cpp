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
#include "velox/exec/ValueStream.h"
#include "velox/common/testutil/TestValue.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {

ValueStream::ValueStream(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::ValueStreamNode> valueStreamNode)
    : SourceOperator(
          driverCtx,
          valueStreamNode->outputType(),
          operatorId,
          valueStreamNode->id(),
          "ValueStream") {
  valueStream_ = valueStreamNode->rowVectorStream();
}

RowVectorPtr ValueStream::getOutput() {
  if (valueStream_->hasNext()) {
    return valueStream_->next();
  } else {
    finished_ = true;
    return nullptr;
  }
}

bool ValueStream::isFinished() {
  return finished_;
}

} // namespace facebook::velox::exec

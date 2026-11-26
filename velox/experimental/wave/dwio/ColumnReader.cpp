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

#include "velox/experimental/wave/dwio/ColumnReader.h"

namespace facebook::velox::wave {

void ColumnReader::makeOp(
    ReadStream* readStream,
    ColumnAction action,
    ColumnOp& op) {
  VELOX_CHECK(
      action == ColumnAction::kValues || action == ColumnAction::kFilter,
      "Only values and filters are supported");
  op.action = action;
  op.reader = this;
  // A column that is only filtered on has no operand and produces no values.
  if (operand_ != nullptr) {
    op.waveVector = readStream->operandVector(operand_->id, requestedType_);
  }
};

bool ColumnReader::hasNonNullFilter() const {
  return scanSpec_->filter() && !scanSpec_->filter()->testNull();
}

} // namespace facebook::velox::wave

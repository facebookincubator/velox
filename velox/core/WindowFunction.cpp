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
#include "velox/core/WindowFunction.h"

namespace facebook::velox::core {

WindowFunction::WindowFunction(WindowFunctionType windowFunctionType)
    : windowFunctionType_(windowFunctionType) {}

TypePtr WindowFunction::getOutputType() const {
  switch (windowFunctionType_) {
    case ROW_NUMBER:
      return BIGINT();
    default:
      VELOX_NYI();
  }
}

void WindowFunction::resetPartition() {
  switch (windowFunctionType_) {
    case ROW_NUMBER:
      windowState_.rowNumber = 1;
      break;
    default:
      VELOX_NYI();
  }
}

void WindowFunction::evaluate(VectorPtr flatOutput, vector_size_t row) {
  switch (windowFunctionType_) {
    case ROW_NUMBER:
      flatOutput->asFlatVector<int64_t>()->mutableRawValues()[row] =
          windowState_.rowNumber++;
      break;
    default:
      VELOX_NYI();
  }
}

} // namespace facebook::velox::core

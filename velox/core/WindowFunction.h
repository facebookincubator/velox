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

#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::core {

class WindowFunction {
 public:
  enum WindowFunctionType {
    ROW_NUMBER = 1,
  };

  WindowFunction(WindowFunctionType windowFunctionType);

  TypePtr getOutputType() const;

  void resetPartition();

  void evaluate(VectorPtr flatOutput, vector_size_t row);

 private:
  struct WindowState {
    int64_t rowNumber;
  };

  WindowFunctionType windowFunctionType_;
  WindowState windowState_;
};

} // namespace facebook::velox::core

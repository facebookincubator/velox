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

#include "velox/functions/remote/utils/restserver/RemoteFunctionRestHandler.h"

namespace facebook::velox::functions {

class RemoteStrTrimHandler : public RemoteFunctionRestHandler {
 public:
  RemoteStrTrimHandler(RowTypePtr inputTypes, TypePtr outputType)
      : RemoteFunctionRestHandler(
            std::move(inputTypes),
            std::move(outputType)) {}

 protected:
  void compute(const RowVectorPtr& inputVector, const VectorPtr& resultVector) {
    auto inputFlat = inputVector->childAt(0)->asFlatVector<StringView>();
    auto outFlat = resultVector->asFlatVector<StringView>();
    const auto numRows = inputVector->size();

    for (vector_size_t i = 0; i < numRows; ++i) {
      if (inputFlat->isNullAt(i)) {
        outFlat->setNull(i, true);
      } else {
        std::string result = inputFlat->valueAt(i).str();
        result.erase(
            std::remove_if(result.begin(), result.end(), ::isspace),
            result.end());
        outFlat->set(i, result.data());
      }
    }
  }
};

} // namespace facebook::velox::functions

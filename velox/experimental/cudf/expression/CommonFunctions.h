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

#include "velox/expression/FunctionSignature.h"

#include <initializer_list>

namespace facebook::velox::cudf_velox {

struct ArrayAccessPolicy {
  bool allowNegativeIndices;
  bool nullOnNegativeIndices;
  bool allowOutOfBound;
  bool indexStartsAtOne;
};

std::vector<exec::FunctionSignaturePtr> arrayAccessSignatures(
    std::initializer_list<const char*> indexTypes);

void registerArrayAccessFunction(
    const std::string& name,
    ArrayAccessPolicy policy,
    std::vector<exec::FunctionSignaturePtr> signatures);

} // namespace facebook::velox::cudf_velox

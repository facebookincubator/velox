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

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Macros.h"

// Forwarding the definitions here so that codegen can still use functions in
// this namespace.
namespace facebook::velox::functions {

template <typename T>
VELOX_GPU_COMPATIBLE T checkedPlus(const T& a, const T& b) {
  return facebook::velox::checkedPlus(a, b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedMinus(const T& a, const T& b) {
  return facebook::velox::checkedMinus(a, b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedMultiply(const T& a, const T& b) {
  return facebook::velox::checkedMultiply(a, b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedDivide(const T& a, const T& b) {
  return facebook::velox::checkedDivide(a, b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedModulus(const T& a, const T& b) {
  return facebook::velox::checkedModulus(a, b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedNegate(const T& a) {
  return facebook::velox::checkedNegate(a);
}

} // namespace facebook::velox::functions

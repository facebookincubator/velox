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

// Adapted from Apache DataSketches

#pragma once

#include "velox/common/base/Exceptions.h"
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace facebook::velox::common::theta {

static inline void ensureMinimumMemory(
    size_t bytes_available,
    size_t min_needed) {
  if (bytes_available < min_needed) {
    auto msg = "Insufficient buffer size detected: bytes available " +
        std::to_string(bytes_available) + ", minimum needed " +
        std::to_string(min_needed);
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        msg,
        error_source::kErrorSourceUser,
        error_code::kGenericUserError,
        false /*retriable*/);
  }
}

static inline void checkMemorySize(size_t requested_index, size_t capacity) {
  if (requested_index > capacity) {
    auto msg = "Attempt to access memory beyond limits: requested index " +
        std::to_string(requested_index) + ", capacity " +
        std::to_string(capacity);
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        msg,
        error_source::kErrorSourceUser,
        error_code::kGenericUserError,
        false /*retriable*/);
  }
}

// note: size is in bytes, not items
static inline size_t copyFromMem(const void* src, void* dst, size_t size) {
  memcpy(dst, src, size);
  return size;
}

// note: size is in bytes, not items
static inline size_t copyToMem(const void* src, void* dst, size_t size) {
  memcpy(dst, src, size);
  return size;
}

template <typename T>
static inline size_t copyFromMem(const void* src, T& item) {
  memcpy(&item, src, sizeof(T));
  return sizeof(T);
}

template <typename T>
static inline size_t copyToMem(T item, void* dst) {
  memcpy(dst, &item, sizeof(T));
  return sizeof(T);
}

} // namespace facebook::velox::common::theta

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

#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"
#include <cstdint>

namespace facebook::velox::gpu {

class GpuStringView {
 public:
  GPU_HOST_DEVICE GpuStringView() = default;
  GPU_HOST_DEVICE GpuStringView(const char* data, int32_t len)
      : data_(data), size_(len) {}

  GPU_HOST_DEVICE const char* data() const { return data_; }
  GPU_HOST_DEVICE int32_t size() const { return size_; }
  GPU_HOST_DEVICE bool empty() const { return size_ == 0; }
  GPU_HOST_DEVICE const char* begin() const { return data_; }
  GPU_HOST_DEVICE const char* end() const { return data_ + size_; }

  GPU_HOST_DEVICE bool operator==(const GpuStringView& o) const {
    if (size_ != o.size_) {
      return false;
    }
    for (int32_t i = 0; i < size_; ++i) {
      if (data_[i] != o.data_[i]) {
        return false;
      }
    }
    return true;
  }

  GPU_HOST_DEVICE bool operator!=(const GpuStringView& o) const {
    return !(*this == o);
  }

 private:
  const char* data_ = nullptr;
  int32_t size_ = 0;
};

} // namespace facebook::velox::gpu

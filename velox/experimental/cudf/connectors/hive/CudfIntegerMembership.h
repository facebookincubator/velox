/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace facebook::velox::cudf_velox::connector::hive {

/// Owns a GPU hash set for one integer filter, reused across probe batches.
/// Keys have no NULLs. emptyKey must be absent from keys; its probe is
/// rejected. Construction and use are stream ordered. The caller must finish
/// pending uses before destroying the set or using it on another stream.
class CudfIntegerHashSet {
 public:
  CudfIntegerHashSet(
      const cudf::column_view& keys,
      int64_t emptyKey,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);
  ~CudfIntegerHashSet();

  void apply(
      const cudf::column_view& input,
      bool nullAllowed,
      std::unique_ptr<cudf::column>& mask,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Intersects a non-nullable retention mask with an exact integer bitmap.
/// UINT32 filter words are based at minimum. Creates the mask if absent.
void applyIntegerBitmapToMask(
    const cudf::column_view& input,
    const cudf::column_view& filter,
    int64_t minimum,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Intersects the mask with an inclusive integer range, preserving NULL policy.
void applyIntegerRangeToMask(
    const cudf::column_view& input,
    int64_t lower,
    int64_t upper,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::connector::hive

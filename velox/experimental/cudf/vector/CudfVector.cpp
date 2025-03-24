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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cmath>

namespace facebook::velox::cudf_velox {
namespace {

static std::size_t estimateSize(
    cudf::column_view const& view,
    rmm::cuda_stream_view stream);

struct ColumnSizeEstimator {
  rmm::cuda_stream_view stream_;
  ColumnSizeEstimator(rmm::cuda_stream_view stream) : stream_(stream) {}
  // fixed width types
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  std::size_t operator()(cudf::column_view const& view) const {
    using storageT = cudf::device_storage_type_t<T>;
    auto bytes = view.size() * sizeof(storageT);
    if (view.nullable()) {
      bytes += cudf::bitmask_allocation_size_bytes(view.size());
    }
    return bytes;
  }
  // dictionary, string, list, struct
  template <
      typename T,
      std::enable_if_t<not cudf::is_fixed_width<T>()>* = nullptr>
  std::size_t operator()(cudf::column_view const& view) const {
    auto bytes = 0;
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      auto const strings_view = cudf::strings_column_view(view);
      auto const chars_size = strings_view.chars_size(stream_);
      bytes += chars_size;
    }
    auto num_children = view.num_children();
    for (auto i = 0; i < num_children; ++i) {
      // recursive call
      bytes += estimateSize(view.child(i), stream_);
    }
    if (view.nullable()) {
      bytes += cudf::bitmask_allocation_size_bytes(view.size());
    }
    return bytes;
  }
};

std::size_t estimateSize(
    cudf::column_view const& view,
    rmm::cuda_stream_view stream) {
  return cudf::type_dispatcher(view.type(), ColumnSizeEstimator{stream}, view);
}

static std::size_t estimateSize(
    cudf::table_view const& view,
    rmm::cuda_stream_view stream) {
  auto bytes = 0;
  for (auto const& column : view) {
    bytes += estimateSize(column, stream);
  }
  return bytes;
}

} // namespace

uint64_t CudfVector::estimateFlatSize() const {
  return estimateSize(table_->view(), stream_);
}

} // namespace facebook::velox::cudf_velox

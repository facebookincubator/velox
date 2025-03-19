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

#include <cudf/reduction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cmath>

static int64_t estimate_size(cudf::table_view const& view) {
  // Compute the size in bits for each row.
  auto const row_sizes = cudf::row_bit_count(view);
  // Accumulate the row sizes to compute a sum.
  auto const agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  cudf::data_type sum_dtype{cudf::type_id::INT64};
  auto const total_size_scalar = cudf::reduce(*row_sizes, *agg, sum_dtype);
  auto const total_size_in_bits =
      static_cast<cudf::numeric_scalar<int64_t>*>(total_size_scalar.get())
          ->value();
  // Convert the size in bits to the size in bytes.
  return static_cast<int64_t>(
      std::ceil(static_cast<double>(total_size_in_bits) / 8));
}

namespace facebook::velox::cudf_velox {

uint64_t CudfVector::estimateFlatSize() const {
  return estimate_size(table_->view());
}

} // namespace facebook::velox::cudf_velox

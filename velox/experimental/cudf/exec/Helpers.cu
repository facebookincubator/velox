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

#include "velox/experimental/cudf/exec/Helpers.h"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <vector>

namespace facebook::velox::cudf_velox {

void sort_join_indices_inplace(
    cudf::mutable_column_view leftJoinIndices,
    cudf::mutable_column_view rightJoinIndices,
    rmm::cuda_stream_view stream) {
  thrust::sort_by_key(
      rmm::exec_policy(stream),
      leftJoinIndices.begin<cudf::size_type>(),
      leftJoinIndices.end<cudf::size_type>(),
      rightJoinIndices.begin<cudf::size_type>());
}
// /// Hash table type
using hash_value_type =
    cudf::size_type; // from cudf/hashing.hpp: hash_value_type
using rhs_index_type = cudf::size_type;

using hasher = cuco::default_hash_function<hash_value_type>;
using probing_scheme_type = cuco::linear_probing<1, hasher>;
using cuco_storage_type = cuco::storage<1>;

using hash_table_type = cuco::static_set<
    hash_value_type,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<hash_value_type>,
    probing_scheme_type,
    cudf::detail::cuco_allocator<char>,
    cuco_storage_type>;

template <typename T>
void print_vector(
    const T* begin,
    int size,
    const std::string& name,
    rmm::cuda_stream_view stream) {
  return;
  using base_T = std::conditional_t<std::is_same_v<T, bool>, char, T>;
  std::vector<base_T> host_data(size, -1);
  cudaMemcpyAsync(
      host_data.data(),
      begin,
      size * sizeof(base_T),
      cudaMemcpyDefault,
      stream);
  stream.synchronize();
  std::cout << name << " = ";
  for (int i = 0; i < size; i++) {
    std::cout << static_cast<T>(host_data[i]) << " ";
  }
  std::cout << std::endl;
}

[[nodiscard]]
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
filtered_indices_again(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightIndices,
    cudf::mutable_column_view& filterColumn,
    rmm::cuda_stream_view stream) {
  auto mr = cudf::get_current_device_resource_ref();
  // if filter true, insert to static_set.
  // for all left indices, if static_set[index] is false, set right_index as
  // INT_MIN. then if right_index is INT_MIN, filterColumn[0] as true. apply
  // boolean mask on left, right indices and return left, right indices.
  hash_table_type hash_table{
      cuco::extent<std::size_t>{leftIndices->size()},
      1.0,
      cuco::empty_key{std::numeric_limits<hash_value_type>::min()},
      cuda::std::equal_to<hash_value_type>{},
      {},
      cuco::thread_scope_device,
      cuco_storage_type{},
      cudf::detail::cuco_allocator<char>{
          rmm::mr::polymorphic_allocator<char>{}, stream},
      cuda::stream_ref{stream.value()}};
  hash_table.insert_if_async(
      leftIndices->begin(),
      leftIndices->end(),
      filterColumn.begin<bool>(),
      cuda::std::identity{},
      stream.value());
  stream.synchronize();

  auto hash_table_ref = hash_table.ref(cuco::insert_and_find);
  thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<std::size_t>(0),
      thrust::make_counting_iterator<std::size_t>(rightIndices->size()),
      [hash_table = hash_table_ref,
       filterColumn = filterColumn.begin<bool>(),
       leftIndices = leftIndices->begin(),
       rightIndices = rightIndices->begin()] __device__(auto i) mutable {
        if (hash_table.insert_and_find(leftIndices[i]).second == true) {
          rightIndices[i] = std::numeric_limits<cudf::size_type>::min();
          filterColumn[i] = true;
        }
        if (rightIndices[i] == std::numeric_limits<cudf::size_type>::min()) {
          filterColumn[i] = true;
        }
      });
  auto leftIndicesCol =
      cudf::column_view{cudf::device_span<cudf::size_type const>{*leftIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightIndices}};
  auto filterTableView = cudf::table_view{
      std::vector<cudf::column_view>{leftIndicesCol, rightIndicesCol}};
  // Remove null mask, because they are made true already.
  auto nonNullFilterColumnView = cudf::column_view(
      cudf::data_type(cudf::type_id::BOOL8),
      filterColumn.size(),
      filterColumn.head<void>(),
      nullptr,
      0);

  auto filteredTable = cudf::apply_boolean_mask(
      filterTableView, nonNullFilterColumnView, stream);
  auto filteredColumns = filteredTable->release();
  return {std::move(filteredColumns[0]), std::move(filteredColumns[1])};
}
} // namespace facebook::velox::cudf_velox

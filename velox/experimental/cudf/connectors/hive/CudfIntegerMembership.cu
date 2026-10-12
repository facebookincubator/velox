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

#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <thrust/transform.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
enum class FilterKind { kRange, kBitmap };

template <typename T, FilterKind kKind>
struct IntegerMembership {
  const T* values;
  const cudf::bitmask_type* nulls;
  cudf::size_type offset;
  const void* filter;
  cudf::size_type filterSize;
  int64_t minimum;
  int64_t maximum;
  bool nullAllowed;
  bool* mask;
  bool initialize;

  __device__ bool operator()(cudf::size_type row) const {
    if (!initialize && !mask[row]) {
      return false;
    }
    if (nulls && !cudf::bit_is_set(nulls, row + offset)) {
      return nullAllowed;
    }
    if constexpr (kKind == FilterKind::kRange) {
      return values[row] >= minimum && values[row] <= maximum;
    } else {
      const auto index =
          static_cast<uint64_t>(values[row]) - static_cast<uint64_t>(minimum);
      const auto* bitmap = static_cast<const uint32_t*>(filter);
      return index < static_cast<uint64_t>(filterSize) * 32 &&
          (bitmap[index / 32] & (uint32_t{1} << (index % 32))) != 0;
    }
  }
};
template <FilterKind kKind>
void applyIntegerFilter(
    const cudf::column_view& input,
    const cudf::column_view& filter,
    int64_t minimum,
    int64_t maximum,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const bool initialize = !mask;
  if (initialize) {
    mask = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::BOOL8},
        input.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
  }
  auto output = mask->mutable_view();
  cudf::type_dispatcher(input.type(), [&]<typename T>() {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      const void* filterData = nullptr;
      if constexpr (kKind == FilterKind::kBitmap) {
        filterData = filter.data<uint32_t>();
      }
      auto rows = cuda::counting_iterator<cudf::size_type>{0};
      thrust::transform(
          rmm::exec_policy_nosync(stream, mr),
          rows,
          rows + input.size(),
          output.data<bool>(),
          IntegerMembership<T, kKind>{
              input.data<T>(),
              input.null_mask(),
              input.offset(),
              filterData,
              filter.size(),
              minimum,
              maximum,
              nullAllowed,
              output.data<bool>(),
              initialize});
    } else {
      CUDF_FAIL("Integer membership requires a signed integer column");
    }
  });
}

template <typename Key>
using IntegerSet = cuco::static_set<
    Key,
    cuco::extent<uint32_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<Key>,
    cuco::linear_probing<1, cuco::default_hash_function<Key>>,
    rmm::mr::polymorphic_allocator<Key>>;

template <typename T>
using HashKey = std::conditional_t<(sizeof(T) < sizeof(int32_t)), int32_t, T>;

// Hash slots use at least 32 bits; promotion happens in registers, not a
// column.
template <typename T>
struct ToHashKey {
  __host__ __device__ HashKey<T> operator()(T value) const {
    return static_cast<HashKey<T>>(value);
  }
};

template <typename T, typename SetRef>
struct HashMembership {
  const T* values;
  const cudf::bitmask_type* nulls;
  cudf::size_type offset;
  SetRef set;
  HashKey<T> emptyKey;
  bool nullAllowed;
  bool* mask;
  bool initialize;

  __device__ bool operator()(cudf::size_type row) const {
    if (!initialize && !mask[row]) {
      return false;
    }
    if (nulls && !cudf::bit_is_set(nulls, row + offset)) {
      return nullAllowed;
    }
    const auto key = static_cast<HashKey<T>>(values[row]);
    return key != emptyKey && set.contains(key);
  }
};
} // namespace

struct CudfIntegerHashSet::Impl {
  cudf::data_type type;
  int64_t emptyKey;
  std::unique_ptr<IntegerSet<int32_t>> set32;
  std::unique_ptr<IntegerSet<int64_t>> set64;

  template <typename Key>
  auto& storage() {
    if constexpr (std::is_same_v<Key, int32_t>) {
      return set32;
    } else {
      return set64;
    }
  }
};

CudfIntegerHashSet::CudfIntegerHashSet(
    const cudf::column_view& keys,
    int64_t emptyKey,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : impl_(std::make_unique<Impl>()) {
  CUDF_EXPECTS(
      keys.null_count() == 0, "Integer hash keys must not contain NULLs");
  impl_->type = keys.type();
  impl_->emptyKey = emptyKey;
  cudf::type_dispatcher(keys.type(), [&]<typename T>() {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      using Key = HashKey<T>;
      CUDF_EXPECTS(
          emptyKey >= std::numeric_limits<Key>::min() &&
              emptyKey <= std::numeric_limits<Key>::max(),
          "Empty key is outside hash storage type");
      auto& set = impl_->storage<Key>();
      set = std::make_unique<IntegerSet<Key>>(
          cuco::extent<uint32_t>{
              static_cast<uint32_t>(std::max(keys.size(), 1))},
          0.25,
          cuco::empty_key<Key>{static_cast<Key>(emptyKey)},
          cuda::std::equal_to<Key>{},
          cuco::linear_probing<1, cuco::default_hash_function<Key>>{},
          cuco::cuda_thread_scope<cuda::thread_scope_device>{},
          cuco::storage<1>{},
          rmm::mr::polymorphic_allocator<Key>{mr},
          stream);
      if (keys.size() > 0) {
        auto first =
            cuda::make_transform_iterator(keys.data<T>(), ToHashKey<T>{});
        set->insert_async(first, first + keys.size(), stream);
      }
    } else {
      CUDF_FAIL("Integer hash membership requires a signed integer column");
    }
  });
}

CudfIntegerHashSet::~CudfIntegerHashSet() = default;

void CudfIntegerHashSet::apply(
    const cudf::column_view& input,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  CUDF_EXPECTS(input.type() == impl_->type, "Integer hash probe type mismatch");
  const bool initialize = !mask;
  if (initialize) {
    mask = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::BOOL8},
        input.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
  }
  auto* output = mask->mutable_view().data<bool>();
  cudf::type_dispatcher(input.type(), [&]<typename T>() {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      using Key = HashKey<T>;
      auto ref = impl_->storage<Key>()->ref(cuco::contains);
      auto rows = cuda::counting_iterator<cudf::size_type>{0};
      thrust::transform(
          rmm::exec_policy_nosync(stream, mr),
          rows,
          rows + input.size(),
          output,
          HashMembership<T, decltype(ref)>{
              input.data<T>(),
              input.null_mask(),
              input.offset(),
              ref,
              static_cast<Key>(impl_->emptyKey),
              nullAllowed,
              output,
              initialize});
    } else {
      CUDF_FAIL("Integer hash membership requires a signed integer column");
    }
  });
}

void applyIntegerBitmapToMask(
    const cudf::column_view& input,
    const cudf::column_view& filter,
    int64_t minimum,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      filter.type().id() == cudf::type_id::UINT32,
      "Integer bitmap must contain UINT32 words");
  applyIntegerFilter<FilterKind::kBitmap>(
      input, filter, minimum, 0, nullAllowed, mask, stream, mr);
}

void applyIntegerRangeToMask(
    const cudf::column_view& input,
    int64_t lower,
    int64_t upper,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  applyIntegerFilter<FilterKind::kRange>(
      input, {}, lower, upper, nullAllowed, mask, stream, mr);
}
} // namespace facebook::velox::cudf_velox::connector::hive

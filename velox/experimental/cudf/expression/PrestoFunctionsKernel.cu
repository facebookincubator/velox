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

#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {
namespace {

template <typename T>
struct CheckedModulusColumnsFunctor {
  const T* lhs;
  const T* rhs;
  T* out;

  __device__ void operator()(int32_t idx) const {
    auto b = rhs[idx];
    out[idx] = (b == -1) ? 0 : (lhs[idx] % b);
  }
};

template <typename T>
struct CheckedModulusColumnScalarFunctor {
  const T* lhs;
  T rhs;
  T* out;

  __device__ void operator()(int32_t idx) const {
    out[idx] = (rhs == -1) ? 0 : (lhs[idx] % rhs);
  }
};

template <typename T>
struct CheckedModulusScalarColumnFunctor {
  T lhs;
  const T* rhs;
  T* out;

  __device__ void operator()(int32_t idx) const {
    auto b = rhs[idx];
    out[idx] = (b == -1) ? 0 : (lhs % b);
  }
};

template <typename T>
std::unique_ptr<cudf::column> integralCheckedModulusTyped(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream, mr);
  auto out = cudf::make_fixed_width_column(
      type, lhs.size(), std::move(nullMask), nullCount, stream, mr);
  CheckedModulusColumnsFunctor<T> functor{
      lhs.data<T>(), rhs.data<T>(), out->mutable_view().data<T>()};
  if (lhs.size() > 0) {
    cub::DeviceFor::ForEachN(
        thrust::counting_iterator<int32_t>(0),
        lhs.size(),
        functor,
        stream.value());
  }
  return out;
}

template <typename T>
std::unique_ptr<cudf::column> integralCheckedModulusTyped(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const& rhsScalar = static_cast<cudf::numeric_scalar<T> const&>(rhs);

  auto out = cudf::make_fixed_width_column(
      type,
      lhs.size(),
      cudf::copy_bitmask(lhs, stream, mr),
      lhs.null_count(),
      stream,
      mr);
  CheckedModulusColumnScalarFunctor<T> functor{
      lhs.data<T>(), rhsScalar.value(stream), out->mutable_view().data<T>()};
  if (lhs.size() > 0) {
    cub::DeviceFor::ForEachN(
        thrust::counting_iterator<int32_t>(0),
        lhs.size(),
        functor,
        stream.value());
  }
  return out;
}

template <typename T>
std::unique_ptr<cudf::column> integralCheckedModulusTyped(
    cudf::data_type type,
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const& lhsScalar = static_cast<cudf::numeric_scalar<T> const&>(lhs);

  auto out = cudf::make_fixed_width_column(
      type,
      rhs.size(),
      cudf::copy_bitmask(rhs, stream, mr),
      rhs.null_count(),
      stream,
      mr);
  CheckedModulusScalarColumnFunctor<T> functor{
      lhsScalar.value(stream), rhs.data<T>(), out->mutable_view().data<T>()};
  if (rhs.size() > 0) {
    cub::DeviceFor::ForEachN(
        thrust::counting_iterator<int32_t>(0),
        rhs.size(),
        functor,
        stream.value());
  }
  return out;
}

template <typename T>
T checkedModulus(T a, T b) {
  if (b == -1) {
    return 0;
  }
  return (a % b);
}

} // namespace

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Integral mod requires equal sizes");
  CUDF_EXPECTS(
      lhs.type().id() == rhs.type().id(),
      "Integral mod requires matching input types");
  switch (type.id()) {
    case cudf::type_id::INT8:
      return integralCheckedModulusTyped<int8_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT16:
      return integralCheckedModulusTyped<int16_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT32:
      return integralCheckedModulusTyped<int32_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT64:
      return integralCheckedModulusTyped<int64_t>(type, lhs, rhs, stream, mr);
    default:
      throw cudf::logic_error("Unsupported type for integral mod");
  }
}

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (type.id()) {
    case cudf::type_id::INT8:
      return integralCheckedModulusTyped<int8_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT16:
      return integralCheckedModulusTyped<int16_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT32:
      return integralCheckedModulusTyped<int32_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT64:
      return integralCheckedModulusTyped<int64_t>(type, lhs, rhs, stream, mr);
    default:
      throw cudf::logic_error("Unsupported type for integral mod");
  }
}

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (type.id()) {
    case cudf::type_id::INT8:
      return integralCheckedModulusTyped<int8_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT16:
      return integralCheckedModulusTyped<int16_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT32:
      return integralCheckedModulusTyped<int32_t>(type, lhs, rhs, stream, mr);
    case cudf::type_id::INT64:
      return integralCheckedModulusTyped<int64_t>(type, lhs, rhs, stream, mr);
    default:
      throw cudf::logic_error("Unsupported type for integral mod");
  }
}

} // namespace facebook::velox::cudf_velox

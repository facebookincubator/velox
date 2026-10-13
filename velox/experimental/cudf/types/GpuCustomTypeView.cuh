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

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

#include <type_traits>

namespace facebook::velox::gpu {

/// How a custom type with a custom comparison orders its physical values,
/// specialised per type tag. The CPU reaches the comparison through the
/// value's Type; a kernel has no Type, so the tag selects it at compile time.
/// A tag with no specialisation has no device comparison, and a function
/// comparing it does not compile.
template <typename Tag>
struct GpuCustomTypeComparison;

/// TIMESTAMP WITH TIME ZONE orders by instant, as
/// TimestampWithTimeZoneType::compare() does: two values naming one instant in
/// different zones are equal.
template <>
struct GpuCustomTypeComparison<TimestampWithTimezoneT> {
  GPU_HOST_DEVICE static int compare(int64_t left, int64_t right) {
    const int64_t leftMillis = unpackMillisUtc(left);
    const int64_t rightMillis = unpackMillisUtc(right);
    return leftMillis < rightMillis ? -1 : leftMillis == rightMillis ? 0 : 1;
  }
};

/// A custom type with a custom comparison as a GPU simple function sees it,
/// offering what exec::CustomTypeWithCustomComparisonView offers a CPU body:
/// the physical value behind operator*, and comparisons in the type's own
/// order. There is no conversion to the physical value: a generic comparison
/// such as EqFunction compares views, and must not fall through to the bits.
template <typename Tag>
class GpuCustomTypeView {
 public:
  using physical_type = typename Tag::type;

  GpuCustomTypeView() = default;

  GPU_HOST_DEVICE explicit GpuCustomTypeView(physical_type value)
      : value_(value) {}

  GPU_HOST_DEVICE physical_type operator*() const {
    return value_;
  }

  GPU_HOST_DEVICE bool operator==(const GpuCustomTypeView& other) const {
    return compare(other) == 0;
  }

  GPU_HOST_DEVICE bool operator!=(const GpuCustomTypeView& other) const {
    return compare(other) != 0;
  }

  GPU_HOST_DEVICE bool operator<(const GpuCustomTypeView& other) const {
    return compare(other) < 0;
  }

  GPU_HOST_DEVICE bool operator<=(const GpuCustomTypeView& other) const {
    return compare(other) <= 0;
  }

  GPU_HOST_DEVICE bool operator>(const GpuCustomTypeView& other) const {
    return compare(other) > 0;
  }

  GPU_HOST_DEVICE bool operator>=(const GpuCustomTypeView& other) const {
    return compare(other) >= 0;
  }

 private:
  GPU_HOST_DEVICE int compare(const GpuCustomTypeView& other) const {
    return GpuCustomTypeComparison<Tag>::compare(value_, other.value_);
  }

  physical_type value_{};
};

template <typename>
struct isGpuCustomTypeView : std::false_type {};

template <typename Tag>
struct isGpuCustomTypeView<GpuCustomTypeView<Tag>> : std::true_type {};

} // namespace facebook::velox::gpu

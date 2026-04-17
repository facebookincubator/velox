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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuda/iterator>
#include <thrust/sequence.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Aliases for cuco's 32 and 64 bit roaring bitmaps.
using Roaring32BitmapType = cuco::experimental::
    roaring_bitmap<cuda::std::uint32_t, rmm::mr::polymorphic_allocator<char>>;
using Roaring64BitmapType = cuco::experimental::
    roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

/// Helper functor to negate a boolean value.
struct NegateBool {
  __device__ bool operator()(bool b) const {
    return !b;
  }
};

} // namespace

/// Opaque wrapper class for cuco's 32 or 64 bit roaring bitmap depending on
/// the input inside a DV-v1 Puffin blob.
struct CudfDeletionVectorReader::RoaringBitmapImpl {
  std::unique_ptr<Roaring32BitmapType> bitmap32;
  std::unique_ptr<Roaring64BitmapType> bitmap64;

  template <class InputIt, class OutputIt>
  void contains_async(
      InputIt first,
      InputIt last,
      OutputIt out,
      rmm::cuda_stream_view stream) {
    if (bitmap64) {
      bitmap64->contains_async(first, last, out, stream);
    } else if (bitmap32) {
      bitmap32->contains_async(first, last, out, stream);
    }
  }

  bool empty() const {
    return !bitmap32 && !bitmap64;
  }
};

/// RoaringBitmapImpl is now fully defined. Define its deleter, and
/// CudfDeletionVectorReader's move constructor and destructor here.
void CudfDeletionVectorReader::RoaringBitmapDeleter::operator()(
    RoaringBitmapImpl* p) const {
  delete p;
}

CudfDeletionVectorReader::~CudfDeletionVectorReader() = default;
CudfDeletionVectorReader::CudfDeletionVectorReader(
    CudfDeletionVectorReader&&) noexcept = default;

/// Constructs the cuco roaring bitmap on the GPU from the roaringBitmapPayload.
template <CudfDeletionVectorReader::BitmapType BitSize>
void CudfDeletionVectorReader::buildBitmap(
    std::string_view roaringBitmapPayload,
    rmm::cuda_stream_view stream) {
  bitmap_ = std::unique_ptr<RoaringBitmapImpl, RoaringBitmapDeleter>(
      new RoaringBitmapImpl());
  auto const* bytes =
      reinterpret_cast<cuda::std::byte const*>(roaringBitmapPayload.data());
  if constexpr (BitSize == BitmapType::k32Bit) {
    bitmap_->bitmap32 = std::make_unique<Roaring32BitmapType>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
  } else {
    bitmap_->bitmap64 = std::make_unique<Roaring64BitmapType>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
  }

  // Mark the roaring bitmap as loaded
  loaded_ = true;
}

/// Instantiate the template for 32 and 64 bit roaring bitmaps
template void CudfDeletionVectorReader::buildBitmap<
    CudfDeletionVectorReader::BitmapType::k32Bit>(
    std::string_view,
    rmm::cuda_stream_view);
template void CudfDeletionVectorReader::buildBitmap<
    CudfDeletionVectorReader::BitmapType::k64Bit>(
    std::string_view,
    rmm::cuda_stream_view);

void CudfDeletionVectorReader::applyDeletes(
    cudf::mutable_column_view const& rowMask,
    std::size_t startRow,
    std::size_t numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  if (numRows == 0) {
    return;
  }

  // Load the cuco roaring bitmap
  loadBitmap(stream);

  // Return early if no bitmap or empty bitmap
  if (not bitmap_ or bitmap_->empty()) {
    return;
  }

  // Helper lambda to probe the cuco roaring bitmap (deletion vector)
  auto probeDeletionVector = [&](auto& bitmap) {
    // Deduce the roaring bitmap value type from the reference.
    using ValueType =
        typename std::remove_reference_t<decltype(bitmap)>::value_type;

    // Construct row index column if needed
    if (not rowIndices_ or rowIndices_->size() < numRows) {
      rowIndices_ = cudf::make_numeric_column(
          cudf::data_type{cudf::type_to_id<ValueType>()},
          static_cast<cudf::size_type>(numRows),
          cudf::mask_state::UNALLOCATED,
          stream,
          temp_mr);
    }

    // Generate row indices
    auto rowIndexIter = rowIndices_->mutable_view().begin<ValueType>();
    thrust::sequence(
        rmm::exec_policy_nosync(stream, temp_mr),
        rowIndexIter,
        rowIndexIter + numRows,
        static_cast<ValueType>(startRow));

    // Probe the roaring bitmap and negate output
    auto rowMaskIter = cuda::make_transform_output_iterator(
        rowMask.begin<bool>(), NegateBool{});
    bitmap.contains_async(
        rowIndexIter, rowIndexIter + numRows, rowMaskIter, stream);
  };

  // Probe the deletion vector
  if (bitmap_->bitmap32) {
    probeDeletionVector(*bitmap_->bitmap32);
  } else {
    probeDeletionVector(*bitmap_->bitmap64);
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg

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
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

using RoaringBitmapType = cuco::experimental::
    roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

struct NegateBool {
  __device__ bool operator()(bool b) const {
    return !b;
  }
};

} // namespace

// ---------------------------------------------------------------------------
// BitmapImpl — opaque wrapper kept out of the header
// ---------------------------------------------------------------------------

struct CudfDeletionVectorReader::BitmapImpl {
  std::unique_ptr<RoaringBitmapType> bitmap;
};

// ---------------------------------------------------------------------------
// Special members (all defined here where BitmapImpl is complete)
// ---------------------------------------------------------------------------

CudfDeletionVectorReader::CudfDeletionVectorReader(
    std::string filePath,
    uint64_t fileSizeInBytes,
    std::unordered_map<int32_t, std::string> lowerBounds,
    std::unordered_map<int32_t, std::string> upperBounds)
    : filePath_(std::move(filePath)),
      fileSizeInBytes_(fileSizeInBytes),
      lowerBounds_(std::move(lowerBounds)),
      upperBounds_(std::move(upperBounds)) {}

CudfDeletionVectorReader::~CudfDeletionVectorReader() = default;
CudfDeletionVectorReader::CudfDeletionVectorReader(
    CudfDeletionVectorReader&&) noexcept = default;
CudfDeletionVectorReader& CudfDeletionVectorReader::operator=(
    CudfDeletionVectorReader&&) noexcept = default;

// ---------------------------------------------------------------------------
// loadAndInitialize — load blob + parse envelope + build GPU bitmap
// ---------------------------------------------------------------------------

void CudfDeletionVectorReader::loadAndInitialize(rmm::cuda_stream_view stream) {
  dvBlobBytes_ = loadBlob();
  parseDvBlobEnvelope();

  CUDF_EXPECTS(
      dvPayloadSize_ > sizeof(uint64_t),
      "Deletion vector Roaring64 payload too small");

  auto const* payloadBytes = reinterpret_cast<cuda::std::byte const*>(
      dvBlobBytes_.data() + dvPayloadOffset_);
  bitmap_ = std::make_unique<BitmapImpl>();
  bitmap_->bitmap = std::make_unique<RoaringBitmapType>(
      payloadBytes, rmm::mr::polymorphic_allocator<char>{}, stream);
}

// ---------------------------------------------------------------------------
// applyDeletionVector — filter deleted rows from a table chunk
// ---------------------------------------------------------------------------

std::unique_ptr<cudf::table> CudfDeletionVectorReader::applyDeletionVector(
    cudf::table_view const& table,
    std::size_t startRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numRows = table.num_rows();
  if (numRows == 0 || !bitmap_ || !bitmap_->bitmap) {
    return std::make_unique<cudf::table>(table, stream, mr);
  }

  auto rowIndices =
      rmm::device_buffer(numRows * sizeof(std::size_t), stream, mr);
  auto* rowIndicesPtr = static_cast<std::size_t*>(rowIndices.data());
  thrust::sequence(
      rmm::exec_policy_nosync(stream),
      rowIndicesPtr,
      rowIndicesPtr + numRows,
      startRow);

  auto rowMask = rmm::device_buffer(numRows * sizeof(bool), stream, mr);
  auto rowMaskIter = thrust::make_transform_output_iterator(
      static_cast<bool*>(rowMask.data()), NegateBool{});

  bitmap_->bitmap->contains(
      rowIndicesPtr, rowIndicesPtr + numRows, rowMaskIter, stream);

  auto maskColumn = cudf::column_view(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      rowMask.data(),
      nullptr,
      0);

  return cudf::apply_boolean_mask(table, maskColumn, stream, mr);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg

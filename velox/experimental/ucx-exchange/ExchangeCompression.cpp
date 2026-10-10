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

#include "velox/experimental/ucx-exchange/ExchangeCompression.h"

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

#include <span>
#include <utility>

#if __has_include(<cudf/detail/fused_for.hpp>)
#define VELOX_UCX_HAS_FUSED_FOR 1
#include <cudf/detail/fused_for.hpp>
#else
#define VELOX_UCX_HAS_FUSED_FOR 0
#endif

namespace facebook::velox::ucx_exchange {
namespace {

#if VELOX_UCX_HAS_FUSED_FOR
constexpr std::size_t kFusedForTileBytes = 32 * 1024;

cudf::detail::fused_for_options fusedForOptions(
    ExchangeCompression compression) {
  VELOX_CHECK(
      compression == ExchangeCompression::kFusedForBitpacked ||
      compression == ExchangeCompression::kFusedForByteAligned);
  return cudf::detail::fused_for_options{
      kFusedForTileBytes,
      compression == ExchangeCompression::kFusedForBitpacked
          ? cudf::detail::fused_for_layout::bitpacked
          : cudf::detail::fused_for_layout::byte_aligned};
}
#endif

[[noreturn]] void fusedForUnavailable() {
  VELOX_UNSUPPORTED(
      "Fused FOR exchange compression is not available in this build");
}

PackedExchangePayload makeRawPayload(cudf::packed_columns packed) {
  const auto logicalDataSize = packed.gpu_data->size();
  return PackedExchangePayload{
      std::make_unique<cudf::packed_columns>(std::move(packed)),
      ExchangePayloadCodec::kNone,
      logicalDataSize,
      0};
}

#if VELOX_UCX_HAS_FUSED_FOR
PackedExchangePayload makeFusedForPayload(
    cudf::detail::fused_for_packed_columns packed) {
  const auto logicalDataSize = packed.logical_data_size;
  const auto segmentCount = packed.segment_count;
  auto metadata = wrapExchangePayloadMetadata(
      std::move(packed.metadata),
      ExchangePayloadCodec::kFusedFor,
      logicalDataSize,
      segmentCount);
  return PackedExchangePayload{
      std::make_unique<cudf::packed_columns>(
          std::move(metadata), std::move(packed.wire_data)),
      ExchangePayloadCodec::kFusedFor,
      logicalDataSize,
      segmentCount};
}
#endif

PackedExchangePayload compressCascaded(
    PackedExchangePayload raw,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  VELOX_CHECK_NOT_NULL(raw.packed);
  VELOX_CHECK(raw.codec == ExchangePayloadCodec::kNone);
  VELOX_CHECK_NOT_NULL(raw.packed->metadata);
  VELOX_CHECK_NOT_NULL(raw.packed->gpu_data);
  if (raw.logicalDataSize == 0) {
    return raw;
  }

  auto options = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  auto plan = cudf::experimental::make_pack_plan_builder(
                  *raw.packed,
                  options,
                  stream,
                  cudf::memory_resources{
                      temporaryMemoryResource, temporaryMemoryResource})
                  .build();
  const auto sizes = plan.sizes();
  VELOX_CHECK_EQ(sizes.uncompressed_payload_bytes, raw.logicalDataSize);
  auto output = std::make_unique<rmm::device_buffer>(
      sizes.payload_bytes, stream, outputMemoryResource);
  auto result = cudf::experimental::pack_into(
      plan,
      std::span<uint8_t>{static_cast<uint8_t*>(output->data()), output->size()},
      cudf::memory_resources{temporaryMemoryResource, temporaryMemoryResource});
  VELOX_CHECK_LE(result.payload_bytes, output->size());

  // The plan completes its reads before returning. Retain the ordinary packed
  // payload when compression does not reduce the bytes sent.
  if (result.payload_bytes >= raw.logicalDataSize) {
    return raw;
  }

  output->resize(result.payload_bytes, stream);
  auto metadata = wrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(std::move(result.metadata)),
      ExchangePayloadCodec::kCascaded,
      raw.logicalDataSize,
      0);
  return PackedExchangePayload{
      std::make_unique<cudf::packed_columns>(
          std::move(metadata), std::move(output)),
      ExchangePayloadCodec::kCascaded,
      raw.logicalDataSize,
      0};
}

std::unique_ptr<PackedTableWithStream> retainPackedPayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    int32_t numRows) {
  auto packedColumns =
      cudf::packed_columns(std::move(metadata), std::move(data));
  auto tableView = cudf::unpack(packedColumns);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});

  // Keep the storage and its consumer stream together. The producer's row
  // count is authoritative because a columnless packed table reports zero.
  return std::make_unique<PackedTableWithStream>(
      std::move(packedTable), stream, numRows);
}

} // namespace

bool fusedForAvailable() {
  return VELOX_UCX_HAS_FUSED_FOR;
}

ExchangeCompression parseExchangeCompression(std::string_view value) {
  if (value == "none") {
    return ExchangeCompression::kNone;
  }
  if (value == "fused-for-bitpacked") {
    VELOX_USER_CHECK(
        fusedForAvailable(),
        "Fused FOR exchange compression is not available in this build");
    return ExchangeCompression::kFusedForBitpacked;
  }
  if (value == "fused-for-byte-aligned") {
    VELOX_USER_CHECK(
        fusedForAvailable(),
        "Fused FOR exchange compression is not available in this build");
    return ExchangeCompression::kFusedForByteAligned;
  }
  if (value == "cascaded") {
    return ExchangeCompression::kCascaded;
  }
  VELOX_USER_FAIL(
      "Unsupported cuDF exchange compression '{}'. Expected none, "
      "fused-for-bitpacked, fused-for-byte-aligned, or cascaded",
      value);
}

PackedExchangePayload packExchangePayload(
    cudf::table_view input,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  if (input.num_columns() == 0) {
    return makeRawPayload(cudf::pack(input, stream, outputMemoryResource));
  }

  switch (compression) {
    case ExchangeCompression::kNone:
      return makeRawPayload(cudf::pack(input, stream, outputMemoryResource));
    case ExchangeCompression::kCascaded:
      return compressCascaded(
          makeRawPayload(cudf::pack(input, stream, outputMemoryResource)),
          stream,
          temporaryMemoryResource,
          outputMemoryResource);
    case ExchangeCompression::kFusedForBitpacked:
    case ExchangeCompression::kFusedForByteAligned:
#if VELOX_UCX_HAS_FUSED_FOR
      return makeFusedForPayload(
          cudf::detail::pack_fused_for(
              input,
              stream,
              outputMemoryResource,
              fusedForOptions(compression)));
#else
      fusedForUnavailable();
#endif
  }
  VELOX_FAIL("Unsupported exchange compression mode");
}

std::vector<PackedExchangePayload> splitExchangePayloads(
    cudf::table_view input,
    const std::vector<cudf::size_type>& splits,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  VELOX_CHECK_GT(input.num_columns(), 0);

  if (compression == ExchangeCompression::kFusedForBitpacked ||
      compression == ExchangeCompression::kFusedForByteAligned) {
#if VELOX_UCX_HAS_FUSED_FOR
    auto packed = cudf::detail::contiguous_split_fused_for(
        input,
        splits,
        stream,
        outputMemoryResource,
        fusedForOptions(compression));
    std::vector<PackedExchangePayload> result;
    result.reserve(packed.size());
    for (auto& partition : packed) {
      result.push_back(makeFusedForPayload(std::move(partition)));
    }
    return result;
#else
    fusedForUnavailable();
#endif
  }

  auto packed =
      cudf::contiguous_split(input, splits, stream, outputMemoryResource);
  std::vector<PackedExchangePayload> result;
  result.reserve(packed.size());
  for (auto& partition : packed) {
    auto payload = makeRawPayload(
        cudf::packed_columns(
            std::move(partition.data.metadata),
            std::move(partition.data.gpu_data)));
    if (compression == ExchangeCompression::kCascaded) {
      payload = compressCascaded(
          std::move(payload),
          stream,
          temporaryMemoryResource,
          outputMemoryResource);
    }
    result.push_back(std::move(payload));
  }
  return result;
}

std::unique_ptr<PackedTableWithStream> restoreExchangePayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    int32_t numRows) {
  auto envelope = unwrapExchangePayloadMetadata(std::move(metadata));
  switch (envelope.codec) {
    case ExchangePayloadCodec::kNone:
      return retainPackedPayload(
          std::move(envelope.cudfMetadata), std::move(data), stream, numRows);
    case ExchangePayloadCodec::kCascaded: {
      auto memoryResource = data->memory_resource();
      auto packed = cudf::experimental::packed_data_view{
          *envelope.cudfMetadata,
          std::span<uint8_t const>{
              static_cast<uint8_t const*>(data->data()), data->size()}};
      // Materialize synchronizes before returning, including on the
      // already-ready same-worker path, so both borrowed buffers remain alive
      // until then.
      auto table = cudf::experimental::materialize(
          packed,
          stream,
          cudf::memory_resources{memoryResource, memoryResource});
      return std::make_unique<PackedTableWithStream>(
          std::move(table), stream, envelope.logicalDataSize, numRows);
    }
    case ExchangePayloadCodec::kFusedFor:
#if VELOX_UCX_HAS_FUSED_FOR
    {
      auto memoryResource = data->memory_resource();
      auto decoded = cudf::detail::decode_fused_for(
          cudf::detail::fused_for_packed_columns{
              std::move(envelope.cudfMetadata),
              std::move(data),
              envelope.auxiliaryCount,
              envelope.logicalDataSize},
          stream,
          memoryResource);
      return retainPackedPayload(
          std::move(decoded.metadata),
          std::move(decoded.gpu_data),
          stream,
          numRows);
    }
#else
      fusedForUnavailable();
#endif
  }
  VELOX_FAIL("Unsupported exchange payload codec");
}

} // namespace facebook::velox::ucx_exchange

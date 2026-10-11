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

#include "velox/experimental/ucx-exchange/ExchangeCompressionWire.h"

#include <cudf/contiguous_split.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace facebook::velox::ucx_exchange {

struct PackedTableWithStream;

/// Compression mode used by one exchange output operator. The two fused
/// frame-of-reference (FOR) modes store transformed integers differently, but
/// both are identified as kFusedFor in the metadata sent to the receiver.
enum class ExchangeCompression {
  kNone,
  kFusedForBitpacked,
  kFusedForByteAligned,
  kCascaded,
};

/// Contains the cuDF metadata and GPU data to send, together with the codec the
/// receiver must use. If Cascaded does not make the data smaller, `codec` is
/// kNone and `packed` contains the original uncompressed data.
struct PackedExchangePayload {
  std::unique_ptr<cudf::packed_columns> packed;
  ExchangePayloadCodec codec{ExchangePayloadCodec::kNone};
  /// Size of the packed GPU data before compression.
  std::size_t logicalDataSize{0};
  /// Number of fused FOR segments. Zero for other codecs.
  std::size_t auxiliaryCount{0};
};

/// Converts a configuration value to a compression mode. Throws a user error
/// for an unknown mode or when fused FOR is not available in this build.
ExchangeCompression parseExchangeCompression(std::string_view value);

/// Packs one table for transfer using `compression`. The returned `codec`
/// tells the receiver how to decode the data. Cascaded may return kNone when
/// its compressed output is not smaller than the original packed data.
PackedExchangePayload packExchangePayload(
    cudf::table_view input,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource);

/// Splits `input` at the requested row offsets and packs each result using
/// `compression`. Fused FOR splits and compresses in one cuDF operation. Other
/// modes split first. Results remain in the same order as the requested ranges.
std::vector<PackedExchangePayload> splitExchangePayloads(
    cudf::table_view input,
    const std::vector<cudf::size_type>& splits,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource);

/// Reconstructs a received table using the codec stored in `metadata`.
/// Uncompressed and fused FOR data remain in packed cuDF storage. Cascaded
/// decompression returns an owning cudf::table. The result retains `stream`
/// and the row count reported by the producer.
std::unique_ptr<PackedTableWithStream> restoreExchangePayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    int32_t numRows);

} // namespace facebook::velox::ucx_exchange

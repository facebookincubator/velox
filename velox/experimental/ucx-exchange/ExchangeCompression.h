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

/// Selects the one compression codec used by an exchange output operator.
/// Fused FOR exposes two layouts as separate configuration values, but both
/// values use the same codec on the wire.
enum class ExchangeCompression {
  kNone,
  kFusedForBitpacked,
  kFusedForByteAligned,
  kCascaded,
};

/// Owns one packed exchange payload and describes the codec actually used.
/// Cascaded may retain an ordinary raw payload when compression does not reduce
/// its size, so `codec` can be kNone even when Cascaded was selected.
struct PackedExchangePayload {
  std::unique_ptr<cudf::packed_columns> packed;
  ExchangePayloadCodec codec{ExchangePayloadCodec::kNone};
  std::size_t logicalDataSize{0};
  std::size_t auxiliaryCount{0};
};

/// Returns true when the linked cuDF build provides fused FOR packing.
bool fusedForAvailable();

/// Parses a configured compression mode and rejects unavailable or unknown
/// values with a user error.
ExchangeCompression parseExchangeCompression(std::string_view value);

/// Packs one table with the selected codec. The codec choice is exclusive and
/// fixed for this call. The result carries the codec that was actually used.
PackedExchangePayload packExchangePayload(
    cudf::table_view input,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource);

/// Splits and packs one table with the selected codec. Fused FOR combines the
/// split and transform. Other modes split first and optionally compress each
/// ordinary packed partition. The result order matches the split order.
std::vector<PackedExchangePayload> splitExchangePayloads(
    cudf::table_view input,
    const std::vector<cudf::size_type>& splits,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource);

/// Restores one received payload according to the single codec recorded in its
/// metadata. Raw and fused FOR payloads retain packed storage. Cascaded returns
/// an owning table.
std::unique_ptr<PackedTableWithStream> restoreExchangePayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    int32_t numRows);

} // namespace facebook::velox::ucx_exchange

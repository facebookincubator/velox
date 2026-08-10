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

#include <cstddef>
#include <cstdint>

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg deletion-vector-v1 blob frame constants (Iceberg V3 spec), shared by
/// DeletionVectorWriter and DeletionVectorReader so the on-disk value can never
/// drift between the write and read paths. On disk a deletion vector blob is:
///   [length: 4B big-endian][magic][roaring bitmap][CRC-32: 4B big-endian]
/// where 'length' and the CRC-32 cover the magic bytes + bitmap. The magic is
/// Iceberg's MAGIC_NUMBER 0x6439D3D1 written little-endian, i.e. the on-disk
/// bytes D1 D3 39 64 (org.apache.iceberg.deletes.BitmapPositionDeleteIndex).
inline constexpr char kDeletionVectorMagic[] = {'\xD1', '\xD3', '\x39', '\x64'};
inline constexpr size_t kDeletionVectorMagicSize = 4;

// Sizes of the big-endian length prefix and trailing CRC-32 that bracket the
// magic + bitmap in the deletion-vector-v1 frame.
inline constexpr size_t kDeletionVectorLengthSize = 4;
inline constexpr size_t kDeletionVectorCrcSize = 4;

/// Largest Roaring64 group key (the high 32 bits of a position) the Iceberg
/// deletion-vector format can represent.
///
/// Iceberg's RoaringPositionBitmap stores the key as a signed 32-bit int and
/// caps it one below Integer.MAX_VALUE, so a spec-compliant reader cannot load
/// a larger key back. Keys at or above 2^31 are worse still: `key << 32`
/// shifts into the sign bit of the int64 position and yields a negative row
/// ordinal. Both the writer (via DeletionVectorWriter::kMaxPosition) and the
/// reader enforce this so neither can accept a blob the other would reject.
inline constexpr uint32_t kMaxRoaring64GroupKey = 2'147'483'646;

} // namespace facebook::velox::connector::hive::iceberg

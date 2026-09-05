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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace facebook::velox::connector::hive::hudi {

/// Six-byte marker that precedes every block in a Hudi log file.
inline constexpr std::string_view kHudiLogMagic{"#HUDI#"};

/// Version of the log file framing. Modern Hudi tables (table version 6+) write
/// V1, which carries a block type, header, content length and footer. V0 is a
/// legacy layout that omits those fields and is not supported here.
enum class LogFormatVersion : uint32_t {
  kV0 = 0,
  kV1 = 1,
};

/// Type of a Hudi log block, as stored in the 4-byte block-type field.
enum class HudiLogBlockType : uint32_t {
  kCommand = 0,
  kDelete = 1,
  kCorrupted = 2,
  kAvroData = 3,
  kHfileData = 4,
  kParquetData = 5,
  kCdcData = 6,
};

/// Ordinal identifying an entry in a block's header or footer metadata map.
enum class HudiLogBlockMetadataKey : uint32_t {
  kInstantTime = 0,
  kTargetInstantTime = 1,
  kSchema = 2,
  kCommandBlockType = 3,
  kCompactedBlockTimes = 4,
  kRecordPositions = 5,
  kBlockIdentifier = 6,
  kIsPartial = 7,
  kBaseFileInstantTimeOfRecordPositions = 8,
};

/// Sub-type of a command block, stored as a decimal string under the
/// CommandBlockType metadata key.
enum class HudiCommandBlockType : uint32_t {
  kRollback = 0,
};

/// Metadata map read from a block header or footer, keyed by ordinal.
using HudiLogBlockMetadata =
    std::unordered_map<HudiLogBlockMetadataKey, std::string>;

/// One parsed block of a Hudi log file. Holds the framing metadata and the raw
/// (still-encoded) content bytes; content decoding is performed by dedicated
/// decoders in later stages.
struct HudiLogBlock {
  /// Framing version of the enclosing log file.
  LogFormatVersion formatVersion{LogFormatVersion::kV1};

  /// Type of this block.
  HudiLogBlockType blockType{HudiLogBlockType::kCorrupted};

  /// Header metadata entries (e.g. instant time, writer schema).
  HudiLogBlockMetadata header;

  /// Footer metadata entries; usually empty in practice.
  HudiLogBlockMetadata footer;

  /// Raw, undecoded content bytes of the block.
  std::string content;

  /// Returns the commit/instant time this block was written at. Fails if the
  /// InstantTime header entry is absent.
  const std::string& instantTime() const;

  /// Returns the instant time targeted by a rollback command block, or nullopt
  /// when the TargetInstantTime header entry is absent.
  std::optional<std::string> targetInstantTime() const;

  /// Returns the writer Avro schema JSON, or nullopt when absent.
  std::optional<std::string> schemaJson() const;

  /// Returns the command sub-type, or nullopt when this is not a command block
  /// or the CommandBlockType header entry is absent.
  std::optional<HudiCommandBlockType> commandBlockType() const;

  /// Returns true when this block carries data records (Avro, Parquet or CDC).
  bool isDataBlock() const;

  /// Returns true when this block carries delete record keys.
  bool isDeleteBlock() const;

  /// Returns true when this block is a rollback command block.
  bool isRollbackBlock() const;
};

} // namespace facebook::velox::connector::hive::hudi

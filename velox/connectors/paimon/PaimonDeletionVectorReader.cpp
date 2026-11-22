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
#include "velox/connectors/paimon/PaimonDeletionVectorReader.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>

#include <folly/lang/Bits.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"

namespace facebook::velox::connector::paimon {
namespace {

// Paimon frames each deletion vector as [length][magic][bitmap][CRC]. The
// length and the CRC are big-endian int32s, and the length counts the magic
// and the bitmap. Paimon Java skips the CRC on read, and so does this reader.
constexpr uint64_t kFrameLengthBytes = 4;
constexpr uint64_t kMagicBytes = 4;
constexpr uint64_t kHeaderBytes = kFrameLengthBytes + kMagicBytes;
constexpr uint64_t kCrcBytes = 4;

// Version 1 frames (Java BitmapDeletionVector) hold a 32-bit RoaringBitmap
// after a big-endian magic number. Their deletion file length is the frame
// length.
constexpr uint32_t kBitmap32Magic = 1'581'511'376;

// Version 2 frames (Java Bitmap64DeletionVector, the Iceberg deletion vector
// layout) hold a Roaring64 bitmap after a little-endian magic number. Their
// deletion file length also counts the frame length and the CRC.
constexpr uint32_t kBitmap64Magic = 1'681'511'377;

} // namespace

PaimonDeletionVectorReader::PaimonDeletionVectorReader(
    const PaimonDeletionFile& deletionFile,
    memory::MemoryPool* pool,
    std::shared_ptr<const config::ConfigBase> connectorConfig,
    std::shared_ptr<filesystems::TokenProvider> tokenProvider)
    : connector::DeletionVectorReader(
          deletionFile.path,
          /*splitOffset=*/0,
          pool,
          std::move(connectorConfig),
          /*fileReadOps=*/{},
          std::move(tokenProvider)),
      deletionFile_(deletionFile) {
  // Paimon Java writes frames of at most 2 GB, so the length fits an int32 in
  // both frame versions.
  static constexpr uint64_t kMaxDeletionVectorLength =
      std::numeric_limits<int32_t>::max();
  VELOX_CHECK_LE(
      deletionFile_.length,
      kMaxDeletionVectorLength,
      "Paimon deletion vector length exceeds sanity limit: {}",
      deletionFile_.length);
  static constexpr uint64_t kMaxDeletionVectorCardinality = 10'000'000'000ULL;
  if (deletionFile_.cardinality.has_value()) {
    VELOX_CHECK_LE(
        *deletionFile_.cardinality,
        kMaxDeletionVectorCardinality,
        "Paimon deletion vector cardinality exceeds sanity limit: {}",
        *deletionFile_.cardinality);
  }
}

PaimonDeletionVectorReader::Bitmap PaimonDeletionVectorReader::readBitmap(
    ReadFile& file,
    const FileIoContext& ioContext) {
  const auto fileSize = file.size();
  const auto offset = deletionFile_.offset;
  VELOX_CHECK_LE(
      offset,
      fileSize,
      "Paimon deletion-vector offset {} exceeds file size {}.",
      offset,
      fileSize);

  // Reads the header and the bitmap in one read. The bitmap of a version 1
  // frame ends 4 bytes past the deletion file length, which leaves out the
  // length field, and that of a version 2 frame 4 bytes before it, since the
  // length counts the length field and the CRC. Reading to the later end takes
  // the CRC of a version 2 frame and may pass the frame, so the read stops at
  // the end of the file. A length too short for the magic number still reads
  // the header, so that the checks below report the frame.
  const auto numFrameBytes = std::min(
      std::max(deletionFile_.length, kMagicBytes) + kFrameLengthBytes,
      fileSize - offset);
  VELOX_CHECK_GE(
      numFrameBytes,
      kHeaderBytes,
      "Paimon deletion-vector frame header is truncated: {}",
      deletionFile_.path);
  auto frameBytes = readBytes(file, offset, numFrameBytes, ioContext);
  const auto* frame = frameBytes->as<char>();

  uint32_t frameLength;
  std::memcpy(&frameLength, frame, sizeof(uint32_t));
  frameLength = folly::Endian::big(frameLength);
  uint32_t magic;
  std::memcpy(&magic, frame + kFrameLengthBytes, sizeof(uint32_t));

  bool roaring64{false};
  if (folly::Endian::big(magic) == kBitmap32Magic) {
    VELOX_CHECK_EQ(
        frameLength,
        deletionFile_.length,
        "Paimon deletion-vector frame length does not match the deletion "
        "file length: {}",
        deletionFile_.path);
  } else if (folly::Endian::little(magic) == kBitmap64Magic) {
    VELOX_CHECK_EQ(
        frameLength + kFrameLengthBytes + kCrcBytes,
        deletionFile_.length,
        "Paimon deletion-vector frame length does not match the deletion "
        "file length: {}",
        deletionFile_.path);
    roaring64 = true;
  } else {
    VELOX_FAIL(
        "Paimon deletion vector has an unknown magic number: {}",
        folly::Endian::big(magic));
  }
  VELOX_CHECK_GE(
      frameLength,
      kMagicBytes,
      "Paimon deletion-vector frame is shorter than its magic number: {}",
      deletionFile_.path);

  // After the length checks, the read holds the whole bitmap unless the file
  // ends inside it.
  const uint64_t bitmapLength = frameLength - kMagicBytes;
  VELOX_CHECK_LE(
      bitmapLength,
      numFrameBytes - kHeaderBytes,
      "Paimon deletion-vector range [{}, {}) exceeds file size {}.",
      offset + kHeaderBytes,
      offset + kHeaderBytes + bitmapLength,
      fileSize);

  dwio::common::RoaringBitmapView bitmap(
      std::string_view(frame + kHeaderBytes, bitmapLength),
      roaring64 ? dwio::common::RoaringBitmapView::Format::k64Bit
                : dwio::common::RoaringBitmapView::Format::k32Bit);
  if (deletionFile_.cardinality.has_value()) {
    VELOX_CHECK_EQ(
        bitmap.cardinality(),
        *deletionFile_.cardinality,
        "Paimon deletion-vector cardinality mismatch: {}",
        deletionFile_.path);
  }
  return {std::move(frameBytes), std::move(bitmap)};
}

} // namespace facebook::velox::connector::paimon
